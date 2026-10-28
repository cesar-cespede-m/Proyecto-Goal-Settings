"""MV-13: comparar en cohorte común y promover sólo evidencia aprobada exacta."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from core.analysis.snapshot_recipe import snapshot_recipe
from core.contracts.promotion import PromotionCohortV1, PromotionEvaluationV1, PromotionReviewV1
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.prediction_batches import PredictionBatchService
from core.domain.value_policy import ValuePolicyService
from core.economics.scenarios import evaluate_scenario_version
from core.ml_engine.decision import selected_row_numbers
from core.ml_engine.scoring import ScoringService


class PromotionError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PromotionService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.capabilities = CapabilityService(repository)
        self.policies = ValuePolicyService(repository)
        self.values = DecisionValueService(repository)
        self.scoring = ScoringService(repository, storage_root)

    def cohort(self, cohort_id: str) -> PromotionCohortV1:
        row = self.connection.execute(
            "SELECT manifest_json,sha256 FROM promotion_cohorts WHERE id=?", (cohort_id,)
        ).fetchone()
        if row is None:
            raise PromotionError("COHORT_NOT_FOUND", "La cohorte sellada no existe")
        result = PromotionCohortV1.model_validate_json(row["manifest_json"])
        if result.id != cohort_id or canonical_hash(result) != row["sha256"]:
            raise PromotionError("COHORT_HASH_MISMATCH", "La cohorte cambió")
        return result

    def evaluation(self, evaluation_id: str) -> PromotionEvaluationV1:
        row = self.connection.execute(
            "SELECT evaluation_json,sha256 FROM promotion_evaluations WHERE id=?",
            (evaluation_id,),
        ).fetchone()
        if row is None:
            raise PromotionError("EVALUATION_NOT_FOUND", "La comparación no existe")
        result = PromotionEvaluationV1.model_validate_json(row["evaluation_json"])
        if result.id != evaluation_id or canonical_hash(result) != row["sha256"]:
            raise PromotionError("EVALUATION_HASH_MISMATCH", "La comparación cambió")
        return result

    def active_policy_id(self, capability_id: str) -> str | None:
        self.capabilities.get_capability(capability_id)
        row = self.connection.execute(
            "SELECT policy_version_id FROM capability_model_assignments "
            "WHERE capability_id=? ORDER BY rowid DESC LIMIT 1",
            (capability_id,),
        ).fetchone()
        return None if row is None else str(row["policy_version_id"])

    def list_cohorts(self, capability_id: str) -> list[PromotionCohortV1]:
        self.capabilities.get_capability(capability_id)
        rows = self.connection.execute(
            "SELECT id FROM promotion_cohorts WHERE capability_id=? ORDER BY rowid DESC",
            (capability_id,),
        ).fetchall()
        return [self.cohort(str(row["id"])) for row in rows]

    def evaluation_for_cohort(self, cohort_id: str) -> PromotionEvaluationV1 | None:
        self.cohort(cohort_id)
        row = self.connection.execute(
            "SELECT id FROM promotion_evaluations WHERE cohort_id=?", (cohort_id,)
        ).fetchone()
        return None if row is None else self.evaluation(str(row["id"]))

    def _policies(self, capability_id: str, current_id: str, candidate_id: str):
        current = self.policies.get(current_id)
        candidate = self.policies.get(candidate_id)
        if (
            current.capability_id != capability_id
            or candidate.capability_id != capability_id
            or current.id == candidate.id
            or not self.policies.is_review_current(current.id)
            or not self.policies.is_review_current(candidate.id)
            or current.content.economic_version_id != candidate.content.economic_version_id
            or current.content.economic_sha256 != candidate.content.economic_sha256
            or current.content.predictive_version_id != candidate.content.predictive_version_id
        ):
            raise PromotionError("POLICIES_NOT_COMPARABLE", "Revisa ambas políticas y economía")
        source = self.capabilities.get_cycle(current.content.cycle_id)
        update = self.capabilities.get_cycle(candidate.content.cycle_id)
        if update.source_cycle_id != source.id or update.capability_id != capability_id:
            raise PromotionError("CANDIDATE_SCOPE_MISMATCH", "La candidata no deriva del vigente")
        active = self.active_policy_id(capability_id)
        if active is not None and active != current.id:
            raise PromotionError("CURRENT_POLICY_CHANGED", "La versión vigente cambió")
        return current, candidate

    def seal(
        self,
        capability_id: str,
        *,
        current_policy_id: str,
        candidate_policy_id: str,
        filename: str,
        content: bytes,
        data_classification: str,
        provenance_basis: str,
        coverage_complete: bool,
        coverage_basis: str,
        observed_until: datetime,
        min_rows: int,
        min_f1_improvement: float,
        actor_label: str,
    ) -> PromotionCohortV1:
        if not actor_label.strip() or observed_until.tzinfo is None:
            raise PromotionError("SEAL_INPUT_REQUIRED", "Indica responsable y corte observado")
        if data_classification not in {"synthetic", "sandbox", "anonymized"}:
            raise PromotionError("DATA_KO", "No se admiten datos productivos reales")
        if not 1 <= len(provenance_basis.strip()) <= 500:
            raise PromotionError("PROVENANCE_REQUIRED", "Declara la procedencia autorizada")
        if not coverage_complete or not 1 <= len(coverage_basis.strip()) <= 500:
            raise PromotionError("COVERAGE_REQUIRED", "Confirma cobertura completa del resultado")
        input_hash = hashlib.sha256(content).hexdigest()
        if (
            self.connection.execute(
                "SELECT 1 FROM promotion_cohorts WHERE capability_id=? AND input_sha256=?",
                (capability_id, input_hash),
            ).fetchone()
            is not None
        ):
            raise PromotionError("COHORT_ALREADY_SEALED", "Este archivo ya tiene criterio sellado")
        current, candidate = self._policies(capability_id, current_policy_id, candidate_policy_id)
        self.scoring.approved_run(current.content.run_id)
        candidate_run = self.repository.get_run(candidate.content.run_id)
        if candidate_run["status"] != "COMPLETED":
            raise PromotionError("CANDIDATE_NOT_READY", "La candidata no terminó")
        strategies = [
            self.repository.get_strategy(
                self.repository.get_run(policy.content.run_id)["strategy_version_id"]
            )
            for policy in (current, candidate)
        ]
        recipes = [snapshot_recipe(self.repository, str(row["dataset_id"])) for row in strategies]
        if any(recipe is None for recipe in recipes):
            raise PromotionError("RECIPE_REQUIRED", "Ambos modelos necesitan receta verificable")
        entity = recipes[0]["spec"]["entity"]
        if recipes[1]["spec"]["entity"] != entity:
            raise PromotionError("ENTITY_MISMATCH", "Los IDs de entidad no son comparables")
        target = self.values.get_predictive_version(
            current.content.predictive_version_id
        ).spec.target_column
        frame = read_validated_frame(validate_csv_bytes(filename, content, IngestionLimits()))
        availability = recipes[0]["spec"].get("label_available_at_output_column")
        if (
            entity not in frame.columns
            or target not in frame.columns
            or availability is None
            or availability not in frame.columns
            or frame[entity].isna().any()
            or not frame[entity].is_unique
            or frame[target].isna().any()
        ):
            raise PromotionError("COHORT_INCOMPLETE", "Exige IDs únicos y resultados completos")
        dates = pd.to_datetime(frame[availability], utc=True, errors="coerce")
        if dates.isna().any() or dates.max().to_pydatetime() > observed_until:
            raise PromotionError("COHORT_NOT_MATURE", "Hay resultados sin ventana observada")
        ids = set(frame[entity].astype(str))
        batch_service = PredictionBatchService(self.repository, self.scoring.storage_root)
        intervention_ids = {
            case.case_id
            for batch in batch_service.list_batches(capability_id)
            for case in batch.cases
        }
        if ids & intervention_ids:
            raise PromotionError("COHORT_INTERVENTION_OVERLAP", "La cohorte repite casos de acción")
        for strategy in strategies:
            from core.domain.strategy import StrategyProposalV1

            proposal = StrategyProposalV1.model_validate_json(strategy["strategy_json"])
            training = self.scoring._training_frame(proposal)
            if training is None or entity not in training.columns:
                raise PromotionError("TRAINING_EVIDENCE_MISSING", "Falta fuente de entrenamiento")
            if ids & set(training[entity].dropna().astype(str)):
                raise PromotionError("COHORT_LEAKAGE", "La cohorte repite entidades de selección")
        now = datetime.now(UTC)
        cohort = PromotionCohortV1(
            id=str(uuid4()),
            capability_id=capability_id,
            current_policy_id=current.id,
            candidate_policy_id=candidate.id,
            current_policy_sha256=current.sha256,
            candidate_policy_sha256=candidate.sha256,
            input_sha256=input_hash,
            entity_ids_sha256=canonical_hash(sorted(ids)),
            rows=len(frame),
            entity_column=entity,
            target_column=target,
            data_classification=data_classification,
            provenance_basis=provenance_basis.strip(),
            coverage_complete=True,
            coverage_basis=coverage_basis.strip(),
            observed_until=observed_until,
            min_rows=min_rows,
            min_f1_improvement=min_f1_improvement,
            actor_label=actor_label.strip(),
            created_at=now,
        )
        with transaction(self.connection):
            self._policies(capability_id, current.id, candidate.id)
            if (
                self.connection.execute(
                    "SELECT 1 FROM promotion_cohorts WHERE capability_id=? AND input_sha256=?",
                    (capability_id, input_hash),
                ).fetchone()
                is not None
            ):
                raise PromotionError(
                    "COHORT_ALREADY_SEALED", "Este archivo ya tiene criterio sellado"
                )
            self.connection.execute(
                "INSERT INTO promotion_cohorts(id,capability_id,current_policy_id,"
                "candidate_policy_id,input_sha256,manifest_json,sha256,actor_label,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    cohort.id,
                    capability_id,
                    current.id,
                    candidate.id,
                    cohort.input_sha256,
                    canonical_json(cohort),
                    canonical_hash(cohort),
                    actor_label.strip(),
                    now.isoformat(),
                ),
            )
        return cohort

    def evaluate(self, cohort_id: str, *, filename: str, content: bytes, actor_label: str):
        cohort = self.cohort(cohort_id)
        if hashlib.sha256(content).hexdigest() != cohort.input_sha256:
            raise PromotionError("COHORT_INPUT_CHANGED", "Sube el archivo sellado exacto")
        row = self.connection.execute(
            "SELECT id FROM promotion_evaluations WHERE cohort_id=?", (cohort_id,)
        ).fetchone()
        if row is not None:
            return self.evaluation(str(row["id"]))
        current, candidate = self._policies(
            cohort.capability_id, cohort.current_policy_id, cohort.candidate_policy_id
        )
        if (
            current.sha256 != cohort.current_policy_sha256
            or candidate.sha256 != cohort.candidate_policy_sha256
        ):
            raise PromotionError("POLICY_CHANGED", "La política sellada cambió")
        current_scores = self.scoring.evaluate(
            current.content.run_id, filename=filename, content=content
        )
        candidate_scores = self.scoring.evaluate(
            candidate.content.run_id,
            filename=filename,
            content=content,
            _candidate_internal=True,
        )
        if (
            current_scores.input_sha256 != candidate_scores.input_sha256
            or current_scores.rows_scored != cohort.rows
            or candidate_scores.rows_scored != cohort.rows
            or current_scores.seen_in_training
            or candidate_scores.seen_in_training
            or current_scores.follow_up is None
            or candidate_scores.follow_up is None
            or current_scores.follow_up.rows_with_outcome != cohort.rows
            or candidate_scores.follow_up.rows_with_outcome != cohort.rows
            or current_scores.follow_up.f1 is None
            or candidate_scores.follow_up.f1 is None
        ):
            raise PromotionError("COMMON_EVIDENCE_INVALID", "Falta evidencia común completa")
        current_ids = {case.case_id for case in current_scores.cases}
        candidate_ids = {case.case_id for case in candidate_scores.cases}
        if (
            None in current_ids
            or current_ids != candidate_ids
            or canonical_hash(sorted(current_ids)) != cohort.entity_ids_sha256
        ):
            raise PromotionError("COMMON_COHORT_MISMATCH", "Los modelos vieron casos distintos")
        selected = []
        for policy, scores in ((current, current_scores), (candidate, candidate_scores)):
            selected.append(
                len(
                    selected_row_numbers(
                        policy.content.plan,
                        scores=[case.score for case in scores.cases],
                        row_numbers=[case.row_number for case in scores.cases],
                    )
                )
            )
        economic = self.values.get_economic_version(current.content.economic_version_id)
        current_scenario = evaluate_scenario_version(economic, selected_cases=selected[0])
        candidate_scenario = evaluate_scenario_version(economic, selected_cases=selected[1])
        current_f1 = current_scores.follow_up.f1
        candidate_f1 = candidate_scores.follow_up.f1
        acceptable_value = candidate_scenario.net_value >= current_scenario.net_value
        recommendation = (
            "INSUFFICIENT"
            if cohort.rows < cohort.min_rows
            else "CANDIDATE_BETTER"
            if candidate_f1 > current_f1 + cohort.min_f1_improvement and acceptable_value
            else "KEEP_CURRENT"
        )
        now = datetime.now(UTC)
        result = PromotionEvaluationV1(
            id=str(uuid4()),
            cohort_id=cohort.id,
            input_sha256=cohort.input_sha256,
            current_run_id=current.content.run_id,
            candidate_run_id=candidate.content.run_id,
            current_report_id=current_scores.report_id,
            candidate_report_id=candidate_scores.report_id,
            rows=cohort.rows,
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            current_selected=selected[0],
            candidate_selected=selected[1],
            current_scenario=current_scenario,
            candidate_scenario=candidate_scenario,
            recommendation=recommendation,
            actor_label=actor_label.strip(),
            created_at=now,
        )
        with transaction(self.connection):
            self._policies(cohort.capability_id, current.id, candidate.id)
            self.connection.execute(
                "INSERT INTO promotion_evaluations(id,cohort_id,evaluation_json,sha256,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    result.id,
                    cohort.id,
                    canonical_json(result),
                    canonical_hash(result),
                    now.isoformat(),
                ),
            )
        return result

    def review(
        self, evaluation_id: str, *, approved_hash: str, decision: str, actor_label: str
    ) -> PromotionReviewV1:
        result = self.evaluation(evaluation_id)
        if approved_hash != canonical_hash(result) or decision not in {"APPROVED", "REJECTED"}:
            raise PromotionError("REVIEW_MISMATCH", "Revisa la comparación exacta")
        if not actor_label.strip():
            raise PromotionError("ACTOR_REQUIRED", "Indica quién revisó")
        now = datetime.now(UTC)
        review = PromotionReviewV1(
            id=str(uuid4()),
            evaluation_id=evaluation_id,
            approved_hash=approved_hash,
            decision=decision,
            actor_label=actor_label.strip(),
            created_at=now,
        )
        with transaction(self.connection):
            cohort = self.cohort(result.cohort_id)
            self._policies(
                cohort.capability_id, cohort.current_policy_id, cohort.candidate_policy_id
            )
            self.connection.execute(
                "INSERT INTO promotion_reviews(id,evaluation_id,approved_hash,decision,"
                "actor_label,created_at) VALUES(?,?,?,?,?,?)",
                (
                    review.id,
                    evaluation_id,
                    approved_hash,
                    decision,
                    actor_label.strip(),
                    now.isoformat(),
                ),
            )
        return review

    def promote(self, evaluation_id: str, *, actor_label: str, approved_hash: str) -> str:
        result = self.evaluation(evaluation_id)
        if result.recommendation != "CANDIDATE_BETTER" or approved_hash != canonical_hash(result):
            raise PromotionError("PROMOTION_NOT_SUPPORTED", "La evidencia no favorece la candidata")
        cohort = self.cohort(result.cohort_id)
        with transaction(self.connection):
            existing = self.connection.execute(
                "SELECT id FROM capability_model_assignments WHERE evaluation_id=?",
                (evaluation_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["id"])
            latest = self.connection.execute(
                "SELECT id FROM promotion_cohorts WHERE capability_id=? "
                "ORDER BY rowid DESC LIMIT 1",
                (cohort.capability_id,),
            ).fetchone()
            if latest["id"] != cohort.id:
                raise PromotionError("COMPARISON_STALE", "Hay una cohorte posterior")
            current, candidate = self._policies(
                cohort.capability_id, cohort.current_policy_id, cohort.candidate_policy_id
            )
            row = self.connection.execute(
                "SELECT * FROM promotion_reviews WHERE evaluation_id=? ORDER BY rowid DESC LIMIT 1",
                (evaluation_id,),
            ).fetchone()
            if (
                row is None
                or row["decision"] != "APPROVED"
                or row["approved_hash"] != approved_hash
            ):
                raise PromotionError("HUMAN_REVIEW_REQUIRED", "Falta revisión humana exacta")
            _run, report, _proposal = self.scoring.approved_run(
                candidate.content.run_id, _for_promotion=True
            )
            if report["id"] != result.candidate_report_id:
                raise PromotionError("REPORT_CHANGED", "El reporte aprobado no es el evaluado")
            _current_run, current_report, _current_proposal = self.scoring.approved_run(
                current.content.run_id
            )
            if current_report["id"] != result.current_report_id:
                raise PromotionError("REPORT_CHANGED", "El reporte vigente no es el evaluado")
            if not actor_label.strip():
                raise PromotionError("ACTOR_REQUIRED", "Indica quién promueve")
            identifier = str(uuid4())
            self.connection.execute(
                "INSERT INTO capability_model_assignments(id,capability_id,previous_policy_id,"
                "policy_version_id,evaluation_id,review_id,actor_label,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    cohort.capability_id,
                    current.id,
                    candidate.id,
                    evaluation_id,
                    row["id"],
                    actor_label.strip(),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return identifier
