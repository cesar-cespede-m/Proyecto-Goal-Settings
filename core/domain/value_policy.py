"""MV-10: política sobre validation y escenario declarado, con revisión separada."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from core.contracts.value_policy import PolicyContentV1, PolicyReviewV1, PolicyVersionV1
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.gates import has_training_approval
from core.domain.predictive_guard import require_capability_training_binding
from core.economics.scenarios import evaluate_scenario_version
from core.ml_engine.decision import by_capacity
from core.ml_engine.models import EvaluationSummaryV1


class ValuePolicyError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise ValuePolicyError("INVALID_LABEL", f"Indica {name}")
    return value.strip()


class ValuePolicyService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.capabilities = CapabilityService(repository)
        self.values = DecisionValueService(repository)

    def get(self, policy_id: str) -> PolicyVersionV1:
        row = self.connection.execute(
            "SELECT * FROM policy_versions WHERE id=?", (policy_id,)
        ).fetchone()
        if row is None:
            raise ValuePolicyError("POLICY_NOT_FOUND", "La política no existe")
        content = PolicyContentV1.model_validate_json(row["content_json"])
        if (
            canonical_hash(content) != row["sha256"]
            or content.capability_id != row["capability_id"]
        ):
            raise ValuePolicyError("POLICY_HASH_MISMATCH", "La política cambió")
        return PolicyVersionV1(
            id=row["id"],
            capability_id=row["capability_id"],
            version_no=row["version_no"],
            based_on_version_id=row["based_on_version_id"],
            sha256=row["sha256"],
            content=content,
            actor_label=row["actor_label"],
            created_at=row["created_at"],
        )

    def list_versions(self, capability_id: str) -> list[PolicyVersionV1]:
        self.capabilities.get_capability(capability_id)
        rows = self.connection.execute(
            "SELECT id FROM policy_versions WHERE capability_id=? ORDER BY version_no",
            (capability_id,),
        ).fetchall()
        return [self.get(row["id"]) for row in rows]

    def latest_review(self, policy_id: str) -> PolicyReviewV1 | None:
        self.get(policy_id)
        row = self.connection.execute(
            "SELECT * FROM policy_review_events WHERE policy_version_id=? "
            "ORDER BY created_at DESC,rowid DESC LIMIT 1",
            (policy_id,),
        ).fetchone()
        return None if row is None else PolicyReviewV1.model_validate(dict(row))

    def _latest_economic(self, capability_id: str):
        versions = self.values.list_economic_versions(capability_id)
        if not versions:
            raise ValuePolicyError("ECONOMIC_SCENARIO_REQUIRED", "Declara un escenario económico")
        return versions[-1]

    def _content(
        self, capability_id: str, cycle_id: str, run_id: str, economic_version_id: str
    ) -> PolicyContentV1:
        self.capabilities.require_scope(capability_id, cycle_id=cycle_id, run_id=run_id)
        run = self.repository.get_run(run_id)
        if run["status"] != "COMPLETED" or not run["selected_model"]:
            raise ValuePolicyError("MODEL_NOT_RECOMMENDED", "Falta un modelo recomendado")
        strategy = self.repository.get_strategy(str(run["strategy_version_id"]))
        if not has_training_approval(self.connection, str(strategy["id"])):
            raise ValuePolicyError(
                "MODEL_APPROVAL_NOT_CURRENT", "La aprobación del modelo no está vigente"
            )
        binding = require_capability_training_binding(self.repository, str(strategy["id"]))
        if binding is None or binding.cycle_id != cycle_id:
            raise ValuePolicyError(
                "MODEL_BINDING_REQUIRED", "El modelo no corresponde a este ciclo"
            )
        economic = self._latest_economic(capability_id)
        if (
            economic.id != economic_version_id
            or not economic.confirmed_human
            or economic.missing_fields
        ):
            raise ValuePolicyError("ECONOMIC_SCENARIO_NOT_CURRENT", "Confirma el escenario vigente")
        scenario = economic.scenario
        if scenario.capacity_cases is None:
            raise ValuePolicyError(
                "CAPACITY_REQUIRED", "Declara capacidad para seleccionar la política"
            )
        evaluation = EvaluationSummaryV1.model_validate_json(run["candidate_metrics_json"])
        if evaluation.selection.selected_model != run["selected_model"]:
            raise ValuePolicyError("MODEL_EVIDENCE_MISMATCH", "La selección no coincide con el run")
        plan = by_capacity(
            evaluation,
            cases_per_period=scenario.capacity_cases,
            expected_cases_per_period=scenario.planned_cases,
        )
        money = evaluate_scenario_version(economic, selected_cases=plan.to_attend)
        if money.status != "scenario_estimate":
            raise ValuePolicyError(
                "ECONOMIC_SCENARIO_INCOMPLETE", "El escenario no permite cálculo"
            )
        return PolicyContentV1(
            capability_id=capability_id,
            cycle_id=cycle_id,
            run_id=run_id,
            strategy_fingerprint=str(strategy["fingerprint"]),
            selected_model=str(run["selected_model"]),
            predictive_version_id=binding.predictive_version_id,
            economic_version_id=economic.id,
            economic_sha256=economic.sha256,
            selection_artifact_sha256=plan.selection_artifact_sha256,
            plan=plan,
            economic_evaluation=money,
        )

    def _replay(self, operation_id: str, event: str, request_sha: str) -> str | None:
        row = self.connection.execute(
            "SELECT * FROM policy_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        if row["event_type"] != event or row["request_sha256"] != request_sha:
            raise ValuePolicyError("POLICY_OPERATION_CONFLICT", "La operación ya tiene otro efecto")
        return str(row["entity_id"])

    def _operation(
        self, operation_id: str, event: str, entity_id: str, request_sha: str, actor: str
    ):
        self.connection.execute(
            "INSERT INTO policy_operations(operation_id,event_type,entity_id,request_sha256,"
            "actor_label,created_at) VALUES(?,?,?,?,?,?)",
            (operation_id, event, entity_id, request_sha, actor, datetime.now(UTC).isoformat()),
        )

    def propose(
        self,
        capability_id: str,
        *,
        cycle_id: str,
        run_id: str,
        economic_version_id: str,
        based_on_version_id: str | None = None,
        actor_label: str,
        operation_id: str,
        verify_current: Callable[[], None] | None = None,
    ) -> PolicyVersionV1:
        actor, operation = _label(actor_label, "quién propone"), _label(operation_id, "operación")
        request = canonical_hash(
            {
                "capability_id": capability_id,
                "cycle_id": cycle_id,
                "run_id": run_id,
                "economic_version_id": economic_version_id,
                "based_on_version_id": based_on_version_id,
                "actor_label": actor,
            }
        )
        with transaction(self.connection):
            replay = self._replay(operation, "POLICY_PROPOSED", request)
            if replay is not None:
                return self.get(replay)
        content = self._content(capability_id, cycle_id, run_id, economic_version_id)
        with transaction(self.connection):
            replay = self._replay(operation, "POLICY_PROPOSED", request)
            if replay is not None:
                return self.get(replay)
            if verify_current is not None:
                verify_current()
            if canonical_hash(
                self._content(capability_id, cycle_id, run_id, economic_version_id)
            ) != canonical_hash(content):
                raise ValuePolicyError("POLICY_INPUT_CHANGED", "La evidencia cambió al guardar")
            versions = self.list_versions(capability_id)
            latest = versions[-1] if versions else None
            if based_on_version_id != (latest.id if latest else None):
                raise ValuePolicyError(
                    "POLICY_VERSION_CONFLICT", "La propuesta debe partir de la última política"
                )
            policy_id = str(uuid4())
            self.connection.execute(
                "INSERT INTO policy_versions(id,capability_id,version_no,based_on_version_id,"
                "content_json,sha256,actor_label,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    policy_id,
                    capability_id,
                    len(versions) + 1,
                    based_on_version_id,
                    canonical_json(content),
                    canonical_hash(content),
                    actor,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._operation(operation, "POLICY_PROPOSED", policy_id, request, actor)
        return self.get(policy_id)

    def review(
        self,
        policy_id: str,
        *,
        approved_hash: str,
        decision: str,
        actor_label: str,
        operation_id: str,
    ) -> PolicyReviewV1:
        actor, operation = _label(actor_label, "quién revisa"), _label(operation_id, "operación")
        if decision not in {"APPROVED", "REJECTED"}:
            raise ValuePolicyError("INVALID_REVIEW", "La revisión debe aprobar o rechazar")
        request = canonical_hash(
            {
                "policy_id": policy_id,
                "approved_hash": approved_hash,
                "decision": decision,
                "actor_label": actor,
            }
        )
        with transaction(self.connection):
            replay = self._replay(operation, "POLICY_REVIEWED", request)
            if replay is not None:
                row = self.connection.execute(
                    "SELECT * FROM policy_review_events WHERE id=?",
                    (replay,),
                ).fetchone()
                return PolicyReviewV1.model_validate(dict(row))
            policy = self.get(policy_id)
            if policy.sha256 != approved_hash:
                raise ValuePolicyError("POLICY_HASH_MISMATCH", "Revisa la versión exacta")
            related = [
                version
                for version in self.list_versions(policy.capability_id)
                if version.content.run_id == policy.content.run_id
            ]
            if related[-1].id != policy_id:
                raise ValuePolicyError("POLICY_VERSION_STALE", "Hay una política posterior")
            if decision == "APPROVED":
                current = self._content(
                    policy.capability_id,
                    policy.content.cycle_id,
                    policy.content.run_id,
                    policy.content.economic_version_id,
                )
                if canonical_hash(current) != policy.sha256:
                    raise ValuePolicyError("POLICY_EVIDENCE_STALE", "La evidencia cambió")
            review_id = str(uuid4())
            self.connection.execute(
                "INSERT INTO policy_review_events(id,policy_version_id,approved_hash,decision,"
                "actor_label,created_at) VALUES(?,?,?,?,?,?)",
                (
                    review_id,
                    policy_id,
                    approved_hash,
                    decision,
                    actor,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._operation(operation, "POLICY_REVIEWED", review_id, request, actor)
        return self.latest_review(policy_id)

    def is_review_current(self, policy_id: str) -> bool:
        policy = self.get(policy_id)
        review = self.latest_review(policy_id)
        if review is None or review.decision != "APPROVED" or review.approved_hash != policy.sha256:
            return False
        related = [
            version
            for version in self.list_versions(policy.capability_id)
            if version.content.run_id == policy.content.run_id
        ]
        if related[-1].id != policy_id:
            return False
        try:
            return (
                canonical_hash(
                    self._content(
                        policy.capability_id,
                        policy.content.cycle_id,
                        policy.content.run_id,
                        policy.content.economic_version_id,
                    )
                )
                == policy.sha256
            )
        except (ValueError, DomainError):
            return False
