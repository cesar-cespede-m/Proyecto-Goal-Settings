"""MV-11: trazabilidad privada de predicción, asignación, actuación y resultado."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from core.contracts.prediction_batch import (
    ActionObservationV1,
    BatchCaseV1,
    BatchSummaryV1,
    PredictionBatchV1,
)
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.errors import DomainError
from core.domain.value_policy import ValuePolicyService
from core.ml_engine.decision import selected_row_numbers
from core.ml_engine.scoring import ScoringResultV1, ScoringService


class PredictionBatchError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise PredictionBatchError("INVALID_INPUT", f"Indica {name}")
    return value.strip()


class PredictionBatchService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.policies = ValuePolicyService(repository)
        self.scoring = ScoringService(repository, storage_root)

    def get(self, batch_id: str) -> PredictionBatchV1:
        row = self.connection.execute(
            "SELECT batch_json,sha256 FROM prediction_batches WHERE id=?", (batch_id,)
        ).fetchone()
        if row is None:
            raise PredictionBatchError("BATCH_NOT_FOUND", "El lote no existe")
        batch = PredictionBatchV1.model_validate_json(row["batch_json"])
        if canonical_hash(batch) != row["sha256"] or batch.id != batch_id:
            raise PredictionBatchError("BATCH_HASH_MISMATCH", "El lote cambió")
        return batch

    def list_batches(self, capability_id: str) -> list[PredictionBatchV1]:
        self.policies.capabilities.get_capability(capability_id)
        rows = self.connection.execute(
            "SELECT id FROM prediction_batches WHERE capability_id=? ORDER BY created_at,id",
            (capability_id,),
        ).fetchall()
        return [self.get(str(row["id"])) for row in rows]

    def _replay(self, operation: str, event: str, request_hash: str) -> str | None:
        row = self.connection.execute(
            "SELECT * FROM prediction_batch_operations WHERE operation_id=?", (operation,)
        ).fetchone()
        if row is None:
            return None
        if row["event_type"] != event or row["request_sha256"] != request_hash:
            raise PredictionBatchError("OPERATION_CONFLICT", "La operación ya tiene otro efecto")
        return str(row["entity_id"])

    def _record_operation(self, operation: str, event: str, entity: str, request_hash: str):
        self.connection.execute(
            "INSERT INTO prediction_batch_operations(operation_id,event_type,entity_id,"
            "request_sha256,created_at) VALUES(?,?,?,?,?)",
            (operation, event, entity, request_hash, datetime.now(UTC).isoformat()),
        )

    def _scoring_event(self, event_id: str, scoring: ScoringResultV1) -> None:
        row = self.connection.execute(
            "SELECT entity_id,detail_json,event_type FROM audit_events WHERE id=?", (event_id,)
        ).fetchone()
        if row is None or row["event_type"] != "MODEL_USED_FOR_SCORING":
            raise PredictionBatchError("SCORING_EVIDENCE_REQUIRED", "Falta uso auditado del modelo")
        detail = json.loads(row["detail_json"])
        if (
            row["entity_id"] != scoring.run_id
            or detail.get("input_sha256") != scoring.input_sha256
            or detail.get("result_sha256") != canonical_hash(scoring)
            or detail.get("report_id") != scoring.report_id
            or detail.get("rows_scored") != scoring.rows_scored
        ):
            raise PredictionBatchError(
                "SCORING_EVIDENCE_MISMATCH", "El lote no coincide con el uso"
            )

    def register(
        self,
        capability_id: str,
        *,
        policy_version_id: str,
        scoring: ScoringResultV1,
        scoring_event_id: str,
        period: str,
        data_classification: str,
        provenance_basis: str,
        actor_label: str,
        operation_id: str,
    ) -> PredictionBatchV1:
        actor = _required(actor_label, "quién registró el lote")
        operation = _required(operation_id, "operación")
        if data_classification not in {"synthetic", "sandbox", "anonymized"}:
            raise PredictionBatchError("DATA_KO", "No se admiten datos productivos reales")
        basis = _required(provenance_basis, "procedencia autorizada")
        request = canonical_hash(
            {
                "capability_id": capability_id,
                "policy_version_id": policy_version_id,
                "scoring_sha256": canonical_hash(scoring),
                "scoring_event_id": scoring_event_id,
                "period": period,
                "data_classification": data_classification,
                "provenance_basis": basis,
                "actor_label": actor,
            }
        )
        with transaction(self.connection):
            replay = self._replay(operation, "BATCH_REGISTERED", request)
            if replay is not None:
                return self.get(replay)
            policy = self.policies.get(policy_version_id)
            if policy.capability_id != capability_id or not self.policies.is_review_current(
                policy.id
            ):
                raise PredictionBatchError("POLICY_NOT_CURRENT", "Revisa la política vigente")
            if scoring.run_id != policy.content.run_id:
                raise PredictionBatchError("RUN_MISMATCH", "El lote usa otro modelo")
            _run, report, _proposal = self.scoring.approved_run(scoring.run_id)
            if scoring.report_id != report["id"]:
                raise PredictionBatchError("REPORT_MISMATCH", "El reporte cambió")
            self._scoring_event(scoring_event_id, scoring)
            if scoring.seen_in_training or scoring.rows_scored != len(scoring.cases):
                raise PredictionBatchError("NON_FRESH_BATCH", "Usa casos nuevos y completos")
            if any(case.case_id is None for case in scoring.cases):
                raise PredictionBatchError("CASE_ID_REQUIRED", "Cada caso necesita un ID")
            selected = selected_row_numbers(
                policy.content.plan,
                scores=[case.score for case in scoring.cases],
                row_numbers=[case.row_number for case in scoring.cases],
            )
            now = datetime.now(UTC)
            batch = PredictionBatchV1(
                id=str(uuid4()),
                capability_id=capability_id,
                cycle_id=policy.content.cycle_id,
                policy_version_id=policy.id,
                policy_sha256=policy.sha256,
                run_id=scoring.run_id,
                report_id=scoring.report_id,
                scoring_event_id=scoring_event_id,
                scoring_input_sha256=scoring.input_sha256,
                period=period,
                data_classification=data_classification,
                provenance_basis=basis,
                cases=[
                    BatchCaseV1(
                        case_id=str(case.case_id),
                        row_number=case.row_number,
                        rank=case.rank,
                        score=case.score,
                        assigned=case.row_number in selected,
                    )
                    for case in scoring.cases
                ],
                actor_label=actor,
                created_at=now,
            )
            self.connection.execute(
                "INSERT INTO prediction_batches(id,capability_id,policy_version_id,"
                "scoring_event_id,period,batch_json,sha256,actor_label,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    batch.id,
                    capability_id,
                    policy.id,
                    scoring_event_id,
                    period,
                    canonical_json(batch),
                    canonical_hash(batch),
                    actor,
                    now.isoformat(),
                ),
            )
            self._record_operation(operation, "BATCH_REGISTERED", batch.id, request)
            self.repository._audit(
                project_id=self.repository.get_run_project_id(scoring.run_id),
                event_type="PREDICTION_BATCH_REGISTERED",
                entity_type="PREDICTION_BATCH",
                entity_id=batch.id,
                detail={
                    "policy_version_id": policy.id,
                    "period": period,
                    "cases": len(batch.cases),
                    "assigned": sum(case.assigned for case in batch.cases),
                    "input_sha256": scoring.input_sha256,
                },
            )
        return batch

    def observations(self, batch_id: str) -> list[ActionObservationV1]:
        self.get(batch_id)
        rows = self.connection.execute(
            "SELECT event_json FROM action_observations WHERE batch_id=? ORDER BY rowid",
            (batch_id,),
        ).fetchall()
        return [ActionObservationV1.model_validate_json(row["event_json"]) for row in rows]

    def record(
        self,
        batch_id: str,
        *,
        case_id: str,
        event_type: str,
        outcome_positive: bool | None = None,
        occurred_at: datetime,
        actor_label: str,
        operation_id: str,
    ) -> ActionObservationV1:
        actor = _required(actor_label, "quién registró el hecho")
        operation = _required(operation_id, "operación")
        identifier = _required(case_id, "ID de caso")
        request = canonical_hash(
            {
                "batch_id": batch_id,
                "case_id": identifier,
                "event_type": event_type,
                "outcome_positive": outcome_positive,
                "occurred_at": occurred_at.isoformat(),
                "actor_label": actor,
            }
        )
        with transaction(self.connection):
            replay = self._replay(operation, "ACTION_OBSERVED", request)
            if replay is not None:
                row = self.connection.execute(
                    "SELECT event_json FROM action_observations WHERE id=?", (replay,)
                ).fetchone()
                return ActionObservationV1.model_validate_json(row["event_json"])
            batch = self.get(batch_id)
            period_start = datetime.fromisoformat(f"{batch.period}-01T00:00:00+00:00")
            if occurred_at.tzinfo is None or occurred_at < period_start:
                raise PredictionBatchError(
                    "INVALID_EVENT_TIME", "El hecho debe ocurrir dentro o después del período"
                )
            case = next((item for item in batch.cases if item.case_id == identifier), None)
            if case is None:
                raise PredictionBatchError("UNKNOWN_CASE", "El caso no pertenece al lote")
            previous = [item for item in self.observations(batch_id) if item.case_id == identifier]
            if event_type == "ACTED":
                if (
                    not case.assigned
                    or any(item.event_type == "ACTED" for item in previous)
                    or any(item.event_type == "OUTCOME_OBSERVED" for item in previous)
                ):
                    raise PredictionBatchError("ACTION_NOT_ALLOWED", "No hay asignación pendiente")
            elif event_type in {"OUTCOME_PENDING", "OUTCOME_OBSERVED"}:
                if any(item.event_type == "OUTCOME_OBSERVED" for item in previous):
                    raise PredictionBatchError(
                        "OUTCOME_ALREADY_OBSERVED", "El resultado ya es final"
                    )
            else:
                raise PredictionBatchError("INVALID_EVENT", "El hecho no está permitido")
            now = datetime.now(UTC)
            event = ActionObservationV1(
                id=str(uuid4()),
                batch_id=batch_id,
                case_id=identifier,
                event_type=event_type,
                outcome_positive=outcome_positive,
                occurred_at=occurred_at,
                actor_label=actor,
                created_at=now,
            )
            self.connection.execute(
                "INSERT INTO action_observations(id,batch_id,case_id,event_type,event_json,"
                "actor_label,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    event.id,
                    batch_id,
                    identifier,
                    event_type,
                    canonical_json(event),
                    actor,
                    now.isoformat(),
                ),
            )
            self._record_operation(operation, "ACTION_OBSERVED", event.id, request)
            self.repository._audit(
                project_id=self.repository.get_run_project_id(batch.run_id),
                event_type="ACTION_OBSERVATION_RECORDED",
                entity_type="PREDICTION_BATCH",
                entity_id=batch.id,
                detail={"event_type": event_type, "period": batch.period, "count": 1},
            )
        return event

    def summary(self, batch_id: str) -> BatchSummaryV1:
        batch = self.get(batch_id)
        events = self.observations(batch_id)
        acted = {item.case_id for item in events if item.event_type == "ACTED"}
        outcomes = {
            item.case_id: item
            for item in events
            if item.event_type in {"OUTCOME_PENDING", "OUTCOME_OBSERVED"}
        }
        assigned = sum(case.assigned for case in batch.cases)
        observed = [item for item in outcomes.values() if item.event_type == "OUTCOME_OBSERVED"]
        return BatchSummaryV1(
            batch_id=batch.id,
            period=batch.period,
            cases=len(batch.cases),
            assigned=assigned,
            acted=len(acted),
            assigned_not_acted=assigned - len(acted),
            observed=len(observed),
            pending=len(batch.cases) - len(observed),
            positive_observed=sum(item.outcome_positive is True for item in observed),
        )
