"""MV-12: datos nuevos abren una candidata aislada; nunca entrenan ni promueven."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.analysis.snapshot import ObservationCoverageV1
from core.analysis.snapshot_recipe import snapshot_recipe
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.prediction_batches import PredictionBatchService
from core.domain.retraining_service import RetrainingService
from core.domain.value_policy import ValuePolicyService
from core.ml_engine.scoring import ScoringService


class CapabilityLearningError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CandidateStarted:
    capability_id: str
    source_cycle_id: str
    cycle_id: str
    project_id: str
    dataset_id: str
    context_version_id: str
    predictive_version_id: str


class CapabilityLearningService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.storage_root = Path(storage_root)
        self.capabilities = CapabilityService(repository)
        self.values = DecisionValueService(repository)
        self.policies = ValuePolicyService(repository)
        self.retraining = RetrainingService(repository, storage_root)
        self.batches = PredictionBatchService(repository, storage_root)

    def _result(self, row: Any) -> CandidateStarted:
        cycle = self.capabilities.get_cycle(str(row["candidate_cycle_id"]))
        context = self.repository.get_latest_context(cycle.project_id)
        binding = self.values.get_cycle_binding(cycle.id)
        if context is None or binding is None:
            raise CapabilityLearningError("CANDIDATE_INCOMPLETE", "La candidata no está completa")
        return CandidateStarted(
            capability_id=cycle.capability_id,
            source_cycle_id=str(cycle.source_cycle_id),
            cycle_id=cycle.id,
            project_id=cycle.project_id,
            dataset_id=str(row["candidate_dataset_id"]),
            context_version_id=str(context["id"]),
            predictive_version_id=binding.predictive_version_id,
        )

    def start_from_snapshot(
        self,
        capability_id: str,
        *,
        source_run_id: str,
        filename: str,
        content: bytes,
        coverage: ObservationCoverageV1 | dict[str, Any],
        reason: str,
        reference_unintervened: bool,
        reference_basis: str,
        actor_label: str,
        data_authorized: bool,
        operation_id: str,
    ) -> CandidateStarted:
        if not data_authorized:
            raise CapabilityLearningError("DATA_AUTHORIZATION_REQUIRED", "Confirma los datos")
        if not reference_unintervened or not 1 <= len(reference_basis.strip()) <= 500:
            raise CapabilityLearningError(
                "REFERENCE_COHORT_REQUIRED",
                "Declara una cohorte de referencia sin intervención y su procedencia",
            )
        if (
            not 1 <= len(reason.strip()) <= 200
            or not 1 <= len(actor_label.strip()) <= 200
            or not 1 <= len(operation_id.strip()) <= 190
        ):
            raise CapabilityLearningError(
                "MISSING_DECISION", "Indica motivo, responsable y operación"
            )
        coverage_model = ObservationCoverageV1.model_validate(coverage)
        data_hash = hashlib.sha256(content).hexdigest()
        request_hash = canonical_hash(
            {
                "capability_id": capability_id,
                "source_run_id": source_run_id,
                "filename": filename,
                "source_data_sha256": data_hash,
                "coverage": coverage_model.model_dump(mode="json"),
                "reason": reason.strip(),
                "reference_unintervened": reference_unintervened,
                "reference_basis": reference_basis.strip(),
                "actor_label": actor_label.strip(),
            }
        )
        existing = self.connection.execute(
            "SELECT * FROM capability_update_operations WHERE operation_id=?", (operation_id,)
        ).fetchone()
        if existing is not None:
            if existing["request_sha256"] != request_hash:
                raise CapabilityLearningError("OPERATION_CONFLICT", "La operación cambió")
            return self._result(existing)
        duplicate = self.connection.execute(
            "SELECT * FROM capability_update_operations WHERE capability_id=? "
            "AND source_run_id=? AND source_data_sha256=?",
            (capability_id, source_run_id, data_hash),
        ).fetchone()
        if duplicate is not None:
            if duplicate["request_sha256"] != request_hash:
                raise CapabilityLearningError(
                    "COHORT_ALREADY_USED", "Esta cohorte ya tiene otra propuesta de actualización"
                )
            return self._result(duplicate)

        source_cycle = self.capabilities.get_cycle_for_project(
            self.repository.get_run_project_id(source_run_id)
        )
        if source_cycle is None or source_cycle.capability_id != capability_id:
            raise CapabilityLearningError("SOURCE_SCOPE_MISMATCH", "El modelo es de otra capacidad")
        policies = self.policies.list_versions(capability_id)
        if (
            not policies
            or policies[-1].content.run_id != source_run_id
            or not self.policies.is_review_current(policies[-1].id)
        ):
            raise CapabilityLearningError("POLICY_NOT_CURRENT", "Revisa la política vigente")
        ScoringService(self.repository, self.storage_root).approved_run(source_run_id)
        binding = self.values.get_cycle_binding(source_cycle.id)
        if binding is None:
            raise CapabilityLearningError("SOURCE_BINDING_REQUIRED", "Falta contrato predictivo")
        strategy = self.repository.get_strategy(
            self.repository.get_run(source_run_id)["strategy_version_id"]
        )
        source_dataset = self.repository.get_dataset(str(strategy["dataset_id"]))
        classification = json.loads(source_dataset["profile_json"]).get("data_classification")
        if classification not in {"synthetic", "sandbox", "anonymized"}:
            raise CapabilityLearningError("DATA_KO", "No se admiten datos productivos reales")
        recipe = snapshot_recipe(self.repository, str(strategy["dataset_id"]))
        if recipe is None:
            raise CapabilityLearningError("SNAPSHOT_RECIPE_REQUIRED", "Falta receta de la cohorte")
        entity_column = recipe["spec"]["entity"]
        events = read_validated_frame(validate_csv_bytes(filename, content, IngestionLimits()))
        if entity_column not in events.columns:
            raise CapabilityLearningError("ENTITY_COLUMN_REQUIRED", "Falta el ID de entidad")
        incoming = set(events[entity_column].dropna().astype(str))
        used = {
            case.case_id
            for batch in self.batches.list_batches(capability_id)
            for case in batch.cases
        }
        if incoming & used:
            raise CapabilityLearningError(
                "INTERVENTION_COHORT_OVERLAP",
                "La actualización debe usar una cohorte separada de los casos intervenidos",
            )
        started = self.retraining.start_from_snapshot(
            source_run_id=source_run_id,
            filename=filename,
            content=content,
            coverage=coverage_model,
            actor_label=actor_label,
            data_authorized=True,
        )
        try:
            cycle = self.capabilities.adopt_update_project(
                capability_id,
                source_cycle_id=source_cycle.id,
                project_id=started.project_id,
                reason=reason.strip(),
                actor_label=actor_label.strip(),
                operation_id=f"{operation_id}:cycle",
            )
            self.values.bind_predictive_to_cycle(
                capability_id,
                cycle_id=cycle.id,
                predictive_version_id=binding.predictive_version_id,
                actor_label=actor_label.strip(),
                operation_id=f"{operation_id}:binding",
            )
            with transaction(self.connection):
                self.connection.execute(
                    "INSERT INTO capability_update_operations(operation_id,capability_id,"
                    "source_run_id,source_data_sha256,request_sha256,candidate_cycle_id,"
                    "candidate_dataset_id,actor_label,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        operation_id,
                        capability_id,
                        source_run_id,
                        data_hash,
                        request_hash,
                        cycle.id,
                        started.dataset_id,
                        actor_label.strip(),
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception as error:
            self.repository.archive_project(started.project_id, actor_label=actor_label.strip())
            raise CapabilityLearningError(
                "CANDIDATE_INCOMPLETE",
                "La candidata quedó archivada e incompleta; el modelo vigente sigue utilizable",
            ) from error
        return CandidateStarted(
            capability_id=capability_id,
            source_cycle_id=source_cycle.id,
            cycle_id=cycle.id,
            project_id=started.project_id,
            dataset_id=started.dataset_id,
            context_version_id=started.context_version_id,
            predictive_version_id=binding.predictive_version_id,
        )
