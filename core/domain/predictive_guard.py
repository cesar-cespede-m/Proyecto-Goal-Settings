"""MV-09: binding y definición vigente de un ciclo, sólo lectura de SQLite.

El ejecutor verifica archivos/maduración y el split real fuera de esta comprobación.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.analysis.snapshot import SNAPSHOT_CUTOFF_COLUMN, SnapshotSpecV1, _outcome_name
from core.analysis.snapshot_recipe import snapshot_recipe
from core.contracts.decision_value import PredictiveSpecVersionV1
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json, strategy_fingerprint
from core.domain.capabilities import CapabilityService
from core.domain.context import BusinessContextV1
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.gates import is_strategy_current
from core.domain.strategy import StrategyProposalV1

_SHA256 = r"^[0-9a-f]{64}$"
_SEMANTIC_FIELDS = {
    "target_column",
    "positive_class",
    "prediction_moment",
    "primary_metric",
    "feature_decisions",
}


class PredictiveGuardError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapabilityTrainingBindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability-training-binding.v1"] = "capability-training-binding.v1"
    capability_id: str
    cycle_id: str
    project_id: str
    binding_id: str
    binding_sha256: str = Field(pattern=_SHA256)
    predictive_version_id: str
    predictive_sha256: str = Field(pattern=_SHA256)
    context_version_id: str
    context_sha256: str = Field(pattern=_SHA256)
    strategy_version_id: str
    strategy_fingerprint: str = Field(pattern=_SHA256)
    dataset_id: str
    dataset_sha256: str = Field(pattern=_SHA256)
    recipe_sha256: str = Field(pattern=_SHA256)
    snapshot_evidence_sha256: str = Field(pattern=_SHA256)
    time_column: Literal["snapshot_cutoff"] = "snapshot_cutoff"
    label_available_at_column: str

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self)


def _context(repository: WorkflowRepository, project_id: str):
    row = repository.get_latest_context(project_id)
    if row is None:
        raise PredictiveGuardError("PREDICTIVE_CONTEXT_REQUIRED", "Falta el contexto del ciclo")
    try:
        payload = json.loads(row["context_json"])
        if canonical_hash(payload) != row["context_sha256"]:
            raise PredictiveGuardError("PREDICTIVE_CONTEXT_HASH_MISMATCH", "El contexto cambió")
        context = BusinessContextV1.model_validate(payload["context"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, PredictiveGuardError):
            raise
        raise PredictiveGuardError(
            "PREDICTIVE_CONTEXT_INVALID", "El contexto no cumple el contrato"
        ) from error
    return row, context


def require_capability_predictive_context(
    repository: WorkflowRepository, *, capability_id: str, cycle_id: str
) -> PredictiveSpecVersionV1:
    """Comprueba el binding de un ciclo antes de proponer o ejecutar una estrategia."""

    capabilities = CapabilityService(repository)
    capabilities.require_scope(capability_id, cycle_id=cycle_id)
    cycle = capabilities.get_cycle(cycle_id)
    capability = capabilities.get_capability(capability_id)
    product = capabilities.get_product(capability.product_id)
    project = repository.connection.execute(
        "SELECT status FROM projects WHERE id=?", (cycle.project_id,)
    ).fetchone()
    if (
        capability.status != "active"
        or product.status != "active"
        or project["status"] == "ARCHIVED"
    ):
        raise PredictiveGuardError("CAPABILITY_NOT_ACTIVE", "El ciclo no está activo")
    values = DecisionValueService(repository)
    binding = values.get_cycle_binding(cycle_id)
    if binding is None:
        raise PredictiveGuardError(
            "PREDICTIVE_BINDING_REQUIRED", "Vincula una definición predictiva confirmada al ciclo"
        )
    version = values.get_predictive_version(binding.predictive_version_id)
    if version.capability_id != capability_id or binding.capability_id != capability_id:
        raise PredictiveGuardError(
            "PREDICTIVE_SCOPE_MISMATCH", "La definición sale de la capacidad"
        )
    if not version.confirmed_human or version.missing_fields or version.spec.missing_fields():
        raise PredictiveGuardError(
            "PREDICTIVE_SPEC_UNCONFIRMED", "La definición requiere confirmación humana completa"
        )
    row, context = _context(repository, cycle.project_id)
    if context.task_type != "binary_classification":
        raise PredictiveGuardError(
            "PREDICTIVE_TASK_NOT_BINARY", "El ciclo requiere un objetivo binario"
        )
    dataset = repository.get_dataset(str(row["dataset_id"]))
    capabilities.require_scope(capability_id, cycle_id=cycle_id, dataset_id=str(dataset["id"]))
    if dataset["admission_status"] != "ACCEPTED" or not dataset["is_active"]:
        raise PredictiveGuardError(
            "PREDICTIVE_DATASET_NOT_CURRENT", "La tabla debe estar admitida y activa"
        )
    current = values.draft_predictive_from_context(capability_id, context_version_id=str(row["id"]))
    expected = version.spec.model_dump(mode="json", include=_SEMANTIC_FIELDS)
    actual = current.model_dump(mode="json", include=_SEMANTIC_FIELDS)
    nested = context.model_dump(mode="json", include=_SEMANTIC_FIELDS)
    if canonical_json(expected) != canonical_json(actual) or canonical_json(
        actual
    ) != canonical_json(nested):
        raise PredictiveGuardError(
            "PREDICTIVE_CONTEXT_MISMATCH", "La definición vinculada contradice el contexto vigente"
        )
    recipe = snapshot_recipe(repository, str(dataset["id"]))
    if recipe is None or version.spec.snapshot_recipe is None:
        raise PredictiveGuardError(
            "PREDICTIVE_RECIPE_REQUIRED", "La tabla y la definición requieren su receta verificable"
        )
    try:
        saved_spec = SnapshotSpecV1.model_validate(recipe["spec"])
    except (KeyError, TypeError, ValueError) as error:
        raise PredictiveGuardError(
            "PREDICTIVE_RECIPE_INVALID", "La receta no cumple el contrato"
        ) from error
    if canonical_hash(saved_spec) != canonical_hash(version.spec.snapshot_recipe):
        raise PredictiveGuardError(
            "PREDICTIVE_RECIPE_MISMATCH", "La receta difiere de la definición vinculada"
        )
    label_column = saved_spec.label_available_at_output_column
    if label_column is None:
        raise PredictiveGuardError(
            "LABEL_AVAILABILITY_REQUIRED",
            "Reconstruye la tabla con disponibilidad declarada en su receta",
        )
    outcome = saved_spec.outcome[0]
    window = version.spec.outcome_window
    if window is None:
        raise PredictiveGuardError(
            "PREDICTIVE_WINDOW_NOT_VERIFIED", "Declara la ventana del resultado"
        )
    if window.kind == "deadline_column":
        matches_window = (
            outcome.op == "deadline_not_reached_stage" and window.column == label_column
        )
    else:
        matches_window = getattr(outcome, "within_days", None) == window.days
    if _outcome_name(outcome) != version.spec.target_column or not matches_window:
        raise PredictiveGuardError(
            "PREDICTIVE_OUTCOME_MISMATCH", "El resultado o su horizonte no coinciden"
        )
    forbidden = {
        saved_spec.entity.casefold(),
        SNAPSHOT_CUTOFF_COLUMN.casefold(),
        label_column.casefold(),
    }
    if forbidden & {name.casefold() for name in context.allowed_features}:
        raise PredictiveGuardError(
            "PREDICTIVE_METADATA_AS_FEATURE",
            "Identidad y fechas de control deben excluirse de las variables",
        )
    for key in ("source_dataset_id", "entity_dataset_id"):
        identifier = recipe.get(key)
        if key == "entity_dataset_id" and identifier is None:
            continue
        if not isinstance(identifier, str) or not identifier:
            raise PredictiveGuardError(
                "PREDICTIVE_SOURCE_REQUIRED", "La receta requiere fuentes registradas"
            )
        capabilities.require_scope(capability_id, cycle_id=cycle_id, dataset_id=identifier)
        source = repository.get_dataset(identifier)
        if source["admission_status"] != "ACCEPTED":
            raise PredictiveGuardError(
                "PREDICTIVE_SOURCE_NOT_ACCEPTED", "Una fuente no está admitida"
            )
        if key == "source_dataset_id" and source["sha256"] != recipe.get("source_sha256"):
            raise PredictiveGuardError(
                "PREDICTIVE_SOURCE_HASH_MISMATCH", "La receta refiere otra fuente"
            )
    return version


def require_capability_training_binding(
    repository: WorkflowRepository, strategy_version_id: str
) -> CapabilityTrainingBindingV1 | None:
    """Se llama también dentro de start_run: sin I/O, LLM ni efectos sobre los gates."""

    strategy = repository.get_strategy(strategy_version_id)
    row = repository.get_context(str(strategy["context_version_id"]))
    capabilities = CapabilityService(repository)
    cycle = capabilities.get_cycle_for_project(str(row["project_id"]))
    if cycle is None:
        return None
    version = require_capability_predictive_context(
        repository, capability_id=cycle.capability_id, cycle_id=cycle.id
    )
    if not is_strategy_current(repository.connection, strategy_version_id):
        raise PredictiveGuardError(
            "PREDICTIVE_STRATEGY_NOT_CURRENT", "La estrategia ya no está vigente"
        )
    current, context = _context(repository, cycle.project_id)
    if current["id"] != row["id"]:
        raise PredictiveGuardError(
            "PREDICTIVE_CONTEXT_NOT_CURRENT", "La estrategia usa otro contexto"
        )
    dataset = repository.get_dataset(str(row["dataset_id"]))
    try:
        proposal = StrategyProposalV1.model_validate_json(strategy["strategy_json"])
    except ValidationError as error:
        raise PredictiveGuardError(
            "PREDICTIVE_STRATEGY_INVALID", "La estrategia no cumple el contrato"
        ) from error
    if proposal.task_type != "binary_classification":
        raise PredictiveGuardError(
            "PREDICTIVE_TASK_NOT_BINARY", "El ciclo requiere una estrategia binaria"
        )
    expected = {
        "dataset_id": dataset["id"],
        "dataset_sha256": dataset["sha256"],
        "context_version_id": row["id"],
        "context_sha256": row["context_sha256"],
        "target_column": context.target_column,
        "positive_class": context.positive_class,
        "feature_columns": sorted(context.allowed_features, key=str.casefold),
        "excluded_columns": sorted(context.excluded_features, key=str.casefold),
        "primary_metric": context.primary_metric.value,
        "guardrails": context.guardrails.model_dump(mode="json"),
    }
    actual = proposal.model_dump(mode="json", include=set(expected))
    actual["feature_columns"] = sorted(actual["feature_columns"], key=str.casefold)
    actual["excluded_columns"] = sorted(actual["excluded_columns"], key=str.casefold)
    if (
        canonical_json(expected) != canonical_json(actual)
        or strategy["dataset_id"] != dataset["id"]
    ):
        raise PredictiveGuardError(
            "PREDICTIVE_STRATEGY_MISMATCH", "La estrategia contradice el contexto vigente"
        )
    expected_fingerprint = strategy_fingerprint(
        str(dataset["sha256"]),
        str(row["context_sha256"]),
        str(strategy["strategy_json"]),
        str(strategy["schema_version"]),
    )
    if expected_fingerprint != strategy["fingerprint"]:
        raise PredictiveGuardError(
            "PREDICTIVE_STRATEGY_HASH_MISMATCH", "La huella de la estrategia cambió"
        )
    recipe_spec = version.spec.snapshot_recipe
    assert recipe_spec is not None and recipe_spec.label_available_at_output_column is not None
    if (
        proposal.split.method != "temporal"
        or proposal.split.time_column != SNAPSHOT_CUTOFF_COLUMN
        or proposal.split.label_available_at_column != recipe_spec.label_available_at_output_column
    ):
        raise PredictiveGuardError(
            "TEMPORAL_LABEL_AVAILABILITY_REQUIRED",
            "La partición debe usar el corte y la disponibilidad de la receta",
        )
    binding = DecisionValueService(repository).get_cycle_binding(cycle.id)
    assert binding is not None
    recipe = snapshot_recipe(repository, str(dataset["id"]))
    return CapabilityTrainingBindingV1(
        capability_id=cycle.capability_id,
        cycle_id=cycle.id,
        project_id=cycle.project_id,
        binding_id=binding.id,
        binding_sha256=canonical_hash(binding),
        predictive_version_id=version.id,
        predictive_sha256=version.sha256,
        context_version_id=str(row["id"]),
        context_sha256=str(row["context_sha256"]),
        strategy_version_id=strategy_version_id,
        strategy_fingerprint=str(strategy["fingerprint"]),
        dataset_id=str(dataset["id"]),
        dataset_sha256=str(dataset["sha256"]),
        recipe_sha256=canonical_hash(recipe_spec),
        snapshot_evidence_sha256=canonical_hash(recipe),
        label_available_at_column=recipe_spec.label_available_at_output_column,
    )
