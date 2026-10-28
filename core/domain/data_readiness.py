"""MV-08: evidencia reproducible de maduración, sin enviar filas al agente."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.analysis.engine import LoadedSource, load_project_source
from core.analysis.snapshot import (
    ObservationCoverageV1,
    SnapshotError,
    SnapshotReadinessV1,
    SnapshotSpecV1,
    build_snapshot,
)
from core.analysis.snapshot_recipe import snapshot_recipe
from core.analysis.validation import AnalysisPlanError
from core.contracts.decision_value import PredictiveSpecVersionV1
from core.contracts.orchestration import ArtifactRefV1
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError

_SHA256 = r"^[0-9a-f]{64}$"
_COVERAGE_PLAN = (
    "Declarar el intervalo de captura continua de esta fuente y verificar que incluye "
    "el corte y toda la ventana del resultado para los casos usados."
)
_PENDING_PLAN = (
    "Mantener los casos pendientes fuera del entrenamiento y observarlos hasta completar "
    "su ventana; declarar la cobertura del nuevo lote antes de reconstruir la tabla."
)
_PROVENANCE_PLAN = (
    "Conservar la fuente admitida, la receta que define el corte y el resultado, y la "
    "cobertura declarada de este lote; reconstruir una tabla sólo con casos maduros."
)


class DataReadinessError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DataReadinessV1(BaseModel):
    """Preparación de etiquetas de la tabla; no es permiso ni suficiencia estadística."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["data-readiness.v1"] = "data-readiness.v1"
    status: Literal["ready", "needs_data", "needs_human"]
    reason_codes: list[str] = Field(default_factory=list)
    reason_counts: dict[str, int] = Field(default_factory=dict)
    snapshot_readiness: SnapshotReadinessV1 | None = None
    table_rows: int = Field(ge=0, strict=True)
    verified_table_rows: int = Field(default=0, ge=0, strict=True)
    pending_source_count: int = Field(default=0, ge=0, strict=True)
    measurement_plan: list[str] = Field(default_factory=list)
    output_refs: list[ArtifactRefV1] = Field(default_factory=list)
    recipe_sha256: str | None = Field(default=None, pattern=_SHA256)
    binding_id: str | None = None
    binding_sha256: str | None = Field(default=None, pattern=_SHA256)
    evidence_sha256: str = Field(pattern=_SHA256)
    grants_training_approval: Literal[False] = False
    statistical_support_checked: Literal[False] = False

    def safe_summary(self) -> dict:
        """Las referencias se entregan aparte para resolverlas como alias del agente."""

        return self.model_dump(mode="json", exclude={"output_refs", "binding_id"})


class DataReadinessService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.capabilities = CapabilityService(repository)
        self.values = DecisionValueService(repository)

    def _load(self, capability_id: str, cycle_id: str, dataset_id: str) -> LoadedSource:
        self.capabilities.require_scope(capability_id, cycle_id=cycle_id, dataset_id=dataset_id)
        try:
            return load_project_source(self.repository, self.storage_root, dataset_id)
        except AnalysisPlanError as error:
            # No propagar columnas, categorías, rutas ni errores de parseo al LLM.
            raise DataReadinessError(
                error.code, "La fuente no pudo verificarse como admitida e íntegra"
            ) from error

    def inspect_label_readiness(
        self,
        capability_id: str,
        *,
        dataset_id: str,
        cycle_id: str | None = None,
        predictive_version_id: str | None = None,
    ) -> DataReadinessV1:
        self.capabilities.require_scope(capability_id, cycle_id=cycle_id, dataset_id=dataset_id)
        dataset = self.repository.get_dataset(dataset_id)
        cycle = self.capabilities.get_cycle_for_project(str(dataset["project_id"]))
        assert cycle is not None  # require_scope ya exige pertenencia a una capacidad.
        table = self._load(capability_id, cycle.id, dataset_id)
        refs = [
            ArtifactRefV1(kind="dataset", id=dataset_id, sha256=table.sha256),
            ArtifactRefV1(kind="cycle", id=cycle.id, sha256=canonical_hash(cycle)),
        ]
        binding = self.values.get_cycle_binding(cycle.id)
        if binding is not None:
            if (
                predictive_version_id is not None
                and predictive_version_id != binding.predictive_version_id
            ):
                raise DataReadinessError(
                    "PREDICTIVE_BINDING_MISMATCH", "El ciclo está vinculado a otra especificación"
                )
            predictive_version_id = binding.predictive_version_id
        if predictive_version_id is None:
            versions = self.values.list_predictive_versions(capability_id)
            predictive = versions[-1] if versions else None
        else:
            predictive = self.values.get_predictive_version(predictive_version_id)
        if predictive is not None:
            if predictive.capability_id != capability_id:
                raise DataReadinessError(
                    "SPEC_SCOPE_MISMATCH", "La especificación pertenece a otra capacidad"
                )
            refs.append(
                ArtifactRefV1(kind="predictive_spec", id=predictive.id, sha256=predictive.sha256)
            )
        recipe_hash = None

        def result(
            status: Literal["ready", "needs_data", "needs_human"],
            codes: list[str],
            plans: list[str],
            readiness: SnapshotReadinessV1 | None = None,
            *,
            verified: int = 0,
        ) -> DataReadinessV1:
            payload = dict(
                status=status,
                reason_codes=sorted(set(codes)),
                reason_counts={} if readiness is None else readiness.reason_counts,
                snapshot_readiness=readiness,
                table_rows=len(table.frame),
                verified_table_rows=verified,
                pending_source_count=0 if readiness is None else readiness.pending_rows,
                measurement_plan=list(dict.fromkeys(plans)),
                output_refs=refs,
                recipe_sha256=recipe_hash,
                binding_id=None if binding is None else binding.id,
                binding_sha256=None if binding is None else canonical_hash(binding),
            )
            draft = DataReadinessV1(**payload, evidence_sha256="0" * 64)
            return draft.model_copy(
                update={
                    "evidence_sha256": canonical_hash(
                        draft.model_dump(mode="json", exclude={"evidence_sha256"})
                    )
                }
            )

        recipe = snapshot_recipe(self.repository, dataset_id)
        if recipe is None:
            return result("needs_human", ["SNAPSHOT_PROVENANCE_MISSING"], [_PROVENANCE_PLAN])
        try:
            spec = SnapshotSpecV1.model_validate(recipe["spec"])
            coverage = (
                None
                if recipe.get("coverage") is None
                else ObservationCoverageV1.model_validate(recipe["coverage"])
            )
            recorded = (
                None
                if recipe.get("readiness") is None
                else SnapshotReadinessV1.model_validate(recipe["readiness"])
            )
            source_id = recipe["source_dataset_id"]
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Invalid source reference")
            entity_id = recipe.get("entity_dataset_id")
            if entity_id is not None and (not isinstance(entity_id, str) or not entity_id):
                raise ValueError("Invalid entity reference")
        except (KeyError, TypeError, ValueError, ValidationError):
            return result("needs_human", ["SNAPSHOT_METADATA_INVALID"], [_PROVENANCE_PLAN])
        recipe_hash = canonical_hash(spec)
        source = self._load(capability_id, cycle.id, source_id)
        if source.sha256 != recipe.get("source_sha256"):
            raise DataReadinessError(
                "SNAPSHOT_SOURCE_HASH_MISMATCH", "La receta refiere otra versión de la fuente"
            )
        refs.append(ArtifactRefV1(kind="dataset", id=source_id, sha256=source.sha256))
        entities = None
        if entity_id is not None:
            entities = self._load(capability_id, cycle.id, entity_id)
            refs.append(ArtifactRefV1(kind="dataset", id=entity_id, sha256=entities.sha256))
        try:
            rebuilt = build_snapshot(
                source.frame,
                spec,
                None if entities is None else entities.frame,
                coverage=coverage,
            )
        except SnapshotError:
            return result("needs_human", ["SNAPSHOT_REBUILD_FAILED"], [_PROVENANCE_PLAN])
        readiness = rebuilt.readiness
        plans = [_PENDING_PLAN] if readiness.pending_rows else []
        if recorded is not None and recorded != readiness:
            return result(
                "needs_human", ["SNAPSHOT_READINESS_MISMATCH"], [_PROVENANCE_PLAN], readiness
            )
        if readiness.eligible_rows == 0:
            return result(
                "needs_data",
                ["NO_MATURE_ROWS", *readiness.reason_counts],
                [_COVERAGE_PLAN, *plans],
                readiness,
            )
        # Esta es la serialización que usan la UI y el reentrenamiento de snapshots.
        # No basta que el CSV tenga etiquetas completas: debe ser exactamente la cohorte madura.
        expected_hash = hashlib.sha256(
            rebuilt.training_frame().to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if expected_hash != table.sha256:
            return result(
                "needs_data" if readiness.pending_rows else "needs_human",
                ["TABLE_REBUILD_MISMATCH"],
                [_PROVENANCE_PLAN, *plans],
                readiness,
            )
        issues = self._predictive_issues(
            predictive, spec, rebuilt.summary.outcome_names, rebuilt.summary.feature_names
        )
        if issues:
            return result(
                "needs_human",
                issues,
                [
                    "Revisar y confirmar la definición predictiva, su receta, el resultado "
                    "y la ventana de observación compatibles con esta tabla.",
                    *plans,
                ],
                readiness,
                verified=readiness.eligible_rows,
            )
        return result(
            "ready",
            ["PENDING_SOURCE_ROWS_EXCLUDED"] if readiness.pending_rows else [],
            plans,
            readiness,
            verified=readiness.eligible_rows,
        )

    @staticmethod
    def _predictive_issues(
        version: PredictiveSpecVersionV1 | None,
        recipe: SnapshotSpecV1,
        outcome_names: list[str],
        feature_names: list[str],
    ) -> list[str]:
        if version is None:
            return ["PREDICTIVE_SPEC_MISSING"]
        issues = []
        if not version.confirmed_human or version.spec.missing_fields():
            issues.append("PREDICTIVE_SPEC_UNCONFIRMED_OR_INCOMPLETE")
        spec = version.spec
        allowed_features = set(feature_names) | set(recipe.entity_attributes or [])
        if any(
            item.column not in allowed_features
            for item in spec.feature_decisions
            if item.use == "include"
        ):
            issues.append("PREDICTIVE_FEATURES_NOT_VERIFIED")
        if spec.snapshot_recipe is None:
            issues.append("PREDICTIVE_RECIPE_NOT_DECLARED")
        elif canonical_hash(spec.snapshot_recipe) != canonical_hash(recipe):
            issues.append("PREDICTIVE_RECIPE_MISMATCH")
        if spec.outcome_window is None:
            issues.append("OUTCOME_WINDOW_NOT_DECLARED")
        elif spec.target_column not in outcome_names:
            issues.append("PREDICTIVE_TARGET_MISMATCH")
        else:
            outcome = recipe.outcome[outcome_names.index(spec.target_column)]
            if spec.outcome_window.kind == "deadline_column":
                if (
                    outcome.op != "deadline_not_reached_stage"
                    or recipe.label_available_at_output_column != spec.outcome_window.column
                ):
                    issues.append("PREDICTIVE_WINDOW_MISMATCH")
            elif outcome.op not in {"not_reached_stage", "reached_stage", "inactive"}:
                issues.append("BINARY_OUTCOME_NOT_VERIFIED")
            elif getattr(outcome, "within_days", None) != spec.outcome_window.days:
                issues.append("PREDICTIVE_WINDOW_MISMATCH")
        return issues
