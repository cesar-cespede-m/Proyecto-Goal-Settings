"""Servicio CTX-01 independiente de Streamlit y de proveedores LLM."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.data_engine.errors import DataValidationError
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import DatasetProfileV1, IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_json
from core.domain.context import (
    MAX_CLASSES,
    MIN_ROWS_PER_CLASS,
    BusinessContextV1,
    ClassSupport,
    ContextValidationSnapshot,
    ContextVersionResult,
    FeatureUse,
)
from core.domain.errors import ContextValidationError


def _python_scalar(value: Any) -> str | int | float | bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str | bool | int | float):
        return value
    raise ContextValidationError(
        "UNSUPPORTED_TARGET_VALUE",
        "El target contiene un valor no soportado",
    )


def _value_key(value: str | int | float | bool) -> str:
    """La representación canónica evita confundir 1, 1.0, True y '1'."""

    return f"{type(value).__name__}:{canonical_json(value)}"


class BusinessContextService:
    def __init__(
        self,
        repository: WorkflowRepository,
        storage_root: str | Path,
        *,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.limits = limits or IngestionLimits()

    def _load_dataset(self, dataset_id: str) -> tuple[Any, DatasetProfileV1, pd.DataFrame]:
        row = self.repository.get_dataset(dataset_id)
        if row["admission_status"] != "ACCEPTED":
            raise ContextValidationError(
                "DATASET_NOT_ACCEPTED",
                "Solo se puede definir contexto para un dataset aceptado",
            )
        if not row["is_active"]:
            raise ContextValidationError(
                "DATASET_NOT_ACTIVE",
                "Solo se puede definir contexto para el snapshot analítico activo",
            )
        root = self.storage_root.resolve()
        path = (root / Path(str(row["stored_path"]))).resolve()
        if root not in path.parents or not path.is_file():
            raise ContextValidationError(
                "DATASET_FILE_UNAVAILABLE",
                "El archivo privado del dataset no está disponible",
            )
        content = path.read_bytes()
        validated = validate_csv_bytes(str(row["original_filename_display"]), content, self.limits)
        try:
            frame = read_validated_frame(validated)
        except DataValidationError as error:
            raise ContextValidationError(
                "DATASET_PARSE_FAILED",
                "El dataset aceptado ya no se pudo interpretar",
            ) from error
        profile = DatasetProfileV1.model_validate(json.loads(row["profile_json"]))
        return row, profile, frame

    def inspect_target(self, dataset_id: str, target_column: str) -> list[ClassSupport]:
        """Devuelve sólo clases y conteos agregados para apoyar el formulario."""

        _, _, frame = self._load_dataset(dataset_id)
        if target_column not in frame.columns:
            raise ContextValidationError("TARGET_NOT_FOUND", "La columna target no existe")
        series = frame[target_column]
        if series.isna().any():
            raise ContextValidationError("TARGET_HAS_MISSING", "El target contiene valores nulos")
        counts = series.value_counts(dropna=False, sort=False)
        return [
            ClassSupport(value=_python_scalar(value), count=int(count))
            for value, count in counts.items()
        ]

    def create_version(
        self,
        *,
        project_id: str,
        dataset_id: str,
        context: BusinessContextV1 | dict[str, Any],
    ) -> ContextVersionResult:
        model = BusinessContextV1.model_validate(context)
        normalized = model.model_dump(mode="json")
        normalized["feature_decisions"] = sorted(
            normalized["feature_decisions"],
            key=lambda item: item["column"].casefold(),
        )
        normalized["guardrails"]["rules"] = sorted(
            normalized["guardrails"]["rules"],
            key=lambda item: item["metric"],
        )
        model = BusinessContextV1.model_validate(normalized)
        dataset, profile, frame = self._load_dataset(dataset_id)
        if dataset["project_id"] != project_id:
            raise ContextValidationError(
                "PROJECT_DATASET_MISMATCH",
                "El dataset no pertenece al proyecto indicado",
            )
        if model.target_column not in frame.columns:
            raise ContextValidationError("TARGET_NOT_FOUND", "La columna target no existe")

        protected_join_columns: set[str] = set()
        auxiliary_prefixes: tuple[str, ...] = ()
        analysis_table = (
            self.repository.get_analysis_table(dataset_id)
            if dataset["dataset_kind"] == "DERIVED"
            else None
        )
        # GA-06: una tabla guardada desde un análisis trae su receta como linaje.
        if dataset["dataset_kind"] == "DERIVED" and analysis_table is None:
            composition = self.repository.get_dataset_composition(dataset_id)
            if composition is None:
                raise ContextValidationError(
                    "COMPOSITION_LINEAGE_MISSING",
                    "El dataset derivado no conserva su linaje",
                )
            summary = json.loads(composition["summary_json"])
            spec = json.loads(composition["spec_json"])
            protected_join_columns = set(summary["protected_join_columns"])
            auxiliary_prefixes = tuple(f"{item['alias']}__" for item in spec["joins"])
            if model.target_column.startswith(auxiliary_prefixes):
                raise ContextValidationError(
                    "TARGET_FROM_AUXILIARY",
                    "El target debe pertenecer a la fuente principal",
                )

        series = frame[model.target_column]
        missing_count = int(series.isna().sum())
        if missing_count:
            raise ContextValidationError("TARGET_HAS_MISSING", "El target contiene valores nulos")
        distribution = _target_distribution(series, model)

        expected_features = set(frame.columns) - {model.target_column}
        decisions = {item.column: item for item in model.feature_decisions}
        missing_decisions = sorted(expected_features - set(decisions))
        unknown_decisions = sorted(set(decisions) - expected_features)
        if missing_decisions:
            raise ContextValidationError(
                "FEATURE_DECISIONS_INCOMPLETE",
                "Faltan decisiones para: " + ", ".join(missing_decisions),
            )
        if unknown_decisions:
            raise ContextValidationError(
                "FEATURE_NOT_FOUND",
                "No existen en el dataset: " + ", ".join(unknown_decisions),
            )
        if not model.allowed_features:
            raise ContextValidationError(
                "NO_ALLOWED_FEATURES",
                "Debe quedar al menos una variable permitida",
            )
        included_join_keys = sorted(protected_join_columns & set(model.allowed_features))
        if included_join_keys:
            raise ContextValidationError(
                "JOIN_KEY_INCLUDED",
                "Las claves de cruce deben excluirse del entrenamiento: "
                + ", ".join(included_join_keys),
            )

        profile_by_name = {item.name: item for item in profile.columns}
        unsafe_included = sorted(
            name
            for name, decision in decisions.items()
            if decision.use is FeatureUse.INCLUDE
            and (profile_by_name[name].possible_identifier or profile_by_name[name].constant)
        )
        if unsafe_included:
            raise ContextValidationError(
                "UNSAFE_FEATURE_INCLUDED",
                "Posibles identificadores o constantes deben excluirse: "
                + ", ".join(unsafe_included),
            )

        # Comprobación matemática mínima para que cada clase aparezca en 60/20/20.
        insufficient = [item for item in distribution if int(item.count * 0.2) < 1]
        if insufficient:
            raise ContextValidationError(
                "INSUFFICIENT_CLASS_SUPPORT",
                "Cada clase debe poder aportar ejemplos a train, validation y test",
            )

        validation = ContextValidationSnapshot(
            dataset_sha256=dataset["sha256"],
            row_count=int(dataset["n_rows"]),
            column_count=int(dataset["n_cols"]),
            target_missing_count=missing_count,
            target_distribution=distribution,
        )
        payload = {
            "schema_version": model.schema_version,
            "target_column": model.target_column,
            "positive_class": model.positive_class,
            "prediction_moment": model.prediction_moment,
            "excluded_columns": model.excluded_features,
            "primary_metric": model.primary_metric.value,
            "guardrails": model.guardrails.model_dump(mode="json"),
            "context": model.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
        }
        latest = self.repository.get_latest_context(project_id)
        if latest is not None and latest["context_json"] == canonical_json(payload):
            raise ContextValidationError(
                "CONTEXT_UNCHANGED",
                "El contexto es idéntico a la versión vigente",
            )
        context_id = self.repository.add_context(
            project_id=project_id,
            dataset_id=dataset_id,
            version_no=None,
            payload=payload,
        )
        stored = self.repository.get_context(context_id)
        return ContextVersionResult(
            context_version_id=context_id,
            project_id=project_id,
            dataset_id=dataset_id,
            version_no=int(stored["version_no"]),
            context_sha256=stored["context_sha256"],
            context=model,
            validation=validation,
        )


def _target_distribution(series: Any, model: BusinessContextV1) -> list[ClassSupport]:
    """Valida el objetivo según el tipo de modelo (ADR-016); sí/no queda como antes."""

    if model.task_type == "regression":
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or values.nunique() < 3:
            raise ContextValidationError(
                "TARGET_NOT_A_QUANTITY",
                "Para anticipar una cantidad el objetivo debe ser numérico y variar",
            )
        return []
    counts = series.value_counts(dropna=False, sort=False)
    distribution = [
        ClassSupport(value=_python_scalar(value), count=int(count))
        for value, count in counts.items()
    ]
    if model.task_type == "multiclass_classification":
        if not 3 <= len(counts) <= MAX_CLASSES:
            raise ContextValidationError(
                "TOO_MANY_CLASSES" if len(counts) > MAX_CLASSES else "NOT_ENOUGH_CLASSES",
                f"Para varias clases el objetivo debe tener entre 3 y {MAX_CLASSES} valores",
            )
        if int(counts.min()) < MIN_ROWS_PER_CLASS:
            raise ContextValidationError(
                "CLASS_TOO_SMALL",
                f"Cada valor del objetivo necesita al menos {MIN_ROWS_PER_CLASS} casos",
            )
        return distribution
    if len(counts) != 2:
        raise ContextValidationError(
            "TARGET_NOT_BINARY",
            "El target debe contener exactamente dos clases",
        )
    available_class_keys = {_value_key(item.value) for item in distribution}
    if _value_key(model.positive_class) not in available_class_keys:
        raise ContextValidationError(
            "POSITIVE_CLASS_NOT_FOUND",
            "La clase positiva debe coincidir exactamente con una clase del target",
        )
    return distribution
