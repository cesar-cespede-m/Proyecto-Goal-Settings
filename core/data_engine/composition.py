"""Composición determinista y trazable de hasta cinco CSV permitidos."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_dtype_equal
from pydantic import ValidationError

from core.data_engine.errors import DatasetCompositionError, DataValidationError
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import (
    CompositionLimits,
    CompositionSourceRefV1,
    DatasetClassification,
    DatasetCompositionSpecV1,
    DatasetCompositionSummaryV1,
    DatasetProfileV1,
    IngestionLimits,
    JoinStatisticsV1,
)
from core.data_engine.profiler import build_profile
from core.data_engine.validator import validate_csv_bytes
from core.domain.canonical import canonical_hash


@dataclass(frozen=True)
class CSVSourceInput:
    filename: str
    content: bytes


@dataclass(frozen=True)
class DatasetCompositionResult:
    content: bytes
    profile: DatasetProfileV1
    summary: DatasetCompositionSummaryV1


def _classification(spec: DatasetCompositionSpecV1) -> DatasetClassification:
    values = {
        spec.primary.classification,
        *(item.source.classification for item in spec.joins),
    }
    if DatasetClassification.ANONYMIZED in values:
        return DatasetClassification.ANONYMIZED
    if DatasetClassification.SANDBOX in values:
        return DatasetClassification.SANDBOX
    return DatasetClassification.SYNTHETIC


# Por debajo de esta proporción de filas principales con coincidencia se advierte.
LOW_MATCH_RATE = 0.5


def _normalized_keys(keys: pd.Series) -> set[str]:
    """Claves comparables sin mayúsculas, guiones, guiones bajos, puntos ni espacios."""

    return set(keys.astype(str).str.casefold().str.replace(r"[\s._-]+", "", regex=True))


def _validated_frame(
    source_ref: CompositionSourceRefV1,
    source: CSVSourceInput,
    ingestion_limits: IngestionLimits,
) -> pd.DataFrame:
    dataset_id = source_ref.dataset_id
    expected_hash = source_ref.dataset_sha256
    observed_hash = hashlib.sha256(source.content).hexdigest()
    if observed_hash != expected_hash:
        raise DatasetCompositionError(
            "SOURCE_HASH_MISMATCH",
            f"El contenido de la fuente {dataset_id} no coincide con su hash aprobado",
        )
    try:
        validated = validate_csv_bytes(source.filename, source.content, ingestion_limits)
        return read_validated_frame(validated)
    except DataValidationError as error:
        raise DatasetCompositionError(
            "SOURCE_INVALID",
            f"La fuente {dataset_id} ya no cumple el contrato CSV: {error.code}",
        ) from error


def _require_column(frame: pd.DataFrame, name: str, *, role: str) -> None:
    if name not in frame.columns:
        raise DatasetCompositionError(
            "JOIN_KEY_NOT_FOUND",
            f"La clave declarada no existe en la fuente {role}",
        )
    if frame[name].isna().any():
        raise DatasetCompositionError(
            "JOIN_KEY_HAS_MISSING",
            f"La clave declarada contiene vacíos en la fuente {role}",
        )


def _protected_names(frame: pd.DataFrame) -> set[str]:
    return {str(name).casefold() for name in frame.columns}


def validate_join_cardinality(
    primary_keys: pd.Series,
    auxiliary_keys: pd.Series,
    cardinality: str,
    *,
    auxiliary_role: str,
) -> None:
    """Aplica los controles compartidos que impiden cruces ambiguos o multiplicativos."""

    if not is_dtype_equal(primary_keys.dtype, auxiliary_keys.dtype):
        raise DatasetCompositionError(
            "JOIN_KEY_TYPE_MISMATCH",
            f"Las claves del cruce con {auxiliary_role} no tienen el mismo tipo",
        )
    if auxiliary_keys.duplicated().any():
        raise DatasetCompositionError(
            "AUXILIARY_KEY_NOT_UNIQUE",
            f"La clave del auxiliar {auxiliary_role} contiene duplicados",
        )
    if cardinality == "one_to_one" and primary_keys.duplicated().any():
        raise DatasetCompositionError(
            "PRIMARY_KEY_NOT_UNIQUE",
            "La clave principal contiene duplicados y el cruce exige one_to_one",
        )


def compose_csv_datasets(
    spec: DatasetCompositionSpecV1 | dict[str, object],
    sources: Mapping[str, CSVSourceInput],
    *,
    ingestion_limits: IngestionLimits | None = None,
    composition_limits: CompositionLimits | None = None,
) -> DatasetCompositionResult:
    """Cruza una fuente principal con hasta cuatro auxiliares sin multiplicar sus filas."""

    try:
        model = DatasetCompositionSpecV1.model_validate(spec)
    except ValidationError as error:
        raise DatasetCompositionError("COMPOSITION_SPEC_INVALID", str(error)) from error
    csv_limits = ingestion_limits or IngestionLimits()
    limits = composition_limits or CompositionLimits()
    expected_ids = {model.primary.dataset_id, *(item.source.dataset_id for item in model.joins)}
    if len(expected_ids) > limits.max_sources:
        raise DatasetCompositionError(
            "TOO_MANY_SOURCES",
            f"La composición admite como máximo {limits.max_sources} datasets",
        )
    if set(sources) != expected_ids:
        raise DatasetCompositionError(
            "SOURCE_SET_MISMATCH",
            "Las fuentes entregadas no coinciden exactamente con la especificación",
        )
    total_input_bytes = sum(len(item.content) for item in sources.values())
    if total_input_bytes > limits.max_total_input_bytes:
        raise DatasetCompositionError(
            "COMPOSITION_INPUT_TOO_LARGE",
            "Las fuentes superan el límite agregado de la composición",
        )

    primary_input = sources[model.primary.dataset_id]
    frame = _validated_frame(model.primary, primary_input, csv_limits)
    primary_columns = set(frame.columns)
    join_statistics: list[JoinStatisticsV1] = []
    protected_columns: set[str] = set()
    join_warnings: list[str] = []

    for index, join in enumerate(model.joins, start=1):
        auxiliary = _validated_frame(
            join.source,
            sources[join.source.dataset_id],
            csv_limits,
        )
        if join.primary_key not in primary_columns:
            raise DatasetCompositionError(
                "JOIN_KEY_NOT_FOUND",
                "La clave declarada no existe en la fuente principal",
            )
        _require_column(frame, join.primary_key, role="principal")
        _require_column(auxiliary, join.auxiliary_key, role=join.alias)
        protected_columns.add(join.primary_key)

        validate_join_cardinality(
            frame[join.primary_key],
            auxiliary[join.auxiliary_key],
            join.cardinality,
            auxiliary_role=join.alias,
        )

        reserved_key = f"__goal_settings_join_key_{index}"
        indicator = f"__goal_settings_merge_{index}"
        if reserved_key in frame.columns or indicator in frame.columns:
            raise DatasetCompositionError(
                "RESERVED_COLUMN_COLLISION",
                "Una columna usa un nombre interno reservado para la composición",
            )
        rename = {
            column: (reserved_key if column == join.auxiliary_key else f"{join.alias}__{column}")
            for column in auxiliary.columns
        }
        # Una columna auxiliar con el nombre de una principal (p. ej. el resultado) puede ser
        # una copia: se advierte como posible fuga para que la persona la revise.
        principal_names = {str(name).casefold() for name in primary_columns}
        repeated = sorted(
            rename[column]
            for column in auxiliary.columns
            if column != join.auxiliary_key and str(column).casefold() in principal_names
        )
        if repeated:
            join_warnings.append("AUXILIARY_REPEATS_PRIMARY_COLUMNS:" + ",".join(repeated))
        renamed = auxiliary.rename(columns=rename)
        new_columns = [name for name in renamed.columns if name != reserved_key]
        existing = _protected_names(frame)
        if any(name.casefold() in existing for name in new_columns):
            raise DatasetCompositionError(
                "OUTPUT_COLUMN_COLLISION",
                f"El alias {join.alias} produce columnas repetidas",
            )
        if len(frame.columns) + len(new_columns) > csv_limits.max_columns:
            raise DatasetCompositionError(
                "COMPOSED_TOO_MANY_COLUMNS",
                f"El resultado superaría el límite de {csv_limits.max_columns} columnas",
            )

        before_rows = len(frame)
        primary_keys = frame[join.primary_key]
        auxiliary_keys = auxiliary[join.auxiliary_key]
        unmatched_auxiliary = int((~auxiliary_keys.isin(primary_keys)).sum())
        try:
            frame = frame.merge(
                renamed,
                how="left",
                left_on=join.primary_key,
                right_on=reserved_key,
                sort=False,
                validate=join.cardinality,
                indicator=indicator,
            )
        except pd.errors.MergeError as error:
            raise DatasetCompositionError(
                "JOIN_CARDINALITY_VIOLATION",
                f"El cruce con {join.alias} no cumple {join.cardinality}",
            ) from error
        matched = int((frame[indicator] == "both").sum())
        frame = frame.drop(columns=[reserved_key, indicator])
        if matched == 0:
            if _normalized_keys(primary_keys) & _normalized_keys(auxiliary_keys):
                raise DatasetCompositionError(
                    "JOIN_KEYS_FORMAT_MISMATCH",
                    f"Ninguna fila coincide con el auxiliar {join.alias}, pero sus claves sí "
                    "coinciden al ignorar mayúsculas, guiones y espacios",
                )
            raise DatasetCompositionError(
                "JOIN_HAS_NO_MATCHES",
                f"Ninguna fila coincide con el auxiliar {join.alias}: los archivos no comparten "
                "ningún caso con esa clave",
            )
        if matched / before_rows < LOW_MATCH_RATE:
            join_warnings.append(f"LOW_JOIN_MATCH:{join.alias}")
        if len(frame) != before_rows:
            raise DatasetCompositionError(
                "ROW_MULTIPLICATION_DETECTED",
                f"El cruce con {join.alias} alteró la cantidad de filas principales",
            )
        join_statistics.append(
            JoinStatisticsV1(
                auxiliary_dataset_id=join.source.dataset_id,
                alias=join.alias,
                primary_rows=before_rows,
                auxiliary_rows=len(auxiliary),
                matched_primary_rows=matched,
                unmatched_primary_rows=before_rows - matched,
                unmatched_auxiliary_rows=unmatched_auxiliary,
                output_rows=len(frame),
            )
        )

    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if len(content) > csv_limits.max_upload_bytes:
        raise DatasetCompositionError(
            "COMPOSED_FILE_TOO_LARGE",
            "El CSV derivado supera el límite permitido",
        )
    try:
        validated_output = validate_csv_bytes("composed.csv", content, csv_limits)
    except DataValidationError as error:
        raise DatasetCompositionError(
            "COMPOSED_DATASET_INVALID",
            f"El resultado no cumple el contrato CSV: {error.code}",
        ) from error
    profile = build_profile(
        validated_output,
        classification=_classification(model),
        limits=csv_limits,
    )
    join_warning = "JOIN_KEYS_MUST_BE_EXCLUDED:" + ",".join(sorted(protected_columns))
    profile = profile.model_copy(
        update={"warnings": [*profile.warnings, join_warning, *join_warnings]}
    )
    output_hash = hashlib.sha256(content).hexdigest()
    summary = DatasetCompositionSummaryV1(
        composition_sha256=canonical_hash(model),
        output_sha256=output_hash,
        source_count=len(expected_ids),
        total_input_bytes=total_input_bytes,
        output_bytes=len(content),
        protected_join_columns=sorted(protected_columns),
        joins=join_statistics,
    )
    return DatasetCompositionResult(content=content, profile=profile, summary=summary)
