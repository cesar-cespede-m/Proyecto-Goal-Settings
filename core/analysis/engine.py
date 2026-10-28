"""Ejecución acotada del plan declarativo de análisis (GA-02a a GA-02d)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import ConfigDict, Field, ValidationError, field_validator

from core.analysis.contracts import (
    AggregateFunction,
    AggregateSpecV1,
    AnalysisPlanV1,
    ColumnType,
    DeriveBucketStepV1,
    DeriveDatetimeStepV1,
    DeriveRatioStepV1,
    FilterStepV1,
    GroupByStepV1,
    JoinStepV1,
    LimitStepV1,
    PivotStepV1,
    RenameStepV1,
    SelectStepV1,
    SortStepV1,
    StrictAnalysisModel,
    ThresholdLabelStepV1,
    UnionStepV1,
)
from core.analysis.validation import AnalysisPlanError, canonical_plan_hash, validate_plan
from core.data_engine import IngestionLimits, analysis_schema
from core.data_engine.composition import validate_join_cardinality
from core.data_engine.errors import DatasetCompositionError, DataValidationError
from core.data_engine.loading import read_validated_frame
from core.data_engine.validator import validate_csv_bytes

RESULT_TRUNCATED = "RESULT_TRUNCATED"
DATETIME_PARSE_FAILED = "DATETIME_PARSE_FAILED"
DIVISION_BY_ZERO = "DIVISION_BY_ZERO"
JOIN_UNMATCHED_ROWS = "JOIN_UNMATCHED_ROWS"
THRESHOLD_VALUE = "THRESHOLD_VALUE"

MAX_PIVOT_COLUMNS = 50
ANALYSIS_ENGINE_VERSION = "analysis-engine.v1"

DAY_PART_LABELS = ("madrugada", "mañana", "tarde", "noche")


@dataclass(frozen=True, slots=True)
class AnalysisLimits:
    max_input_rows: int = 100_000
    max_output_rows: int = 5_000
    max_seconds: float = 20

    def __post_init__(self) -> None:
        if self.max_input_rows <= 0 or self.max_output_rows <= 0 or self.max_seconds <= 0:
            raise ValueError("Los límites del análisis deben ser mayores que cero")


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """Fuente validada junto con la versión y el esquema que autorizan su uso."""

    frame: pd.DataFrame
    sha256: str
    schema: Mapping[str, ColumnType]


class AnalysisColumnV1(StrictAnalysisModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    type: ColumnType


type JSONScalar = str | int | float | bool | None


class AnalysisResultV1(StrictAnalysisModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    columns: list[AnalysisColumnV1]
    rows: list[list[JSONScalar]]
    total_rows: int = Field(ge=0)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes: dict[str, str]
    engine_version: str = Field(default=ANALYSIS_ENGINE_VERSION, min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    warnings: list[str]

    @field_validator("created_at")
    @classmethod
    def created_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha de creación del resultado debe incluir zona horaria")
        return value


def _internal_path(storage_root: str | Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise AnalysisPlanError("INVALID_INTERNAL_PATH", "La ruta privada no es válida")
    root = Path(storage_root).resolve()
    path = (root / Path(pure)).resolve()
    if root != path and root not in path.parents:
        raise AnalysisPlanError("INVALID_INTERNAL_PATH", "La ruta privada salió del storage")
    return path


def load_project_source(repository: Any, storage_root: str | Path, dataset_id: str) -> LoadedSource:
    """Carga una fuente aceptada e íntegra usando su perfil persistido como esquema."""

    try:
        dataset = repository.get_dataset(dataset_id)
    except (LookupError, ValueError) as error:
        raise AnalysisPlanError(
            "SOURCE_NOT_FOUND", "La fuente solicitada no está registrada"
        ) from error
    if dataset["admission_status"] != "ACCEPTED":
        raise AnalysisPlanError("DATASET_NOT_ACCEPTED", "La fuente no está aceptada")

    path = _internal_path(storage_root, str(dataset["stored_path"]))
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise AnalysisPlanError(
            "ACCEPTED_DATASET_MISSING",
            "La fuente aceptada no está disponible en el storage privado",
        ) from error

    observed_hash = hashlib.sha256(content).hexdigest()
    if observed_hash != str(dataset["sha256"]):
        raise AnalysisPlanError(
            "DATASET_HASH_MISMATCH",
            "El contenido de la fuente ya no coincide con la versión aceptada",
        )

    try:
        validated = validate_csv_bytes(
            str(dataset["original_filename_display"]),
            content,
            IngestionLimits(),
        )
        profile = json.loads(str(dataset["profile_json"]))
        schema = {
            name: ColumnType(column_type) for name, column_type in analysis_schema(profile).items()
        }
    except (DataValidationError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise AnalysisPlanError(
            "TYPE_MISMATCH", "La fuente aceptada o su perfil guardado no son válidos"
        ) from error

    return LoadedSource(
        frame=read_validated_frame(validated),
        sha256=observed_hash,
        schema=schema,
    )


class AnalysisEngine:
    def __init__(self, limits: AnalysisLimits | None = None) -> None:
        self.limits = limits or AnalysisLimits()

    def _check_time(self, started_at: float) -> None:
        if perf_counter() - started_at > self.limits.max_seconds:
            raise AnalysisPlanError(
                "ANALYSIS_TIME_LIMIT_EXCEEDED",
                "El análisis superó el tiempo máximo permitido",
            )

    def run(
        self,
        plan: AnalysisPlanV1 | Mapping[str, object],
        sources: Mapping[str, LoadedSource],
    ) -> AnalysisResultV1:
        """Valida y ejecuta las operaciones implementadas, en el orden declarado."""

        started_at = perf_counter()
        if not isinstance(plan, AnalysisPlanV1):
            try:
                plan = AnalysisPlanV1.model_validate(plan)
            except ValidationError as error:
                raise AnalysisPlanError(
                    "TYPE_MISMATCH", "El contrato del plan no es válido"
                ) from error

        schemas = {name: source.schema for name, source in sources.items()}
        result_schema = validate_plan(plan, schemas)
        self._check_time(started_at)

        for source_name in plan.sources:
            row_count = len(sources[source_name].frame)
            if row_count > self.limits.max_input_rows:
                raise AnalysisPlanError(
                    "ANALYSIS_INPUT_ROW_LIMIT_EXCEEDED",
                    f"La fuente {source_name!r} supera el límite de filas de entrada",
                )

        frame = sources[plan.sources[0]].frame.copy()
        runtime_schema = {
            column: ColumnType(column_type)
            for column, column_type in schemas[plan.sources[0]].items()
        }
        warnings: list[str] = []
        for step in plan.steps:
            self._check_time(started_at)
            if isinstance(step, SelectStepV1):
                frame = frame.loc[:, step.columns].copy()
                runtime_schema = {column: runtime_schema[column] for column in step.columns}
            elif isinstance(step, RenameStepV1):
                frame = frame.rename(columns=step.mapping)
                runtime_schema = {
                    step.mapping.get(column, column): column_type
                    for column, column_type in runtime_schema.items()
                }
            elif isinstance(step, FilterStepV1):
                frame = frame.loc[_filter_mask(frame, step)].copy()
            elif isinstance(step, DeriveDatetimeStepV1):
                derived, failed_count = _derive_datetime(frame[step.column], step.part)
                frame[step.output_column] = derived
                runtime_schema[step.output_column] = _datetime_part_type(step.part)
                if failed_count:
                    warnings.append(f"{DATETIME_PARSE_FAILED}:{failed_count}")
            elif isinstance(step, DeriveBucketStepV1):
                if step.method == "calendar":
                    assert step.unit is not None
                    frame[step.output_column] = _derive_calendar_bucket(
                        frame[step.column], step.unit
                    )
                else:
                    assert step.bins is not None
                    frame[step.output_column] = _derive_numeric_bucket(
                        frame[step.column], step.method, step.bins
                    )
                runtime_schema[step.output_column] = ColumnType.CATEGORY
            elif isinstance(step, DeriveRatioStepV1):
                derived, zero_count = _derive_ratio(frame[step.numerator], frame[step.denominator])
                frame[step.output_column] = derived
                runtime_schema[step.output_column] = ColumnType.NUMBER
                if zero_count:
                    warnings.append(f"{DIVISION_BY_ZERO}:{zero_count}")
            elif isinstance(step, UnionStepV1):
                frame = pd.concat(
                    [frame, sources[step.source].frame.loc[:, frame.columns]],
                    ignore_index=True,
                )
            elif isinstance(step, JoinStepV1):
                frame, unmatched = _join(frame, sources[step.source].frame, step)
                runtime_schema = _joined_schema(runtime_schema, sources[step.source].schema, step)
                if unmatched:
                    warnings.append(f"{JOIN_UNMATCHED_ROWS}:{unmatched}")
            elif isinstance(step, GroupByStepV1):
                frame = _group_by(frame, step)
                runtime_schema = {
                    **{column: runtime_schema[column] for column in step.by},
                    **{
                        aggregation.output_column: _aggregate_output_type(
                            runtime_schema, aggregation
                        )
                        for aggregation in step.aggregations
                    },
                }
            elif isinstance(step, SortStepV1):
                frame = frame.sort_values(
                    by=step.by,
                    ascending=not step.descending,
                    kind="stable",
                    na_position="last",
                )
            elif isinstance(step, LimitStepV1):
                frame = frame.head(step.rows).copy()
            elif isinstance(step, PivotStepV1):
                frame, runtime_schema = _pivot(frame, runtime_schema, step)
            elif isinstance(step, ThresholdLabelStepV1):
                cutoff = step.threshold
                if step.percentile is not None:
                    populated = frame[step.column].dropna()
                    if populated.empty:
                        raise AnalysisPlanError(
                            "THRESHOLD_VALUE_UNAVAILABLE",
                            "No hay valores para calcular el percentil solicitado",
                        )
                    cutoff = float(populated.quantile(step.percentile))
                    warnings.append(f"{THRESHOLD_VALUE}:{_format_boundary(cutoff)}")
                assert cutoff is not None
                frame[step.output_column] = _threshold_labels(
                    frame[step.column], cutoff, step.labels
                )
                runtime_schema[step.output_column] = ColumnType.CATEGORY
            else:
                raise AssertionError(f"Operación validada no cubierta por el motor: {step.op}")
            self._check_time(started_at)

        if list(frame.columns) != list(runtime_schema):
            raise AnalysisPlanError(
                "RESULT_SCHEMA_MISMATCH",
                "Las columnas ejecutadas no coinciden con el esquema calculado",
            )
        if not any(isinstance(step, PivotStepV1) for step in plan.steps):
            if runtime_schema != result_schema:
                raise AnalysisPlanError(
                    "RESULT_SCHEMA_MISMATCH",
                    "Los tipos ejecutados no coinciden con el esquema validado",
                )

        total_rows = len(frame)
        if total_rows > self.limits.max_output_rows:
            frame = frame.head(self.limits.max_output_rows)
            warnings.append(RESULT_TRUNCATED)

        columns = [AnalysisColumnV1(name=name, type=runtime_schema[name]) for name in frame.columns]
        rows = [
            [_json_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        self._check_time(started_at)
        return AnalysisResultV1(
            columns=columns,
            rows=rows,
            total_rows=total_rows,
            plan_hash=canonical_plan_hash(plan),
            source_hashes={name: sources[name].sha256 for name in plan.sources},
            warnings=warnings,
        )


def _filter_mask(frame: pd.DataFrame, step: FilterStepV1) -> pd.Series:
    values = frame[step.column]
    operator = step.operator
    if operator == "eq":
        return values.eq(step.value)
    if operator == "ne":
        return values.ne(step.value)
    if operator == "gt":
        return values.gt(step.value)
    if operator == "gte":
        return values.ge(step.value)
    if operator == "lt":
        return values.lt(step.value)
    if operator == "lte":
        return values.le(step.value)
    if operator == "in":
        return values.isin(step.value)
    if operator == "not_in":
        return ~values.isin(step.value)
    if operator == "is_null":
        return values.isna()
    if operator == "not_null":
        return values.notna()
    raise AssertionError(f"Operador de filtro no cubierto por el contrato: {operator}")


def _datetime_part_type(part: str) -> ColumnType:
    if part in {"datetime", "date", "week"}:
        return ColumnType.DATETIME
    if part == "day_part":
        return ColumnType.CATEGORY
    return ColumnType.NUMBER


def _joined_schema(
    left: Mapping[str, ColumnType],
    right: Mapping[str, ColumnType],
    step: JoinStepV1,
) -> dict[str, ColumnType]:
    result = dict(left)
    for column, column_type in right.items():
        if column == step.right_on and step.right_on == step.left_on:
            continue
        result[f"{step.right_prefix}{column}"] = ColumnType(column_type)
    return result


def _unique_internal_column(*frames: pd.DataFrame) -> str:
    existing = {str(column) for frame in frames for column in frame.columns}
    candidate = "__analysis_join_key"
    while candidate in existing:
        candidate += "_"
    return candidate


def _join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    step: JoinStepV1,
) -> tuple[pd.DataFrame, int]:
    try:
        validate_join_cardinality(
            left[step.left_on],
            right[step.right_on],
            step.cardinality,
            auxiliary_role=step.source,
        )
    except DatasetCompositionError as error:
        raise AnalysisPlanError(error.code, str(error)) from error

    # Claves con formatos distintos («7» y 7, « 7 » y 7.0) se cruzan por su texto normalizado.
    left_keys, right_keys = left[step.left_on], right[step.right_on]
    if is_numeric_dtype(left_keys) != is_numeric_dtype(right_keys):
        left_keys, right_keys = _key_text(left_keys), _key_text(right_keys)
    unmatched = int((~left_keys.isin(right_keys)).sum())
    reserved_key = _unique_internal_column(left, right)
    left_key = reserved_key + "_left"
    renamed = right.rename(
        columns={
            column: (reserved_key if column == step.right_on else f"{step.right_prefix}{column}")
            for column in right.columns
        }
    ).assign(**{reserved_key: right_keys.to_numpy()})
    before_rows = len(left)
    try:
        result = left.assign(**{left_key: left_keys.to_numpy()}).merge(
            renamed,
            how=step.how,
            left_on=left_key,
            right_on=reserved_key,
            sort=False,
            validate=step.cardinality,
        )
        result = result.drop(columns=left_key)
    except pd.errors.MergeError as error:
        raise AnalysisPlanError(
            "JOIN_CARDINALITY_VIOLATION",
            f"El cruce con {step.source!r} no cumple {step.cardinality}",
        ) from error
    if len(result) > before_rows:
        raise AnalysisPlanError(
            "ROW_MULTIPLICATION_DETECTED",
            f"El cruce con {step.source!r} multiplicó filas",
        )
    if step.right_on == step.left_on:
        result = result.drop(columns=reserved_key)
    else:
        result = result.rename(columns={reserved_key: f"{step.right_prefix}{step.right_on}"})
    return result, unmatched


def _key_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _aggregate_output_type(
    schema: Mapping[str, ColumnType],
    aggregation: AggregateSpecV1,
) -> ColumnType:
    if aggregation.function in {
        AggregateFunction.COUNT,
        AggregateFunction.NUNIQUE,
        AggregateFunction.SUM,
        AggregateFunction.MEAN,
        AggregateFunction.MEDIAN,
        AggregateFunction.SHARE,
    }:
        return ColumnType.NUMBER
    assert aggregation.column is not None
    return schema[aggregation.column]


def _grouped_aggregation(
    grouped: Any,
    sizes: pd.Series,
    aggregation: AggregateSpecV1,
    frame: pd.DataFrame,
) -> pd.Series:
    function = aggregation.function
    column = aggregation.column
    if function is AggregateFunction.COUNT:
        return sizes if column is None else grouped[column].count()
    if function is AggregateFunction.SHARE:
        numerator = sizes if column is None else grouped[column].count()
        denominator = len(frame) if column is None else int(frame[column].notna().sum())
        if denominator == 0:
            return numerator.astype(float) * 0.0
        return numerator.astype(float).div(denominator).clip(lower=0.0, upper=1.0)
    assert column is not None
    if function is AggregateFunction.NUNIQUE:
        return grouped[column].nunique(dropna=True)
    return grouped[column].agg(function.value)


def _scalar_aggregation(frame: pd.DataFrame, aggregation: AggregateSpecV1) -> Any:
    function = aggregation.function
    column = aggregation.column
    if function is AggregateFunction.COUNT:
        return len(frame) if column is None else int(frame[column].count())
    if function is AggregateFunction.SHARE:
        denominator = len(frame) if column is None else int(frame[column].notna().sum())
        return 1.0 if denominator else 0.0
    assert column is not None
    if function is AggregateFunction.NUNIQUE:
        return int(frame[column].nunique(dropna=True))
    return frame[column].agg(function.value)


def _group_by(frame: pd.DataFrame, step: GroupByStepV1) -> pd.DataFrame:
    if not step.by:
        return pd.DataFrame(
            [
                {
                    aggregation.output_column: _scalar_aggregation(frame, aggregation)
                    for aggregation in step.aggregations
                }
            ]
        )

    grouped = frame.groupby(step.by, dropna=False, observed=True, sort=False)
    sizes = grouped.size()
    result = sizes.index.to_frame(index=False)
    for aggregation in step.aggregations:
        values = _grouped_aggregation(grouped, sizes, aggregation, frame)
        result[aggregation.output_column] = values.reindex(sizes.index).to_numpy()
    return result


def _pivot_column_name(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _pivot_column_name(value.item())
    return str(value)


def _pivot(
    frame: pd.DataFrame,
    schema: Mapping[str, ColumnType],
    step: PivotStepV1,
) -> tuple[pd.DataFrame, dict[str, ColumnType]]:
    distinct_columns = int(frame[step.columns].nunique(dropna=True))
    if len(step.index) + distinct_columns > MAX_PIVOT_COLUMNS:
        raise AnalysisPlanError(
            "PIVOT_COLUMN_LIMIT_EXCEEDED",
            f"La tabla dinámica superaría el máximo de {MAX_PIVOT_COLUMNS} columnas",
        )

    working = frame
    values = step.values
    aggregation = step.aggregation.value
    if step.aggregation in {AggregateFunction.COUNT, AggregateFunction.SHARE}:
        working = frame.copy()
        pivot_value = _unique_internal_column(working)
        if values is None:
            working[pivot_value] = 1.0
        else:
            working[pivot_value] = working[values].notna().astype(float)
        values = pivot_value
        aggregation = "sum"

    assert values is not None
    pivoted = working.pivot_table(
        index=step.index,
        columns=step.columns,
        values=values,
        aggfunc=aggregation,
        observed=True,
        sort=False,
    )
    if step.aggregation is AggregateFunction.SHARE:
        denominator = len(frame) if step.values is None else int(frame[step.values].notna().sum())
        pivoted = pivoted.astype(float) * 0.0 if denominator == 0 else pivoted.div(denominator)
    if step.aggregation in {AggregateFunction.COUNT, AggregateFunction.SHARE}:
        pivoted = pivoted.fillna(0.0)

    dynamic_columns = [_pivot_column_name(column) for column in pivoted.columns]
    if len(set(dynamic_columns)) != len(dynamic_columns) or set(dynamic_columns) & set(step.index):
        raise AnalysisPlanError(
            "PIVOT_COLUMN_COLLISION",
            "Los valores de la columna dinámica producen nombres de salida repetidos",
        )
    pivoted.columns = dynamic_columns
    result = pivoted.reset_index()
    if len(result.columns) > MAX_PIVOT_COLUMNS:
        raise AnalysisPlanError(
            "PIVOT_COLUMN_LIMIT_EXCEEDED",
            f"La tabla dinámica superó el máximo de {MAX_PIVOT_COLUMNS} columnas",
        )

    if step.aggregation in {
        AggregateFunction.COUNT,
        AggregateFunction.NUNIQUE,
        AggregateFunction.SUM,
        AggregateFunction.MEAN,
        AggregateFunction.MEDIAN,
        AggregateFunction.SHARE,
    }:
        value_type = ColumnType.NUMBER
    else:
        assert step.values is not None
        value_type = schema[step.values]
    result_schema = {
        **{column: schema[column] for column in step.index},
        **{column: value_type for column in dynamic_columns},
    }
    return result, result_schema


def _threshold_labels(
    values: pd.Series,
    cutoff: float,
    labels: tuple[str, str],
) -> pd.Series:
    result = values.ge(cutoff).map({False: labels[0], True: labels[1]})
    return result.where(values.notna()).astype("category")


def _derive_datetime(values: pd.Series, part: str) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    failed_count = int((values.notna() & parsed.isna()).sum())
    if part == "datetime":
        return parsed, failed_count
    if part == "hour_of_day":
        return parsed.dt.hour.astype("Int64"), failed_count
    if part == "day_of_week":
        return parsed.dt.dayofweek.astype("Int64"), failed_count
    if part == "day_of_month":
        return parsed.dt.day.astype("Int64"), failed_count
    if part == "month":
        return parsed.dt.month.astype("Int64"), failed_count
    if part == "year":
        return parsed.dt.year.astype("Int64"), failed_count
    if part == "date":
        return parsed.dt.normalize(), failed_count
    if part == "week":
        return (
            parsed.dt.normalize() - pd.to_timedelta(parsed.dt.dayofweek, unit="D"),
            failed_count,
        )
    if part == "day_part":
        return (
            pd.cut(
                parsed.dt.hour,
                bins=[-1, 5, 11, 18, 23],
                labels=DAY_PART_LABELS,
            ),
            failed_count,
        )
    raise AssertionError(f"Parte de fecha no cubierta por el contrato: {part}")


def _format_boundary(value: float) -> str:
    if value == 0:
        value = 0.0
    return format(value, ".12g")


def _derive_numeric_bucket(values: pd.Series, method: str, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    populated = numeric.dropna()
    if populated.empty:
        return pd.Series(pd.Categorical([None] * len(values)), index=values.index)
    if populated.nunique() == 1:
        boundary = _format_boundary(float(populated.iloc[0]))
        label = f"{boundary}–{boundary}"
        result = pd.Series(pd.NA, index=values.index, dtype="string")
        result.loc[numeric.notna()] = label
        return result.astype("category")

    if method == "equal_width":
        bucketed = pd.cut(numeric, bins=bins, duplicates="drop", precision=12)
    elif method == "quantile":
        bucketed = pd.qcut(numeric, q=bins, duplicates="drop", precision=12)
    else:
        raise AssertionError(f"Método de tramos no cubierto por el contrato: {method}")

    categories = bucketed.cat.categories
    minimum = float(populated.min())
    maximum = float(populated.max())
    labels = []
    for index, interval in enumerate(categories):
        left = minimum if index == 0 else float(interval.left)
        right = maximum if index == len(categories) - 1 else float(interval.right)
        labels.append(f"{_format_boundary(left)}–{_format_boundary(right)}")
    return bucketed.cat.rename_categories(labels)


def _derive_calendar_bucket(values: pd.Series, unit: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if unit == "hour":
        return parsed.dt.floor("h")
    if unit == "day":
        return parsed.dt.normalize()
    if unit == "week":
        return parsed.dt.normalize() - pd.to_timedelta(parsed.dt.dayofweek, unit="D")
    if unit == "month":
        return parsed.dt.to_period("M").dt.start_time
    if unit == "quarter":
        return parsed.dt.to_period("Q").dt.start_time
    if unit == "year":
        return parsed.dt.to_period("Y").dt.start_time
    raise AssertionError(f"Unidad de calendario no cubierta por el contrato: {unit}")


def _derive_ratio(numerator: pd.Series, denominator: pd.Series) -> tuple[pd.Series, int]:
    zero_mask = denominator.notna() & denominator.eq(0)
    safe_denominator = denominator.mask(zero_mask)
    return numerator.div(safe_denominator), int(zero_mask.sum())


def _json_value(value: Any) -> JSONScalar:
    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
