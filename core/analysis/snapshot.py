"""Tabla de punto en el tiempo a partir de historias de eventos.

El contrato sólo admite operaciones declarativas conocidas. Las variables se calculan con
eventos estrictamente anteriores al corte y los resultados con eventos estrictamente
posteriores.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import Field, ValidationError, field_validator, model_validator

from core.analysis.contracts import ColumnName, ScalarValue, StrictAnalysisModel

SNAPSHOT_CUTOFF_COLUMN = "snapshot_cutoff"


class SnapshotError(RuntimeError):
    """La tabla de punto en el tiempo no se puede construir de forma segura."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class FixedCutoffV1(StrictAnalysisModel):
    mode: Literal["fixed"]
    at: datetime


class PerEntityFirstEventCutoffV1(StrictAnalysisModel):
    mode: Literal["per_entity_first_event_plus_days"]
    days: int = Field(ge=0, le=36_500)


type CutoffSpecV1 = datetime | FixedCutoffV1 | PerEntityFirstEventCutoffV1


class StageTargetV1(StrictAnalysisModel):
    column: ColumnName
    value: ScalarValue


class FeatureBaseV1(StrictAnalysisModel):
    output_column: ColumnName | None = None


class CountEventsFeatureV1(FeatureBaseV1):
    op: Literal["count_events"]
    window_days: int = Field(gt=0, le=36_500)


class DaysSinceLastEventFeatureV1(FeatureBaseV1):
    op: Literal["days_since_last_event"]


class LastValueFeatureV1(FeatureBaseV1):
    op: Literal["last_value"]
    column: ColumnName


class StageFeatureBaseV1(FeatureBaseV1):
    stage: ColumnName
    order: list[ScalarValue] = Field(min_length=1, max_length=1_000)

    @field_validator("order")
    @classmethod
    def order_is_unique(cls, value: list[ScalarValue]) -> list[ScalarValue]:
        typed = [(type(item).__name__, repr(item)) for item in value]
        if len(typed) != len(set(typed)):
            raise ValueError("Cada etapa debe aparecer una sola vez en order")
        return value


class StagesReachedFeatureV1(StageFeatureBaseV1):
    op: Literal["stages_reached"]


class CurrentStageFeatureV1(StageFeatureBaseV1):
    op: Literal["current_stage"]


class DaysInCurrentStageFeatureV1(StageFeatureBaseV1):
    op: Literal["days_in_current_stage"]


class NumericWindowFeatureV1(FeatureBaseV1):
    op: Literal["sum", "mean", "max"]
    column: ColumnName
    window_days: int = Field(gt=0, le=36_500)


class CountCategoryFeatureV1(FeatureBaseV1):
    op: Literal["count_category"]
    column: ColumnName
    value: ScalarValue
    window_days: int = Field(gt=0, le=36_500)


type SnapshotFeatureV1 = Annotated[
    CountEventsFeatureV1
    | DaysSinceLastEventFeatureV1
    | LastValueFeatureV1
    | StagesReachedFeatureV1
    | CurrentStageFeatureV1
    | DaysInCurrentStageFeatureV1
    | NumericWindowFeatureV1
    | CountCategoryFeatureV1,
    Field(discriminator="op"),
]


class OutcomeBaseV1(StrictAnalysisModel):
    output_column: ColumnName | None = None


class NotReachedStageOutcomeV1(OutcomeBaseV1):
    op: Literal["not_reached_stage"]
    stage: StageTargetV1
    within_days: int = Field(gt=0, le=36_500)


class DeadlineNotReachedStageOutcomeV1(OutcomeBaseV1):
    op: Literal["deadline_not_reached_stage"]
    stage: StageTargetV1
    deadline_column: ColumnName


class InactiveOutcomeV1(OutcomeBaseV1):
    op: Literal["inactive"]
    within_days: int = Field(gt=0, le=36_500)


class ReachedStageOutcomeV1(OutcomeBaseV1):
    op: Literal["reached_stage"]
    stage: StageTargetV1
    within_days: int = Field(gt=0, le=36_500)


class LastStageReachedOutcomeV1(OutcomeBaseV1):
    op: Literal["last_stage_reached"]
    stage: ColumnName
    within_days: int = Field(gt=0, le=36_500)


class DaysUntilStageOutcomeV1(OutcomeBaseV1):
    op: Literal["days_until_stage"]
    stage: StageTargetV1


class FutureSumOutcomeV1(OutcomeBaseV1):
    op: Literal["future_sum"]
    column: ColumnName
    within_days: int = Field(gt=0, le=36_500)


type SnapshotOutcomeV1 = Annotated[
    NotReachedStageOutcomeV1
    | DeadlineNotReachedStageOutcomeV1
    | InactiveOutcomeV1
    | ReachedStageOutcomeV1
    | LastStageReachedOutcomeV1
    | DaysUntilStageOutcomeV1
    | FutureSumOutcomeV1,
    Field(discriminator="op"),
]


class SnapshotSpecV1(StrictAnalysisModel):
    """Receta cerrada para convertir eventos en una fila por entidad."""

    schema_version: Literal["snapshot.v1"] = "snapshot.v1"
    entity: ColumnName
    moment: ColumnName
    cutoff: CutoffSpecV1
    windows_days: list[int] = Field(min_length=1, max_length=100)
    features: list[SnapshotFeatureV1] = Field(min_length=1, max_length=100)
    outcome: list[SnapshotOutcomeV1] = Field(min_length=1, max_length=20)
    entity_attributes: list[ColumnName] | None = None
    # Opt-in: conservar el JSON y CSV históricos cuando no se declara metadata temporal.
    label_available_at_output_column: ColumnName | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("windows_days")
    @classmethod
    def windows_are_positive_and_unique(cls, value: list[int]) -> list[int]:
        if any(days <= 0 for days in value):
            raise ValueError("Las ventanas deben tener una cantidad positiva de días")
        if len(value) != len(set(value)):
            raise ValueError("Cada ventana debe declararse una sola vez")
        return value

    @field_validator("entity_attributes")
    @classmethod
    def attributes_are_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Cada atributo fijo debe declararse una sola vez")
        return value

    @model_validator(mode="after")
    def feature_windows_are_declared(self) -> SnapshotSpecV1:
        declared = set(self.windows_days)
        for feature in self.features:
            window = getattr(feature, "window_days", None)
            if window is not None and window not in declared:
                raise ValueError(
                    f"La ventana de {window} días debe aparecer también en windows_days"
                )
        availability = self.label_available_at_output_column
        if any(isinstance(item, DeadlineNotReachedStageOutcomeV1) for item in self.outcome):
            if availability is None or len(self.outcome) != 1:
                raise ValueError("La promesa por caso requiere una columna de disponibilidad")
        if availability is not None:
            if len(self.outcome) != 1 or self.outcome[0].op not in {
                "not_reached_stage",
                "deadline_not_reached_stage",
                "reached_stage",
                "inactive",
            }:
                raise ValueError(
                    "La disponibilidad requiere un único resultado binario con horizonte fijo"
                )
            reserved = [
                self.entity,
                self.moment,
                SNAPSHOT_CUTOFF_COLUMN,
                *(self.entity_attributes or []),
                *(_feature_name(feature) for feature in self.features),
                *(_outcome_name(outcome) for outcome in self.outcome),
            ]
            if availability.casefold() in {name.casefold() for name in reserved}:
                raise ValueError("La disponibilidad requiere una columna de metadata exclusiva")
        return self


class ObservationCoverageV1(StrictAnalysisModel):
    """Cobertura declarada de este lote, no deducida de la última fecha de evento.

    `complete` declara captura continua para todas las entidades y eventos de la fuente
    durante el intervalo. Fechas sin zona se interpretan como UTC, igual que los eventos.
    La cobertura de un archivo nunca se reutiliza automáticamente para otro archivo.
    """

    schema_version: Literal["observation-coverage.v1"] = "observation-coverage.v1"
    observed_from: datetime
    observed_until: datetime
    complete: bool = False

    @field_validator("observed_from", "observed_until")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> ObservationCoverageV1:
        if self.observed_until < self.observed_from:
            raise ValueError("El fin de cobertura no puede ser anterior a su inicio")
        return self


class SnapshotReadinessV1(StrictAnalysisModel):
    """Evidencia agregada serializable, sin IDs ni filas de la fuente."""

    schema_version: Literal["snapshot-readiness.v1"] = "snapshot-readiness.v1"
    total_rows: int = Field(ge=0)
    eligible_rows: int = Field(ge=0)
    pending_rows: int = Field(ge=0)
    reason_counts: dict[str, int]
    coverage: ObservationCoverageV1 | None = None

    @model_validator(mode="after")
    def counts_are_consistent(self) -> SnapshotReadinessV1:
        if self.eligible_rows + self.pending_rows != self.total_rows:
            raise ValueError("Los casos aptos y pendientes deben sumar el total")
        if self.eligible_rows and (self.coverage is None or not self.coverage.complete):
            raise ValueError("Los casos aptos requieren una cobertura completa declarada")
        if self.pending_rows and not self.reason_counts:
            raise ValueError("Los casos pendientes necesitan un motivo")
        if any(count < 0 or count > self.pending_rows for count in self.reason_counts.values()):
            raise ValueError("Los motivos deben contar casos pendientes, sin duplicarlos")
        return self


class OutcomeSummaryV1(StrictAnalysisModel):
    kind: Literal["boolean", "numeric", "category"]
    total: int = Field(ge=0)
    non_null: int = Field(ge=0)
    proportion_true: float | None = Field(default=None, ge=0, le=1)
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    value_counts: dict[str, int] | None = None


class SnapshotSummaryV1(StrictAnalysisModel):
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    feature_names: list[str]
    outcome_names: list[str]
    outcome_summaries: dict[str, OutcomeSummaryV1]

    @property
    def row_count(self) -> int:
        return self.rows

    @property
    def column_count(self) -> int:
        return self.columns

    @property
    def variable_names(self) -> list[str]:
        return self.feature_names


@dataclass(frozen=True, slots=True)
class SnapshotResultV1:
    dataframe: pd.DataFrame
    summary: SnapshotSummaryV1
    readiness: SnapshotReadinessV1
    row_readiness: pd.DataFrame

    @property
    def frame(self) -> pd.DataFrame:
        return self.dataframe

    def training_frame(self) -> pd.DataFrame:
        """Sólo casos con todos los resultados observables; no agrega metadata como features."""

        eligible = self.row_readiness["eligible_for_training"].to_numpy(dtype=bool)
        return self.dataframe.loc[eligible].copy()


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SnapshotError(
            "SNAPSHOT_COLUMN_NOT_FOUND",
            f"La columna {missing[0]!r} no existe en la tabla de {source}",
        )


def _require_numeric(frame: pd.DataFrame, column: str) -> None:
    _require_columns(frame, [column], source="eventos")
    if is_bool_dtype(frame[column]) or not is_numeric_dtype(frame[column]):
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"La columna {column!r} debe ser numérica para esta operación",
        )


def _feature_name(feature: SnapshotFeatureV1) -> str:
    if feature.output_column is not None:
        return feature.output_column
    if isinstance(feature, CountEventsFeatureV1):
        return f"count_events_{feature.window_days}d"
    if isinstance(feature, DaysSinceLastEventFeatureV1):
        return "days_since_last_event"
    if isinstance(feature, LastValueFeatureV1):
        return f"last_{feature.column}"
    if isinstance(feature, StagesReachedFeatureV1):
        return f"stages_reached_{feature.stage}"
    if isinstance(feature, CurrentStageFeatureV1):
        return f"current_stage_{feature.stage}"
    if isinstance(feature, DaysInCurrentStageFeatureV1):
        return f"days_in_current_stage_{feature.stage}"
    if isinstance(feature, NumericWindowFeatureV1):
        return f"{feature.op}_{feature.column}_{feature.window_days}d"
    return f"count_{feature.column}_{feature.window_days}d"


def _outcome_name(outcome: SnapshotOutcomeV1) -> str:
    if outcome.output_column is not None:
        return outcome.output_column
    if isinstance(outcome, DaysUntilStageOutcomeV1):
        return "outcome_days_until_stage"
    if isinstance(outcome, DeadlineNotReachedStageOutcomeV1):
        return "outcome_deadline_not_reached_stage"
    if isinstance(outcome, LastStageReachedOutcomeV1):
        return f"outcome_last_stage_{outcome.within_days}d"
    if isinstance(outcome, FutureSumOutcomeV1):
        return f"outcome_future_sum_{outcome.column}_{outcome.within_days}d"
    return f"outcome_{outcome.op}_{outcome.within_days}d"


def _timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
    except (TypeError, ValueError) as error:
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"{label} debe contener fechas y horas válidas",
        ) from error
    if isinstance(parsed, pd.DatetimeIndex):
        raise SnapshotError("SNAPSHOT_TYPE_MISMATCH", f"{label} debe ser una fecha única")
    return pd.Timestamp(parsed)


def _parsed_moments(values: pd.Series, column: str) -> pd.Series:
    if is_bool_dtype(values) or is_numeric_dtype(values):
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"La columna de momento {column!r} debe contener fechas y horas",
        )
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"La columna de momento {column!r} contiene fechas vacías o inválidas",
        )
    return parsed


def _same_value(left: Any, right: ScalarValue) -> bool:
    if pd.isna(left):
        return False
    return bool(left == right)


def _stage_position(value: Any, order: list[ScalarValue]) -> int | None:
    for position, expected in enumerate(order):
        if _same_value(value, expected):
            return position
    return None


def _validate_stage_features(events: pd.DataFrame, features: list[SnapshotFeatureV1]) -> None:
    checked: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for feature in features:
        if not isinstance(feature, StageFeatureBaseV1):
            continue
        _require_columns(events, [feature.stage], source="eventos")
        signature = (
            feature.stage,
            tuple((type(value).__name__, repr(value)) for value in feature.order),
        )
        if signature in checked:
            continue
        checked.add(signature)
        outside = [
            value
            for value in events[feature.stage].drop_duplicates().tolist()
            if _stage_position(value, feature.order) is None
        ]
        if outside:
            raise SnapshotError(
                "SNAPSHOT_STAGE_OUT_OF_ORDER",
                f"La etapa {outside[0]!r} de {feature.stage!r} no aparece en order",
            )


def _validate_operation_columns(events: pd.DataFrame, spec: SnapshotSpecV1) -> None:
    for feature in spec.features:
        if isinstance(feature, LastValueFeatureV1 | CountCategoryFeatureV1):
            _require_columns(events, [feature.column], source="eventos")
        if isinstance(feature, NumericWindowFeatureV1):
            _require_numeric(events, feature.column)
    for outcome in spec.outcome:
        if isinstance(
            outcome,
            NotReachedStageOutcomeV1
            | DeadlineNotReachedStageOutcomeV1
            | ReachedStageOutcomeV1
            | DaysUntilStageOutcomeV1,
        ):
            _require_columns(events, [outcome.stage.column], source="eventos")
            if isinstance(outcome, DeadlineNotReachedStageOutcomeV1):
                _require_columns(events, [outcome.deadline_column], source="eventos")
        elif isinstance(outcome, LastStageReachedOutcomeV1):
            _require_columns(events, [outcome.stage], source="eventos")
        elif isinstance(outcome, FutureSumOutcomeV1):
            _require_numeric(events, outcome.column)
    _validate_stage_features(events, spec.features)


def _cutoffs(
    events: pd.DataFrame,
    spec: SnapshotSpecV1,
    parsed_moment: str,
    entities: list[Any],
) -> dict[Any, pd.Timestamp]:
    if isinstance(spec.cutoff, PerEntityFirstEventCutoffV1):
        first = events.groupby(spec.entity, sort=False)[parsed_moment].min()
        delta = pd.Timedelta(days=spec.cutoff.days)
        return {entity: pd.Timestamp(first.loc[entity] + delta) for entity in entities}
    fixed = spec.cutoff.at if isinstance(spec.cutoff, FixedCutoffV1) else spec.cutoff
    cutoff = _timestamp(fixed, label="El momento de corte")
    return dict.fromkeys(entities, cutoff)


def _groups(frame: pd.DataFrame, entity: str) -> dict[Any, pd.DataFrame]:
    return {key: group for key, group in frame.groupby(entity, sort=False)}


def _window(
    history: pd.DataFrame,
    moment_column: str,
    cutoff: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    lower = cutoff - pd.Timedelta(days=days)
    return history.loc[history[moment_column].ge(lower)]


def _current_stage(
    history: pd.DataFrame,
    feature: StageFeatureBaseV1,
) -> tuple[Any, int] | None:
    positions = history[feature.stage].map(lambda value: _stage_position(value, feature.order))
    if positions.empty:
        return None
    position = int(positions.max())
    return feature.order[position], position


def _feature_value(
    history: pd.DataFrame,
    feature: SnapshotFeatureV1,
    *,
    cutoff: pd.Timestamp,
    moment_column: str,
) -> Any:
    if isinstance(feature, CountEventsFeatureV1):
        return int(len(_window(history, moment_column, cutoff, feature.window_days)))
    if isinstance(feature, DaysSinceLastEventFeatureV1):
        if history.empty:
            return None
        return (cutoff - history[moment_column].max()).total_seconds() / 86_400
    if isinstance(feature, LastValueFeatureV1):
        return None if history.empty else history.iloc[-1][feature.column]
    if isinstance(feature, StagesReachedFeatureV1):
        return int(
            len(
                {
                    position
                    for value in history[feature.stage]
                    if (position := _stage_position(value, feature.order)) is not None
                }
            )
        )
    if isinstance(feature, CurrentStageFeatureV1):
        current = _current_stage(history, feature)
        return None if current is None else current[0]
    if isinstance(feature, DaysInCurrentStageFeatureV1):
        current = _current_stage(history, feature)
        if current is None:
            return None
        entered = history.loc[
            history[feature.stage].map(lambda value: _same_value(value, current[0])),
            moment_column,
        ].min()
        return (cutoff - entered).total_seconds() / 86_400
    if isinstance(feature, NumericWindowFeatureV1):
        values = _window(history, moment_column, cutoff, feature.window_days)[
            feature.column
        ].dropna()
        if values.empty:
            return 0.0 if feature.op == "sum" else None
        return float(getattr(values, feature.op)())
    window = _window(history, moment_column, cutoff, feature.window_days)
    return int(window[feature.column].map(lambda value: _same_value(value, feature.value)).sum())


def _horizon(
    future: pd.DataFrame,
    moment_column: str,
    cutoff: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    return future.loc[future[moment_column].le(cutoff + pd.Timedelta(days=days))]


def _frozen_deadline(
    history: pd.DataFrame, column: str, cutoff: pd.Timestamp
) -> tuple[pd.Timestamp | None, str | None]:
    raw = history[column].dropna()
    if raw.empty:
        return None, "DEADLINE_MISSING"
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if parsed.isna().any():
        return None, "DEADLINE_INVALID"
    dates = parsed.drop_duplicates()
    if len(dates) != 1:
        return None, "DEADLINE_CHANGED_BEFORE_CUTOFF"
    deadline = pd.Timestamp(dates.iloc[0])
    if deadline <= cutoff:
        return None, "DEADLINE_BEFORE_CUTOFF"
    return deadline, None


def _reached(frame: pd.DataFrame, target: StageTargetV1) -> bool:
    return bool(frame[target.column].map(lambda value: _same_value(value, target.value)).any())


def _outcome_value(
    future: pd.DataFrame,
    outcome: SnapshotOutcomeV1,
    *,
    cutoff: pd.Timestamp,
    moment_column: str,
    deadline: pd.Timestamp | None = None,
) -> Any:
    if isinstance(outcome, DeadlineNotReachedStageOutcomeV1):
        assert deadline is not None
        within = future.loc[future[moment_column].le(deadline)]
        return not _reached(within, outcome.stage)
    if isinstance(outcome, DaysUntilStageOutcomeV1):
        reached = future.loc[
            future[outcome.stage.column].map(lambda value: _same_value(value, outcome.stage.value))
        ]
        if reached.empty:
            return None
        return (reached[moment_column].min() - cutoff).total_seconds() / 86_400
    within = _horizon(future, moment_column, cutoff, outcome.within_days)
    if isinstance(outcome, NotReachedStageOutcomeV1):
        return not _reached(within, outcome.stage)
    if isinstance(outcome, InactiveOutcomeV1):
        return within.empty
    if isinstance(outcome, ReachedStageOutcomeV1):
        return _reached(within, outcome.stage)
    if isinstance(outcome, LastStageReachedOutcomeV1):
        return None if within.empty else within.iloc[-1][outcome.stage]
    return float(within[outcome.column].sum())


def _observable_outcome(
    future: pd.DataFrame,
    outcome: SnapshotOutcomeV1,
    *,
    cutoff: pd.Timestamp,
    moment_column: str,
    coverage: ObservationCoverageV1 | None,
    deadline: pd.Timestamp | None = None,
    deadline_reason: str | None = None,
) -> tuple[Any, str | None]:
    """Nunca convierte falta de seguimiento en un resultado negativo o una suma cero."""

    if coverage is None:
        return None, "COVERAGE_NOT_DECLARED"
    if not coverage.complete:
        return None, "COVERAGE_INCOMPLETE"
    if pd.Timestamp(coverage.observed_from) > cutoff:
        return None, "COVERAGE_STARTS_AFTER_CUTOFF"
    if isinstance(outcome, DeadlineNotReachedStageOutcomeV1) and deadline_reason is not None:
        return None, deadline_reason
    observed_until = pd.Timestamp(coverage.observed_until)
    horizon = getattr(outcome, "within_days", None)
    # Todos los outcomes acotados esperan su ventana completa, incluso si ya apareció
    # un evento positivo: evita que el entrenamiento seleccione sólo positivos tempranos.
    label_end = (
        deadline
        if isinstance(outcome, DeadlineNotReachedStageOutcomeV1)
        else cutoff + pd.Timedelta(days=horizon)
        if horizon is not None
        else cutoff
    )
    if observed_until < label_end or (horizon is None and observed_until <= cutoff):
        return None, "OUTCOME_WINDOW_NOT_MATURE"
    observed_future = future.loc[future[moment_column].le(observed_until)]
    value = _outcome_value(
        observed_future,
        outcome,
        cutoff=cutoff,
        moment_column=moment_column,
        deadline=deadline,
    )
    if value is None or pd.isna(value):
        # Tiempo hasta evento sin evento observado está censurado. Tampoco inventar una
        # categoría cuando no se observó ninguna etapa en una ventana completa.
        return None, "OUTCOME_NOT_OBSERVED"
    return value, None


def _join_entities(
    result: pd.DataFrame,
    entities: pd.DataFrame | None,
    spec: SnapshotSpecV1,
) -> pd.DataFrame:
    if entities is None:
        if spec.entity_attributes is not None:
            raise SnapshotError(
                "SNAPSHOT_ENTITY_TABLE_REQUIRED",
                "Se declararon atributos fijos, pero no se entregó la tabla de entidades",
            )
        return result
    _require_columns(entities, [spec.entity], source="entidades")
    if entities[spec.entity].isna().any() or entities[spec.entity].duplicated().any():
        raise SnapshotError(
            "SNAPSHOT_ENTITY_NOT_UNIQUE",
            f"La columna {spec.entity!r} debe ser única y no tener vacíos en entidades",
        )
    attributes = spec.entity_attributes
    if attributes is None:
        attributes = [column for column in entities.columns if column != spec.entity]
    _require_columns(entities, attributes, source="entidades")
    selected = [spec.entity, *attributes]
    collisions = (set(selected) - {spec.entity}) & set(result.columns)
    if collisions:
        name = sorted(collisions)[0]
        raise SnapshotError(
            "SNAPSHOT_OUTPUT_COLLISION",
            f"El atributo fijo {name!r} repite una columna del resultado",
        )
    if not result[spec.entity].isin(entities[spec.entity]).all():
        raise SnapshotError(
            "SNAPSHOT_ENTITY_NOT_FOUND",
            "Hay entidades de los eventos que no aparecen en la tabla de atributos fijos",
        )
    try:
        joined = result.merge(
            entities.loc[:, selected],
            on=spec.entity,
            how="left",
            validate="one_to_one",
        )
    except (TypeError, ValueError) as error:
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            "La clave de entidad tiene tipos incompatibles entre las dos tablas",
        ) from error
    return joined


def _summary_for(series: pd.Series, outcome: SnapshotOutcomeV1) -> OutcomeSummaryV1:
    present = series.dropna()
    common = {"total": len(series), "non_null": len(present)}
    if isinstance(
        outcome,
        NotReachedStageOutcomeV1
        | DeadlineNotReachedStageOutcomeV1
        | InactiveOutcomeV1
        | ReachedStageOutcomeV1,
    ):
        proportion = None if present.empty else float(present.astype(bool).mean())
        return OutcomeSummaryV1(kind="boolean", proportion_true=proportion, **common)
    if isinstance(outcome, DaysUntilStageOutcomeV1 | FutureSumOutcomeV1):
        if present.empty:
            return OutcomeSummaryV1(kind="numeric", **common)
        numeric = pd.to_numeric(present)
        return OutcomeSummaryV1(
            kind="numeric",
            mean=float(numeric.mean()),
            minimum=float(numeric.min()),
            maximum=float(numeric.max()),
            **common,
        )
    counts = {str(value): int(count) for value, count in present.value_counts().items()}
    return OutcomeSummaryV1(kind="category", value_counts=counts, **common)


def build_snapshot(
    events: pd.DataFrame,
    spec: SnapshotSpecV1 | Mapping[str, object],
    entities: pd.DataFrame | None = None,
    *,
    coverage: ObservationCoverageV1 | Mapping[str, object] | None = None,
) -> SnapshotResultV1:
    """Construye features y resultados observables; `training_frame()` excluye pendientes.

    Una receta antigua sin cobertura sigue siendo legible, pero no produce etiquetas
    asumidas: todas quedan pendientes hasta declarar cobertura para este lote.
    """

    if not isinstance(events, pd.DataFrame):
        raise SnapshotError("SNAPSHOT_TYPE_MISMATCH", "events debe ser una tabla de pandas")
    if entities is not None and not isinstance(entities, pd.DataFrame):
        raise SnapshotError("SNAPSHOT_TYPE_MISMATCH", "entities debe ser una tabla de pandas")
    if not isinstance(spec, SnapshotSpecV1):
        try:
            spec = SnapshotSpecV1.model_validate(spec)
        except ValidationError as error:
            detail = error.errors(include_url=False)[0]["msg"]
            raise SnapshotError(
                "SNAPSHOT_SPEC_INVALID",
                f"El contrato del snapshot no es válido: {detail}",
            ) from error
    if coverage is not None:
        try:
            coverage = ObservationCoverageV1.model_validate(coverage)
        except ValidationError as error:
            detail = error.errors(include_url=False)[0]["msg"]
            raise SnapshotError(
                "SNAPSHOT_COVERAGE_INVALID", f"La cobertura no es válida: {detail}"
            ) from error

    _require_columns(events, [spec.entity, spec.moment], source="eventos")
    if events.empty:
        raise SnapshotError("SNAPSHOT_EMPTY_EVENTS", "La tabla de eventos no contiene filas")
    if events[spec.entity].isna().any():
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"La columna de entidad {spec.entity!r} no puede tener valores vacíos",
        )
    try:
        entity_values = events[spec.entity].drop_duplicates().tolist()
        set(entity_values)
    except TypeError as error:
        raise SnapshotError(
            "SNAPSHOT_TYPE_MISMATCH",
            f"La columna de entidad {spec.entity!r} debe contener valores simples",
        ) from error

    _validate_operation_columns(events, spec)
    feature_names = [_feature_name(feature) for feature in spec.features]
    outcome_names = [_outcome_name(outcome) for outcome in spec.outcome]
    generated = feature_names + outcome_names
    if len(generated) != len(set(generated)) or {spec.entity, SNAPSHOT_CUTOFF_COLUMN} & set(
        generated
    ):
        raise SnapshotError(
            "SNAPSHOT_OUTPUT_COLLISION",
            "Los nombres de salida deben ser únicos y no repetir columnas reservadas",
        )

    work = events.copy()
    moment_column = "__goal_settings_snapshot_moment__"
    position_column = "__goal_settings_snapshot_position__"
    while moment_column in work.columns:
        moment_column += "_"
    while position_column in work.columns or position_column == moment_column:
        position_column += "_"
    work[moment_column] = _parsed_moments(work[spec.moment], spec.moment)
    work[position_column] = range(len(work))
    work = work.sort_values([moment_column, position_column], kind="stable")

    cutoffs = _cutoffs(work, spec, moment_column, entity_values)
    row_cutoffs = work[spec.entity].map(cutoffs)
    history_mask = work[moment_column].lt(row_cutoffs)
    future_mask = work[moment_column].gt(row_cutoffs)
    history_groups = _groups(work.loc[history_mask], spec.entity)
    future_groups = _groups(work.loc[future_mask], spec.entity)

    result = pd.DataFrame(
        {
            spec.entity: entity_values,
            SNAPSHOT_CUTOFF_COLUMN: [cutoffs[entity] for entity in entity_values],
        }
    )
    result = _join_entities(result, entities, spec)
    empty = work.iloc[0:0]
    for feature, name in zip(spec.features, feature_names, strict=True):
        result[name] = [
            _feature_value(
                history_groups.get(entity, empty),
                feature,
                cutoff=cutoffs[entity],
                moment_column=moment_column,
            )
            for entity in entity_values
        ]
    pending_reasons: list[set[str]] = [set() for _ in entity_values]
    deadline_values: dict[Any, tuple[pd.Timestamp | None, str | None]] = {}
    for outcome in spec.outcome:
        if isinstance(outcome, DeadlineNotReachedStageOutcomeV1):
            deadline_values = {
                entity: _frozen_deadline(
                    history_groups.get(entity, empty), outcome.deadline_column, cutoffs[entity]
                )
                for entity in entity_values
            }
    for outcome, name in zip(spec.outcome, outcome_names, strict=True):
        observed = [
            _observable_outcome(
                future_groups.get(entity, empty),
                outcome,
                cutoff=cutoffs[entity],
                moment_column=moment_column,
                coverage=coverage,
                deadline=deadline_values[entity][0]
                if isinstance(outcome, DeadlineNotReachedStageOutcomeV1)
                else None,
                deadline_reason=deadline_values[entity][1]
                if isinstance(outcome, DeadlineNotReachedStageOutcomeV1)
                else None,
            )
            for entity in entity_values
        ]
        result[name] = [value for value, _reason in observed]
        for reasons, (_value, reason) in zip(pending_reasons, observed, strict=True):
            if reason is not None:
                reasons.add(reason)

    if spec.label_available_at_output_column is not None:
        # Incluso un positivo temprano espera la ventana completa de la receta. Esta fecha
        # no acredita cobertura: los pendientes siguen excluidos por training_frame().
        deadline_outcome = isinstance(spec.outcome[0], DeadlineNotReachedStageOutcomeV1)
        horizon = None if deadline_outcome else spec.outcome[0].within_days
        try:
            result[spec.label_available_at_output_column] = (
                [deadline_values[entity][0] for entity in entity_values]
                if deadline_outcome
                else result[SNAPSHOT_CUTOFF_COLUMN] + pd.Timedelta(days=horizon)
            )
        except (OverflowError, ValueError) as error:
            raise SnapshotError(
                "LABEL_AVAILABILITY_OUT_OF_RANGE", "La disponibilidad excede el rango de fechas"
            ) from error

    eligible = [not reasons for reasons in pending_reasons]
    row_readiness = result[[spec.entity, SNAPSHOT_CUTOFF_COLUMN]].copy()
    row_readiness["eligible_for_training"] = eligible
    row_readiness["pending_reasons"] = [sorted(reasons) for reasons in pending_reasons]
    reason_counts = Counter(reason for reasons in pending_reasons for reason in reasons)

    summaries = {
        name: _summary_for(result[name], outcome)
        for outcome, name in zip(spec.outcome, outcome_names, strict=True)
    }
    return SnapshotResultV1(
        dataframe=result,
        readiness=SnapshotReadinessV1(
            total_rows=len(result),
            eligible_rows=sum(eligible),
            pending_rows=len(result) - sum(eligible),
            reason_counts=dict(sorted(reason_counts.items())),
            coverage=coverage,
        ),
        row_readiness=row_readiness,
        summary=SnapshotSummaryV1(
            rows=len(result),
            columns=len(result.columns),
            feature_names=feature_names,
            outcome_names=outcome_names,
            outcome_summaries=summaries,
        ),
    )


__all__ = [
    "CountCategoryFeatureV1",
    "CountEventsFeatureV1",
    "CurrentStageFeatureV1",
    "DaysInCurrentStageFeatureV1",
    "DaysSinceLastEventFeatureV1",
    "DaysUntilStageOutcomeV1",
    "DeadlineNotReachedStageOutcomeV1",
    "FixedCutoffV1",
    "FutureSumOutcomeV1",
    "InactiveOutcomeV1",
    "LastStageReachedOutcomeV1",
    "LastValueFeatureV1",
    "NotReachedStageOutcomeV1",
    "NumericWindowFeatureV1",
    "ObservationCoverageV1",
    "OutcomeSummaryV1",
    "PerEntityFirstEventCutoffV1",
    "ReachedStageOutcomeV1",
    "SNAPSHOT_CUTOFF_COLUMN",
    "SnapshotError",
    "SnapshotFeatureV1",
    "SnapshotOutcomeV1",
    "SnapshotReadinessV1",
    "SnapshotResultV1",
    "SnapshotSpecV1",
    "SnapshotSummaryV1",
    "StageTargetV1",
    "StagesReachedFeatureV1",
    "build_snapshot",
]
