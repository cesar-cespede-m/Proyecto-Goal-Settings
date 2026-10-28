"""Contratos estrictos y genéricos para planes declarativos de análisis."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ColumnName = Annotated[str, Field(min_length=1, max_length=255)]
SourceName = Annotated[str, Field(min_length=1, max_length=100)]
type ScalarValue = str | int | float | bool


class StrictAnalysisModel(BaseModel):
    """Base común: el JSON declarativo no admite extensiones implícitas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ColumnType(StrEnum):
    NUMBER = "number"
    CATEGORY = "category"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    PERSONAL = "personal"


class AggregateFunction(StrEnum):
    COUNT = "count"
    NUNIQUE = "nunique"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SHARE = "share"


class AggregateSpecV1(StrictAnalysisModel):
    function: AggregateFunction
    output_column: ColumnName
    column: ColumnName | None = None


class SelectStepV1(StrictAnalysisModel):
    op: Literal["select"]
    columns: list[ColumnName] = Field(min_length=1)


class RenameStepV1(StrictAnalysisModel):
    op: Literal["rename"]
    mapping: dict[ColumnName, ColumnName] = Field(min_length=1)


class FilterStepV1(StrictAnalysisModel):
    op: Literal["filter"]
    column: ColumnName
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "not_in",
        "is_null",
        "not_null",
    ]
    value: ScalarValue | list[ScalarValue] | None = None

    @model_validator(mode="after")
    def operator_matches_value(self) -> FilterStepV1:
        if self.operator in {"is_null", "not_null"} and self.value is not None:
            raise ValueError("Los filtros de vacíos no aceptan un valor")
        if self.operator in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError("Los filtros de pertenencia requieren una lista")
        if self.operator not in {"is_null", "not_null", "in", "not_in"} and (
            self.value is None or isinstance(self.value, list)
        ):
            raise ValueError("El operador requiere un valor escalar")
        return self


class DeriveDatetimeStepV1(StrictAnalysisModel):
    op: Literal["derive_datetime"]
    column: ColumnName
    output_column: ColumnName
    part: Literal[
        "datetime",
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "date",
        "week",
        "day_part",
    ] = "datetime"


class DeriveBucketStepV1(StrictAnalysisModel):
    op: Literal["derive_bucket"]
    column: ColumnName
    output_column: ColumnName
    method: Literal["equal_width", "quantile", "calendar"]
    bins: int | None = Field(default=None, ge=2, le=100)
    unit: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None

    @model_validator(mode="after")
    def method_has_one_parameter(self) -> DeriveBucketStepV1:
        if self.method == "calendar" and (self.unit is None or self.bins is not None):
            raise ValueError("El agrupamiento de calendario requiere unit y no acepta bins")
        if self.method != "calendar" and (self.bins is None or self.unit is not None):
            raise ValueError("El agrupamiento numérico requiere bins y no acepta unit")
        return self


class DeriveRatioStepV1(StrictAnalysisModel):
    op: Literal["derive_ratio"]
    numerator: ColumnName
    denominator: ColumnName
    output_column: ColumnName


class UnionStepV1(StrictAnalysisModel):
    op: Literal["union"]
    source: SourceName


class JoinStepV1(StrictAnalysisModel):
    op: Literal["join"]
    source: SourceName
    left_on: ColumnName
    right_on: ColumnName
    how: Literal["left", "inner"] = "left"
    cardinality: Literal["many_to_one", "one_to_one"] = "many_to_one"
    right_prefix: str = Field(default="", max_length=40, pattern=r"^[a-zA-Z0-9_]*$")


class GroupByStepV1(StrictAnalysisModel):
    op: Literal["group_by"]
    by: list[ColumnName]
    aggregations: list[AggregateSpecV1] = Field(min_length=1)


class SortStepV1(StrictAnalysisModel):
    op: Literal["sort"]
    by: list[ColumnName] = Field(min_length=1)
    descending: bool = False


class LimitStepV1(StrictAnalysisModel):
    op: Literal["limit"]
    rows: int = Field(gt=0, le=5_000)


class PivotStepV1(StrictAnalysisModel):
    op: Literal["pivot"]
    index: list[ColumnName] = Field(min_length=1)
    columns: ColumnName
    values: ColumnName | None = None
    aggregation: AggregateFunction


class ThresholdLabelStepV1(StrictAnalysisModel):
    op: Literal["threshold_label"]
    column: ColumnName
    output_column: ColumnName
    threshold: float | None = None
    percentile: float | None = Field(default=None, ge=0, le=1)
    labels: tuple[str, str] = Field(default=("no", "sí"))

    @model_validator(mode="after")
    def uses_exactly_one_cutoff(self) -> ThresholdLabelStepV1:
        if (self.threshold is None) == (self.percentile is None):
            raise ValueError("Se requiere exactamente un umbral o percentil")
        if self.labels[0] == self.labels[1]:
            raise ValueError("Las dos etiquetas deben ser distintas")
        return self


type AnalysisStepV1 = Annotated[
    SelectStepV1
    | RenameStepV1
    | FilterStepV1
    | DeriveDatetimeStepV1
    | DeriveBucketStepV1
    | DeriveRatioStepV1
    | UnionStepV1
    | JoinStepV1
    | GroupByStepV1
    | SortStepV1
    | LimitStepV1
    | PivotStepV1
    | ThresholdLabelStepV1,
    Field(discriminator="op"),
]


class AnalysisOutputV1(StrictAnalysisModel):
    chart: Literal["table", "bar", "line", "scatter", "heatmap"] = "table"
    x: ColumnName | None = None
    y: ColumnName | None = None
    color: ColumnName | None = None


class AnalysisPlanV1(StrictAnalysisModel):
    schema_version: Literal["analysis-plan.v1"] = "analysis-plan.v1"
    sources: list[SourceName] = Field(min_length=1)
    steps: list[AnalysisStepV1]
    output: AnalysisOutputV1

    @model_validator(mode="after")
    def sources_are_unique(self) -> AnalysisPlanV1:
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("Cada fuente debe declararse una sola vez")
        return self
