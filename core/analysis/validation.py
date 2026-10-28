"""Validación de esquemas para planes de análisis, sin cargar ni ejecutar datos."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from core.analysis.contracts import (
    AggregateFunction,
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
    ThresholdLabelStepV1,
    UnionStepV1,
)
from core.domain.canonical import canonical_hash

MAX_ANALYSIS_STEPS = 25

DATETIME_PART_TYPES = {
    "datetime": ColumnType.DATETIME,
    "hour_of_day": ColumnType.NUMBER,
    "day_of_week": ColumnType.NUMBER,
    "day_of_month": ColumnType.NUMBER,
    "month": ColumnType.NUMBER,
    "year": ColumnType.NUMBER,
    "date": ColumnType.DATETIME,
    "week": ColumnType.DATETIME,
    "day_part": ColumnType.CATEGORY,
}

Schema = dict[str, ColumnType]
SchemaCatalog = Mapping[str, Mapping[str, ColumnType | str]]
JOIN_TEXT_TYPES = {ColumnType.CATEGORY, ColumnType.IDENTIFIER, ColumnType.TEXT}


class AnalysisPlanError(RuntimeError):
    """El plan no puede validarse de forma segura contra los esquemas declarados."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _normalize_schema(source: str, schema: Mapping[str, ColumnType | str]) -> Schema:
    normalized: Schema = {}
    for column, raw_type in schema.items():
        try:
            normalized[column] = ColumnType(raw_type)
        except ValueError as error:
            raise AnalysisPlanError(
                "TYPE_MISMATCH",
                f"La columna {column!r} de la fuente {source!r} tiene un tipo no admitido",
            ) from error
    return normalized


def _require_source(source: str, schemas: SchemaCatalog) -> Schema:
    if source not in schemas:
        raise AnalysisPlanError(
            "SOURCE_NOT_FOUND",
            f"La fuente declarada {source!r} no tiene un esquema disponible",
        )
    return _normalize_schema(source, schemas[source])


def _require_columns(schema: Schema, columns: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in schema]
    if missing:
        raise AnalysisPlanError(
            "COLUMN_NOT_FOUND",
            f"La columna {missing[0]!r} no existe en el esquema resultante",
        )


def _require_type(
    schema: Schema,
    column: str,
    allowed: set[ColumnType],
    *,
    operation: str,
) -> None:
    _require_columns(schema, [column])
    if schema[column] not in allowed:
        expected = ", ".join(sorted(item.value for item in allowed))
        raise AnalysisPlanError(
            "TYPE_MISMATCH",
            f"La operación {operation!r} requiere {expected} en la columna {column!r}",
        )


def _add_derived_column(
    schema: Schema,
    output_column: str,
    output_type: ColumnType,
    *,
    replaceable: set[str] | None = None,
) -> None:
    if output_column in schema and output_column not in (replaceable or set()):
        raise AnalysisPlanError(
            "TYPE_MISMATCH",
            f"La columna derivada {output_column!r} ya existe",
        )
    schema[output_column] = output_type


def _aggregate_type(
    schema: Schema,
    function: AggregateFunction,
    column: str | None,
) -> ColumnType:
    if function in {AggregateFunction.COUNT, AggregateFunction.SHARE}:
        if column is not None:
            _require_columns(schema, [column])
        return ColumnType.NUMBER
    if column is None:
        raise AnalysisPlanError(
            "TYPE_MISMATCH",
            f"El agregado {function.value!r} requiere una columna",
        )
    _require_columns(schema, [column])
    if function is AggregateFunction.NUNIQUE:
        return ColumnType.NUMBER
    if function in {
        AggregateFunction.SUM,
        AggregateFunction.MEAN,
        AggregateFunction.MEDIAN,
    }:
        _require_type(schema, column, {ColumnType.NUMBER}, operation=function.value)
        return ColumnType.NUMBER
    _require_type(
        schema,
        column,
        {ColumnType.DATETIME, ColumnType.NUMBER},
        operation=function.value,
    )
    return schema[column]


def _validate_filter(step: FilterStepV1, schema: Schema) -> None:
    _require_columns(schema, [step.column])
    if step.operator in {"gt", "gte", "lt", "lte"}:
        _require_type(
            schema,
            step.column,
            {ColumnType.DATETIME, ColumnType.NUMBER},
            operation="filter",
        )
    if schema[step.column] is ColumnType.NUMBER and step.value is not None:
        values = step.value if isinstance(step.value, list) else [step.value]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise AnalysisPlanError(
                "TYPE_MISMATCH",
                f"El filtro numérico de {step.column!r} recibió un valor incompatible",
            )


def _validate_group_by(step: GroupByStepV1, schema: Schema) -> Schema:
    _require_columns(schema, step.by)
    result = {column: schema[column] for column in step.by}
    for aggregation in step.aggregations:
        if aggregation.output_column in result:
            raise AnalysisPlanError(
                "TYPE_MISMATCH",
                f"La salida agregada {aggregation.output_column!r} está repetida",
            )
        result[aggregation.output_column] = _aggregate_type(
            schema,
            aggregation.function,
            aggregation.column,
        )
    return result


def _validate_join(step: JoinStepV1, schema: Schema, schemas: SchemaCatalog) -> Schema:
    right = _require_source(step.source, schemas)
    _require_columns(schema, [step.left_on])
    _require_columns(right, [step.right_on])
    left_type = schema[step.left_on]
    right_type = right[step.right_on]
    if (
        left_type is not right_type
        and not {left_type, right_type} <= JOIN_TEXT_TYPES
        # Un código numérico que se repite cruza con la clave única del otro archivo.
        and {left_type, right_type} != {ColumnType.NUMBER, ColumnType.IDENTIFIER}
    ):
        raise AnalysisPlanError(
            "JOIN_KEY_TYPE_MISMATCH",
            "Las claves del cruce tienen tipos distintos",
        )
    result = dict(schema)
    for column, column_type in right.items():
        if column == step.right_on and step.right_on == step.left_on:
            continue
        output_column = f"{step.right_prefix}{column}"
        if output_column in result:
            raise AnalysisPlanError(
                "TYPE_MISMATCH",
                f"El cruce produciría la columna repetida {output_column!r}",
            )
        result[output_column] = column_type
    return result


def _validate_pivot(step: PivotStepV1, schema: Schema) -> Schema:
    required = [*step.index, step.columns]
    if step.values is not None:
        required.append(step.values)
    _require_columns(schema, required)
    value_type = _aggregate_type(schema, step.aggregation, step.values)
    result = {column: schema[column] for column in step.index}
    # La clave de columns describe el tipo común de cada columna dinámica resultante.
    result[step.columns] = value_type
    return result


def validate_plan(
    plan: AnalysisPlanV1 | Mapping[str, object],
    schemas: SchemaCatalog,
) -> Schema:
    """Valida el plan paso a paso y devuelve su esquema lógico final."""

    if not isinstance(plan, AnalysisPlanV1):
        try:
            plan = AnalysisPlanV1.model_validate(plan)
        except ValidationError as error:
            raise AnalysisPlanError("TYPE_MISMATCH", "El contrato del plan no es válido") from error
    if len(plan.steps) > MAX_ANALYSIS_STEPS:
        raise AnalysisPlanError(
            "TOO_MANY_STEPS",
            f"El plan supera el máximo de {MAX_ANALYSIS_STEPS} pasos",
        )
    available = {source: _require_source(source, schemas) for source in plan.sources}
    current = dict(available[plan.sources[0]])

    for step in plan.steps:
        if isinstance(step, SelectStepV1):
            _require_columns(current, step.columns)
            current = {column: current[column] for column in step.columns}
        elif isinstance(step, RenameStepV1):
            _require_columns(current, list(step.mapping))
            renamed = {step.mapping.get(column, column): kind for column, kind in current.items()}
            if len(renamed) != len(current):
                raise AnalysisPlanError(
                    "TYPE_MISMATCH",
                    "El renombrado produciría columnas repetidas",
                )
            current = renamed
        elif isinstance(step, FilterStepV1):
            _validate_filter(step, current)
        elif isinstance(step, DeriveDatetimeStepV1):
            _require_type(
                current,
                step.column,
                {ColumnType.DATETIME, ColumnType.TEXT},
                operation=step.op,
            )
            _add_derived_column(
                current,
                step.output_column,
                DATETIME_PART_TYPES[step.part],
                replaceable={step.column},
            )
        elif isinstance(step, DeriveBucketStepV1):
            allowed = {ColumnType.DATETIME} if step.method == "calendar" else {ColumnType.NUMBER}
            _require_type(current, step.column, allowed, operation=step.op)
            _add_derived_column(current, step.output_column, ColumnType.CATEGORY)
        elif isinstance(step, DeriveRatioStepV1):
            _require_type(current, step.numerator, {ColumnType.NUMBER}, operation=step.op)
            _require_type(current, step.denominator, {ColumnType.NUMBER}, operation=step.op)
            _add_derived_column(current, step.output_column, ColumnType.NUMBER)
        elif isinstance(step, UnionStepV1):
            union_schema = _require_source(step.source, available)
            if current != union_schema:
                raise AnalysisPlanError(
                    "UNION_SCHEMA_MISMATCH",
                    "La unión requiere exactamente las mismas columnas y tipos",
                )
        elif isinstance(step, JoinStepV1):
            current = _validate_join(step, current, available)
        elif isinstance(step, GroupByStepV1):
            current = _validate_group_by(step, current)
        elif isinstance(step, SortStepV1):
            _require_columns(current, step.by)
        elif isinstance(step, LimitStepV1):
            pass
        elif isinstance(step, PivotStepV1):
            current = _validate_pivot(step, current)
        elif isinstance(step, ThresholdLabelStepV1):
            _require_type(current, step.column, {ColumnType.NUMBER}, operation=step.op)
            _add_derived_column(current, step.output_column, ColumnType.CATEGORY)

    output_columns = [
        column for column in (plan.output.x, plan.output.y, plan.output.color) if column is not None
    ]
    _require_columns(current, output_columns)
    return current


def canonical_plan_hash(plan: AnalysisPlanV1 | Mapping[str, object]) -> str:
    """Devuelve el SHA-256 del JSON canónico del contrato validado."""

    model = plan if isinstance(plan, AnalysisPlanV1) else AnalysisPlanV1.model_validate(plan)
    return canonical_hash(model)
