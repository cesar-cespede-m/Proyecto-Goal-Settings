"""Pruebas de GA-00 con esquemas leídos desde los CSV sintéticos del repositorio."""

from __future__ import annotations

import ast
import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.analysis import (
    AnalysisPlanError,
    AnalysisPlanV1,
    ColumnType,
    canonical_plan_hash,
    validate_plan,
)
from core.analysis.contracts import ThresholdLabelStepV1

SYNTHETIC_DIR = Path(__file__).parents[2] / "data" / "synthetic"


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return any(separator in value for separator in ("-", "/", ":"))


def _infer_type(values: Iterable[str]) -> ColumnType:
    populated = [value.strip() for value in values if value.strip()]
    if populated and all(_is_number(value) for value in populated):
        return ColumnType.NUMBER
    if populated and all(_is_datetime(value) for value in populated):
        return ColumnType.DATETIME
    unique = set(populated)
    if populated and len(unique) == len(populated):
        return ColumnType.IDENTIFIER
    if len(unique) <= 20:
        return ColumnType.CATEGORY
    return ColumnType.TEXT


@pytest.fixture(scope="module")
def synthetic_schemas() -> dict[str, dict[str, ColumnType]]:
    files = sorted(SYNTHETIC_DIR.glob("*.csv"))[:3]
    assert len(files) == 3
    schemas: dict[str, dict[str, ColumnType]] = {}
    for index, path in enumerate(files):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            rows = [row for _, row in zip(range(100), reader, strict=False)]
            assert reader.fieldnames
        schemas[f"source_{index}"] = {
            column: _infer_type(row[column] for row in rows) for column in reader.fieldnames
        }
    return schemas


def _plan(
    source: str,
    steps: list[dict[str, object]],
    *,
    output: dict[str, object] | None = None,
    sources: list[str] | None = None,
) -> AnalysisPlanV1:
    return AnalysisPlanV1.model_validate(
        {
            "sources": sources or [source],
            "steps": steps,
            "output": output or {"chart": "table"},
        }
    )


def _column_of_type(
    schema: dict[str, ColumnType],
    accepted: set[ColumnType],
) -> str:
    return next(column for column, kind in schema.items() if kind in accepted)


def test_contract_is_strict_and_discriminates_every_catalog_operation() -> None:
    operations: list[dict[str, object]] = [
        {"op": "select", "columns": ["field_a"]},
        {"op": "rename", "mapping": {"field_a": "field_b"}},
        {"op": "filter", "column": "field_a", "operator": "eq", "value": 1},
        {
            "op": "derive_datetime",
            "column": "field_a",
            "output_column": "field_b",
        },
        {
            "op": "derive_bucket",
            "column": "field_a",
            "output_column": "field_b",
            "method": "equal_width",
            "bins": 4,
        },
        {
            "op": "derive_ratio",
            "numerator": "field_a",
            "denominator": "field_b",
            "output_column": "field_c",
        },
        {"op": "union", "source": "source_1"},
        {
            "op": "join",
            "source": "source_1",
            "left_on": "field_a",
            "right_on": "field_b",
        },
        {
            "op": "group_by",
            "by": ["field_a"],
            "aggregations": [
                {"function": "count", "output_column": "row_count"},
                {"function": "nunique", "column": "field_b", "output_column": "unique_count"},
                {"function": "sum", "column": "field_b", "output_column": "total"},
                {"function": "mean", "column": "field_b", "output_column": "average"},
                {"function": "median", "column": "field_b", "output_column": "middle"},
                {"function": "min", "column": "field_b", "output_column": "minimum"},
                {"function": "max", "column": "field_b", "output_column": "maximum"},
                {"function": "share", "output_column": "share"},
            ],
        },
        {"op": "sort", "by": ["field_a"], "descending": True},
        {"op": "limit", "rows": 50},
        {
            "op": "pivot",
            "index": ["field_a"],
            "columns": "field_b",
            "aggregation": "count",
        },
        {
            "op": "threshold_label",
            "column": "field_a",
            "output_column": "field_b",
            "threshold": 0.5,
        },
    ]
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": ["source_0", "source_1"],
            "steps": operations,
            "output": {"chart": "heatmap", "x": "field_a", "y": "field_b"},
        }
    )

    assert [step.op for step in plan.steps] == [item["op"] for item in operations]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AnalysisPlanV1.model_validate(
            {
                "sources": ["source_0"],
                "steps": [],
                "output": {"chart": "table", "unexpected": True},
            }
        )


def test_validate_plan_reads_three_distinct_synthetic_schemas(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    plan = AnalysisPlanV1(
        sources=list(synthetic_schemas),
        steps=[],
        output={"chart": "table"},
    )

    assert validate_plan(plan, synthetic_schemas) == synthetic_schemas["source_0"]
    assert all(schema for schema in synthetic_schemas.values())


def test_validation_calculates_schema_for_derived_and_aggregated_columns() -> None:
    schemas = {
        "source_0": {
            "measure_a": "number",
            "measure_b": "number",
            "raw_time": "text",
            "segment": "category",
        }
    }
    plan = _plan(
        "source_0",
        [
            {
                "op": "derive_datetime",
                "column": "raw_time",
                "output_column": "event_time",
            },
            {
                "op": "derive_bucket",
                "column": "event_time",
                "output_column": "period",
                "method": "calendar",
                "unit": "month",
            },
            {
                "op": "derive_ratio",
                "numerator": "measure_a",
                "denominator": "measure_b",
                "output_column": "ratio",
            },
            {"op": "filter", "column": "ratio", "operator": "gte", "value": 0},
            {
                "op": "group_by",
                "by": ["period"],
                "aggregations": [
                    {"function": "count", "output_column": "rows"},
                    {"function": "sum", "column": "ratio", "output_column": "total"},
                    {"function": "mean", "column": "ratio", "output_column": "average"},
                    {"function": "median", "column": "ratio", "output_column": "median"},
                    {"function": "min", "column": "ratio", "output_column": "minimum"},
                    {"function": "max", "column": "ratio", "output_column": "maximum"},
                    {"function": "nunique", "column": "ratio", "output_column": "unique"},
                    {"function": "share", "output_column": "share"},
                ],
            },
            {
                "op": "threshold_label",
                "column": "average",
                "output_column": "band",
                "percentile": 0.5,
            },
            {"op": "sort", "by": ["average"], "descending": True},
            {"op": "limit", "rows": 100},
        ],
        output={"chart": "bar", "x": "period", "y": "average", "color": "band"},
    )

    assert validate_plan(plan, schemas) == {
        "period": ColumnType.CATEGORY,
        "rows": ColumnType.NUMBER,
        "total": ColumnType.NUMBER,
        "average": ColumnType.NUMBER,
        "median": ColumnType.NUMBER,
        "minimum": ColumnType.NUMBER,
        "maximum": ColumnType.NUMBER,
        "unique": ColumnType.NUMBER,
        "share": ColumnType.NUMBER,
        "band": ColumnType.CATEGORY,
    }


@pytest.mark.parametrize(
    ("part", "expected_type"),
    [
        ("datetime", ColumnType.DATETIME),
        ("hour_of_day", ColumnType.NUMBER),
        ("day_of_week", ColumnType.NUMBER),
        ("day_of_month", ColumnType.NUMBER),
        ("month", ColumnType.NUMBER),
        ("year", ColumnType.NUMBER),
        ("date", ColumnType.DATETIME),
        ("week", ColumnType.DATETIME),
        ("day_part", ColumnType.CATEGORY),
    ],
)
@pytest.mark.parametrize("source_type", [ColumnType.DATETIME, ColumnType.TEXT])
def test_derive_datetime_part_calculates_result_type(
    part: str,
    expected_type: ColumnType,
    source_type: ColumnType,
) -> None:
    plan = _plan(
        "source_0",
        [
            {
                "op": "derive_datetime",
                "column": "raw_time",
                "output_column": "derived",
                "part": part,
            }
        ],
    )

    assert validate_plan(plan, {"source_0": {"raw_time": source_type}}) == {
        "raw_time": source_type,
        "derived": expected_type,
    }


@pytest.mark.parametrize(
    "part",
    [
        "datetime",
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "date",
        "week",
        "day_part",
    ],
)
def test_derive_datetime_part_rejects_non_temporal_source(part: str) -> None:
    plan = _plan(
        "source_0",
        [
            {
                "op": "derive_datetime",
                "column": "measure",
                "output_column": "derived",
                "part": part,
            }
        ],
    )

    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(plan, {"source_0": {"measure": ColumnType.NUMBER}})
    assert captured.value.code == "TYPE_MISMATCH"


def test_derive_datetime_defaults_to_datetime_part() -> None:
    plan = _plan(
        "source_0",
        [
            {
                "op": "derive_datetime",
                "column": "raw_time",
                "output_column": "parsed_time",
            }
        ],
    )

    assert plan.steps[0].part == "datetime"
    assert validate_plan(plan, {"source_0": {"raw_time": "text"}})["parsed_time"] is (
        ColumnType.DATETIME
    )


def test_threshold_label_defaults_to_spanish_labels() -> None:
    step = ThresholdLabelStepV1(
        op="threshold_label",
        column="measure",
        output_column="label",
        threshold=1,
    )

    assert step.labels == ("no", "sí")


def test_canonical_plan_hash_is_stable_and_sensitive_to_the_plan() -> None:
    first = _plan("source_0", [{"op": "limit", "rows": 10}])
    equivalent = AnalysisPlanV1.model_validate(first.model_dump(mode="json"))
    changed = _plan("source_0", [{"op": "limit", "rows": 11}])

    assert canonical_plan_hash(first) == canonical_plan_hash(equivalent)
    assert canonical_plan_hash(first) != canonical_plan_hash(changed)
    assert len(canonical_plan_hash(first)) == 64


def test_rejects_missing_source_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(_plan("source_missing", []), synthetic_schemas)
    assert captured.value.code == "SOURCE_NOT_FOUND"


def test_rejects_missing_column_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    source = next(iter(synthetic_schemas))
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(
            _plan(source, [{"op": "select", "columns": ["missing_field"]}]),
            synthetic_schemas,
        )
    assert captured.value.code == "COLUMN_NOT_FOUND"


def test_rejects_incompatible_type_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    source, schema = next(
        (name, schema)
        for name, schema in synthetic_schemas.items()
        if any(kind is not ColumnType.NUMBER for kind in schema.values())
    )
    incompatible = _column_of_type(schema, set(ColumnType) - {ColumnType.NUMBER})
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(
            _plan(
                source,
                [
                    {
                        "op": "derive_ratio",
                        "numerator": incompatible,
                        "denominator": incompatible,
                        "output_column": "ratio",
                    }
                ],
            ),
            synthetic_schemas,
        )
    assert captured.value.code == "TYPE_MISMATCH"


def test_group_by_without_columns_calculates_a_single_total_schema(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    source = next(iter(synthetic_schemas))
    plan = _plan(
        source,
        [
            {
                "op": "group_by",
                "by": [],
                "aggregations": [{"function": "count", "output_column": "rows"}],
            }
        ],
    )
    assert validate_plan(plan, synthetic_schemas) == {"rows": ColumnType.NUMBER}


def test_rejects_join_keys_with_different_types_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    items = list(synthetic_schemas.items())
    left_source, left_schema = items[0]
    right_source, right_schema = items[1]
    left_key = _column_of_type(left_schema, {ColumnType.NUMBER})
    right_key = _column_of_type(
        right_schema,
        {ColumnType.CATEGORY, ColumnType.DATETIME, ColumnType.TEXT},
    )
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": [left_source, right_source],
            "steps": [
                {
                    "op": "join",
                    "source": right_source,
                    "left_on": left_key,
                    "right_on": right_key,
                }
            ],
            "output": {"chart": "table"},
        }
    )
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(plan, synthetic_schemas)
    assert captured.value.code == "JOIN_KEY_TYPE_MISMATCH"


def test_join_accepts_compatible_textual_key_roles() -> None:
    schemas = {
        "left": {"key": ColumnType.CATEGORY},
        "right": {"key": ColumnType.IDENTIFIER, "value": ColumnType.NUMBER},
    }
    plan = _plan(
        "left",
        [
            {
                "op": "join",
                "source": "right",
                "left_on": "key",
                "right_on": "key",
                "right_prefix": "right_",
            }
        ],
        sources=["left", "right"],
    )

    assert validate_plan(plan, schemas) == {
        "key": ColumnType.CATEGORY,
        "right_value": ColumnType.NUMBER,
    }


def test_rejects_union_with_different_columns_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    first, second = list(synthetic_schemas)[:2]
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": [first, second],
            "steps": [{"op": "union", "source": second}],
            "output": {"chart": "table"},
        }
    )
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(plan, synthetic_schemas)
    assert captured.value.code == "UNION_SCHEMA_MISMATCH"


def test_rejects_more_than_twenty_five_steps_with_code(
    synthetic_schemas: dict[str, dict[str, ColumnType]],
) -> None:
    source = next(iter(synthetic_schemas))
    plan = _plan(source, [{"op": "limit", "rows": 1} for _ in range(26)])
    with pytest.raises(AnalysisPlanError) as captured:
        validate_plan(plan, synthetic_schemas)
    assert captured.value.code == "TOO_MANY_STEPS"


def test_production_contract_does_not_embed_synthetic_file_or_column_names() -> None:
    production_literals: set[str] = set()
    for path in (Path(__file__).parents[2] / "core" / "analysis").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        production_literals.update(
            node.value.casefold()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    forbidden: set[str] = set()
    for path in SYNTHETIC_DIR.glob("*.csv"):
        forbidden.add(path.stem.casefold())
        with path.open(encoding="utf-8-sig", newline="") as source:
            forbidden.update(column.casefold() for column in next(csv.reader(source)))

    assert not forbidden & production_literals
