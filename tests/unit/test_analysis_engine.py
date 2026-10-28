"""Pruebas de GA-02a/b contra negocios intercambiables de la suite sintética."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import core.analysis.engine as engine_module
from core.analysis import (
    AnalysisEngine,
    AnalysisLimits,
    AnalysisPlanError,
    AnalysisPlanV1,
    ColumnType,
    LoadedSource,
    load_project_source,
    validate_plan,
)
from core.data_engine import (
    DataIngestionService,
    DatasetAdmissionRequest,
    IngestionLimits,
    analysis_schema,
    validate_and_profile,
)
from core.data_engine.loading import read_validated_frame
from core.data_engine.validator import validate_csv_bytes
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.support.domains import TestDomain as DomainCase


def _plan(source: str | list[str], steps: list[dict[str, object]]) -> AnalysisPlanV1:
    return AnalysisPlanV1.model_validate(
        {
            "sources": [source] if isinstance(source, str) else source,
            "steps": steps,
            "output": {"chart": "table"},
        }
    )


def _python_scalar(value: Any) -> str | int | float | bool:
    return value.item() if hasattr(value, "item") else value


def _load_path(path: Path) -> LoadedSource:
    content = path.read_bytes()
    profile = validate_and_profile(path.name, content)
    validated = validate_csv_bytes(path.name, content, IngestionLimits())
    return LoadedSource(
        frame=read_validated_frame(validated),
        sha256=hashlib.sha256(content).hexdigest(),
        schema={
            name: ColumnType(kind)
            for name, kind in analysis_schema(profile.model_dump(mode="json")).items()
        },
    )


def _load_domain(domain: DomainCase) -> LoadedSource:
    return _load_path(domain.files[0])


@pytest.fixture(params=DOMAINS[:3], ids=lambda domain: domain.key)
def domain_source(request: pytest.FixtureRequest) -> LoadedSource:
    return _load_domain(request.param)


def test_select_rename_sort_and_limit_run_in_declared_order(domain_source: LoadedSource) -> None:
    numeric = next(name for name, kind in domain_source.schema.items() if kind is ColumnType.NUMBER)
    companion = next(name for name in domain_source.schema if name != numeric)
    plan = _plan(
        "source",
        [
            {"op": "select", "columns": [numeric, companion]},
            {"op": "rename", "mapping": {numeric: "measure"}},
            {"op": "sort", "by": ["measure"], "descending": True},
            {"op": "limit", "rows": 3},
        ],
    )

    result = AnalysisEngine().run(plan, {"source": domain_source})

    assert [column.name for column in result.columns] == ["measure", companion]
    assert result.total_rows == 3
    assert len(result.rows) == 3
    assert [row[0] for row in result.rows] == sorted([row[0] for row in result.rows], reverse=True)


@pytest.mark.parametrize(
    "operator",
    ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_null", "not_null"],
)
def test_every_filter_operator_builds_a_mask_in_code(
    domain_source: LoadedSource,
    operator: str,
) -> None:
    column = next(name for name, kind in domain_source.schema.items() if kind is ColumnType.NUMBER)
    frame = domain_source.frame.copy()
    populated = frame[column].dropna()
    scalar = _python_scalar(populated.iloc[len(populated) // 2])
    if operator in {"is_null", "not_null"}:
        frame.loc[frame.index[0], column] = pd.NA
        value: object = None
        expected = frame[column].isna() if operator == "is_null" else frame[column].notna()
    elif operator in {"in", "not_in"}:
        value = [scalar]
        expected = frame[column].isin(value)
        if operator == "not_in":
            expected = ~expected
    else:
        value = scalar
        comparisons = {
            "eq": frame[column].eq(scalar),
            "ne": frame[column].ne(scalar),
            "gt": frame[column].gt(scalar),
            "gte": frame[column].ge(scalar),
            "lt": frame[column].lt(scalar),
            "lte": frame[column].le(scalar),
        }
        expected = comparisons[operator]
    source = LoadedSource(frame=frame, sha256=domain_source.sha256, schema=domain_source.schema)
    step: dict[str, object] = {"op": "filter", "column": column, "operator": operator}
    if value is not None:
        step["value"] = value

    result = AnalysisEngine().run(_plan("source", [step]), {"source": source})

    assert result.total_rows == int(expected.sum())


def test_output_is_json_safe_and_truncated_with_a_warning() -> None:
    frame = pd.DataFrame(
        {
            "measure": [1.0, float("nan"), 3.0],
            "moment": [pd.Timestamp("2026-01-01"), pd.NaT, pd.Timestamp("2026-01-03")],
        }
    )
    source = LoadedSource(
        frame=frame,
        sha256="a" * 64,
        schema={"measure": ColumnType.NUMBER, "moment": ColumnType.DATETIME},
    )

    result = AnalysisEngine(AnalysisLimits(max_output_rows=2)).run(
        _plan("source", []), {"source": source}
    )

    assert result.total_rows == 3
    assert len(result.rows) == 2
    assert result.rows[0][1] == "2026-01-01T00:00:00"
    assert result.rows[1] == [None, None]
    assert result.warnings == ["RESULT_TRUNCATED"]
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_input_row_limit_is_enforced(domain_source: LoadedSource) -> None:
    with pytest.raises(AnalysisPlanError) as captured:
        AnalysisEngine(AnalysisLimits(max_input_rows=1)).run(
            _plan("source", []), {"source": domain_source}
        )

    assert captured.value.code == "ANALYSIS_INPUT_ROW_LIMIT_EXCEEDED"


def test_time_limit_is_enforced(
    domain_source: LoadedSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(engine_module, "perf_counter", lambda: next(ticks))

    with pytest.raises(AnalysisPlanError) as captured:
        AnalysisEngine(AnalysisLimits(max_seconds=0.5)).run(
            _plan("source", []), {"source": domain_source}
        )

    assert captured.value.code == "ANALYSIS_TIME_LIMIT_EXCEEDED"


def test_count_by_group_runs_on_three_interchangeable_businesses(
    domain_source: LoadedSource,
) -> None:
    group_column = min(
        domain_source.schema,
        key=lambda column: domain_source.frame[column].nunique(dropna=False),
    )
    plan = _plan(
        "source",
        [
            {
                "op": "group_by",
                "by": [group_column],
                "aggregations": [{"function": "count", "output_column": "row_count"}],
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": domain_source})

    assert sum(row[-1] for row in result.rows) == len(domain_source.frame)
    assert {column.name: column.type for column in result.columns} == validate_plan(
        plan, {"source": domain_source.schema}
    )


def test_union_stacks_same_schema_even_when_source_column_order_differs() -> None:
    source = _load_domain(DOMAINS[2])
    split = len(source.frame) // 2
    first = LoadedSource(
        frame=source.frame.iloc[:split].copy(),
        sha256="1" * 64,
        schema=source.schema,
    )
    reversed_columns = list(reversed(source.frame.columns))
    second = LoadedSource(
        frame=source.frame.iloc[split:].loc[:, reversed_columns].copy(),
        sha256="2" * 64,
        schema={column: source.schema[column] for column in reversed_columns},
    )
    plan = _plan(["first", "second"], [{"op": "union", "source": "second"}])

    result = AnalysisEngine().run(plan, {"first": first, "second": second})

    assert result.total_rows == len(source.frame)
    assert [column.name for column in result.columns] == list(source.frame.columns)
    assert {column.name: column.type for column in result.columns} == validate_plan(
        plan, {"first": first.schema, "second": second.schema}
    )


def test_group_by_supports_every_aggregation_and_one_total() -> None:
    source = _load_domain(DOMAINS[0])
    numeric = next(name for name, kind in source.schema.items() if kind is ColumnType.NUMBER)
    group = min(
        (
            name
            for name, kind in source.schema.items()
            if kind in {ColumnType.CATEGORY, ColumnType.TEXT}
        ),
        key=lambda column: source.frame[column].nunique(dropna=False),
    )
    aggregations = [
        {"function": "count", "output_column": "rows"},
        {"function": "nunique", "column": numeric, "output_column": "unique"},
        {"function": "sum", "column": numeric, "output_column": "total"},
        {"function": "mean", "column": numeric, "output_column": "average"},
        {"function": "median", "column": numeric, "output_column": "middle"},
        {"function": "min", "column": numeric, "output_column": "minimum"},
        {"function": "max", "column": numeric, "output_column": "maximum"},
        {"function": "share", "output_column": "share"},
    ]
    grouped_plan = _plan(
        "source",
        [{"op": "group_by", "by": [group], "aggregations": aggregations}],
    )
    total_plan = _plan(
        "source",
        [{"op": "group_by", "by": [], "aggregations": aggregations}],
    )

    grouped = AnalysisEngine().run(grouped_plan, {"source": source})
    total = AnalysisEngine().run(total_plan, {"source": source})

    share_position = next(
        index for index, column in enumerate(grouped.columns) if column.name == "share"
    )
    assert sum(row[share_position] for row in grouped.rows) == pytest.approx(1.0)
    assert total.total_rows == 1
    assert total.rows[0][0] == len(source.frame)
    assert total.rows[0][-1] == 1.0
    assert {column.name: column.type for column in grouped.columns} == validate_plan(
        grouped_plan, {"source": source.schema}
    )
    assert {column.name: column.type for column in total.columns} == validate_plan(
        total_plan, {"source": source.schema}
    )


def test_pivot_uses_values_as_columns_and_validated_cell_type() -> None:
    source = _load_domain(DOMAINS[1])
    categorical = [
        name
        for name, kind in source.schema.items()
        if kind in {ColumnType.CATEGORY, ColumnType.TEXT}
        and source.frame[name].nunique(dropna=True) <= 10
    ]
    index, columns = categorical[:2]
    values = next(name for name, kind in source.schema.items() if kind is ColumnType.NUMBER)
    plan = _plan(
        "source",
        [
            {
                "op": "pivot",
                "index": [index],
                "columns": columns,
                "values": values,
                "aggregation": "mean",
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})
    validated = validate_plan(plan, {"source": source.schema})

    assert result.columns[0].name == index
    assert result.columns[0].type is source.schema[index]
    assert all(column.type is validated[columns] for column in result.columns[1:])
    assert len(result.columns) <= 50


@pytest.mark.parametrize(
    ("aggregation", "values"),
    [
        ("count", None),
        ("nunique", "measure"),
        ("sum", "measure"),
        ("mean", "measure"),
        ("median", "measure"),
        ("min", "measure"),
        ("max", "measure"),
        ("share", None),
    ],
)
def test_pivot_executes_every_supported_aggregation(
    aggregation: str,
    values: str | None,
) -> None:
    source = LoadedSource(
        frame=pd.DataFrame(
            {
                "row": ["a", "a", "b"],
                "column": ["x", "y", "x"],
                "measure": [1.0, 2.0, 3.0],
            }
        ),
        sha256="8" * 64,
        schema={
            "row": ColumnType.CATEGORY,
            "column": ColumnType.CATEGORY,
            "measure": ColumnType.NUMBER,
        },
    )
    step: dict[str, object] = {
        "op": "pivot",
        "index": ["row"],
        "columns": "column",
        "aggregation": aggregation,
    }
    if values is not None:
        step["values"] = values
    plan = _plan("source", [step])

    result = AnalysisEngine().run(plan, {"source": source})
    validated = validate_plan(plan, {"source": source.schema})

    assert [column.name for column in result.columns] == ["row", "x", "y"]
    assert all(column.type is validated["column"] for column in result.columns[1:])
    if aggregation in {"count", "share"}:
        assert result.rows[1][-1] == 0.0
    if aggregation == "share":
        assert sum(value for row in result.rows for value in row[1:]) == pytest.approx(1.0)


def _joinable_sources_with_datetime() -> tuple[LoadedSource, LoadedSource, str, str]:
    domain = next(item for item in DOMAINS if item.multi_file and item.has_datetimes)
    loaded = [_load_path(path) for path in domain.files]
    for left in loaded:
        datetime_columns = [
            name for name, kind in left.schema.items() if kind is ColumnType.DATETIME
        ]
        if not datetime_columns:
            continue
        for right in loaded:
            if right is left:
                continue
            for key in set(left.schema) & set(right.schema):
                key_types = {left.schema[key], right.schema[key]}
                if (
                    (
                        len(key_types) == 1
                        or key_types
                        <= {ColumnType.CATEGORY, ColumnType.IDENTIFIER, ColumnType.TEXT}
                    )
                    and right.frame[key].notna().all()
                    and right.frame[key].is_unique
                ):
                    return left, right, key, datetime_columns[0]
    raise AssertionError("El fixture multiarchivo no expuso un cruce válido con fecha")


def test_multifile_join_counts_by_group_and_hour_then_labels_percentile() -> None:
    left, right, key, datetime_column = _joinable_sources_with_datetime()
    right_group = min(
        (name for name in right.schema if name != key),
        key=lambda column: right.frame[column].nunique(dropna=False),
    )
    prefixed_group = f"right_{right_group}"
    plan = _plan(
        ["left", "right"],
        [
            {
                "op": "join",
                "source": "right",
                "left_on": key,
                "right_on": key,
                "how": "left",
                "cardinality": "many_to_one",
                "right_prefix": "right_",
            },
            {
                "op": "derive_datetime",
                "column": datetime_column,
                "output_column": "hour",
                "part": "hour_of_day",
            },
            {
                "op": "group_by",
                "by": [prefixed_group, "hour"],
                "aggregations": [{"function": "count", "output_column": "rows"}],
            },
            {
                "op": "threshold_label",
                "column": "rows",
                "output_column": "at_or_above_cutoff",
                "percentile": 0.75,
            },
        ],
    )

    result = AnalysisEngine(AnalysisLimits(max_output_rows=100_000)).run(
        plan, {"left": left, "right": right}
    )

    row_count_position = next(
        index for index, column in enumerate(result.columns) if column.name == "rows"
    )
    label_position = next(
        index for index, column in enumerate(result.columns) if column.name == "at_or_above_cutoff"
    )
    assert sum(row[row_count_position] for row in result.rows) == len(left.frame)
    assert {row[label_position] for row in result.rows} <= {"no", "sí"}
    assert any(warning.startswith("THRESHOLD_VALUE:") for warning in result.warnings)
    assert {column.name: column.type for column in result.columns} == validate_plan(
        plan, {"left": left.schema, "right": right.schema}
    )


@pytest.mark.parametrize(("how", "expected_rows"), [("left", 3), ("inner", 2)])
def test_join_warns_for_unmatched_rows_without_multiplication(
    how: str,
    expected_rows: int,
) -> None:
    left = LoadedSource(
        frame=pd.DataFrame({"key": [1, 2, 3], "measure": [10.0, 20.0, 30.0]}),
        sha256="3" * 64,
        schema={"key": ColumnType.IDENTIFIER, "measure": ColumnType.NUMBER},
    )
    right = LoadedSource(
        frame=pd.DataFrame({"key": [1, 2], "label": ["a", "b"]}),
        sha256="4" * 64,
        schema={"key": ColumnType.IDENTIFIER, "label": ColumnType.CATEGORY},
    )
    plan = _plan(
        ["left", "right"],
        [
            {
                "op": "join",
                "source": "right",
                "left_on": "key",
                "right_on": "key",
                "how": how,
                "cardinality": "many_to_one",
                "right_prefix": "right_",
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"left": left, "right": right})

    assert result.total_rows == expected_rows
    assert result.total_rows <= len(left.frame)
    assert result.warnings == ["JOIN_UNMATCHED_ROWS:1"]


def test_join_rejects_duplicate_right_keys_clearly() -> None:
    left = LoadedSource(
        frame=pd.DataFrame({"key": [1, 2]}),
        sha256="5" * 64,
        schema={"key": ColumnType.IDENTIFIER},
    )
    right = LoadedSource(
        frame=pd.DataFrame({"key": [1, 1], "value": [10.0, 20.0]}),
        sha256="6" * 64,
        schema={"key": ColumnType.IDENTIFIER, "value": ColumnType.NUMBER},
    )
    plan = _plan(
        ["left", "right"],
        [
            {
                "op": "join",
                "source": "right",
                "left_on": "key",
                "right_on": "key",
                "cardinality": "many_to_one",
                "right_prefix": "right_",
            }
        ],
    )

    with pytest.raises(AnalysisPlanError) as captured:
        AnalysisEngine().run(plan, {"left": left, "right": right})

    assert captured.value.code == "AUXILIARY_KEY_NOT_UNIQUE"


def test_one_to_one_join_rejects_duplicate_left_keys() -> None:
    left = LoadedSource(
        frame=pd.DataFrame({"key": [1, 1]}),
        sha256="9" * 64,
        schema={"key": ColumnType.IDENTIFIER},
    )
    right = LoadedSource(
        frame=pd.DataFrame({"key": [1], "value": [10.0]}),
        sha256="a" * 64,
        schema={"key": ColumnType.IDENTIFIER, "value": ColumnType.NUMBER},
    )
    plan = _plan(
        ["left", "right"],
        [
            {
                "op": "join",
                "source": "right",
                "left_on": "key",
                "right_on": "key",
                "cardinality": "one_to_one",
                "right_prefix": "right_",
            }
        ],
    )

    with pytest.raises(AnalysisPlanError) as captured:
        AnalysisEngine().run(plan, {"left": left, "right": right})

    assert captured.value.code == "PRIMARY_KEY_NOT_UNIQUE"


def test_fixed_threshold_uses_two_custom_labels_without_warning() -> None:
    source = LoadedSource(
        frame=pd.DataFrame({"measure": [1.0, 2.0, None]}),
        sha256="b" * 64,
        schema={"measure": ColumnType.NUMBER},
    )
    plan = _plan(
        "source",
        [
            {
                "op": "threshold_label",
                "column": "measure",
                "output_column": "label",
                "threshold": 2.0,
                "labels": ["below", "at_or_above"],
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    assert [row[-1] for row in result.rows] == ["below", "at_or_above", None]
    assert result.warnings == []
    assert result.columns[-1].type is ColumnType.CATEGORY


def test_pivot_rejects_more_than_fifty_result_columns() -> None:
    source = LoadedSource(
        frame=pd.DataFrame(
            {
                "row": ["all"] * 50,
                "column": [f"value_{index}" for index in range(50)],
            }
        ),
        sha256="7" * 64,
        schema={"row": ColumnType.CATEGORY, "column": ColumnType.CATEGORY},
    )
    plan = _plan(
        "source",
        [
            {
                "op": "pivot",
                "index": ["row"],
                "columns": "column",
                "aggregation": "count",
            }
        ],
    )

    with pytest.raises(AnalysisPlanError) as captured:
        AnalysisEngine().run(plan, {"source": source})

    assert captured.value.code == "PIVOT_COLUMN_LIMIT_EXCEEDED"


@pytest.fixture
def dated_domain_source() -> tuple[LoadedSource, str]:
    domain = next(item for item in DOMAINS if item.has_datetimes and item.multi_file)
    for path in domain.files:
        source = _load_path(path)
        columns = [name for name, kind in source.schema.items() if kind is ColumnType.DATETIME]
        if columns:
            return source, columns[0]
    raise AssertionError("El negocio de prueba con fechas no expuso una fecha")


def test_every_datetime_part_matches_validated_type(
    dated_domain_source: tuple[LoadedSource, str],
) -> None:
    source, datetime_column = dated_domain_source
    parts = {
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
    steps = [
        {
            "op": "derive_datetime",
            "column": datetime_column,
            "output_column": f"derived_{part}",
            "part": part,
        }
        for part in parts
    ]
    plan = _plan("source", steps)

    result = AnalysisEngine().run(plan, {"source": source})

    result_types = {column.name: column.type for column in result.columns}
    validated = validate_plan(plan, {"source": source.schema})
    assert result_types == validated
    assert {result_types[f"derived_{part}"] for part in parts} == set(parts.values())
    assert all(row[-1] in {"madrugada", "mañana", "tarde", "noche", None} for row in result.rows)
    first_position = int(source.frame[datetime_column].first_valid_index())
    first = pd.Timestamp(source.frame.iloc[first_position][datetime_column])
    row = dict(
        zip(
            (column.name for column in result.columns),
            result.rows[first_position],
            strict=True,
        )
    )
    expected_day_part = ("madrugada", "mañana", "tarde", "noche")[
        0 if first.hour <= 5 else 1 if first.hour <= 11 else 2 if first.hour <= 18 else 3
    ]
    assert row["derived_datetime"] == first.isoformat()
    assert row["derived_hour_of_day"] == first.hour
    assert row["derived_day_of_week"] == first.dayofweek
    assert row["derived_day_of_month"] == first.day
    assert row["derived_month"] == first.month
    assert row["derived_year"] == first.year
    assert row["derived_date"] == first.normalize().isoformat()
    assert (
        row["derived_week"] == (first.normalize() - pd.Timedelta(days=first.dayofweek)).isoformat()
    )
    assert row["derived_day_part"] == expected_day_part


def test_activity_can_be_counted_by_hour_and_monday_based_weekday(
    dated_domain_source: tuple[LoadedSource, str],
) -> None:
    source, datetime_column = dated_domain_source
    plan = _plan(
        "source",
        [
            {
                "op": "derive_datetime",
                "column": datetime_column,
                "output_column": "activity_hour",
                "part": "hour_of_day",
            },
            {
                "op": "derive_datetime",
                "column": datetime_column,
                "output_column": "activity_weekday",
                "part": "day_of_week",
            },
        ],
    )

    result = AnalysisEngine(AnalysisLimits(max_output_rows=100_000)).run(plan, {"source": source})

    observed_hours = Counter(row[-2] for row in result.rows if row[-2] is not None)
    observed_weekdays = Counter(row[-1] for row in result.rows if row[-1] is not None)
    parsed = pd.to_datetime(source.frame[datetime_column], errors="coerce", format="mixed")
    assert observed_hours == Counter(parsed.dt.hour.dropna().astype(int))
    assert observed_weekdays == Counter(parsed.dt.dayofweek.dropna().astype(int))
    assert set(observed_hours) <= set(range(24))
    assert set(observed_weekdays) <= set(range(7))


def test_datetime_text_parse_failures_are_empty_and_counted() -> None:
    source = LoadedSource(
        frame=pd.DataFrame({"raw_moment": ["2026-01-05 08:15:00", "unreadable", None]}),
        sha256="b" * 64,
        schema={"raw_moment": ColumnType.TEXT},
    )
    plan = _plan(
        "source",
        [
            {
                "op": "derive_datetime",
                "column": "raw_moment",
                "output_column": "parsed_moment",
                "part": "datetime",
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    assert result.rows[0][-1] == "2026-01-05T08:15:00"
    assert result.rows[1][-1] is None
    assert result.rows[2][-1] is None
    assert result.warnings == ["DATETIME_PARSE_FAILED:1"]


def test_day_part_boundaries_and_monday_week_start_are_exact() -> None:
    moments = pd.to_datetime(
        [
            "2026-01-05 00:00:00",
            "2026-01-05 05:59:59",
            "2026-01-05 06:00:00",
            "2026-01-05 11:59:59",
            "2026-01-05 12:00:00",
            "2026-01-05 18:59:59",
            "2026-01-05 19:00:00",
            "2026-01-11 23:59:59",
        ]
    )
    source = LoadedSource(
        frame=pd.DataFrame({"moment": moments}),
        sha256="c" * 64,
        schema={"moment": ColumnType.DATETIME},
    )
    plan = _plan(
        "source",
        [
            {
                "op": "derive_datetime",
                "column": "moment",
                "output_column": "period",
                "part": "day_part",
            },
            {
                "op": "derive_datetime",
                "column": "moment",
                "output_column": "week_start",
                "part": "week",
            },
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    assert [row[-2] for row in result.rows] == [
        "madrugada",
        "madrugada",
        "mañana",
        "mañana",
        "tarde",
        "tarde",
        "noche",
        "noche",
    ]
    assert {row[-1] for row in result.rows} == {"2026-01-05T00:00:00"}


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("hour", "2026-05-17T13:00:00"),
        ("day", "2026-05-17T00:00:00"),
        ("week", "2026-05-11T00:00:00"),
        ("month", "2026-05-01T00:00:00"),
        ("quarter", "2026-04-01T00:00:00"),
        ("year", "2026-01-01T00:00:00"),
    ],
)
def test_calendar_bucket_truncates_each_unit(unit: str, expected: str) -> None:
    source = LoadedSource(
        frame=pd.DataFrame({"moment": [pd.Timestamp("2026-05-17 13:45:59")]}),
        sha256="d" * 64,
        schema={"moment": ColumnType.DATETIME},
    )
    plan = _plan(
        "source",
        [
            {
                "op": "derive_bucket",
                "column": "moment",
                "output_column": "calendar_period",
                "method": "calendar",
                "unit": unit,
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    assert result.rows[0][-1] == expected
    assert result.columns[-1].type is ColumnType.CATEGORY


@pytest.mark.parametrize("method", ["equal_width", "quantile"])
def test_numeric_buckets_have_readable_ranges_on_another_business(method: str) -> None:
    source = _load_domain(DOMAINS[0])
    numeric = next(name for name, kind in source.schema.items() if kind is ColumnType.NUMBER)
    plan = _plan(
        "source",
        [
            {
                "op": "derive_bucket",
                "column": numeric,
                "output_column": "range_label",
                "method": method,
                "bins": 4,
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    labels = {row[-1] for row in result.rows if row[-1] is not None}
    assert labels
    assert all("–" in label for label in labels)
    assert len(labels) <= 4
    assert result.columns[-1].type is ColumnType.CATEGORY


def test_ratio_uses_another_business_and_warns_for_zero_denominators() -> None:
    source = _load_domain(DOMAINS[0])
    numeric_columns = [name for name, kind in source.schema.items() if kind is ColumnType.NUMBER]
    numerator, denominator = numeric_columns[:2]
    frame = source.frame.copy()
    frame.loc[frame.index[:2], denominator] = [0, -0.0]
    source = LoadedSource(frame=frame, sha256=source.sha256, schema=source.schema)
    plan = _plan(
        "source",
        [
            {
                "op": "derive_ratio",
                "numerator": numerator,
                "denominator": denominator,
                "output_column": "ratio",
            }
        ],
    )

    result = AnalysisEngine().run(plan, {"source": source})

    assert result.rows[0][-1] is None
    assert result.rows[1][-1] is None
    populated_position = next(
        position
        for position, (_, row) in enumerate(frame.iterrows())
        if row[numerator] is not None
        and not pd.isna(row[numerator])
        and row[denominator] is not None
        and not pd.isna(row[denominator])
        and row[denominator] != 0
    )
    assert result.rows[populated_position][-1] == pytest.approx(
        frame.iloc[populated_position][numerator] / frame.iloc[populated_position][denominator]
    )
    assert result.warnings == ["DIVISION_BY_ZERO:2"]
    assert result.columns[-1].type is ColumnType.NUMBER


@pytest.fixture
def persisted_domain(
    tmp_path: Path,
) -> tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes]:
    connection = get_connection(tmp_path / "analysis.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("GA-02a synthetic fixture")
    storage_root = tmp_path / "private"
    path = DOMAINS[0].files[0]
    content = path.read_bytes()
    result = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request=DatasetAdmissionRequest(
            original_filename_display=path.name,
            source_name="Synthetic pytest fixture",
            source_url="https://example.invalid/analysis",
            license="test-only",
            classification="synthetic",
            authorization_status="approved",
        ),
        content=content,
    )
    yield connection, repository, storage_root, result.metadata.dataset_id, content
    connection.close()


def test_load_project_source_uses_registered_hash_and_profile(
    persisted_domain: tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes],
) -> None:
    _, repository, storage_root, dataset_id, content = persisted_domain

    source = load_project_source(repository, storage_root, dataset_id)

    assert source.sha256 == hashlib.sha256(content).hexdigest()
    assert len(source.frame) > 0
    assert set(source.frame.columns) == set(source.schema)


def test_load_project_source_rejects_changed_content(
    persisted_domain: tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes],
) -> None:
    _, repository, storage_root, dataset_id, content = persisted_domain
    row = repository.get_dataset(dataset_id)
    (storage_root / str(row["stored_path"])).write_bytes(content + b"\n")

    with pytest.raises(AnalysisPlanError) as captured:
        load_project_source(repository, storage_root, dataset_id)

    assert captured.value.code == "DATASET_HASH_MISMATCH"


def test_load_project_source_rejects_source_that_is_no_longer_accepted(
    persisted_domain: tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes],
) -> None:
    connection, repository, storage_root, dataset_id, _ = persisted_domain
    connection.execute(
        "UPDATE datasets SET admission_status = 'QUARANTINED', is_active = 0 WHERE id = ?",
        (dataset_id,),
    )

    with pytest.raises(AnalysisPlanError) as captured:
        load_project_source(repository, storage_root, dataset_id)

    assert captured.value.code == "DATASET_NOT_ACCEPTED"


def test_load_project_source_rejects_missing_accepted_file(
    persisted_domain: tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes],
) -> None:
    _, repository, storage_root, dataset_id, _ = persisted_domain
    row = repository.get_dataset(dataset_id)
    (storage_root / str(row["stored_path"])).unlink()

    with pytest.raises(AnalysisPlanError) as captured:
        load_project_source(repository, storage_root, dataset_id)

    assert captured.value.code == "ACCEPTED_DATASET_MISSING"


def test_load_project_source_rejects_missing_registered_source(
    persisted_domain: tuple[sqlite3.Connection, WorkflowRepository, Path, str, bytes],
) -> None:
    _, repository, storage_root, _, _ = persisted_domain

    with pytest.raises(AnalysisPlanError) as captured:
        load_project_source(repository, storage_root, "missing")

    assert captured.value.code == "SOURCE_NOT_FOUND"
