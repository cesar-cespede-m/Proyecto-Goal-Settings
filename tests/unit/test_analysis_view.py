"""Pantalla genérica del resultado de un análisis: gráfico por tipo de columna y tabla."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.analysis_view import OTHERS_LABEL, build_chart, result_frame
from app.ui.tokens import CHART_SERIES
from core.analysis.contracts import AnalysisOutputV1, ColumnType
from core.analysis.engine import RESULT_TRUNCATED, AnalysisResultV1

SHA = "a" * 64


def _result(columns, rows, *, total=None, warnings=()) -> AnalysisResultV1:
    return AnalysisResultV1(
        columns=[{"name": name, "type": ColumnType(kind)} for name, kind in columns],
        rows=rows,
        total_rows=len(rows) if total is None else total,
        plan_hash=SHA,
        source_hashes={"fuente": SHA},
        warnings=list(warnings),
    )


BY_HOUR = _result(
    [("grupo", "category"), ("hora", "number"), ("casos", "number")],
    [[f"g{group}", hour, group * 10 + hour] for group in range(6) for hour in range(24)],
)


@pytest.mark.parametrize("chart", ["bar", "line", "scatter", "heatmap"])
def test_every_chart_kind_builds_from_column_types(chart: str) -> None:
    output = AnalysisOutputV1(chart=chart, x="hora", y="casos", color="grupo")
    spec = build_chart(BY_HOUR, output).to_dict()

    assert spec["encoding"]["x"]["type"] == "ordinal"  # pocos valores numéricos: orden
    assert spec["config"]["background"] == "transparent"


def test_table_only_when_the_plan_asks_for_a_table_or_axes_are_missing() -> None:
    assert build_chart(BY_HOUR, AnalysisOutputV1(chart="table")) is None
    assert build_chart(BY_HOUR, AnalysisOutputV1(chart="bar", x="hora")) is None


def test_series_beyond_the_palette_fold_into_others() -> None:
    output = AnalysisOutputV1(chart="bar", x="hora", y="casos", color="grupo")
    spec = build_chart(BY_HOUR, output).to_dict()
    dataset = next(iter(spec["datasets"].values()))
    groups = {row["grupo"] for row in dataset}

    assert len(groups) == len(CHART_SERIES)
    assert OTHERS_LABEL in groups
    assert spec["encoding"]["color"]["scale"]["range"] == list(CHART_SERIES)


def test_dates_use_a_time_axis() -> None:
    result = _result(
        [("fecha", "datetime"), ("casos", "number")],
        [[f"2026-01-{day:02d}T00:00:00", day] for day in range(1, 29)],
    )
    spec = build_chart(result, AnalysisOutputV1(chart="line", x="fecha", y="casos")).to_dict()

    assert spec["encoding"]["x"]["type"] == "temporal"


def test_screen_shows_chart_table_and_truncation_note() -> None:
    def render() -> None:
        from app.ui.analysis_view import render_analysis_result
        from core.analysis.contracts import AnalysisOutputV1 as Output
        from core.analysis.contracts import ColumnType as Kind
        from core.analysis.engine import AnalysisResultV1 as Result

        render_analysis_result(
            Result(
                columns=[
                    {"name": "grupo", "type": Kind.CATEGORY},
                    {"name": "casos", "type": Kind.NUMBER},
                ],
                rows=[["a", 3], ["b", 5]],
                total_rows=9000,
                plan_hash="a" * 64,
                source_hashes={"fuente": "a" * 64},
                warnings=["RESULT_TRUNCATED"],
            ),
            Output(chart="bar", x="grupo", y="casos"),
        )

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    assert len(app.dataframe) == 1
    assert "9.000 filas" in app.caption[0].value and "primeras 2" in app.caption[0].value


def test_empty_result_explains_instead_of_drawing() -> None:
    def render() -> None:
        from app.ui.analysis_view import render_analysis_result
        from core.analysis.contracts import AnalysisOutputV1 as Output
        from core.analysis.contracts import ColumnType as Kind
        from core.analysis.engine import AnalysisResultV1 as Result

        render_analysis_result(
            Result(
                columns=[{"name": "grupo", "type": Kind.CATEGORY}],
                rows=[],
                total_rows=0,
                plan_hash="a" * 64,
                source_hashes={},
                warnings=[],
            ),
            Output(chart="bar", x="grupo", y="grupo"),
        )

    app = AppTest.from_function(render).run(timeout=30)

    assert "no devolvió casos" in app.info[0].value


def test_real_engine_result_renders_for_a_test_business(tmp_path: Path) -> None:
    from core.analysis.engine import AnalysisEngine, LoadedSource
    from core.data_engine import IngestionLimits, analysis_schema, validate_and_profile
    from core.data_engine.loading import read_validated_frame
    from core.data_engine.validator import validate_csv_bytes
    from tests.support.domains import DOMAINS

    path = DOMAINS[0].files[0]
    content = path.read_bytes()
    schema = analysis_schema(validate_and_profile(path.name, content).model_dump())
    number = next(name for name, kind in schema.items() if kind == "number")
    category = next(name for name, kind in schema.items() if kind == "category")
    source = LoadedSource(
        frame=read_validated_frame(validate_csv_bytes(path.name, content, IngestionLimits())),
        sha256=SHA,
        schema=schema,
    )
    result = AnalysisEngine().run(
        {
            "sources": ["base"],
            "steps": [
                {"op": "select", "columns": [category, number]},
                {"op": "sort", "by": [number], "descending": True},
                {"op": "limit", "rows": 20},
            ],
            "output": {"chart": "scatter", "x": number, "y": number, "color": category},
        },
        {"base": source},
    )

    assert len(result_frame(result)) == 20
    assert build_chart(result, AnalysisOutputV1(chart="scatter", x=number, y=number)) is not None
    assert RESULT_TRUNCATED not in result.warnings


def test_bars_keep_the_result_order_and_readable_axis_titles() -> None:
    """Recorrido 22/10: la tabla iba de mayor a menor y el gráfico en orden alfabético."""
    result = _result(
        [("dia_semana", "category"), ("casos_atrasados", "number")],
        [["sáb", 454], ["vie", 419], ["jue", 395]],
    )
    spec = build_chart(result, AnalysisOutputV1(chart="bar", x="dia_semana", y="casos_atrasados"))
    encoding = spec.to_dict()["encoding"]

    assert encoding["x"]["sort"] is None
    assert encoding["x"]["title"] == "dia semana"
    assert encoding["y"]["title"] == "casos atrasados"


def test_ranges_and_axis_numbers_read_in_spanish() -> None:
    """Recorrido 22/10: «12.022» parecía doce mil y los ejes decían «1,000»."""
    result = _result(
        [("peso (tramo)", "category"), ("casos", "number")],
        [["0.08–3.41", 700], ["8.06–12.022", 699], ["texto.libre", 1]],
    )
    frame = result_frame(result)
    assert list(frame["peso (tramo)"]) == ["0,08–3,41", "8,06–12,022", "texto.libre"]

    spec = build_chart(result, AnalysisOutputV1(chart="bar", x="peso (tramo)", y="casos"))
    assert spec.to_dict()["config"]["locale"]["number"]["decimal"] == ","
