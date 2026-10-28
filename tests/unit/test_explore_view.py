"""«Analizar tus datos»: el constructor guiado funciona igual en cualquier negocio."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.explore_view import COUNT, DATE_PARTS, GROUPABLE, build_plan
from core.analysis.validation import validate_plan
from core.data_engine import analysis_schema, validate_and_profile
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.unit.test_catalog import ingest

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def _schema(path: Path) -> dict[str, str]:
    return analysis_schema(validate_and_profile(path.name, path.read_bytes()).model_dump())


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_choice_builds_a_valid_plan_in_every_business(domain) -> None:
    for path in domain.files:
        schema = _schema(path)
        numbers = [name for name, kind in schema.items() if kind == "number"]
        for column, kind in schema.items():
            if kind not in GROUPABLE:
                continue
            parts = list(DATE_PARTS) if kind == "datetime" else [None]
            for part in parts:
                plan = build_plan(schema, group_by=column, date_part=part)
                validate_plan(plan, {"archivo": schema})
                if numbers:
                    plan = build_plan(
                        schema, group_by=column, date_part=part, measure=numbers[0], function="mean"
                    )
                    validate_plan(plan, {"archivo": schema})


def test_dates_draw_lines_and_categories_draw_ranked_bars() -> None:
    schema = {"cuando": "datetime", "grupo": "category", "valor": "number"}

    by_hour = build_plan(schema, group_by="cuando", date_part="hour_of_day")
    by_group = build_plan(schema, group_by="grupo", measure="valor", function="sum")

    assert by_hour.output.chart == "line"
    assert by_hour.steps[0].op == "derive_datetime"
    assert by_group.output.chart == "bar"
    assert by_group.steps[-2].descending is True
    assert by_group.output.y == "suma de valor"
    assert build_plan(schema, group_by="grupo").output.y == "casos"
    assert COUNT not in by_group.model_dump_json()


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def test_screen_analyzes_a_file_and_shows_chart_and_table(runtime_dir: Path) -> None:
    domain = next(item for item in DOMAINS if item.has_datetimes and item.has_outcome)
    path = next(item for item in domain.files if "datetime" in _schema(item).values())
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Analizar")
    ingest(repository, runtime_dir, project_id, path)
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    assert any(item.value == "### Analizar tus datos" for item in app.markdown)
    datetime_column = next(name for name, kind in _schema(path).items() if kind == "datetime")
    next(item for item in app.selectbox if item.label == "Agrupar por").set_value(
        datetime_column
    ).run(timeout=60)
    next(item for item in app.radio if item.label == "Parte de la fecha").set_value(
        "hour_of_day"
    ).run(timeout=60)
    next(item for item in app.button if item.label == "Analizar").click().run(timeout=60)

    assert not app.exception, app.exception
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    hours = app.dataframe[-1].value
    assert list(hours.columns) == [f"{datetime_column} (hora del día)", "casos"]
    assert hours["casos"].sum() > 0


def test_group_by_offers_categories_before_continuous_numbers(runtime_dir: Path) -> None:
    """Recorrido 22/10: el valor por defecto agrupaba por un número continuo."""
    domain = next(item for item in DOMAINS if item.has_outcome)
    path = next(
        item for item in domain.files if {"number", "category"} <= set(_schema(item).values())
    )
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    ingest(repository, runtime_dir, repository.create_project("Agrupar"), path)
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    group = next(item for item in app.selectbox if item.label == "Agrupar por")
    assert _schema(path)[group.value] != "number"
