"""Proyectos: crear, elegir y eliminar desde la barra lateral; stepper según el proyecto."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.ml_workflow import csv_bytes

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def _repository(runtime_dir: Path) -> tuple:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    return connection, WorkflowRepository(connection)


def _run(app: AppTest) -> AppTest:
    app.run(timeout=30)
    assert not app.exception, app.exception
    return app


def _sidebar_html(app: AppTest) -> str:
    return " ".join(item.value for item in app.sidebar.markdown)


def test_archived_project_leaves_the_list_but_keeps_its_audit(tmp_path: Path) -> None:
    connection, repository = _repository(tmp_path)
    project_id = repository.create_project("Para eliminar")

    with pytest.raises(ValueError):
        repository.archive_project(project_id, actor_label=" ")
    repository.archive_project(project_id, actor_label="Responsable")

    assert [row["id"] for row in repository.list_active_projects()] == []
    assert [row["id"] for row in repository.list_projects()] == [project_id]
    event = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'PROJECT_ARCHIVED'"
    ).fetchone()
    assert json.loads(event["detail_json"])["actor_label"] == "Responsable"
    with pytest.raises(LookupError):
        repository.archive_project(project_id, actor_label="Responsable")
    connection.close()


def test_project_is_created_from_the_sidebar_and_its_steps_unlock(runtime_dir: Path) -> None:
    app = _run(AppTest.from_file(str(APP_PATH)))
    app.sidebar.radio(key="workspace-section").set_value("Datos y modelos")
    _run(app)
    assert any("Crea o elige un proyecto" in item.value for item in app.sidebar.caption)

    form_input = next(
        item for item in app.sidebar.text_input if item.label == "Nombre del proyecto"
    )
    form_input.input("Entregas de octubre")
    next(item for item in app.sidebar.button if item.label == "Crear proyecto").click()
    _run(app)

    html = _sidebar_html(app)
    assert "Proyecto seleccionado" in html and "Entregas de octubre" in html
    assert "Paso 1 de 5 · Tus datos" in html
    assert 'aria-current="step"' in html
    assert html.count('aria-disabled="true"') == 5


def test_selected_project_drives_the_stepper_and_can_be_deleted(runtime_dir: Path) -> None:
    connection, repository = _repository(runtime_dir)
    repository.create_project("Sin archivo")
    with_file = repository.create_project("Con archivo")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=with_file,
        request={
            "original_filename_display": "sintetico.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=csv_bytes(),
    )
    connection.close()

    app = _run(AppTest.from_file(str(APP_PATH)))
    html = _sidebar_html(app)
    assert "Con archivo" in html and "Paso 2 de 5 · Qué quieres anticipar" in html
    assert html.count('aria-disabled="true"') == 4

    next(item for item in app.sidebar.text_input if item.label == "¿Quién lo elimina?").input(
        "Responsable"
    )
    next(
        item for item in app.sidebar.checkbox if "Confirmo que quiero eliminar" in item.label
    ).check()
    next(item for item in app.sidebar.button if item.label == "Eliminar proyecto").click()
    _run(app)

    assert any("eliminado de la lista" in item.value for item in app.success)
    html = _sidebar_html(app)
    assert "Sin archivo" in html and "Con archivo" not in html
    labels = app.sidebar.selectbox[0].options
    assert labels == ["Sin archivo"]


def test_delete_requires_confirmation(runtime_dir: Path) -> None:
    connection, repository = _repository(runtime_dir)
    repository.create_project("Proyecto único")
    connection.close()

    app = _run(AppTest.from_file(str(APP_PATH)))
    next(item for item in app.sidebar.text_input if item.label == "¿Quién lo elimina?").input(
        "Responsable"
    )
    next(item for item in app.sidebar.button if item.label == "Eliminar proyecto").click()
    _run(app)

    assert any("Marca la confirmación" in item.value for item in app.sidebar.error)
    assert app.sidebar.selectbox[0].options == ["Proyecto único"]
