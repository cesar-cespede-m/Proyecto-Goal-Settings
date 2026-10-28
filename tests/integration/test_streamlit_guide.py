"""V2-02/03/04: una Guía con la siguiente acción y lo opcional separado del camino principal."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
CHURN = Path(__file__).parents[2] / "data" / "synthetic" / "abandono_suscripcion.csv"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def test_with_a_file_the_guide_says_what_comes_next(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Abandono")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": CHURN.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=CHURN.read_bytes(),
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)

    assert not app.exception
    texts = " ".join(item.value for item in app.markdown)
    assert "Guía de Goal Settings" in texts
    assert "Siguiente paso: [Definir qué quieres anticipar" in texts
    labels = [tab.label for tab in app.tabs]
    assert labels == ["Crear y usar el modelo", "Explorar tus datos"]
    explore = app.tabs[1]
    explore_titles = " ".join(item.value for item in explore.markdown)
    assert "Analizar tus datos" in explore_titles and "Pregúntale a tus datos" in explore_titles
    main_titles = " ".join(item.value for item in app.tabs[0].markdown)
    assert "Pregúntale a tus datos" not in main_titles


def test_choosing_what_to_anticipate_warns_about_revealing_columns(runtime_dir: Path) -> None:

    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Abandono")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": CHURN.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=CHURN.read_bytes(),
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
    picker = next(i for i in app.selectbox if i.label == "Lo que quieres anticipar (opcional)")
    picker.set_value("abandona")
    app.run(timeout=120)

    assert not app.exception
    assert any("encuesta_salida_completada" in item.value for item in app.warning)


def test_journey_roles_are_proposed_and_the_funnel_appears(runtime_dir: Path) -> None:
    events = Path(__file__).parents[2] / "data" / "synthetic" / "recorridos" / "altas_eventos.csv"
    attributes = events.with_name("altas_entidades.csv")
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Recorridos")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": events.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=events.read_bytes(),
    )
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": attributes.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=attributes.read_bytes(),
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
    explore = next(item for item in app.selectbox if item.label == "Archivo")
    explore.select_index(next(i for i, label in enumerate(explore.options) if "eventos" in label))
    app.run(timeout=120)
    assert any(i.label == "Recorridos: hasta dónde llega cada caso" for i in app.expander)
    next(item for item in app.checkbox if item.label == "Sí, así son mis datos").check()
    app.run(timeout=120)

    assert not app.exception, app.exception
    assert any("Donde más se quedan" in item.value for item in app.markdown)
    assert any("Los que empezaron en" in item.value for item in app.caption)

    # V2-24: construir «no avanza» y guardarlo para entrenar.
    picker = next(
        i for i in app.selectbox if i.label == "Sumar datos fijos de cada caso desde (opcional)"
    )
    picker.select_index(1)
    app.run(timeout=120)
    # MV-01: sin declaración de cobertura no se fabrican resultados de abandono.
    next(item for item in app.button if item.label == "Armar la tabla").click()
    app.run(timeout=120)
    assert not app.exception, app.exception
    assert any("Todavía no hay casos" in item.value for item in app.info)
    assert not any(item.label == "Guardar la tabla" for item in app.button)
    next(i for i in app.text_input if i.label == "Inicio de observación completa (UTC)").input(
        "2026-01-01T00:00:00Z"
    )
    next(i for i in app.text_input if i.label == "Fin de observación completa (UTC)").input(
        "2026-10-01T00:00:00Z"
    )
    next(i for i in app.checkbox if "fuente cubre" in i.label).check()
    next(item for item in app.button if item.label == "Armar la tabla").click()
    app.run(timeout=120)
    assert not app.exception, app.exception
    assert any("Tabla lista" in item.value for item in app.success)
    next(item for item in app.text_input if item.label == "¿Quién la guarda?").input("Ana")
    next(item for item in app.checkbox if item.label == "Usarla para entrenar").check()
    next(item for item in app.button if item.label == "Guardar la tabla").click()
    app.run(timeout=120)
    assert not app.exception, app.exception
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    active = connection.execute(
        "SELECT original_filename_display, id, stored_path FROM datasets WHERE is_active = 1"
    ).fetchone()
    from core.analysis.snapshot_recipe import snapshot_recipe

    recipe = snapshot_recipe(WorkflowRepository(connection), active["id"])
    assert recipe["spec"]["label_available_at_output_column"] == "resultado_disponible_desde"
    assert (
        "resultado_disponible_desde"
        in (runtime_dir / active["stored_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    connection.close()
    assert active[0].startswith("no-avanza-a-")


def test_advanced_plan_editor_validates_and_runs(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Avanzado")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": CHURN.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=CHURN.read_bytes(),
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["advanced-mode"] = True
    app.run(timeout=120)
    next(item for item in app.button if item.label == "Validar y ejecutar").click()
    app.run(timeout=120)
    assert not app.exception, app.exception
    assert not app.error, [item.value for item in app.error]

    editor = next(item for item in app.text_area if item.label == "Plan (JSON)")
    editor.input('{"sources": ["no_existe"], "steps": [], "output": {"chart": "table"}}')
    next(item for item in app.button if item.label == "Validar y ejecutar").click()
    app.run(timeout=120)
    assert not app.exception
    assert any("Qué puedes hacer" in item.value for item in app.error)
