"""DATA-02.3: recorrido UI para previsualizar y confirmar un cruce de CSV."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    for name in ("LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


def _ingest(
    service: DataIngestionService,
    project_id: str,
    filename: str,
    content: bytes,
) -> str:
    return service.ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": filename,
            "source_name": "Fixture sintético DATA-02.3",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=content,
    ).metadata.dataset_id


def _project_with_sources(runtime_dir: Path) -> tuple[str, str, str]:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Cruce UI sintético")
    service = DataIngestionService(repository, runtime_dir)
    primary_id = _ingest(
        service,
        project_id,
        "casos.csv",
        b"case_id,target,value\n1,0,10\n2,1,20\n3,0,30\n4,1,40\n",
    )
    auxiliary_id = _ingest(
        service,
        project_id,
        "segmentos.csv",
        b"key,segment\n1,A\n2,B\n3,A\n4,B\n",
    )
    connection.close()
    return project_id, primary_id, auxiliary_id


def _by_label(widgets, label: str):
    return next(item for item in widgets if item.label == label)


def _run(app: AppTest) -> AppTest:
    app.run(timeout=30)
    assert not app.exception, app.exception
    return app


def test_user_previews_and_activates_a_two_source_composition(runtime_dir: Path) -> None:
    project_id, primary_id, auxiliary_id = _project_with_sources(runtime_dir)
    app = _run(AppTest.from_file(str(APP_PATH)))

    assert any(item.value == "### Cruzar archivos" for item in app.markdown)
    _by_label(app.selectbox, "Fuente principal").set_value(primary_id)
    _by_label(app.multiselect, "Fuentes auxiliares (máximo 4)").set_value([auxiliary_id])
    _run(app)

    _by_label(app.selectbox, "Columna de relación en la principal").set_value("case_id")
    _by_label(app.selectbox, "Columna equivalente en esta auxiliar").set_value("key")
    _by_label(app.text_input, "Responsable de confirmar el cruce").input("Data steward pytest")
    _by_label(app.button, "Previsualizar cruce").click()
    _run(app)

    assert any("Previsualización lista: 2 fuentes" in item.value for item in app.success)
    _by_label(app.checkbox, "Revisé los conteos y confirmo este cruce").check()
    _run(app)
    create = _by_label(app.button, "Crear y activar dataset cruzado")
    assert not create.disabled
    create.click()
    _run(app)

    assert any("Cruce creado con 2 fuentes" in item.value for item in app.success)
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    repository = WorkflowRepository(connection)
    active = repository.get_active_dataset(project_id)
    assert active is not None
    assert active["dataset_kind"] == "DERIVED"
    composition = repository.get_dataset_composition(active["id"])
    assert composition is not None
    assert composition["actor_label"] == "Data steward pytest"
    connection.close()


def test_preview_asks_for_the_responsible_person_in_plain_language(runtime_dir: Path) -> None:
    _, primary_id, auxiliary_id = _project_with_sources(runtime_dir)
    app = _run(AppTest.from_file(str(APP_PATH)))
    _by_label(app.selectbox, "Fuente principal").set_value(primary_id)
    _by_label(app.multiselect, "Fuentes auxiliares (máximo 4)").set_value([auxiliary_id])
    _run(app)
    _by_label(app.selectbox, "Columna de relación en la principal").set_value("case_id")
    _by_label(app.selectbox, "Columna equivalente en esta auxiliar").set_value("key")

    _by_label(app.button, "Previsualizar cruce").click()
    _run(app)

    messages = " ".join(item.value for item in app.error)
    assert "Escribe el nombre o rol de quien confirma el cruce" in messages
    assert "pydantic" not in messages and "actor_label" not in messages
    assert not any("Previsualización lista" in item.value for item in app.success)
