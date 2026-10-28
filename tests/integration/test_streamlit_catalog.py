"""GA-01: la pantalla lista los archivos del proyecto y cambia la tabla para entrenar."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.unit.test_catalog import ingest

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def _run(app: AppTest) -> AppTest:
    app.run(timeout=60)
    assert not app.exception, app.exception
    return app


def test_catalog_lists_every_file_and_switches_the_modeling_table(runtime_dir: Path) -> None:
    multi = next(domain for domain in DOMAINS if domain.multi_file and domain.has_outcome)
    files = [path for path in multi.files if path.stat().st_size < 500_000]
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Catálogo UI")
    for path in files:
        ingest(repository, runtime_dir, project_id, path)
    connection.close()

    app = _run(AppTest.from_file(str(APP_PATH)))
    table = app.dataframe[0].value
    assert sorted(table["Archivo"]) == sorted(path.name for path in files)
    assert (table["Uso"] == "Para entrenar").sum() == 1
    assert any(f"Tabla para entrenar: **{files[0].name}**" in item.value for item in app.markdown)

    chosen = next(item for item in app.selectbox if item.label == "Archivo para entrenar").value
    next(item for item in app.text_input if item.label == "¿Quién hace el cambio?").input(
        "Responsable"
    )
    next(item for item in app.button if item.label == "Usar para entrenar").click()
    _run(app)

    assert any("es ahora la tabla para entrenar" in item.value for item in app.success)
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    assert WorkflowRepository(connection).get_active_dataset(project_id)["id"] == chosen
    connection.close()
