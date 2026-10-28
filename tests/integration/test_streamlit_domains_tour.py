"""QA-02: recorrido preventivo de las pantallas de análisis en todos los negocios."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.support.domains import TestDomain as DomainFixture
from tests.unit.test_catalog import ingest

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
TIMEOUT = 120


def _assert_healthy(app: AppTest) -> None:
    assert not app.exception, app.exception
    unexpected = [
        str(item.value) for item in app.error if "inesperad" in str(item.value).casefold()
    ]
    assert not unexpected, unexpected


def _run(app: AppTest) -> AppTest:
    app.run(timeout=TIMEOUT)
    _assert_healthy(app)
    return app


def _create_project(runtime: Path, domain: DomainFixture) -> str:
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project(f"Recorrido {domain.key}")
    dataset_ids = [ingest(repository, runtime, project_id, path) for path in domain.files]
    connection.close()
    return dataset_ids[0]


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_domain_completes_the_analysis_tour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: DomainFixture
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    analysis_dataset_id = _create_project(tmp_path, domain)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
    _assert_healthy(app)
    next(item for item in app.selectbox if item.label == "Archivo").set_value(analysis_dataset_id)
    _run(app)

    automatic = next(item for item in app.radio if item.label == "Qué quieres ver")
    for kind in automatic.options:
        next(item for item in app.radio if item.label == "Qué quieres ver").set_value(kind)
        _run(app)
        next(item for item in app.button if item.label == "Ver análisis").click()
        _run(app)

    next(item for item in app.button if item.label == "Analizar").click()
    _run(app)

    suggestion = next(item for item in app.button if (item.key or "").endswith("-suggestion-0"))
    suggestion.click()
    _run(app)
    next(item for item in app.button if item.label == "Preguntar a los datos").click()
    _run(app)

    next(item for item in app.button if item.label == "Pedir propuestas").click()
    _run(app)

    if len(domain.files) >= 2:
        next(item for item in app.button if item.label == "Buscar relaciones").click()
        _run(app)

    table_name = f"Cálculo {domain.key}"
    next(item for item in app.text_input if item.label == "Nombre de la tabla").input(table_name)
    next(item for item in app.text_input if item.label == "¿Quién la guarda?").input(
        "QA automatizado"
    )
    next(item for item in app.button if item.label == "Guardar tabla").click()
    _run(app)

    catalog = next(item.value for item in app.dataframe if "Archivo" in item.value.columns)
    assert f"{table_name}.csv" in catalog["Archivo"].tolist()
