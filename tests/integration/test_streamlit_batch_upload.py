"""V2-08/V2-09: varios archivos con los datos comunes una sola vez; uno malo no bloquea."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
TRANSPORT = sorted((Path(__file__).parents[2] / "data" / "synthetic" / "transporte").glob("*.csv"))


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def test_review_then_upload_only_the_apt_files(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("Lote")
    connection.close()
    files = [(path.name, path.read_bytes()) for path in TRANSPORT]
    files.append(("roto.csv", b"\xff\xfe\x00no es una tabla"))
    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"batch-upload-{project_id}"] = {
        "step": 3,
        "files": files,
        "shared": {
            "source_name": "Datos sintéticos del equipo",
            "license": None,
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": True,
            "owner_label": "Analista",
            "approver_label": "Jefatura",
        },
        "overrides": {},
    }
    app.run(timeout=120)

    assert not app.exception, app.exception
    assert sum("Apto" in item.value for item in app.success) == len(TRANSPORT)
    assert any(
        "roto.csv" in item.value and "Qué puedes hacer" in item.value for item in app.warning
    )
    next(item for item in app.button if item.label.startswith("Subir 4")).click()
    app.run(timeout=120)

    assert not app.exception, app.exception
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    repository = WorkflowRepository(connection)
    accepted = repository.list_datasets(project_id, admission_status="ACCEPTED")
    roles = repository.get_project_roles(project_id)
    connection.close()
    assert len(accepted) == len(TRANSPORT)
    assert roles["owner_label"] == "Analista" and roles["approver_label"] == "Jefatura"


def test_non_csv_is_explained_and_excluded_from_upload(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("Extensiones")
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)

    uploader = next(
        item for item in app.get("file_uploader") if item.label == "Archivos de datos (CSV)"
    )
    uploader.set_value(
        [
            ("README.md", b"# instrucciones", "text/markdown"),
            ("tabla.csv", b"caso,resultado\n1,si\n", "text/csv"),
        ]
    )
    app.run(timeout=120)

    assert not app.exception, app.exception
    page = " ".join(item.value for item in app.markdown)
    assert "Se subirán" in page
    assert "No se subirán (y por qué)" in page
    assert any(
        "Este archivo no es una tabla CSV: «README.md»." in item.value
        and "Guardar como → CSV" in item.value
        for item in app.warning
    )
    next(item for item in app.button if item.label == "Continuar").click()
    app.run(timeout=120)

    assert not app.exception, app.exception
    assert app.session_state[f"batch-upload-{project_id}"]["files"] == [
        ("tabla.csv", b"caso,resultado\n1,si\n")
    ]
    assert any(item.value == "Subir archivos · parte 2 de 3: Datos comunes" for item in app.caption)
    assert any("Vas a subir **1 archivo**: «tabla.csv»" in item.value for item in app.markdown)
    license_input = next(
        item for item in app.text_input if item.label == "Condiciones de uso (opcional)"
    )
    assert license_input.placeholder == "Ej.: sólo para esta prueba interna"


def test_review_uses_spanish_number_format_and_singular_copy(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("Formato")
    connection.close()
    rows = b"caso,resultado\n" + b"1,si\n" * 3_500
    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"batch-upload-{project_id}"] = {
        "step": 3,
        "files": [("grande.csv", rows)],
        "shared": {
            "source_name": "Datos sintéticos del equipo",
            "license": None,
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": True,
            "owner_label": "Analista",
            "approver_label": "Jefatura",
        },
        "overrides": {},
    }
    app.run(timeout=120)

    assert not app.exception, app.exception
    assert any("Apto · 3.500 filas, 2 columnas" in item.value for item in app.success)
    assert any(item.label == "Subir 1 archivo apto" for item in app.button)


def test_going_back_from_review_keeps_what_was_written(runtime_dir: Path) -> None:
    """Recorrido 22/10: «Volver a los datos comunes» borraba lo escrito y las confirmaciones."""
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("Volver")
    connection.close()
    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"batch-upload-{project_id}"] = {
        "step": 3,
        "files": [(TRANSPORT[0].name, TRANSPORT[0].read_bytes())],
        "shared": {
            "source_name": "Datos sintéticos del equipo",
            "license": "Sólo prueba interna",
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": True,
            "owner_label": "Analista",
            "approver_label": "Jefatura",
        },
        "overrides": {},
    }
    app.run(timeout=120)
    next(item for item in app.button if item.label == "Volver a los datos comunes").click()
    app.run(timeout=120)

    assert not app.exception, app.exception
    texts = {item.label: item.value for item in app.text_input}
    assert texts["¿De dónde salieron?"] == "Datos sintéticos del equipo"
    assert texts["Condiciones de uso (opcional)"] == "Sólo prueba interna"
    assert texts["Responsable del proyecto"] == "Analista"
    assert next(item for item in app.selectbox if item.label == "¿Qué tipo de datos son?").value
    checks = {item.label: item.value for item in app.checkbox}
    assert checks["Confirmo que tengo permiso para usar estos datos en esta prueba"]
    assert checks["Confirmo que no tienen datos que identifiquen a personas reales"]
