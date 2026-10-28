"""Decisiones críticas de Streamlit sin valores preseleccionados."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.ml_engine import MLRunService
from core.reporting import ReportService
from tests.support.ml_workflow import csv_bytes, prepared_strategy

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path) -> Iterator[Path]:
    previous = os.environ.get("GOAL_SETTINGS_RUNTIME_DIR")
    os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if previous is None:
            os.environ.pop("GOAL_SETTINGS_RUNTIME_DIR", None)
        else:
            os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = previous


def _selectbox(app: AppTest, label: str):
    return next(item for item in app.selectbox if item.label == label)


def _radio(app: AppTest, label: str):
    return next(item for item in app.radio if item.label == label)


def _text_input(app: AppTest, label: str):
    return next(item for item in app.text_input if item.label == label)


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_first_approval_starts_without_selection(runtime_dir: Path) -> None:
    connection, _, _, _, _ = prepared_strategy(runtime_dir, storage_subdir=".")
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert _radio(app, "Decisión sobre el entrenamiento").value is None

    _radio(app, "Decisión sobre el entrenamiento").set_value("REJECTED")
    _text_input(app, "Responsable de revisar la estrategia").input("pytest-reviewer")
    _button(app, "Registrar decisión").click().run(timeout=30)
    assert any("Explica el motivo del rechazo" in item.value for item in app.error)


def test_second_approval_starts_without_selection(runtime_dir: Path) -> None:
    connection, repository, _, strategy_id, _ = prepared_strategy(
        runtime_dir,
        storage_subdir=".",
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    result = MLRunService(repository, runtime_dir).execute(strategy_id)
    ReportService(repository, narrative=None).create_report(result.run_id)
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert _radio(app, "Decisión sobre el reporte").value is None

    _radio(app, "Decisión sobre el reporte").set_value("REJECTED")
    _text_input(app, "Responsable de revisar el reporte").input("pytest-reviewer")
    _button(app, "Registrar decisión del reporte").click().run(timeout=30)
    assert any("Explica el motivo del rechazo" in item.value for item in app.error)


def test_approval_requires_reading_review_points(runtime_dir: Path) -> None:
    """UX-05: con puntos para revisar, aprobar exige marcar que se leyeron."""
    # Filas repetidas: un punto para revisar que no depende de columnas ya excluidas (RU-04).
    lines = csv_bytes().decode("utf-8").splitlines()
    content = "\n".join([*lines, *lines[1:4]]) + "\n"
    connection, _, _, _, _ = prepared_strategy(
        runtime_dir, storage_subdir=".", content=content.encode("utf-8")
    )
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    warnings = " ".join(item.value for item in app.warning)
    assert "advertencia(s)" not in warnings
    assert "detalle técnico" not in warnings
    assert [item for item in app.checkbox if item.label.startswith("Leí los puntos para revisar")]
    assert "puntos para revisar" in warnings or "punto para revisar" in warnings

    _radio(app, "Decisión sobre el entrenamiento").set_value("APPROVED")
    _text_input(app, "Responsable de revisar la estrategia").input("pytest-reviewer")
    _button(app, "Registrar decisión").click().run(timeout=30)
    assert any("Antes de aprobar, lee y marca" in item.value for item in app.error)

    next(
        item for item in app.checkbox if item.label.startswith("Leí los puntos para revisar")
    ).check()
    _radio(app, "Decisión sobre el entrenamiento").set_value("APPROVED")
    _text_input(app, "Responsable de revisar la estrategia").input("pytest-reviewer")
    _button(app, "Registrar decisión").click().run(timeout=30)
    assert not any("Antes de aprobar, lee y marca" in item.value for item in app.error)
