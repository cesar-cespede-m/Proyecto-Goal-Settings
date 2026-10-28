"""Recorrido UI: entrenar, generar reporte, aprobarlo y publicarlo sin terminal."""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.ai_engine import LLMRequest, LLMResponse, register_provider
from tests.support.ml_workflow import prepared_strategy

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


def _text_input(app: AppTest, label: str):
    return next(item for item in app.text_input if item.label == label)


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def _radio(app: AppTest, label: str):
    return next(item for item in app.radio if item.label == label)


def test_ui_trains_reports_and_publishes_only_after_second_approval(runtime_dir: Path) -> None:
    """Sin LLM configurado: modo plantilla etiquetado."""
    connection, repository, _, strategy_id, _ = prepared_strategy(runtime_dir, storage_subdir=".")
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert any(item.label == "Entrenar y comparar modelos" for item in app.button)

    app.button(key=f"train-{strategy_id}").click().run(timeout=60)
    assert not app.exception
    assert any("Entrenamiento terminado: Completada" in item.value for item in app.success)
    assert any(item.value == "Reporte (segunda aprobación)" for item in app.subheader)

    _button(app, "Generar reporte verificable").click().run(timeout=30)
    assert not app.exception
    assert any("Pendiente de revisión" in item.value for item in app.caption)
    assert not [item for item in app.button if item.label == "Publicar reporte aprobado"]

    _radio(app, "Decisión sobre el reporte").set_value("APPROVED")
    _text_input(app, "Responsable de revisar el reporte").input("pytest-reviewer")
    _button(app, "Registrar decisión del reporte").click().run(timeout=30)
    assert not app.exception
    assert any("Aprobado" in item.value for item in app.caption)

    _button(app, "Publicar reporte aprobado").click().run(timeout=30)
    assert not app.exception
    assert not any("Reporte publicado" in item.value for item in app.success)
    _button(app, "Cancelar").click().run(timeout=30)
    assert not [item for item in app.button if item.label == "Sí, publicar en la demo"]
    _button(app, "Publicar reporte aprobado").click().run(timeout=30)
    _button(app, "Sí, publicar en la demo").click().run(timeout=30)
    assert not app.exception
    assert any("Reporte publicado" in item.value for item in app.success)


class _EchoProvider:
    """Proveedor falso: reescribe los hallazgos de referencia citando la misma evidencia."""

    name = "pytest_ui_fake"
    model = "echo-model"

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        start = request.user.index("[", request.user.index("PLANTILLA_DE_HALLAZGOS"))
        end = request.user.index("\n\nLIMITACIONES")
        reference = json.loads(request.user[start:end])
        insights = [{**item, "claim": "En simple: " + item["claim"]} for item in reference]
        return LLMResponse(
            content=json.dumps({"insights": insights}),
            provider=self.name,
            model=self.model,
            latency_ms=1,
        )


def test_switching_provider_is_configuration_only(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_provider("pytest_ui_fake", lambda settings: _EchoProvider())
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_ui_fake")
    monkeypatch.setenv("LLM_MODEL", "echo-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    connection, repository, _, strategy_id, _ = prepared_strategy(runtime_dir, storage_subdir=".")
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.button(key=f"train-{strategy_id}").click().run(timeout=60)
    assert any("Redacción con IA generativa activa" in item.value for item in app.caption)
    _button(app, "Generar reporte verificable").click().run(timeout=30)

    assert not app.exception
    assert any("Redactado por IA generativa" in item.value for item in app.success)
    assert any("En simple:" in item.value for item in app.markdown)
    assert not [item for item in app.button if item.label == "Publicar reporte aprobado"]
