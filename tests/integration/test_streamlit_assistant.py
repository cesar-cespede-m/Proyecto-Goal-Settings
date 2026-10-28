"""UX-01.9: asistente de contexto por UI, con IA falsa y sin IA, hasta crear el contexto."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.ai_engine import LLMProviderError, LLMRequest, LLMResponse, register_provider
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.ml_workflow import csv_bytes

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
ACTOR = "Responsable de prueba"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    for name in ("LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


def _project_with_dataset(runtime_dir: Path, content: bytes | None = None) -> str:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Asistente UI")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "sintetico.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=csv_bytes() if content is None else content,
    )
    connection.close()
    return project_id


def _by_key_suffix(widgets, suffix: str):
    return next(item for item in widgets if item.key and item.key.endswith(suffix))


def _by_label(widgets, label: str):
    return next(item for item in widgets if item.label == label)


def _run(app: AppTest) -> AppTest:
    app.run(timeout=30)
    assert not app.exception, app.exception
    return app


def _set_actor(app: AppTest) -> None:
    field = next(item for item in app.text_input if (item.key or "").startswith("actor-input-"))
    field.input(ACTOR)
    _by_key_suffix(app.button, "-actor").click()
    _run(app)


def _save_text(app: AppTest, field: str, value: str) -> None:
    next(item for item in app.text_area if f"-{field}-r" in (item.key or "")).input(value)
    _by_key_suffix(app.button, f"submit-{_draft_id(app)}-{field}").click()
    _run(app)


def _draft_id(app: AppTest) -> str:
    key = next(item.key for item in app.button if (item.key or "").startswith("submit-"))
    return key.split("-", 1)[1].rsplit("-", 1)[0]


def _choose(app: AppTest, field: str, value: object, *, radio: bool = False) -> None:
    widgets = app.radio if radio else app.selectbox
    next(item for item in widgets if f"-{field}-r" in (item.key or "")).set_value(value)
    _run(app)
    # Elegir guarda al instante; sólo con advertencia aparece «Confirmar de todas formas».
    extra = [item for item in app.button if (item.key or "") == f"save-{_draft_id(app)}-{field}"]
    if extra:
        extra[0].click()
        _run(app)


def test_a_goal_typed_as_the_actor_is_rejected(runtime_dir: Path) -> None:
    """Reportado el 20/10: el objetivo quedó guardado como responsable y luego como revisor."""
    _project_with_dataset(runtime_dir)
    app = _run(AppTest.from_file(str(APP_PATH)))
    field = next(item for item in app.text_input if (item.key or "").startswith("actor-input-"))
    # El campo admite hasta 60 caracteres: el navegador corta lo pegado en ese largo.
    field.input("Necesito comprender cómo los usuarios antiguos abandonan")
    _by_key_suffix(app.button, "-actor").click()
    _run(app)
    assert any("parece una descripción" in item.value for item in app.error)
    assert any((item.key or "").startswith("actor-input-") for item in app.text_input)


def test_upload_is_hidden_once_the_project_has_its_file(runtime_dir: Path) -> None:
    _project_with_dataset(runtime_dir)
    app = _run(AppTest.from_file(str(APP_PATH)))
    assert not [item for item in app.button if item.label == "Validar e ingresar"]
    assert any("Tabla para entrenar: **sintetico.csv**" in item.value for item in app.markdown)


def test_offline_assistant_reaches_a_context_with_explicit_confirmations(
    runtime_dir: Path,
) -> None:
    _project_with_dataset(runtime_dir)
    app = _run(AppTest.from_file(str(APP_PATH)))
    assert not [item for item in app.text_area if (item.key or "").endswith("-message")]
    _set_actor(app)

    target = next(item for item in app.selectbox if "-target_column-r" in (item.key or ""))
    assert target.value is None
    assert not [item for item in app.button if item.label.lower().startswith("confirmar todo")]

    # Sin IA activa no se ofrece la caja para conversar: se responde directo abajo.
    assert not [item for item in app.text_area if (item.key or "").endswith("-message")]
    assert any("La IA no está activa" in item.value for item in app.info)

    _save_text(app, "objective", "Priorizar revisiones del fixture")
    _choose(app, "target_column", "target")
    event = next(item for item in app.radio if "-positive_class-r" in (item.key or ""))
    assert event.value is None
    assert all("'" not in event.format_func(option) for option in event.options)
    assert any("Que «target» sea «1» (100 casos)" in label for label in event.options)
    _choose(app, "positive_class", 1, radio=True)
    _save_text(app, "prediction_moment", "Antes del evento sintético")
    _save_text(app, "false_positive_consequence", "Revisión innecesaria")
    _save_text(app, "false_negative_consequence", "Evento no detectado")
    _choose(app, "primary_metric", "f1", radio=True)
    _save_text(app, "metric_rationale", "Equilibrar detectar y acertar")
    _choose(app, "guardrail_mode", "no_quantitative_guardrail", radio=True)

    finalize = _by_label(app.button, "Guardar mis decisiones")
    assert finalize.disabled
    # UX-18: una sola respuesta por columna, sin botón extra.
    for column, answer in (
        ("fixture_id", "known_skip"),
        ("signal", "use"),
        ("noise", "known_skip"),
        ("category", "use"),
    ):
        next(
            i for i in app.radio if f"feature-{_draft_id(app)}-{column}-r" in (i.key or "")
        ).set_value(answer)
        _run(app)

    _by_label(app.button, "Guardar mis decisiones").click()
    _run(app)
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    row = connection.execute("SELECT context_json FROM context_versions").fetchone()
    connection.close()
    context = json.loads(row["context_json"])["context"]
    assert context["decision_actor_label"] == ACTOR
    assert sorted(
        item["column"] for item in context["feature_decisions"] if item["use"] == "include"
    ) == ["category", "signal"]


class _AssistantProvider:
    name = "pytest_ui_assistant"
    model = "fake-assistant"
    requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        type(self).requests.append(request)
        if request.schema_name != "context_assistant_turn":
            raise RuntimeError("sólo el asistente usa este proveedor de prueba")
        turn = {
            "schema_version": "assistant-turn.v1",
            "assistant_message": "Propongo anticipar la columna C005.",
            "questions": [],
            "proposals": [
                {"field": "target_column", "value": "C005", "explanation": "Es binaria."},
                {
                    "field": "objective",
                    "value": "Priorizar revisiones del fixture",
                    "explanation": "Resume tu objetivo.",
                },
            ],
            "feature_notes": [],
            "status": "NEEDS_INPUT",
        }
        return LLMResponse(
            content=json.dumps(turn), provider=self.name, model=self.model, latency_ms=1
        )


def test_choosing_an_option_saves_it_and_only_a_warning_asks_again(runtime_dir: Path) -> None:
    """Recorrido 22/10: marcar el evento no bastaba; había que encontrar un segundo botón."""
    rows = ["fixture_id,signal,target"] + [
        f"fixture-{index},{index % 7},{1 if index % 4 == 0 else 0}" for index in range(200)
    ]
    _project_with_dataset(runtime_dir, ("\n".join(rows) + "\n").encode("utf-8"))
    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    # La forma viene marcada por defecto y no figura como «Falta responder».
    shape = next(item for item in app.radio if "-task_type-r" in (item.key or ""))
    assert shape.value == "binary_classification"
    assert any("Por defecto" in item.value for item in app.caption)
    next(item for item in app.selectbox if "-target_column-r" in (item.key or "")).set_value(
        "target"
    )
    _run(app)
    assert not [item for item in app.button if item.label == "Confirmar esta elección"]

    # 0 es el resultado más frecuente (150 de 200): se advierte y se pide confirmar.
    next(item for item in app.radio if "-positive_class-r" in (item.key or "")).set_value(0)
    _run(app)
    assert any("resultado más común" in item.value for item in app.warning)
    assert any(item.label == "Confirmar de todas formas" for item in app.button)

    next(item for item in app.radio if "-positive_class-r" in (item.key or "")).set_value(1)
    _run(app)
    assert not [item for item in app.button if item.label == "Confirmar de todas formas"]
    event = next(item for item in app.radio if "-positive_class-r" in (item.key or ""))
    assert event.value == 1


def test_llm_proposals_need_explicit_adoption_and_excluded_columns_are_not_sent(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_provider("pytest_ui_assistant", lambda settings: _AssistantProvider())
    _AssistantProvider.requests = []
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_ui_assistant")
    monkeypatch.setenv("LLM_MODEL", "fake-assistant")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    _project_with_dataset(runtime_dir)

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    next(item for item in app.text_area if (item.key or "").endswith("-message")).input(
        "Quiero anticipar el evento del fixture."
    )
    assert not [item for item in app.button if item.label == "Enviar a la IA"]
    _by_label(app.button, "Revisar antes de enviar").click()
    _run(app)
    next(item for item in app.multiselect if (item.key or "").endswith("-excluded")).set_value(
        ["C001"]
    )
    _run(app)
    _by_label(app.button, "Enviar a la IA").click()
    _run(app)

    assert len(_AssistantProvider.requests) == 1
    sent = _AssistantProvider.requests[0].user
    assert "fixture_id" not in sent and "C001" not in sent
    assert any("La IA respondió" in item.value for item in app.success)
    # UX-17: la persona ve el nombre de la columna, no la referencia interna.
    shown = " ".join(item.value for item in app.markdown)
    assert "C005" not in shown and "«target»" in shown
    target = next(item for item in app.selectbox if "-target_column-r" in (item.key or ""))
    assert target.value is None  # la propuesta no se adopta sola

    _by_key_suffix(app.button, "accept-" + _draft_id(app) + "-target_column").click()
    _run(app)
    target = next(item for item in app.selectbox if "-target_column-r" in (item.key or ""))
    assert target.value == "target"


def test_confirming_an_unchanged_ai_text_keeps_its_ai_origin(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX-13: un solo «Confirmar»; sin cambios, la auditoría sigue diciendo «Sugerido por IA»."""
    register_provider("pytest_ui_assistant", lambda settings: _AssistantProvider())
    _AssistantProvider.requests = []
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_ui_assistant")
    monkeypatch.setenv("LLM_MODEL", "fake-assistant")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    _project_with_dataset(runtime_dir)

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    next(item for item in app.text_area if (item.key or "").endswith("-message")).input(
        "Quiero anticipar el evento del fixture."
    )
    _by_label(app.button, "Revisar antes de enviar").click()
    _run(app)
    _by_label(app.button, "Enviar a la IA").click()
    _run(app)
    draft_id = _draft_id(app)
    assert not [item for item in app.button if item.key == f"accept-{draft_id}-objective"]

    _by_key_suffix(app.button, f"submit-{draft_id}-objective").click()
    _run(app)

    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    row = connection.execute("SELECT draft_json FROM context_drafts").fetchone()
    connection.close()
    objective = json.loads(row["draft_json"])["fields"]["objective"]
    assert objective["status"] == "CONFIRMED"
    assert objective["source"] == "LLM"
    assert objective["confirmed_by"] == ACTOR


def test_after_an_accepted_upload_the_file_and_next_step_are_shown(runtime_dir: Path) -> None:
    _project_with_dataset(runtime_dir)
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["upload-accepted-flash"] = "Archivo aceptado y perfilado."
    _run(app)
    assert any(item.value == "Archivo aceptado y perfilado." for item in app.success)
    assert not any("No hay archivos aceptados" in item.value for item in app.markdown)
    assert any((item.key or "").startswith("actor-input-") for item in app.text_input), (
        "el paso de contexto debe estar disponible"
    )


def test_hiding_every_column_from_the_ai_offers_only_to_continue_without_it(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_provider("pytest_ui_assistant", lambda settings: _AssistantProvider())
    _AssistantProvider.requests = []
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_ui_assistant")
    monkeypatch.setenv("LLM_MODEL", "fake-assistant")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    _project_with_dataset(runtime_dir)

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    next(item for item in app.text_area if (item.key or "").endswith("-message")).input(
        "Quiero anticipar el evento del fixture."
    )
    _by_label(app.button, "Revisar antes de enviar").click()
    _run(app)
    excluded = next(item for item in app.multiselect if (item.key or "").endswith("-excluded"))
    excluded.set_value([f"C{index:03d}" for index in range(1, 6)])  # las 5 columnas
    _run(app)

    assert any("Ocultaste todas las columnas" in item.value for item in app.info)
    assert _by_label(app.button, "Enviar a la IA").disabled
    _by_label(app.button, "Continuar sin IA").click()
    _run(app)
    assert _AssistantProvider.requests == []
    assert any("Seguiste sin IA" in item.value for item in app.info)


def _multiclass_csv() -> bytes:
    rows = ["signal,category,nivel"]
    for index in range(120):
        level = ("bajo", "medio", "alto")[index % 3]
        rows.append(f"{index % 7 + (index % 3) * 10},{'ab'[index % 2]},{level}")
    return ("\n".join(rows) + "\n").encode()


def test_assistant_asks_the_outcome_shape_and_reaches_a_multiclass_plan(
    runtime_dir: Path,
) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Varias categorías")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "niveles.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=_multiclass_csv(),
    )
    connection.close()

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)

    _save_text(app, "objective", "Anticipar el nivel de cada caso")
    _choose(app, "task_type", "multiclass_classification", radio=True)
    target = next(item for item in app.selectbox if "-target_column-r" in (item.key or ""))
    assert "nivel" in target.options and "category" not in target.options
    _choose(app, "target_column", "nivel")
    assert not [item for item in app.radio if "-positive_class-r" in (item.key or "")]
    _save_text(app, "prediction_moment", "Antes de conocer el nivel")
    _save_text(app, "false_positive_consequence", "Atención mal dirigida")
    _save_text(app, "false_negative_consequence", "Caso sin atender")
    metric = next(item for item in app.radio if "-primary_metric-r" in (item.key or ""))
    assert (
        set(metric.options)
        == {metric.format_func(value) for value in ("macro_f1", "balanced_accuracy")}
        or len(metric.options) == 2
    )
    _choose(app, "primary_metric", "macro_f1", radio=True)
    _save_text(app, "metric_rationale", "Todas las categorías importan")
    _choose(app, "guardrail_mode", "no_quantitative_guardrail", radio=True)
    for column in ("signal", "category"):
        next(
            i for i in app.radio if f"feature-{_draft_id(app)}-{column}-r" in (i.key or "")
        ).set_value("use")
        _run(app)
    _by_label(app.button, "Guardar mis decisiones").click()
    _run(app)
    assert not app.error, [item.value for item in app.error]

    _by_label(app.button, "Preparar el plan de entrenamiento").click()
    _run(app)
    assert not app.error, [item.value for item in app.error]
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    context = json.loads(
        connection.execute("SELECT context_json FROM context_versions").fetchone()[0]
    )["context"]
    strategy = json.loads(
        connection.execute("SELECT strategy_json FROM strategy_versions").fetchone()[0]
    )
    connection.close()
    assert context["task_type"] == "multiclass_classification"
    assert context["positive_class"] is None
    assert strategy["task_type"] == "multiclass_classification"

    # Aprobar el plan y entrenar desde la pantalla.
    next(i for i in app.radio if i.label == "Decisión sobre el entrenamiento").set_value("APPROVED")
    _by_label(app.text_input, "Responsable de revisar la estrategia").input(ACTOR)
    for box in app.checkbox:
        if box.label == "Leí los puntos para revisar de este plan":
            box.check()
    _by_label(app.button, "Registrar decisión").click()
    _run(app)
    _by_label(app.button, "Entrenar y comparar modelos").click()
    app.run(timeout=180)
    assert not app.exception, app.exception
    assert not app.error, [item.value for item in app.error]
    assert any(
        "Se recomienda" in item.value or "Ninguna alternativa" in item.value
        for item in [*app.success, *app.info]
    )


def test_column_recommendations_are_accepted_in_one_click_and_undone(
    runtime_dir: Path,
) -> None:
    churn = Path(__file__).parents[2] / "data" / "synthetic" / "abandono_suscripcion.csv"
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Abandono")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": churn.name,
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=churn.read_bytes(),
    )
    connection.close()

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    _choose(app, "task_type", "binary_classification", radio=True)
    _choose(app, "target_column", "abandona")

    accept = next(item for item in app.button if item.label.startswith("Aceptar recomendaciones"))
    accept.click()
    _run(app)
    assert not app.error, [item.value for item in app.error]
    texts = " ".join(item.value for item in app.markdown)
    assert "**id_suscriptor** · confirmado" in texts
    assert any(item.label == "Deshacer las recomendaciones aceptadas" for item in app.button)

    _by_label(app.button, "Deshacer las recomendaciones aceptadas").click()
    _run(app)
    texts = " ".join(item.value for item in app.markdown)
    assert "**id_suscriptor** · confirmado" not in texts


class _SlowProvider:
    name = "pytest_ui_slow"
    model = "fake-slow"

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("PROVIDER_TIMEOUT")


def test_a_slow_ai_is_not_reported_as_switched_off(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorrido 22/10: si la IA tardaba, la pantalla decía «Modo sin IA generativa»."""
    register_provider("pytest_ui_slow", lambda settings: _SlowProvider())
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_ui_slow")
    monkeypatch.setenv("LLM_MODEL", "fake-slow")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    _project_with_dataset(runtime_dir)

    app = _run(AppTest.from_file(str(APP_PATH)))
    _set_actor(app)
    next(item for item in app.text_area if (item.key or "").endswith("-message")).input(
        "Quiero anticipar el evento del fixture."
    )
    _by_label(app.button, "Revisar antes de enviar").click()
    _run(app)
    _by_label(app.button, "Enviar a la IA").click()
    _run(app)

    warnings = " ".join(item.value for item in app.warning)
    assert "La IA no pudo ayudar esta vez: la IA tardó demasiado en responder" in warnings
    assert "Modo sin IA generativa" not in warnings + " ".join(item.value for item in app.markdown)
