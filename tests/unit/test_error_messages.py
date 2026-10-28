"""Toda falla visible tiene texto claro, código de incidente rastreable y ayuda de la IA."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.incidents import new_incident_id, record_incident
from app.ui.messages import ERROR_MESSAGES, FALLBACK_REASONS
from core.ai_engine.error_help import ErrorHelper, sanitize_technical
from core.ai_engine.provider import LLMProviderError, LLMResponse

ROOT = Path(__file__).parents[2]

# Códigos de core/ que no llegan a la pantalla como falla: los usan los scripts de terminal
# de la integración SDLC, o son razones de la salud del modelo (MON-02).
NOT_SHOWN_AS_UI_ERRORS = {
    "EMPTY_PLAN",  # sólo vuelve a la IA como motivo del reintento (GA-04a)
    "MODELING_TABLE_NOT_ROW_LEVEL",  # ídem, en el modo «preparar tabla para entrenar»
    "AGGREGATION_REQUIRES_GROUP_BY",
    "ANALYSIS_INPUT_ROW_LIMIT_EXCEEDED",
    "ANALYSIS_TIME_LIMIT_EXCEEDED",
    "APPROVAL_MISSING",
    "ARTIFACT_HASH_MISMATCH",
    "ARTIFACT_MISSING",
    "ARTIFACT_PATH_INVALID",
    "COLUMN_NOT_FOUND",
    "EVIDENCE_INPUT_INVALID",
    "EXPORT_ACTOR_REQUIRED",
    "GH_NOT_AVAILABLE",
    "GH_SECRET_UPDATE_FAILED",
    "HOLDOUT_MISSING",
    "MODEL_NOT_RELEASEABLE",
    "PUBLICATION_NOT_APPROVED",
    "PUBLIC_KEY_INVALID",
    "REPORT_NOT_PUBLISHED",
    "SOURCE_NOT_FOUND",
    "SIGNING_KEY_INVALID",
    "TOO_MANY_STEPS",
    "TYPE_MISMATCH",
    "UNION_SCHEMA_MISMATCH",
    "VALIDATION_MISSING",
    "VARIABLE_NAME_INVALID",
    "DATA_CHANGED_A_LOT",
    "DATA_CHANGING",
    "NOT_ENOUGH_OUTCOMES",
    "NO_OUTCOMES_YET",
    "OPERATION_NOT_IMPLEMENTED",
    "PIVOT_COLUMN_COLLISION",
    "PIVOT_COLUMN_LIMIT_EXCEEDED",
    "RESULT_SCHEMA_MISMATCH",
    "THRESHOLD_VALUE_UNAVAILABLE",
}


def _core_codes() -> set[str]:
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "core").rglob("*.py"))
    codes = set(re.findall(r'(?:Error|Exception)\(\s*\n?\s*f?"([A-Z][A-Z0-9_]{3,})"', sources))
    return codes | set(re.findall(r'code\s*=\s*"([A-Z][A-Z0-9_]{3,})"', sources))


def test_every_error_raised_by_core_has_a_plain_message() -> None:
    covered = set(ERROR_MESSAGES) | set(FALLBACK_REASONS) | NOT_SHOWN_AS_UI_ERRORS
    missing = _core_codes() - covered

    assert not missing, f"Falta texto para: {sorted(missing)}"


def test_every_fallback_code_used_by_the_screens_has_a_message() -> None:
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app").rglob("*.py"))
    used = set(re.findall(r'fallback_code="([A-Z_]+)"', sources))

    assert used <= set(ERROR_MESSAGES), sorted(used - set(ERROR_MESSAGES))


def test_messages_are_plain_spanish_without_codes() -> None:
    for code, common in ERROR_MESSAGES.items():
        text = f"{common.message} {common.action}"
        assert not re.search(r"\b[A-Z]{3,}_[A-Z_]+\b", text), code
        assert "pydantic" not in text.lower() and "traceback" not in text.lower(), code
        assert not re.search(
            r"(snapshot|holdout|pipeline|hash|fingerprint|baseline|dataset|storage)",
            text,
            re.IGNORECASE,
        ), code


def test_incident_is_recorded_redacted_and_found_by_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    try:
        raise ValueError(
            "token TEST_SECRET_TOKEN_INCIDENT_123456789012 falló para persona@example.test"
        )
    except ValueError as error:
        incident_id = record_incident(error, code="SCORING_FAILED", where="Usar el modelo")

    assert re.fullmatch(r"GSA-\d{6}-[0-9A-F]{4}", incident_id)
    stored = (tmp_path / "logs" / "incidents.jsonl").read_text(encoding="utf-8")
    assert "TEST_SECRET_TOKEN_INCIDENT_123456789012" not in stored
    assert "persona@example.test" not in stored
    entry = json.loads(stored)
    assert entry["code"] == "SCORING_FAILED" and "Traceback" in entry["traceback"]

    found = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "find_incident.py"), incident_id.lower()]
        + ["--runtime", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 0
    assert incident_id in found.stdout and "Usar el modelo" in found.stdout


def test_incident_ids_differ() -> None:
    assert len({new_incident_id() for _ in range(50)}) > 45


def test_technical_detail_sent_to_the_ai_drops_typed_values() -> None:
    detail = (
        "1 validation error for DatasetCompositionRequestV1\nactor_label\n  String should have "
        "at least 1 character [type=string_too_short, input_value='Ana 912345678', "
        "input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12"
    )
    clean = sanitize_technical(detail)

    assert "Ana" not in clean and "912345678" not in clean and "errors.pydantic.dev" not in clean
    assert "actor_label" in clean and "string_too_short" in clean


class _Provider:
    name = "fake"
    model = "fake-help"

    def __init__(self, content: str | None = None, error: str | None = None) -> None:
        self.content, self.error, self.requests = content, error, []

    def complete_json(self, request):
        self.requests.append(request)
        if self.error:
            raise LLMProviderError(self.error)
        return LLMResponse(content=self.content, provider=self.name, model=self.model, latency_ms=1)


def _explain(provider):
    return ErrorHelper(provider).explain(
        code="INVALID_DATASET_COMPOSITION",
        user_message="No pudimos crear este cruce.",
        user_action="Revisa las claves.",
        technical_detail="actor_label String should have at least 1 character, input_value=''",
        where="Cruzar archivos",
    )


def test_ai_explains_the_failure_in_plain_language() -> None:
    provider = _Provider(
        json.dumps(
            {
                "explanation": "Falta escribir quién confirma el cruce.",
                "steps": ["Escribe tu nombre en «Responsable de confirmar el cruce»."],
            }
        )
    )
    help_ = _explain(provider)

    assert help_.mode == "LLM"
    assert "quién confirma" in help_.explanation
    assert "DATOS_NO_CONFIABLES" in provider.requests[0].user


def test_ai_help_falls_back_without_breaking_the_screen() -> None:
    assert _explain(None).mode == "DISABLED"
    assert _explain(_Provider(error="PROVIDER_TIMEOUT")).fallback_reason == "PROVIDER_TIMEOUT"
    assert _explain(_Provider("no es json")).fallback_reason == "SCHEMA_INVALID"
    jargon = json.dumps({"explanation": "Error JOIN_HAS_NO_MATCHES.", "steps": ["Reintenta."]})
    assert _explain(_Provider(jargon)).fallback_reason == "TECHNICAL_JARGON"


def test_screen_shows_message_ai_help_and_incident_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))

    def render() -> None:
        import json as json_module

        import app.ui.incidents as incidents
        from app.ui.messages import render_error
        from core.ai_engine.provider import LLMResponse as Response

        class Provider:
            name, model = "fake", "fake-help"

            def complete_json(self, request):
                content = {"explanation": "Falta el responsable.", "steps": ["Escríbelo."]}
                return Response(
                    content=json_module.dumps(content), provider="fake", model="m", latency_ms=1
                )

        incidents._provider = lambda: Provider()
        render_error(ValueError("actor_label vacío"), fallback_code="INVALID_DATASET_COMPOSITION")

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert "No pudimos crear este cruce" in app.error[0].value
    texts = " ".join(item.value for item in app.markdown)
    assert "La IA revisó esta falla" in texts and "Escríbelo." in texts
    assert any("Código de incidente: `GSA-" in item.value for item in app.caption)
    # V2-07: qué pasó y qué hacer, separados, y siempre una salida.
    assert "**Qué puedes hacer:**" in app.error[0].value
    assert any(item.label == "Volver a intentar" for item in app.button)


def test_missing_feature_error_names_column_and_suggests_similar_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_ENABLED", "false")

    def render() -> None:
        from app.ui.messages import render_error
        from core.ml_engine.scoring import MissingFeatureColumnsError

        render_error(
            MissingFeatureColumnsError(
                missing_columns=["distancia_km"],
                available_columns=["caso_id", "distancia"],
            ),
            fallback_code="SCORING_FAILED",
        )

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert "Falta «distancia_km». ¿Es «distancia»?" in app.error[0].value
    assert "Cámbiale el nombre y vuelve a subirlo." in app.error[0].value


def test_every_visible_error_has_an_action_distinct_from_its_message() -> None:
    for code, common in ERROR_MESSAGES.items():
        assert common.action.strip() and common.action != common.message, code


def test_unexpected_failure_never_shows_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_ENABLED", "false")

    def render() -> None:
        import app.main as main

        def broken(runtime_root):
            raise KeyError("columna_inexistente")

        main._render_workflow = broken
        main.render_app()

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert any("Algo falló inesperadamente" in item.value for item in app.error)
    assert any("Código de incidente" in item.value for item in app.caption)
    log = (tmp_path / "logs" / "incidents.jsonl").read_text(encoding="utf-8")
    assert "UNEXPECTED_ERROR" in log and "columna_inexistente" in log
