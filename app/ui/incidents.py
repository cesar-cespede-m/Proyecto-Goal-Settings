"""Código de incidente por cada falla, registro local para encontrarla y ayuda de la IA.

Cada falla mostrada recibe un código como `GSA-261020-7F3A`. El detalle técnico y la traza
quedan en `runtime/logs/incidents.jsonl` (fuera de Git), con secretos y datos personales
redactados, para buscarlo con `scripts/find_incident.py <código>`.
"""

from __future__ import annotations

import json
import os
import secrets
import traceback
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from core.ai_engine.config import LLMConfigError, LLMSettings
from core.ai_engine.error_help import ErrorHelp, ErrorHelper, sanitize_technical
from core.ai_engine.outgoing_review import redact
from core.ai_engine.provider import build_provider

_HELP_CACHE = "incident-ai-help"


def incidents_log() -> Path:
    return Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", "runtime")) / "logs" / "incidents.jsonl"


def new_incident_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%y%m%d")
    return f"GSA-{stamp}-{secrets.token_hex(2).upper()}"


def record_incident(error: BaseException, *, code: str, where: str) -> str:
    """Guarda la falla con su traza redactada; nunca rompe la pantalla si no puede escribir."""

    incident_id = new_incident_id()
    entry = {
        "incident_id": incident_id,
        "created_at": datetime.now(UTC).isoformat(),
        "code": code,
        "where": where,
        "error_type": type(error).__name__,
        "detail": sanitize_technical(str(error)),
        "traceback": redact(
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )[0][-6000:],
    }
    try:
        path = incidents_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return incident_id


def _provider():
    try:
        env_file = Path(os.environ.get("GOAL_SETTINGS_ENV_FILE", ".env"))
        settings = LLMSettings.from_env(env_file=env_file)
        return build_provider(settings)
    except LLMConfigError:
        return None


# Proveedor configurado en este equipo (o None), compartido por las pantallas con IA.
configured_provider = _provider


def ai_help(*, code: str, message: str, action: str, technical: str, where: str) -> ErrorHelp:
    """Pide la explicación a la IA una sola vez por falla idéntica en la sesión."""

    cache = st.session_state.setdefault(_HELP_CACHE, {})
    key = f"{code}|{where}|{technical}"
    if key not in cache:
        provider = _provider()
        if provider is None:
            cache[key] = ErrorHelp(mode="DISABLED").model_dump()
        else:
            with st.spinner("La IA está revisando qué pasó…"):
                cache[key] = (
                    ErrorHelper(provider)
                    .explain(
                        code=code,
                        user_message=message,
                        user_action=action,
                        technical_detail=technical,
                        where=where,
                    )
                    .model_dump()
                )
    return ErrorHelp.model_validate(cache[key])


def render_incident(
    error: BaseException, *, code: str, message: str, action: str, where: str
) -> str:
    """Mensaje fijo, explicación de la IA si está activa, código y detalle técnico."""

    incident_id = record_incident(error, code=code, where=where)
    # V2-07: qué pasó y qué puedes hacer, separados; siempre hay una salida.
    st.error(f"{message}\n\n**Qué puedes hacer:** {action}")
    help_ = ai_help(code=code, message=message, action=action, technical=str(error), where=where)
    if help_.mode == "LLM" and help_.explanation:
        with st.container(border=True):
            st.markdown("**La IA revisó esta falla**")
            st.write(help_.explanation)
            for step in help_.steps:
                st.markdown(f"- {step}")
            st.caption("Es una sugerencia de la IA; no cambió nada en tus datos ni aprobaciones.")
    st.caption(
        f"Código de incidente: `{incident_id}`. Si la falla sigue, compártelo para encontrar "
        "la causa exacta."
    )
    with st.expander("Detalle técnico"):
        st.code(f"{incident_id} · {code}: {sanitize_technical(str(error))}")
    if st.button("Volver a intentar", key=f"retry-{incident_id}"):
        st.rerun()
    return incident_id
