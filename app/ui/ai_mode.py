"""V2-05: decir con honestidad si hay IA generativa activa y qué hace en cada caso."""

import streamlit as st

from app.ui.incidents import configured_provider

ACTIVE = (
    "IA generativa activa: propone, explica y responde con tus palabras. Las cifras siempre las "
    "calcula y verifica Goal Settings, y tú confirmas cada decisión."
)
INACTIVE = (
    "Modo sin IA generativa: Goal Settings te guía con reglas y plantillas verificables. Las "
    "preguntas libres y las explicaciones redactadas por IA no están disponibles."
)


def ai_active() -> bool:
    return configured_provider() is not None


def render_ai_mode() -> None:
    if ai_active():
        st.caption(ACTIVE)
    else:
        st.info(INACTIVE)
