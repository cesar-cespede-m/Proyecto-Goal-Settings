"""Declaración humana de cobertura del lote, independiente de su último evento."""

from typing import Any

import streamlit as st


def coverage_inputs(key: str) -> dict[str, Any] | None:
    st.caption(
        "Indica el período durante el que esta fuente registró todos los eventos relevantes. "
        "La última actividad del archivo no demuestra hasta cuándo se observó. "
        "Sin esta confirmación, los resultados quedan pendientes."
    )
    start = st.text_input(
        "Inicio de observación completa (UTC)",
        placeholder="2026-01-01T00:00:00Z",
        key=f"{key}-coverage-start",
    )
    end = st.text_input(
        "Fin de observación completa (UTC)",
        placeholder="2026-02-28T23:59:59Z",
        key=f"{key}-coverage-end",
    )
    complete = st.checkbox(
        "Confirmo que la fuente cubre todos los eventos relevantes en ese período",
        key=f"{key}-coverage-complete",
    )
    if not start.strip() and not end.strip() and not complete:
        return None
    return {"observed_from": start.strip(), "observed_until": end.strip(), "complete": complete}
