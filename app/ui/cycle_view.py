"""C2-07: «Cómo va el ciclo» del proyecto, con los próximos pasos sugeridos."""

from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.copy import model_name
from core.domain.cycle_summary import cycle_summary

_HEALTH = {
    "NO_USES": "todavía sin usos medidos",
    "STABLE": "estable",
    "WATCH": "en observación",
    "REVIEW": "conviene revisarlo",
}


def render_cycle_summary(repository: Any, storage_root: Path, project_id: str) -> None:
    summary = cycle_summary(repository, storage_root, project_id)
    with st.expander("Cómo va el ciclo de este proyecto", expanded=bool(summary.next_steps)):
        columns = st.columns(4)
        columns[0].metric("Archivos", summary.accepted_files)
        columns[1].metric("Modelos en uso", len(summary.models_in_use))
        columns[2].metric(
            "Decisiones con resultado", f"{summary.decisions_with_outcome}/{summary.decisions}"
        )
        columns[3].metric("Eventos recibidos", summary.events_received)
        for model in summary.models_in_use:
            st.caption(
                f"Modelo ({model_name(model.selected_model)}): "
                f"{_HEALTH.get(model.health_status, model.health_status.lower())}."
            )
        for step in summary.next_steps:
            st.markdown(f"- {step}")
