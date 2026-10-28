"""C2-02: «Decisiones tomadas con tus datos» y lo que pasó después (ciclo 2, B3/B4)."""

from typing import Any

import streamlit as st

from app.ui.messages import render_error
from core.domain.decisions import DecisionRecordError, DecisionService

BASED_ON = {
    "analysis": "Un análisis o una pregunta a los datos",
    "prediction": "Una predicción del modelo",
    "report": "El reporte de un modelo",
}


def render_decisions(repository: Any, project_id: str) -> None:
    service = DecisionService(repository)
    st.markdown("### Decisiones tomadas con tus datos")
    st.caption(
        "Anota qué decidiste a partir de un análisis, una predicción o un reporte, y más adelante "
        "qué pasó. Así se puede medir si los datos y los modelos ayudan de verdad."
    )
    with st.expander("Anotar una decisión"):
        with st.form(f"decision-{project_id}", clear_on_submit=True):
            decision = st.text_area("Qué decidiste", max_chars=2000)
            based_on = st.selectbox("En qué te basaste", list(BASED_ON), format_func=BASED_ON.get)
            reference = st.text_input(
                "Referencia (nombre del análisis, del archivo o del modelo)", max_chars=100
            )
            actor = st.text_input("¿Quién decide?", placeholder="Nombre o rol")
            submitted = st.form_submit_button("Guardar decisión")
        if submitted:
            try:
                service.record(
                    project_id,
                    decision=decision,
                    based_on=based_on,
                    reference_id=reference.strip() or "sin referencia",
                    actor_label=actor,
                )
            except DecisionRecordError as error:
                render_error(error, fallback_code="DECISION_RECORD_FAILED")
            else:
                st.success("Decisión guardada.")
    for stored in service.list(project_id)[:20]:
        record = stored.record
        with st.container(border=True):
            st.markdown(f"**{record.decision}**")
            st.caption(
                f"{BASED_ON[record.based_on]} · «{record.reference_id}» · "
                f"{record.actor_label} · {record.decided_at:%Y-%m-%d}"
            )
            for outcome in stored.outcomes:
                st.markdown(f"- Resultado observado: {outcome['observed_result']}")
            with st.popover("Anotar qué pasó"):
                with st.form(f"outcome-{stored.id}", clear_on_submit=True):
                    result = st.text_area("Qué pasó después", max_chars=2000)
                    observer = st.text_input("¿Quién lo observó?")
                    saved = st.form_submit_button("Guardar resultado")
                if saved:
                    try:
                        service.observe(stored.id, observed_result=result, actor_label=observer)
                    except DecisionRecordError as error:
                        render_error(error, fallback_code="DECISION_RECORD_FAILED")
                    else:
                        st.success("Resultado guardado.")
