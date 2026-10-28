"""C2-01: datos que llegan desde otros sistemas y su conversión en archivo del proyecto."""

from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.messages import render_error


def _entities(repository: Any, project_id: str) -> list[Any]:
    return repository.connection.execute(
        """
        SELECT entity, COUNT(DISTINCT entity_id) AS cases, COUNT(*) AS events,
               MAX(received_at) AS last_received
        FROM observation_events WHERE project_id = ?
        GROUP BY entity ORDER BY entity
        """,
        (project_id,),
    ).fetchall()


def render_incoming_events(
    repository: Any, storage_root: Path, project_id: str, on_change: Any = None
) -> None:
    entities = _entities(repository, project_id)
    if not entities:
        return
    st.markdown("### Datos que llegan desde otros sistemas")
    st.caption(
        "Otros sistemas envían hechos con `scripts/push_events.py`. Conviértelos en un archivo del "
        "proyecto para analizarlos y entrenar con ellos; cada conversión es una foto del momento."
    )
    st.dataframe(
        [
            {
                "Entidad": row["entity"],
                "Casos": row["cases"],
                "Eventos recibidos": row["events"],
                "Último recibido": row["last_received"],
            }
            for row in entities
        ],
        hide_index=True,
        width="stretch",
    )
    with st.form(f"materialize-{project_id}", clear_on_submit=True):
        entity = st.selectbox("Entidad", [row["entity"] for row in entities])
        name = st.text_input("Nombre del archivo", value="eventos")
        classification = st.radio(
            "Origen de los datos",
            ["synthetic", "sandbox"],
            format_func=lambda value: (
                "Sintéticos" if value == "synthetic" else "De prueba (sandbox)"
            ),
            horizontal=True,
        )
        actor = st.text_input("¿Quién lo convierte?", placeholder="Nombre o rol")
        submitted = st.form_submit_button("Convertir en archivo")
    if not submitted:
        return
    # Import tardío: el paquete de integración también carga la firma de evidencia, que la app
    # no necesita para arrancar (modo autónomo sin la librería de cifrado).
    from core.integration.events import EventIngestionService

    try:
        EventIngestionService(repository, storage_root).materialize(
            project_id,
            entity=entity,
            actor_label=actor,
            display_name=name,
            classification=classification,
        )
    except (ValueError, LookupError) as error:
        render_error(error, fallback_code="INVALID_DATASET")
        return
    message = f"Los eventos de «{entity}» ya están en los archivos del proyecto."
    if on_change is not None:
        on_change(message)
    else:
        st.success(message)
