"""GA-01: catálogo de archivos del proyecto y elección de la tabla para entrenar."""

import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.messages import render_error
from app.ui.table_review_view import review_table_change
from core.data_engine import analysis_schema
from core.domain.table_guard import dataset_origin

MAX_PROJECT_FILES = 20
_TYPE_LABELS = (
    ("datetime", "fecha", "fechas"),
    ("number", "número", "números"),
    ("category", "categoría", "categorías"),
    ("text", "texto", "textos"),
    ("identifier", "identificador", "identificadores"),
    ("personal", "posible dato personal", "posibles datos personales"),
)


def column_types_summary(profile: dict[str, Any]) -> str:
    """Resumen en palabras de los tipos de columna de un archivo."""

    schema = analysis_schema(profile)
    counts = pd.Series(list(schema.values())).value_counts()
    parts = [
        f"{int(counts[kind])} {singular if counts[kind] == 1 else plural}"
        for kind, singular, plural in _TYPE_LABELS
        if kind in counts
    ]
    return " · ".join(parts)


def _usage(dataset: Any, origin: str = "source") -> str:
    if dataset["is_active"]:
        return "Para entrenar"
    if origin == "computed":
        return "Tabla calculada"
    if origin == "composition" or dataset["dataset_kind"] == "DERIVED":
        return "Cruce de archivos"
    return "Disponible para analizar"


def render_project_catalog(
    repository: Any,
    project_id: str,
    *,
    render_upload: Callable[..., None],
    on_change: Callable[[str], None],
) -> None:
    """Todos los archivos aceptados del proyecto; uno es la tabla para entrenar."""

    datasets = repository.list_datasets(project_id, admission_status="ACCEPTED")
    st.markdown("### Archivos del proyecto")
    st.caption(
        "Puedes subir varios archivos, aunque tengan columnas distintas. Todos sirven para "
        "analizar y cruzar; uno se usa para entrenar el modelo."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Archivo": dataset["original_filename_display"],
                    "Filas": dataset["n_rows"],
                    "Columnas": dataset["n_cols"],
                    "Tipos de columna": column_types_summary(json.loads(dataset["profile_json"])),
                    "Uso": _usage(dataset, dataset_origin(repository, str(dataset["id"]))),
                }
                for dataset in datasets
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    if len(datasets) < MAX_PROJECT_FILES:
        with st.expander("Agregar otro archivo"):
            render_upload(heading="Agregar archivo al proyecto", form_key="catalog-upload")
    else:
        st.info(f"El proyecto llegó al máximo de {MAX_PROJECT_FILES} archivos.")

    if len(datasets) < 2:
        return
    labels = {str(row["id"]): str(row["original_filename_display"]) for row in datasets}
    active = next((str(row["id"]) for row in datasets if row["is_active"]), None)
    with st.expander("Cambiar la tabla para entrenar"):
        st.caption(
            "Las decisiones y modelos del archivo anterior no se borran: quedan guardados con su "
            "historial. Con la tabla nueva empiezas decisiones nuevas."
        )
        with st.form(f"modeling-table-{project_id}", clear_on_submit=True):
            options = [value for value in labels if value != active]
            chosen = st.selectbox(
                "Archivo para entrenar", options, format_func=lambda value: labels[value]
            )
            actor = st.text_input("¿Quién hace el cambio?", placeholder="Nombre o rol")
            understood = st.checkbox(
                "Revisé los avisos y quiero cambiarla igual (si aparece alguno)"
            )
            submitted = st.form_submit_button("Usar para entrenar")
        if submitted:
            if not actor.strip():
                st.error("Escribe quién hace el cambio; queda en el registro de auditoría.")
                return
            level = review_table_change(repository, project_id, chosen)
            if level == "block" or (level == "warning" and not understood):
                if level == "warning":
                    st.info("Marca que revisaste los avisos para confirmar el cambio.")
                return
            try:
                repository.set_modeling_table(project_id, chosen, actor_label=actor)
            except (LookupError, ValueError) as error:
                render_error(error, fallback_code="MODELING_TABLE_FAILED")
                return
            on_change(f"«{labels[chosen]}» es ahora la tabla para entrenar.")
