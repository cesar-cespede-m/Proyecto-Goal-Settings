"""GA-06: guardar el último cálculo como tabla del proyecto y, si se quiere, usarla para entrenar.

Lo comparten «Analizar tus datos» y «Pregúntale a tus datos». El cálculo se recuerda en la
sesión para que guardar no obligue a repetir la pregunta.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.messages import render_error
from app.ui.table_review_view import review_table_change
from core.analysis.contracts import AnalysisPlanV1
from core.analysis.guided import describe_plan
from core.analysis.tables import AnalysisTableService
from core.analysis.validation import AnalysisPlanError


def remember_result(key: str, plan: AnalysisPlanV1, source_ids: Mapping[str, str]) -> None:
    st.session_state[key] = {
        "plan": plan.model_dump(mode="json"),
        "sources": {name: source_ids[name] for name in plan.sources},
    }


def render_save_table(
    repository: Any,
    storage_root: Path,
    project_id: str,
    key: str,
    *,
    on_saved: Callable[[str], None] | None = None,
) -> None:
    remembered = st.session_state.get(key)
    if not remembered:
        return
    plan = AnalysisPlanV1.model_validate(remembered["plan"])
    with st.expander("Guardar este cálculo como tabla"):
        st.caption(
            "La tabla queda en los archivos del proyecto junto con su receta: qué archivos y qué "
            "pasos la forman. Así puedes volver a analizarla, usarla para entrenar o rehacerla "
            "cuando lleguen datos nuevos. Si lo que quieres anticipar no es una columna, pide en "
            "el cálculo que lo marque como sí/no desde un umbral."
        )
        for line in describe_plan(plan):
            st.caption(f"· {line}")
        with st.form(f"{key}-form"):
            name = st.text_input("Nombre de la tabla", max_chars=80)
            actor = st.text_input("¿Quién la guarda?", placeholder="Nombre o rol")
            use_for_training = st.checkbox(
                "Usarla para entrenar (después igual revisas y apruebas el entrenamiento)"
            )
            submitted = st.form_submit_button("Guardar tabla")
        if not submitted:
            return
        try:
            saved = AnalysisTableService(repository, storage_root).save(
                project_id=project_id,
                plan=plan,
                source_ids=remembered["sources"],
                display_name=name,
                actor_label=actor,
            )
            if use_for_training:
                level = review_table_change(repository, project_id, saved.dataset_id)
                if level != "ok":
                    # V2-11: se guarda, pero no se activa sin revisión.
                    st.info(
                        f"Guardé «{saved.display_name}», pero no la activé para entrenar por los "
                        "avisos de arriba. Si igual la quieres usar, cámbiala desde «Archivos "
                        "del proyecto» → «Cambiar la tabla para entrenar»."
                    )
                    return
                repository.set_modeling_table(project_id, saved.dataset_id, actor_label=actor)
        except (AnalysisPlanError, LookupError, ValueError) as error:
            render_error(error, fallback_code="ANALYSIS_TABLE_FAILED")
            return
        st.session_state.pop(key, None)
        message = (
            f"Guardé «{saved.display_name}» con {saved.rows} filas y {saved.columns} columnas."
            + (" Es ahora la tabla para entrenar." if use_for_training else "")
        )
        if on_saved is not None:
            on_saved(message)
        else:
            st.success(message)
