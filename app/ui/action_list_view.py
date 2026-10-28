"""V2-17: de la lista priorizada a una lista de acción, y después medir si actuar ayudó."""

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.decision_view import chosen_plan
from app.ui.messages import render_error
from core.domain.action_lists import ActionListError, ActionListService
from core.ml_engine.decision import selected_row_numbers


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def render_action_list_offer(repository: Any, storage_root: Path, run: Any, result: Any) -> None:
    st.markdown("#### Crear la lista de acción")
    if result.id_column is None:
        st.info(
            "Para crear una lista de acción y medir después si ayudó, el archivo debe traer una "
            "columna que identifique cada caso (distinta en cada fila).\n\n**Qué puedes hacer:** "
            "agrega un identificador al archivo y vuelve a priorizar."
        )
        return
    decision = chosen_plan(result.run_id)
    selected_rows = (
        selected_row_numbers(
            decision[0],
            scores=[case.score for case in result.cases],
            row_numbers=[case.row_number for case in result.cases],
        )
        if decision
        else {case.row_number for case in result.cases if case.alert}
    )
    attend = [
        case.case_id for case in result.cases if case.row_number in selected_rows and case.case_id
    ]
    if not attend:
        st.info(
            "La regla seleccionada no propone intervenir en este lote. No hay una lista que crear."
        )
        return
    st.caption(
        f"Según tu decisión, atenderías {len(attend)} de {result.rows_scored} casos. Si reservas "
        "un grupo de control (casos que se iban a atender y no se atienden, elegidos al azar), "
        "después podrás estudiar el efecto de actuar en ese grupo. La asignación no confirma "
        "que la atención se haya efectuado."
    )
    with st.form(f"action-list-{result.run_id}"):
        control = st.slider("Grupo de control (%)", min_value=0, max_value=30, value=10, step=5)
        actor = st.text_input("¿Quién decide atender estos casos?", placeholder="Nombre o rol")
        submitted = st.form_submit_button("Crear lista de acción", type="primary")
    if not submitted:
        return
    try:
        created = ActionListService(repository, storage_root).create(
            str(repository.get_run_project_id(result.run_id)),
            result.run_id,
            scoring_input_sha256=result.input_sha256,
            case_ids=[case.case_id for case in result.cases if case.case_id],
            attended_ids=attend,
            control_share=control / 100,
            seed=len(attend),
            plan=decision[0] if decision else None,
            actor_label=actor,
        )
    except (ActionListError, ValueError) as error:
        render_error(error, fallback_code="ACTION_LIST_FAILED")
        return
    groups = (
        [(case_id, "atender") for case_id in created.attended_ids]
        + [(case_id, "control (no atender)") for case_id in created.control_ids]
        + [(case_id, "no atender") for case_id in created.not_attended_ids]
    )
    st.success(
        f"Lista creada: {len(created.attended_ids)} para atender, {len(created.control_ids)} de "
        "control. Quedó registrada como una decisión."
    )
    st.download_button(
        "Descargar la lista de acción",
        data=pd.DataFrame(groups, columns=[result.id_column, "grupo"])
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="lista-de-accion.csv",
        mime="text/csv",
        key=f"action-list-download-{created.id}",
    )


def render_action_list_measure(
    repository: Any, storage_root: Path, run: Any, target: str, positive_class: Any
) -> None:
    rows = repository.connection.execute(
        "SELECT id, action_json, created_at FROM action_lists WHERE run_id = ? "
        "ORDER BY created_at DESC",
        (run["id"],),
    ).fetchall()
    if not rows:
        return
    st.markdown("#### ¿Sirvió actuar?")
    st.caption(
        "Cuando sepas qué pasó con los casos de una lista, sube un CSV con el identificador y la "
        f"columna «{target}». Se compara el grupo atendido con el de control."
    )
    labels = {
        str(row["id"]): f"Lista del {str(row['created_at'])[:16]} · "
        f"{len(json.loads(row['action_json'])['attended_ids'])} atendidos"
        for row in rows
    }
    with st.form(f"action-measure-{run['id']}"):
        chosen = st.selectbox("Lista", list(labels), format_func=labels.get)
        upload = st.file_uploader("Resultados reales (CSV)", type=["csv"])
        actor = st.text_input("¿Quién registra el resultado?", placeholder="Nombre o rol")
        submitted = st.form_submit_button("Medir")
    if not submitted:
        return
    if upload is None:
        st.error("Sube el CSV con los resultados reales.")
        return
    frame = pd.read_csv(io.BytesIO(upload.getvalue()))
    if target not in frame.columns:
        st.error(
            f"El archivo no trae la columna «{target}».\n\n**Qué puedes hacer:** agrégala con el "
            "resultado real de cada caso."
        )
        return
    id_column = next((c for c in frame.columns if c != target and frame[c].is_unique), None)
    if id_column is None:
        st.error("El archivo no trae una columna que identifique cada caso.")
        return
    try:
        outcome = ActionListService(repository, storage_root).measure(
            chosen,
            outcomes=dict(zip(frame[id_column].astype(str), frame[target], strict=True)),
            positive_class=positive_class,
            actor_label=actor,
        )
    except (ActionListError, ValueError) as error:
        render_error(error, fallback_code="ACTION_LIST_FAILED")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Grupo": name,
                    "Casos con resultado": group.observed,
                    "Con el resultado": _pct(group.proportion),
                }
                for name, group in (
                    ("Atendidos", outcome.attended),
                    ("Control", outcome.control),
                    ("No atendidos", outcome.not_attended),
                )
            ]
        ),
        hide_index=True,
    )
    st.info(outcome.summary)
    for warning in outcome.warnings:
        st.caption(warning)
