"""V2-18/V2-20: entender recorridos. Goal Settings propone qué columna es el caso, el momento y la
etapa (por cómo se comportan los datos), la persona confirma y aparece el embudo."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.advanced_view import advanced_enabled
from app.ui.coverage_view import coverage_inputs
from app.ui.incidents import configured_provider
from app.ui.messages import render_error
from app.ui.table_review_view import review_table_change
from core.ai_engine.journey_proposal import propose_outcome
from core.analysis.engine import load_project_source
from core.analysis.funnel import cohorts, funnel, suggest_outcome
from core.analysis.roles import infer_roles
from core.analysis.snapshot import SnapshotError, build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.analysis.validation import AnalysisPlanError
from core.data_engine import DataIngestionService, DataValidationError
from core.data_engine.models import DatasetProfileV1

_ROLE_NAMES = {"entity": "el caso", "moment": "el momento", "stage": "la etapa"}


def render_journey(repository: Any, storage_root: Path, dataset: Any) -> None:
    try:
        source = load_project_source(repository, storage_root, str(dataset["id"]))
    except AnalysisPlanError:
        return
    profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))
    roles = infer_roles(source.frame, profile)
    found = {
        role: next((item for item in roles.items if item.role == role), None)
        for role in _ROLE_NAMES
    }
    if any(item is None for item in found.values()):
        return  # una fila por caso: no hay recorrido que mostrar
    entity, moment, stage = found["entity"], found["moment"], found["stage"]
    key = f"journey-{dataset['id']}"
    with st.expander("Recorridos: hasta dónde llega cada caso", expanded=True):
        st.caption(
            "Este archivo parece tener la actividad de cada caso en varias filas. Goal Settings "
            "propone qué papel cumple cada columna; confírmalo para ver dónde se quedan los casos."
        )
        for item in (entity, moment, stage):
            st.markdown(f"- **«{item.column}»** sería {_ROLE_NAMES[item.role]}: {item.reason}")
        st.markdown("Orden de las etapas: " + " → ".join(f"«{name}»" for name in stage.stage_order))
        if not st.checkbox("Sí, así son mis datos", key=f"{key}-ok"):
            st.caption("Si no es así, este análisis no aplica a este archivo todavía.")
            return
        result = funnel(
            source.frame,
            entity=entity.column,
            moment=moment.column,
            stage=stage.column,
            order=stage.stage_order,
        )
        st.markdown(f"**{result.summary}**")
        st.bar_chart(
            pd.DataFrame(
                {"Casos que llegan": [step.reached for step in result.steps]},
                index=[step.stage for step in result.steps],
            ),
            horizontal=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Etapa": step.stage,
                        "Llegan": step.reached,
                        "Del total": f"{step.share_of_start * 100:.0f} %",
                        "Se quedan aquí": step.stayed_here,
                        "Avanzan": "—"
                        if step.advance_rate is None
                        else f"{step.advance_rate * 100:.0f} %",
                        "Horas hasta la siguiente (mediana)": "—"
                        if step.median_hours_to_next is None
                        else round(step.median_hours_to_next, 1),
                    }
                    for step in result.steps
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Muestra dónde se quedan los casos, no por qué.")
        groups = cohorts(
            source.frame,
            entity=entity.column,
            moment=moment.column,
            stage=stage.column,
            order=stage.stage_order,
        )
        st.markdown("**Por mes de entrada**")
        st.caption(groups.summary)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Empezaron en": row.period,
                        "Casos": row.entities,
                        **{
                            f"Llegan a «{name}»": f"{share * 100:.0f} %"
                            for name, share in row.reached_share.items()
                        },
                    }
                    for row in groups.rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        measures = [item.column for item in roles.items if item.role == "measure"]
        _render_build_outcome(
            repository,
            storage_root,
            dataset,
            source.frame,
            entity,
            moment,
            stage,
            measures,
            _ai_or_rules_suggestion(repository, dataset, result, stage.column),
        )
        render_save_snapshot(repository, storage_root, dataset)


def _render_build_outcome(
    repository: Any,
    storage_root: Path,
    dataset: Any,
    frame: pd.DataFrame,
    entity: Any,
    moment: Any,
    stage: Any,
    measures: list[str],
    suggestion: Any = None,
) -> None:
    """V2-24: construir lo que se quiere anticipar desde el recorrido, sin usar el futuro."""

    key = f"journey-build-{dataset['id']}"
    st.markdown("**Anticipar quién se quedará**")
    st.caption(
        "Goal Settings arma una fila por caso: mira los primeros días para calcular "
        "su comportamiento, y lo que pasó después para saber si avanzó. Los resultados sólo "
        "se consideran completos cuando terminó su plazo y la fuente cubre la observación."
    )
    order = list(stage.stage_order)
    if suggestion is not None:
        st.info(f"Sugerencia de la Guía: {suggestion.reason} Puedes cambiarla.")
    columns = st.columns(3)
    target_stage = columns[0].selectbox(
        "¿Qué etapa debería alcanzar?",
        order[1:],
        index=order[1:].index(suggestion.target_stage) if suggestion else 0,
        key=f"{key}-stage",
    )
    observe_days = columns[1].number_input(
        "Días que se miran al inicio",
        min_value=1,
        max_value=90,
        value=suggestion.observe_days if suggestion else 3,
        key=f"{key}-observe",
    )
    within_days = columns[2].number_input(
        "Plazo para alcanzarla (días)",
        min_value=1,
        max_value=365,
        value=suggestion.within_days if suggestion else 30,
        key=f"{key}-within",
    )
    attributes = _entity_files(repository, storage_root, dataset, entity.column)
    labels = {key_: label for key_, (label, _, _) in attributes.items()}
    chosen = (
        st.selectbox(
            "Sumar datos fijos de cada caso desde (opcional)",
            [None, *labels],
            format_func=lambda value: "Ninguno" if value is None else labels[value],
            key=f"{key}-entities",
            help="Otro archivo del proyecto con una fila por caso y la misma columna de caso.",
        )
        if attributes
        else None
    )
    entities_frame = attributes[chosen][1] if chosen else None
    spec = {
        "entity": entity.column,
        "moment": moment.column,
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": int(observe_days)},
        "windows_days": [int(observe_days)],
        "features": [
            {"op": "count_events", "window_days": int(observe_days)},
            {"op": "days_since_last_event"},
            {"op": "stages_reached", "stage": stage.column, "order": order},
            {"op": "current_stage", "stage": stage.column, "order": order},
            {"op": "days_in_current_stage", "stage": stage.column, "order": order},
            *[
                {"op": "sum", "column": column, "window_days": int(observe_days)}
                for column in measures[:5]
            ],
        ],
        "outcome": [
            {
                "op": "not_reached_stage",
                "stage": {"column": stage.column, "value": target_stage},
                "within_days": int(within_days),
                "output_column": "no_avanza",
            }
        ],
    }
    if st.checkbox(
        "Conservar cuándo se conoce cada resultado para entrenar por fecha",
        value=True,
        key=f"{key}-label-availability",
        help=(
            "Añade una fecha de maduración. Se usa para separar los períodos, nunca como predictor."
        ),
    ):
        spec["label_available_at_output_column"] = "resultado_disponible_desde"
    if chosen:
        spec["entity_attributes"] = attributes[chosen][2]
    coverage = coverage_inputs(key)
    if advanced_enabled():
        # V2-31: la receta completa, editable; se valida igual que la guiada.
        text = st.text_area(
            "Receta (JSON, modo avanzado)",
            value=json.dumps(spec, ensure_ascii=False, indent=1),
            height=240,
            key=f"{key}-spec",
        )
    if not st.button("Armar la tabla", key=f"{key}-build"):
        return
    # Una reconstrucción fallida no puede dejar guardable el resultado anterior.
    st.session_state.pop(f"{key}-result", None)
    if advanced_enabled():
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as error:
            st.error(
                f"La receta no es JSON válido (línea {error.lineno}).\n\n**Qué puedes hacer:** "
                "revisa comas, comillas y llaves."
            )
            return
    try:
        result = build_snapshot(frame, spec, entities_frame, coverage=coverage)
    except (SnapshotError, ValueError) as error:
        render_error(error, fallback_code="SNAPSHOT_SPEC_INVALID")
        return
    outcome = result.summary.outcome_summaries.get("no_avanza")
    share = outcome.proportion_true if outcome else None
    st.success(
        f"Tabla lista: {result.summary.rows} casos y {result.summary.columns} columnas."
        + (
            f" Entre los resultados completos, {share * 100:.0f} % no alcanzó "
            f"«{target_stage}» en {int(within_days)} días."
            if share is not None
            else ""
        )
    )
    st.dataframe(result.dataframe.head(20), hide_index=True)
    st.caption(
        f"Resultados completos: {result.readiness.eligible_rows}. "
        f"Pendientes de observación: {result.readiness.pending_rows}. "
        "Sólo los resultados completos se guardan en la tabla para entrenar."
    )
    st.session_state[f"{key}-result"] = (result, target_stage, spec, chosen)


def render_save_snapshot(repository: Any, storage_root: Path, dataset: Any) -> None:
    """Guardar la tabla armada como archivo del proyecto y, si se quiere, usarla para entrenar."""

    key = f"journey-build-{dataset['id']}"
    built = st.session_state.get(f"{key}-result")
    if built is None:
        return
    result, target_stage, spec, entity_dataset_id = built
    frame = result.training_frame()
    if frame.empty:
        st.info(
            "Todavía no hay casos con observación completa para guardar una tabla de "
            "entrenamiento. Revisa la cobertura o incorpora resultados posteriores."
        )
        return
    with st.form(f"{key}-save"):
        name = st.text_input("Nombre de la tabla", value=f"no-avanza-a-{target_stage}")
        actor = st.text_input("¿Quién la guarda?", placeholder="Nombre o rol")
        use = st.checkbox("Usarla para entrenar")
        submitted = st.form_submit_button("Guardar la tabla")
    if not submitted:
        return
    profile = json.loads(dataset["profile_json"])
    try:
        if not actor.strip():
            raise ValueError("Escribe quién guarda la tabla")
        admitted = DataIngestionService(repository, storage_root).ingest_bytes(
            project_id=str(dataset["project_id"]),
            request={
                "original_filename_display": f"{name.strip() or 'recorrido'}.csv",
                "source_name": "Construida desde el recorrido de "
                f"«{dataset['original_filename_display']}»",
                "classification": profile.get("data_classification") or "synthetic",
                "authorization_status": "approved",
                "anonymization_declared": True,
            },
            content=frame.to_csv(index=False).encode("utf-8"),
        )
        if admitted.metadata.admission_status == "ACCEPTED":
            record_snapshot_recipe(
                repository,
                project_id=str(dataset["project_id"]),
                table_dataset_id=admitted.metadata.dataset_id,
                source_dataset_id=str(dataset["id"]),
                spec=spec,
                actor_label=actor,
                entity_dataset_id=entity_dataset_id,
                coverage=result.readiness.coverage,
                readiness=result.readiness,
            )
        if use and admitted.metadata.admission_status == "ACCEPTED":
            if (
                review_table_change(
                    repository, str(dataset["project_id"]), admitted.metadata.dataset_id
                )
                == "block"
            ):
                return
            repository.set_modeling_table(
                str(dataset["project_id"]), admitted.metadata.dataset_id, actor_label=actor
            )
    except (DataValidationError, ValueError, LookupError) as error:
        render_error(error, fallback_code="SNAPSHOT_SPEC_INVALID")
        return
    st.session_state.pop(f"{key}-result", None)
    st.success(
        "Tabla guardada."
        + (
            " Es ahora la tabla para entrenar: sigue en «Qué quieres anticipar» con «no_avanza»."
            if use
            else ""
        )
    )


def _entity_files(
    repository: Any, storage_root: Path, dataset: Any, entity: str
) -> dict[str, tuple[str, pd.DataFrame, list[str]]]:
    """Archivos del proyecto con una fila por caso y la columna del caso (datos fijos)."""

    found: dict[str, tuple[str, pd.DataFrame, list[str]]] = {}
    for row in repository.list_datasets(str(dataset["project_id"]), admission_status="ACCEPTED"):
        if str(row["id"]) == str(dataset["id"]):
            continue
        try:
            other = load_project_source(repository, storage_root, str(row["id"])).frame
        except AnalysisPlanError:
            continue
        if entity not in other.columns or not other[entity].is_unique:
            continue
        columns = [
            column
            for column in other.columns
            if column != entity and other[column].nunique(dropna=True) < len(other) * 0.9
        ]
        if columns:
            found[str(row["id"])] = (str(row["original_filename_display"]), other, columns)
    return found


def _ai_or_rules_suggestion(repository: Any, dataset: Any, result: Any, stage_column: str) -> Any:
    """V2-24e: con IA, la persona cuenta su objetivo y la IA propone; si no, reglas."""

    key = f"journey-ai-{dataset['id']}"
    provider = configured_provider()
    if provider is not None:
        objective = st.text_input(
            "Tu objetivo con tus palabras (opcional)",
            placeholder="Ej.: quiero que más clientes hagan su primera compra",
            key=f"{key}-objective",
        )
        st.caption(
            "Se envían a la IA tu objetivo y el resumen del embudo (nombres de las etapas y "
            "conteos), nunca filas."
        )
        if st.button("Pedir propuesta a la IA", key=f"{key}-ask"):
            answer = propose_outcome(provider, objective=objective, funnel_result=result)
            if answer.mode == "LLM":
                repository.record_ai_data_shared(
                    project_id=str(dataset["project_id"]),
                    purpose="journey_proposal",
                    columns=[stage_column],
                    rows=0,
                )
            st.session_state[key] = answer.suggestion
    return st.session_state.get(key) or suggest_outcome(result)
