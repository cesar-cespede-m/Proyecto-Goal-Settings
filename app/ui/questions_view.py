"""«Pregúntale a tus datos» (GA-04c): la IA planifica e interpreta; el motor calcula.

Genérico: todo sale del catálogo del proyecto y de los tipos de columna. Por ADR-014 la
persona elige qué columnas puede ver la IA y si ve los valores de las categorías; antes de
enviar ve exactamente qué sale y queda registro de columnas y filas enviadas.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.advanced_view import render_plan_editor
from app.ui.analysis_view import render_analysis_result
from app.ui.incidents import configured_provider
from app.ui.messages import fallback_reason_text, render_error
from app.ui.table_save_view import remember_result, render_save_table
from core.ai_engine.data_questions import DataQuestionService, interpret
from core.ai_engine.proposals import ProposalService
from core.analysis.engine import AnalysisEngine, LoadedSource, load_project_source
from core.analysis.guided import describe_plan, suggested_questions
from core.analysis.kinds import ANALYSIS_KINDS
from core.analysis.payload import omitted_columns, to_llm_payload
from core.analysis.relations import describe_relations, find_relations
from core.analysis.validation import AnalysisPlanError
from core.data_engine import analysis_schema

MAX_FILES = 5
HIDDEN_BY_DEFAULT = {"personal", "identifier"}
_KEY = "data-questions"


def source_name(filename: str, used: set[str]) -> str:
    """Nombre corto y estable para cada archivo dentro del plan."""

    base = re.sub(r"[^0-9A-Za-z_]+", "_", Path(filename).stem).strip("_").lower() or "archivo"
    name, counter = base, 2
    while name in used:
        name, counter = f"{base}_{counter}", counter + 1
    used.add(name)
    return name


def _category_values(source: LoadedSource, columns: list[str]) -> dict[str, list[str]]:
    return {
        column: sorted(source.frame[column].dropna().astype(str).unique().tolist())[:30]
        for column in columns
        if source.schema.get(column) == "category"
    }


def render_questions(
    repository: Any,
    storage_root: Path,
    project_id: str,
    *,
    on_saved: Callable[[str], None] | None = None,
) -> None:
    datasets = repository.list_datasets(project_id, admission_status="ACCEPTED")
    if not datasets:
        return
    st.markdown("### Pregúntale a tus datos")
    if configured_provider() is None:
        st.caption(
            "Sin IA activa puedes usar las ideas para empezar: Goal Settings calcula y resume con "
            "cifras verificadas. Las preguntas con tus palabras necesitan IA."
        )
    else:
        st.caption(
            "Escribe una pregunta. La IA arma el cálculo, Goal Settings lo hace con "
            "tus datos y la IA te explica el resultado con cifras verificadas."
        )
    labels = {str(row["id"]): str(row["original_filename_display"]) for row in datasets}
    chosen = st.multiselect(
        "Archivos que se pueden usar",
        list(labels),
        default=list(labels)[:MAX_FILES],
        max_selections=MAX_FILES,
        format_func=lambda value: labels[value],
        key=f"{_KEY}-files-{project_id}",
    )
    if not chosen:
        st.info("Elige al menos un archivo.")
        return
    used: set[str] = set()
    names = {dataset_id: source_name(labels[dataset_id], used) for dataset_id in chosen}
    if len(chosen) > 1:
        _render_relations(repository, storage_root, project_id, names)
    profiles = {
        names[str(row["id"])]: analysis_schema(json.loads(row["profile_json"]))
        for row in datasets
        if str(row["id"]) in chosen
    }
    every_column = [
        f"{source}.{column}" for source, schema in profiles.items() for column in schema
    ]
    default_visible = [
        f"{source}.{column}"
        for source, schema in profiles.items()
        for column, kind in schema.items()
        if kind not in HIDDEN_BY_DEFAULT
    ]
    with st.expander("Qué puede ver la IA", expanded=False):
        st.warning(
            "Tú decides qué columnas ve la IA. No envíes datos personales a un servicio de IA "
            "que no esté autorizado para ellos: la responsabilidad es tuya. Por defecto quedan "
            "fuera las columnas que parecen personales o identificadores."
        )
        visible = st.multiselect(
            "Columnas que la IA puede ver",
            every_column,
            default=default_visible,
            key=f"{_KEY}-columns-{project_id}",
        )
        share_values = st.checkbox(
            "Permitir que la IA vea los valores de las categorías (sirve para filtrar, por "
            "ejemplo «sólo los casos con X»)",
            key=f"{_KEY}-values-{project_id}",
        )
    visible_set = set(visible)
    schemas = {
        source: {
            column: kind for column, kind in schema.items() if f"{source}.{column}" in visible_set
        }
        for source, schema in profiles.items()
    }
    schemas = {source: schema for source, schema in schemas.items() if schema}
    if not schemas:
        st.info("La IA no tiene columnas para ver. Habilita al menos una en «Qué puede ver la IA».")
        return

    notes = _render_notes(repository, project_id)
    _render_proposals(repository, project_id, schemas, notes)
    suggestions = suggested_questions(schemas)
    if suggestions:
        st.caption("Ideas para empezar:")
        columns = st.columns(min(3, len(suggestions)))
        for index, suggestion in enumerate(suggestions[:3]):
            if columns[index].button(suggestion.text, key=f"{_KEY}-suggestion-{index}"):
                st.session_state[f"{_KEY}-question-{project_id}"] = suggestion.text
                st.session_state[f"{_KEY}-template-{project_id}"] = index
    prepare = (
        st.radio(
            "Qué quieres hacer",
            ["answer", "prepare"],
            format_func=lambda value: (
                "Responder una pregunta"
                if value == "answer"
                else "Preparar una tabla para entrenar (una fila por caso)"
            ),
            horizontal=True,
            key=f"{_KEY}-mode-{project_id}",
        )
        == "prepare"
    )
    if prepare:
        st.caption(
            "Describe qué quieres anticipar y con qué archivos; por ejemplo, «cada caso con los "
            "datos del otro archivo y marca si el valor supera el promedio». La IA arma una tabla "
            "con una fila por caso que luego guardas para entrenar."
        )
    question = st.text_area(
        "Tu pregunta",
        key=f"{_KEY}-question-{project_id}",
        placeholder="Ej.: ¿en qué grupo y en qué momento se concentran más casos?",
    )
    if st.button("Preguntar a los datos", key=f"{_KEY}-ask-{project_id}", type="primary"):
        _answer(
            repository,
            storage_root,
            project_id,
            question=question,
            source_ids={names[dataset_id]: dataset_id for dataset_id in chosen},
            schemas=schemas,
            profiles=profiles,
            visible_set=visible_set,
            share_values=share_values,
            suggestions=suggestions,
            notes=notes,
            prepare=prepare,
        )
    render_plan_editor(
        repository, storage_root, project_id, profiles, names, f"{_KEY}-table-{project_id}"
    )
    render_save_table(
        repository, storage_root, project_id, f"{_KEY}-table-{project_id}", on_saved=on_saved
    )


def _render_relations(
    repository: Any, storage_root: Path, project_id: str, names: dict[str, str]
) -> None:
    with st.expander("Cómo se relacionan tus archivos"):
        st.caption(
            "Goal Settings compara los valores de las columnas para encontrar por dónde cruzar o "
            "apilar archivos. Es una sugerencia: tú decides si tiene sentido para tu negocio."
        )
        if not st.button("Buscar relaciones", key=f"{_KEY}-relations-{project_id}"):
            return
        try:
            sources = {
                name: load_project_source(repository, storage_root, dataset_id)
                for dataset_id, name in names.items()
            }
        except AnalysisPlanError as error:
            render_error(error, fallback_code="ANALYSIS_FAILED")
            return
        lines = describe_relations(find_relations(sources))
        if not lines:
            st.info(
                "No encontré columnas con valores en común ni archivos con las mismas columnas."
            )
        for line in lines:
            st.markdown(f"- {line}")


def _confirmed_outcome(
    repository: Any, project_id: str, source_ids: dict[str, str]
) -> tuple[str, str] | None:
    """Archivo y columna del resultado confirmado en las decisiones vigentes, si se usan aquí."""

    context = repository.get_latest_context(project_id)
    if context is None:
        return None
    for name, dataset_id in source_ids.items():
        if dataset_id == context["dataset_id"]:
            return name, str(context["target_column"])
    return None


def _answer(
    repository: Any,
    storage_root: Path,
    project_id: str,
    *,
    question: str,
    source_ids: dict[str, str],
    schemas: dict[str, dict[str, str]],
    profiles: dict[str, dict[str, str]],
    visible_set: set[str],
    share_values: bool,
    suggestions: list[Any],
    notes: str = "",
    prepare: bool = False,
) -> None:
    if not question.strip():
        st.error("Escribe una pregunta o elige una idea para empezar.")
        return

    sources: dict[str, LoadedSource] = {}
    try:
        for name, dataset_id in source_ids.items():
            if name in schemas:
                sources[name] = load_project_source(repository, storage_root, dataset_id)
    except AnalysisPlanError as error:
        render_error(error, fallback_code="ANALYSIS_FAILED")
        return
    values = (
        {
            source: _category_values(sources[source], list(schema))
            for source, schema in schemas.items()
        }
        if share_values
        else None
    )
    provider = configured_provider()
    template_index = st.session_state.pop(f"{_KEY}-template-{project_id}", None)
    template = (
        suggestions[template_index]
        if template_index is not None
        and template_index < len(suggestions)
        and suggestions[template_index].text == question
        else None
    )
    if template is not None:
        plan = template.plan
    else:
        with st.spinner("La IA está armando el cálculo…"):
            answer = DataQuestionService(provider).plan(
                question,
                schemas,
                context_notes=notes or None,
                category_values=values,
                modeling_table=prepare,
                outcome=_confirmed_outcome(repository, project_id, source_ids),
            )
        if answer.mode == "NOT_ANSWERABLE":
            st.info(f"No se puede responder con estos datos: {answer.reason}")
            return
        if answer.plan is None:
            reason = fallback_reason_text(answer.fallback_reason)
            st.warning(
                f"No pude armar el cálculo con IA ({reason}). Usa una idea para empezar o "
                "«Analizar tus datos»."
            )
            return
        plan = answer.plan
        repository.record_ai_data_shared(
            project_id=project_id,
            purpose="plan_question",
            columns=[
                f"{source}.{column}" for source, schema in schemas.items() for column in schema
            ],
            rows=sum(len(columns) for columns in (values or {}).values()),
        )

    st.markdown("**Cómo lo calculé**")
    for line in describe_plan(plan):
        st.markdown(f"- {line}")
    try:
        result = AnalysisEngine().run(plan, sources)
    except AnalysisPlanError as error:
        render_error(error, fallback_code="ANALYSIS_FAILED")
        return
    render_analysis_result(result, plan.output)
    remember_result(f"{_KEY}-table-{project_id}", plan, source_ids)
    if prepare:
        st.info(
            f"La tabla tiene {result.total_rows} filas. Revísala y, si está bien, guárdala abajo "
            "en «Guardar este cálculo como tabla» marcando «Usarla para entrenar»."
        )
        return

    hidden = {
        column
        for source, schema in profiles.items()
        for column in schema
        if f"{source}.{column}" not in visible_set
    }
    allowed = [column.name for column in result.columns if column.name not in hidden]
    payload = to_llm_payload(result, allowed)
    left_out = omitted_columns(result, allowed)
    with st.spinner("La IA está leyendo el resultado…"):
        reading = interpret(
            provider,
            question=question,
            plan_description=describe_plan(plan),
            payload=payload,
            total_rows=result.total_rows,
        )
    if reading.mode == "LLM":
        repository.record_ai_data_shared(
            project_id=project_id,
            purpose="interpret_result",
            columns=allowed,
            rows=payload.sent_row_count,
        )
    st.markdown("**Qué responde**")
    st.write(reading.explanation)
    for finding in reading.findings:
        st.markdown(f"- {finding}")
    if reading.recommendation:
        st.markdown(f"**Para decidir:** {reading.recommendation}")
    for caveat in reading.caveats:
        st.caption(f"Límite: {caveat}")
    if reading.mode == "FALLBACK":
        st.caption("Resumen sin IA: " + fallback_reason_text(reading.fallback_reason) + ".")
    if left_out:
        st.caption("No se enviaron a la IA: " + ", ".join(f"«{name}»" for name in left_out) + ".")


_TASK_NAMES = {
    "binary_classification": "sí/no",
    "multiclass_classification": "varias categorías",
    "regression": "un número",
}


def _render_proposals(
    repository: Any, project_id: str, schemas: dict[str, dict[str, str]], notes: str = ""
) -> None:
    """GA-09: la IA propone qué preguntar, analizar y entrenar; la persona decide."""

    title = "Qué te propone la IA" if configured_provider() else "Ideas de Goal Settings"
    with st.expander(title):
        st.caption(
            "Con las columnas que la IA puede ver, propone preguntas, análisis y modelos posibles. "
            "Son ideas para decidir: nada se ejecuta ni se entrena sin ti."
        )
        if not st.button("Pedir propuestas", key=f"{_KEY}-propose-{project_id}"):
            return
        with st.spinner("La IA está revisando tus columnas…"):
            answer = ProposalService(configured_provider()).propose(
                schemas, context_notes=notes or None
            )
        if answer.mode == "LLM":
            repository.record_ai_data_shared(
                project_id=project_id,
                purpose="propose",
                columns=[
                    f"{source}.{column}" for source, schema in schemas.items() for column in schema
                ],
                rows=0,
            )
        else:
            st.caption(
                "Propuestas armadas sin IA a partir de los tipos de columna"
                + (
                    f" ({fallback_reason_text(answer.fallback_reason)})."
                    if answer.fallback_reason
                    else "."
                )
            )
        kinds = {kind.key: kind.title for kind in ANALYSIS_KINDS}
        if answer.proposals.questions:
            st.markdown("**Preguntas para hacerle a tus datos**")
            for question in answer.proposals.questions:
                st.markdown(f"- {question}")
        if answer.proposals.analyses:
            st.markdown("**Análisis que conviene mirar** (en «Análisis automáticos»)")
            for item in answer.proposals.analyses:
                st.markdown(f"- {kinds.get(item.kind, item.kind)}: {item.reason}")
        for model in answer.proposals.models:
            st.markdown(
                f"**Modelo posible:** anticipar «{model.target_column}» "
                f"({_TASK_NAMES.get(model.task_type, model.task_type)})"
            )
            st.caption(model.reason)
            if model.suggested_columns:
                st.caption("Con: " + ", ".join(f"«{name}»" for name in model.suggested_columns))
            for risk in model.leakage_risks:
                st.caption(f"Revisa: {risk}")


def _render_notes(repository: Any, project_id: str) -> str:
    """GA-13: notas libres del negocio que la IA usa como contexto; quedan versionadas."""

    current = repository.latest_project_notes(project_id)
    with st.expander("Notas de tu negocio (opcional)"):
        st.caption(
            "Cuenta con tus palabras lo que la IA debería saber para entender tus datos: qué "
            "significa cada archivo, qué te preocupa, qué no se debe usar. Se envían a la IA "
            "junto con tus preguntas y propuestas. No escribas datos personales."
        )
        with st.form(f"{_KEY}-notes-{project_id}"):
            body = st.text_area("Notas", value=current, max_chars=4000, height=150)
            actor = st.text_input("¿Quién las escribe?", placeholder="Nombre o rol")
            saved = st.form_submit_button("Guardar notas")
        if saved:
            try:
                repository.save_project_notes(project_id, body, actor_label=actor)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Notas guardadas. La versión anterior queda en el historial.")
                current = body.strip()
    return current
