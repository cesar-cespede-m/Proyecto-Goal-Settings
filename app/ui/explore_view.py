"""«Analizar tus datos»: constructor guiado de análisis, genérico y sin IA (GA-03).

La persona elige archivo, qué contar o medir y por qué agrupar. Las opciones salen de los tipos
de columna (nunca de sus nombres) y se traducen a un `AnalysisPlanV1` que ejecuta el motor de
`core/analysis`. Las preguntas con IA (GA-04) usarán el mismo motor y la misma vista.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.analysis_view import render_analysis_result
from app.ui.journey_view import render_journey
from app.ui.messages import render_error
from app.ui.table_save_view import remember_result, render_save_table
from core.analysis.engine import AnalysisEngine, load_project_source
from core.analysis.guided import (
    COUNT,
    DATE_PARTS,
    GROUPABLE,
    MEASURE_LABELS,
    SOURCE,
    build_guided_plan,
)
from core.analysis.kinds import applicable_kinds, missing_counts
from core.analysis.quality import describe_findings, quality_findings
from core.analysis.validation import AnalysisPlanError
from core.data_engine import analysis_schema

# El constructor vive en el núcleo para que la pantalla y las preguntas sugeridas coincidan.
build_plan = build_guided_plan
MEASURES = MEASURE_LABELS
MAX_QUALITY_FILES = 5


def render_explore(
    repository: Any,
    storage_root: Path,
    project_id: str,
    *,
    on_saved: Callable[[str], None] | None = None,
) -> None:
    datasets = repository.list_datasets(project_id, admission_status="ACCEPTED")
    if not datasets:
        return
    st.markdown("### Analizar tus datos")
    st.caption(
        "Cuenta o mide por grupos para entender tus datos antes de decidir qué anticipar. "
        "Nada de esto cambia tus archivos ni entrena un modelo."
    )
    labels = {str(row["id"]): str(row["original_filename_display"]) for row in datasets}
    dataset_id = st.selectbox(
        "Archivo", list(labels), format_func=lambda value: labels[value], key="explore-file"
    )
    dataset = next(row for row in datasets if str(row["id"]) == dataset_id)
    schema = analysis_schema(json.loads(dataset["profile_json"]))
    # Recorrido 22/10: por defecto se agrupaba por un número continuo (50 grupos de 0 a 6
    # casos); primero se ofrecen categorías y fechas, que dan grupos legibles.
    groupable = sorted(
        (name for name, kind in schema.items() if kind in GROUPABLE),
        key=lambda name: schema[name] == "number",
    )
    numbers = [name for name, kind in schema.items() if kind == "number"]
    categories = [name for name, kind in schema.items() if kind == "category"]
    render_journey(repository, storage_root, dataset)
    _render_automatic(repository, storage_root, dataset, schema)
    if not groupable:
        st.info("Este archivo no tiene columnas para agrupar: sólo identificadores o texto libre.")
        return

    columns = st.columns(2)
    measure = columns[0].selectbox(
        "Qué contar o medir",
        [COUNT, *numbers],
        format_func=lambda value: "Cantidad de casos" if value == COUNT else value,
        key=f"explore-measure-{dataset_id}",
    )
    function = "count"
    if measure != COUNT:
        function = columns[1].selectbox(
            "Cómo resumirlo",
            list(MEASURES),
            format_func=MEASURES.get,
            key=f"explore-function-{dataset_id}",
        )
    group_by = st.selectbox("Agrupar por", groupable, key=f"explore-group-{dataset_id}")
    date_part = None
    if schema[group_by] == "datetime":
        date_part = st.radio(
            "Parte de la fecha",
            list(DATE_PARTS),
            format_func=DATE_PARTS.get,
            horizontal=True,
            key=f"explore-part-{dataset_id}",
        )
    split_by = st.selectbox(
        "Separar además por (opcional)",
        [None, *[name for name in categories if name != group_by]],
        format_func=lambda value: "Sin separar" if value is None else value,
        key=f"explore-split-{dataset_id}",
    )
    if st.button("Analizar", key=f"explore-run-{dataset_id}", type="primary"):
        _run(
            repository,
            storage_root,
            project_id,
            dataset_id,
            schema,
            group_by,
            date_part,
            measure,
            function,
            split_by,
        )
    render_save_table(
        repository, storage_root, project_id, f"explore-table-{project_id}", on_saved=on_saved
    )


def _run(
    repository: Any,
    storage_root: Path,
    project_id: str,
    dataset_id: str,
    schema: dict[str, str],
    group_by: str,
    date_part: str | None,
    measure: str,
    function: str,
    split_by: str | None,
) -> None:
    try:
        plan = build_plan(
            schema,
            group_by=group_by,
            date_part=date_part,
            measure=measure,
            function=function,
            split_by=split_by,
        )
        source = load_project_source(repository, storage_root, dataset_id)
        result = AnalysisEngine().run(plan, {SOURCE: source})
    except AnalysisPlanError as error:
        render_error(error, fallback_code="ANALYSIS_FAILED")
        return
    render_analysis_result(result, plan.output)
    remember_result(f"explore-table-{project_id}", plan, {SOURCE: dataset_id})


def _render_automatic(
    repository: Any, storage_root: Path, dataset: Any, schema: dict[str, str]
) -> None:
    """Análisis automáticos (GA-03): Goal Settings elige los cálculos según los tipos de columna."""

    dataset_id = str(dataset["id"])
    with st.expander("Análisis automáticos"):
        categories = [name for name, kind in schema.items() if kind == "category"]
        outcome = st.selectbox(
            "Lo que quieres anticipar (opcional)",
            [None, *categories],
            format_func=lambda value: "Todavía no lo sé" if value is None else value,
            key=f"auto-outcome-{dataset_id}",
        )
        if outcome is not None:
            # V2-18: advertir al elegir, no después, columnas que revelan el resultado.
            revealing = sorted(
                {
                    item["feature_column"]
                    for item in json.loads(dataset["profile_json"]).get("outcome_associations", [])
                    if item["target_column"] == outcome
                }
            )
            if revealing:
                st.warning(
                    "Estas columnas casi adivinan «"
                    + outcome
                    + "»: "
                    + ", ".join(f"«{name}»" for name in revealing)
                    + ". Puede que se conozcan recién después del resultado.\n\n"
                    "**Qué puedes hacer:** úsalas para entender, pero no para anticipar; en tus "
                    "decisiones se propondrá dejarlas fuera."
                )
        kinds = {kind.key: kind for kind in applicable_kinds(schema, outcome=outcome)}
        chosen = st.radio(
            "Qué quieres ver",
            list(kinds),
            format_func=lambda key: kinds[key].title,
            horizontal=True,
            key=f"auto-kind-{dataset_id}",
        )
        st.caption(kinds[chosen].description)
        if not st.button("Ver análisis", key=f"auto-run-{dataset_id}"):
            return
        plans = kinds[chosen].plans(
            schema,
            outcome=outcome,
            missing=missing_counts(json.loads(dataset["profile_json"])),
        )
        if chosen == "quality":
            _render_quality_findings(repository, storage_root, dataset)
        if not plans:
            if chosen != "quality":
                st.info("No hay nada que mostrar aquí.")
            return
        try:
            source = load_project_source(repository, storage_root, dataset_id)
            for item in plans:
                st.markdown(f"**{item.title}**")
                render_analysis_result(
                    AnalysisEngine().run(item.plan, {SOURCE: source}), item.plan.output
                )
        except AnalysisPlanError as error:
            render_error(error, fallback_code="ANALYSIS_FAILED")


def _render_quality_findings(repository: Any, storage_root: Path, dataset: Any) -> None:
    """GA-03b: atípicos, fechas imposibles, signos dudosos, categorías casi repetidas y
    valores sin pareja en los demás archivos del proyecto."""

    datasets = repository.list_datasets(str(dataset["project_id"]), admission_status="ACCEPTED")
    try:
        sources = {
            Path(str(row["original_filename_display"])).stem: load_project_source(
                repository, storage_root, str(row["id"])
            )
            for row in datasets[:MAX_QUALITY_FILES]
        }
    except AnalysisPlanError as error:
        render_error(error, fallback_code="ANALYSIS_FAILED")
        return
    own = Path(str(dataset["original_filename_display"])).stem
    findings = quality_findings(sources)
    lines = [
        line
        for finding, line in zip(findings, describe_findings(findings), strict=True)
        if finding.source == own or finding.kind == "KEY_MISMATCH"
    ]
    st.markdown("**Revisión de calidad**")
    if not lines:
        st.success("No encontré valores extremos, fechas imposibles ni categorías casi repetidas.")
    for line in lines:
        st.markdown(f"- {line}")
