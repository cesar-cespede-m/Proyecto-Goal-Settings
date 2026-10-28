"""Pantallas de las extensiones A/B/C/D, sin lógica de dominio duplicada."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from app.ui.action_list_view import render_action_list_measure, render_action_list_offer
from app.ui.components import render_field_card
from app.ui.copy import METRIC_LABELS
from app.ui.coverage_view import coverage_inputs
from app.ui.decision_view import chosen_plan, render_decision
from app.ui.formatting import event_cases_label, format_number, format_pct
from app.ui.messages import ERROR_MESSAGES, fallback_reason_text, render_error
from app.ui.prediction_batch_view import (
    render_prediction_batch_offer,
    render_prediction_batch_tracking,
)
from core.ai_engine.provider import LLMProvider
from core.ai_engine.results_qa import ResultsQA
from core.analysis.snapshot import SnapshotError
from core.analysis.snapshot_recipe import snapshot_recipe
from core.data_engine import DataValidationError
from core.domain.capabilities import CapabilityService
from core.domain.capability_learning import CapabilityLearningError, CapabilityLearningService
from core.domain.models import ReportPayload
from core.domain.recipe_bundle import export_bundle
from core.domain.retraining_service import RetrainingService
from core.ml_engine import RunBlockedError
from core.ml_engine.decision import risk_level_for_rank, selected_row_numbers
from core.ml_engine.drift import ColumnDriftV1, DriftReportV1
from core.ml_engine.health import HealthReason, ModelHealthV1
from core.ml_engine.scoring import ScoringResultV1, ScoringService
from core.reporting.comparison import ExperimentRow, context_changes, list_experiments
from core.reporting.export import render_report_html

_MODEL_LABELS = {
    "DummyClassifier": "Modelo simple de referencia",
    "LogisticRegression": "Alternativa lineal",
    "RandomForestClassifier": "Alternativa de reglas combinadas",
}
_ANSWER_KIND_LABELS = {
    "FACT": "Dato comprobado",
    "INTERPRETATION": "Lectura",
    "HYPOTHESIS": "Hipótesis",
    "RECOMMENDATION": "Recomendación",
    "CANNOT_ANSWER": "No hay evidencia suficiente",
}
_REPORT_STATUS_LABELS = {
    "DRAFT": "Pendiente de revisión",
    "APPROVED": "Aprobado",
    "REJECTED": "Rechazado",
    "PUBLISHED": "Publicado",
}


def _pct(value: float | None) -> str:
    return format_pct(value)


def _model_label(value: str | None) -> str:
    if value is None:
        return "Sin modelo recomendado"
    return _MODEL_LABELS.get(value, "Modelo seleccionado")


def _experiment_label(row: ExperimentRow) -> str:
    date = row.started_at.replace("T", " ")[:16]
    return f"Contexto v{row.context_version_no} · estrategia v{row.strategy_version_no} · {date}"


def _change_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "ninguna"
    if isinstance(value, dict):
        mode = value.get("mode")
        return "sin condición cuantitativa" if mode == "no_quantitative_guardrail" else str(value)
    if isinstance(value, str) and value in METRIC_LABELS:
        return METRIC_LABELS[value]
    return str(value)


def render_experiment_comparison(repository: Any, project_id: str) -> list[ExperimentRow]:
    """Muestra EXT-B dentro de resultados cuando hay dos ejecuciones terminadas."""

    experiments = list_experiments(repository, project_id)
    if len(experiments) < 2:
        return experiments

    st.subheader("Comparar versiones")
    render_field_card("compare_versions", state=f"{len(experiments)} versiones")
    st.write(
        "Compara resultados ya calculados cuando cambiaste una decisión. "
        "Cada fila corresponde a una ejecución terminada."
    )
    rows = [
        {
            "Versión": f"Contexto v{row.context_version_no}",
            "Vigente": "Sí" if row.is_current else "No",
            "Medida principal": METRIC_LABELS.get(row.primary_metric, row.primary_metric),
            "Modelo": _model_label(row.selected_model),
            "Aciertan sus avisos": _pct(row.holdout_precision),
            "Detecta": _pct(row.holdout_recall),
            "Calidad al ordenar": _pct(row.holdout_average_precision),
        }
        for row in experiments
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    render_field_card("compare_selection", state="Tú eliges")
    by_id = {row.run_id: row for row in experiments}
    before_id = st.selectbox(
        "Versión anterior",
        list(by_id),
        index=0,
        format_func=lambda value: _experiment_label(by_id[value]),
        key=f"compare-before-{project_id}",
    )
    after_options = [run_id for run_id in by_id if run_id != before_id]
    after_id = st.selectbox(
        "Versión posterior",
        after_options,
        index=len(after_options) - 1,
        format_func=lambda value: _experiment_label(by_id[value]),
        key=f"compare-after-{project_id}-{before_id}",
    )
    changes = context_changes(repository, before_id, after_id)
    before_model = _model_label(by_id[before_id].selected_model)
    after_model = _model_label(by_id[after_id].selected_model)
    if not changes:
        st.info("No cambiaron las decisiones de negocio registradas entre estas ejecuciones.")
    else:
        for change in changes:
            st.markdown(
                f"- Cambiaste **{change.label.lower()}** de "
                f"**{_change_value(change.before)}** a **{_change_value(change.after)}** "
                f"→ el modelo pasó de **{before_model}** a **{after_model}**."
            )
    return experiments


def _render_answer(answer: Any) -> None:
    kind = _ANSWER_KIND_LABELS.get(answer.kind, "Respuesta")
    st.markdown(f"**{kind}.** {answer.answer}")
    if answer.suggestion:
        st.info(f"Sugerencia: {answer.suggestion}")
    if answer.mode == "LLM":
        st.success("Respuesta de IA verificada contra las cifras disponibles.")
    else:
        reason = fallback_reason_text(answer.fallback_reason)
        st.warning(f"Modo sin IA generativa: {reason}.")
    with st.expander("Ver evidencia y revisión del envío"):
        if answer.evidence_refs:
            for reference in answer.evidence_refs:
                st.code(reference)
        else:
            st.caption("La respuesta no cita cifras porque no encontró evidencia suficiente.")
        if answer.question_sent is not None:
            st.caption("Pregunta revisada que se envió:")
            st.code(answer.question_sent, language=None)


def _results_payload_for_provider(payload: ReportPayload) -> ReportPayload:
    """Quita nombres de variables; EXT-C sólo necesita cifras agregadas."""

    data = payload.model_dump(mode="json")
    data["metrics"] = dict(data["metrics"])
    data["metrics"].pop("permutation_importance", None)
    return ReportPayload.model_validate(data)


def render_results_questions(
    repository: Any,
    project_id: str,
    payload: ReportPayload,
    provider: LLMProvider | None,
) -> None:
    """Muestra EXT-C sin persistir preguntas ni respuestas."""

    if provider is None:
        # V2-05: sin IA no se promete una conversación que no existe.
        return
    st.subheader("Pregúntale a la IA sobre estos resultados")
    render_field_card("results_question", state="No se guarda")
    st.info(
        "Se envían la pregunta revisada y cifras agregadas del reporte. "
        "Nunca se envían filas, valores de celdas ni el archivo. La conversación no se guarda."
    )
    question = st.text_area(
        "¿Qué quieres entender de estos resultados?",
        placeholder="Ej.: ¿Cuántos eventos reales detectó el modelo?",
        key=f"results-question-{project_id}",
    )
    if not st.button("Preguntar", key=f"ask-results-{project_id}"):
        return
    if not question.strip():
        st.error("Escribe una pregunta antes de continuar.")
        return
    experiments = list_experiments(repository, project_id)
    comparison = None
    if len(experiments) >= 2:
        comparison = []
        for row in experiments:
            safe_row = row.model_dump(mode="json")
            for key in ("features_used", "positive_class", "run_id"):
                safe_row.pop(key, None)
            comparison.append(safe_row)
    answer = ResultsQA(provider).ask(
        question,
        _results_payload_for_provider(payload),
        comparison,
    )
    _render_answer(answer)


def render_report_download(repository: Any, report: Any) -> None:
    """Muestra EXT-D para la versión exacta del reporte visible."""

    st.subheader("Descargar reporte")
    render_field_card("report_download", state=_REPORT_STATUS_LABELS.get(report["status"], "Listo"))
    html = render_report_html(repository, report["id"])
    st.download_button(
        "Descargar reporte en HTML",
        data=html,
        file_name="reporte-goal_settings.html",
        mime="text/html",
        key=f"download-report-{report['id']}",
    )
    st.caption("Ábrelo en tu navegador y usa Imprimir para guardarlo como PDF.")


_DRIFT_LEVEL_LABELS = {
    "STABLE": "Sin cambios relevantes",
    "WATCH": "Cambió algo",
    "REVIEW": "Cambió mucho",
    "INSUFFICIENT_DATA": "Muy pocos casos para saberlo",
}
_DRIFT_SUMMARIES = {
    "STABLE": "Los datos de este archivo se parecen a los que se usaron para entrenar el modelo.",
    "WATCH": (
        "Algunas columnas cambiaron algo respecto del entrenamiento. Las prioridades se pueden "
        "usar, pero conviene vigilar si el cambio sigue en los próximos archivos."
    ),
    "REVIEW": (
        "Hay columnas que cambiaron mucho respecto del entrenamiento. Antes de confiar en estas "
        "prioridades, revisa si es un cambio real en la operación o un error de origen, como "
        "otra unidad de medida o un campo mal llenado."
    ),
    "INSUFFICIENT_DATA": (
        "El archivo tiene muy pocos casos para saber si los datos cambiaron; las diferencias que "
        "se ven pueden ser casualidad."
    ),
}


def drift_column_text(column: ColumnDriftV1) -> str:
    """Explica en una frase qué cambió en una columna, sin valores de celdas."""

    parts = []
    if "OUT_OF_TRAINING_RANGE" in column.reasons and column.out_of_range_rate is not None:
        parts.append(
            f"{_pct(column.out_of_range_rate)} de los valores queda fuera del rango visto al "
            "entrenar"
        )
    if "NEW_CATEGORIES" in column.reasons and column.new_category_rate is not None:
        parts.append(f"{_pct(column.new_category_rate)} trae una categoría que nunca se vio")
    if "DISTRIBUTION_SHIFT" in column.reasons:
        if column.reference_mean is not None and column.current_mean is not None:
            parts.append(
                f"el promedio pasó de {format_number(column.reference_mean)} a "
                f"{format_number(column.current_mean)}"
            )
        else:
            parts.append("las proporciones entre categorías son distintas")
    if "MISSING_RATE_CHANGE" in column.reasons:
        parts.append(
            f"los vacíos pasaron de {_pct(column.reference_missing_rate)} a "
            f"{_pct(column.current_missing_rate)}"
        )
    text = "; ".join(parts) or "su distribución cambió"
    return text[0].upper() + text[1:] + "."


def _render_drift(drift: DriftReportV1 | None) -> None:
    st.markdown("#### ¿Los datos cambiaron?")
    if drift is None:
        st.caption(
            "No se pudo comparar con el archivo de entrenamiento porque ya no está disponible o "
            "cambió. El uso del modelo no se bloquea por esto."
        )
        return
    show = {"STABLE": st.success, "WATCH": st.warning, "REVIEW": st.error}.get(drift.level, st.info)
    show(_DRIFT_SUMMARIES[drift.level])
    changed = drift.columns_at("REVIEW", "WATCH")
    if changed and drift.level != "INSUFFICIENT_DATA":
        rows = [
            {
                "Columna": column.column,
                "Qué cambió": drift_column_text(column),
                "Nivel": _DRIFT_LEVEL_LABELS[column.level],
            }
            for column in sorted(changed, key=lambda item: (item.level != "REVIEW", -item.psi))
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "Se compara cada columna que usa el modelo con el archivo con que se entrenó: cómo se "
        "reparten sus valores, si aparecen valores fuera del rango conocido, categorías nuevas "
        "o más vacíos. Que los datos cambien no prueba que el modelo acierte menos; eso se mide "
        "con casos que traen el resultado real."
    )
    with st.expander("Detalle técnico"):
        st.caption(
            "El cambio en la distribución se resume con el índice de estabilidad (PSI). Se marca "
            "para vigilar desde 0,10 y para revisar desde 0,25; también se consideran valores "
            "fuera de rango, categorías nuevas y cambios en la proporción de vacíos."
        )


def _render_scored_result(result: ScoringResultV1, holdout: dict[str, Any]) -> None:
    columns = st.columns(3)
    columns[0].metric("Casos evaluados", result.rows_scored)
    columns[1].metric("Alertas", result.alerts)
    columns[2].metric("Punto de corte", format_number(result.threshold, decimals=3))
    st.caption("El punto de corte es el puntaje mínimo desde el que un caso se marca como alerta.")
    if result.seen_in_training:
        st.warning(
            f"{result.seen_in_training} de estos {result.rows_scored} casos ya estaban en el "
            "archivo con el que se entrenó y evaluó el modelo. Para esos casos el resultado no "
            "indica cómo funcionará con casos realmente nuevos, por eso no se cuentan en "
            "«Cómo le fue al modelo»."
        )
    decision = chosen_plan(result.run_id)
    selected_rows = (
        selected_row_numbers(
            decision[0],
            scores=[case.score for case in result.cases],
            row_numbers=[case.row_number for case in result.cases],
        )
        if decision
        else set()
    )
    table = [
        {
            "Prioridad": case.rank,
            **({"Identificador": case.case_id} if result.id_column else {}),
            "Puntaje": format_number(case.score, decimals=4),
            "Alerta": "Sí" if case.alert else "No",
            **({"Por qué": case.reasons} if case.reasons else {}),
            **(
                {
                    "Nivel": risk_level_for_rank(case.rank, result.rows_scored),
                    "Atender": "Sí" if case.row_number in selected_rows else "No",
                }
                if decision
                else {}
            ),
            **(
                {"Ya usado al entrenar": "Sí" if case.seen_in_training else "No"}
                if result.seen_in_training
                else {}
            ),
        }
        for case in result.cases[:20]
    ]
    st.dataframe(pd.DataFrame(table), hide_index=True)
    if decision:
        st.caption(
            "Las alertas son todos los casos que superan el punto de corte. Los casos por "
            "atender son sólo los que caben en tu capacidad o en la decisión que elegiste; por "
            "eso las cifras pueden ser distintas."
        )
    if any(case.reasons for case in result.cases[:20]):
        st.caption(
            "«Por qué» muestra qué datos de cada caso suben o bajan su puntaje frente a un caso "
            "típico. Es lo que el modelo aprendió a relacionar, no una causa."
        )
    st.download_button(
        "Descargar todos los casos priorizados",
        data=result.to_csv(),
        file_name="casos-priorizados.csv",
        mime="text/csv",
        key=f"download-scoring-{result.run_id}",
    )
    if result.ignored_columns:
        st.caption("Columnas ignoradas: " + ", ".join(result.ignored_columns) + ".")
    _render_drift(result.drift)
    if result.follow_up is None and result.target_present and result.seen_in_training:
        st.info(
            "No se mide «Cómo le fue al modelo»: todos los casos con resultado conocido ya se "
            "usaron al entrenar. Carga casos nuevos con su resultado real para medirlo."
        )
    if result.follow_up is not None:
        follow_up = result.follow_up
        st.markdown("#### Cómo le fue al modelo con estos casos")
        st.write(f"Casos con resultado conocido: {follow_up.rows_with_outcome}.")
        follow_rows = [
            {
                "Medida": "Proporción del evento",
                "Estos casos": _pct(follow_up.prevalence),
                "Prueba final": _pct(
                    holdout.get("positive_support", 0) / holdout["support"]
                    if holdout.get("support")
                    else None
                ),
            },
            {
                "Medida": "Aciertan sus avisos",
                "Estos casos": _pct(follow_up.precision),
                "Prueba final": _pct(holdout.get("precision")),
            },
            {
                "Medida": "Detecta",
                "Estos casos": _pct(follow_up.recall),
                "Prueba final": _pct(holdout.get("recall")),
            },
            {
                "Medida": "Calidad al ordenar",
                "Estos casos": _pct(follow_up.average_precision),
                "Prueba final": _pct(follow_up.training_average_precision),
            },
        ]
        st.dataframe(pd.DataFrame(follow_rows), hide_index=True)


def _render_scoring_history(repository: Any, run_id: str) -> None:
    events = repository.list_scorings(run_id)
    if not events:
        st.caption("Todavía no hay usos registrados para este modelo.")
        return
    rows = []
    for event in events:
        detail = json.loads(event["detail_json"])
        threshold = detail.get("threshold")
        rows.append(
            {
                "Fecha": event["created_at"],
                "Responsable declarado": detail.get("actor_label", "—"),
                "Casos evaluados": detail.get("rows_scored", "—"),
                "Alertas": detail.get("alerts", "—"),
                "Punto de corte": (
                    format_number(float(threshold), decimals=3) if threshold is not None else "—"
                ),
                "Cambio en los datos": _DRIFT_LEVEL_LABELS.get(
                    (detail.get("drift") or {}).get("level"), "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


_HEALTH_HEADLINES = {
    "NO_USES": "Todavía no hay usos registrados; la salud del modelo se evalúa desde el primero.",
    "STABLE": (
        "Sin alertas con la evidencia disponible. Esto no garantiza que el modelo conserve "
        "sus aciertos: revisa cuántos casos y resultados se han podido evaluar."
    ),
    "WATCH": "Vigilar. Hay cambios que todavía no justifican tocar el modelo.",
    "REVIEW": "Revisar. Hay cambios importantes en los datos o en los aciertos del modelo.",
}
_HEALTH_RECOMMENDATIONS = {
    "KEEP_USING": (
        "Mantén la revisión humana de los casos y registra sus resultados para comprobar "
        "cómo se comporta el modelo."
    ),
    "KEEP_WATCHING": "Sigue usándolo y mira si el cambio continúa en los próximos archivos.",
    "CHECK_DATA_SOURCE": (
        "Antes de pensar en reentrenar, confirma con quien entrega los datos si el cambio es real "
        "o un error de origen, como otra unidad de medida o un campo mal llenado. Si es real, "
        "carga casos con el resultado conocido para medir si el modelo acierta menos."
    ),
    "PROPOSE_RETRAINING": (
        "El modelo acierta menos con los datos actuales. Conviene preparar un reentrenamiento con "
        "casos recientes; una persona debe aprobarlo antes de entrenar y antes de publicar."
    ),
}


def _join_columns(columns: list[str]) -> str:
    quoted = [f"«{column}»" for column in columns]
    if len(quoted) <= 1:
        return "".join(quoted) or "algunas columnas"
    return ", ".join(quoted[:-1]) + " y " + quoted[-1]


def health_reason_text(reason: HealthReason) -> str:
    """Explica una razón de la salud del modelo en una frase, con sus cifras."""

    if reason.code == "DATA_CHANGED_A_LOT":
        return (
            f"En el último uso, {_join_columns(reason.columns)} cambió mucho respecto del "
            "entrenamiento."
        )
    if reason.code == "DATA_CHANGING":
        return (
            f"En el último uso, {_join_columns(reason.columns)} cambió algo respecto del "
            "entrenamiento."
        )
    if reason.code in {"PERFORMANCE_DROPPED", "PERFORMANCE_SLIPPING"}:
        # Recorrido 22/10: decía «el modelo sus avisos acertaron…».
        what = (
            "el modelo detectó {observed} de los casos reales"
            if reason.metric == "recall"
            else "los avisos del modelo acertaron {observed} de las veces"
        ).format(observed=_pct(reason.observed))
        size = "bajó" if reason.code == "PERFORMANCE_DROPPED" else "bajó un poco"
        return (
            f"Con {reason.rows_with_outcome} casos de resultado conocido, {what}, frente "
            f"a {_pct(reason.reference)} en la prueba final: {size}."
        )
    if reason.code == "NOT_ENOUGH_OUTCOMES":
        return (
            f"El último archivo con resultado real tenía {reason.rows_with_outcome} casos; hacen "
            "falta al menos 30 para decir si el modelo acierta menos."
        )
    return (
        "Todavía no se cargaron casos con el resultado real, así que no se puede saber si el "
        "modelo acierta menos."
    )


def render_model_health(health: ModelHealthV1) -> None:
    st.markdown("#### Salud del modelo")
    show = {"WATCH": st.warning, "REVIEW": st.error}.get(health.status, st.info)
    show(_HEALTH_HEADLINES[health.status])
    if health.status == "NO_USES":
        return
    for reason in health.reasons:
        st.markdown(f"- {health_reason_text(reason)}")
    if health.recommendation:
        st.markdown(f"**Qué hacer:** {_HEALTH_RECOMMENDATIONS[health.recommendation]}")
    rows = [
        {
            "Fecha": point.created_at,
            "Casos": point.rows_scored,
            "Alertas": point.alerts,
            "Datos": _DRIFT_LEVEL_LABELS.get(point.drift_level, "—"),
            "Detecta": _pct(point.recall) if point.rows_with_outcome else "Sin resultado real",
            "Aciertan sus avisos": (
                _pct(point.precision) if point.rows_with_outcome else "Sin resultado real"
            ),
        }
        for point in reversed(health.points)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"En la prueba final el modelo detectó {_pct(health.reference_recall)} de los casos "
        f"reales y sus avisos acertaron {_pct(health.reference_precision)} de las veces. "
        "Reglas: se habla de caída con al menos 30 casos de resultado conocido; una baja de "
        "8 puntos o más pide vigilar y de 15 o más pide revisar."
    )


def render_retraining_start(
    repository: Any,
    storage_root: Path,
    run: Any,
    health: ModelHealthV1,
    on_started: Callable[[str, str], None] | None,
) -> None:
    """MON-03: proponer un reentrenamiento; nunca entrena por su cuenta."""

    recommended = health.recommendation == "PROPOSE_RETRAINING"
    with st.expander("Reentrenar con casos recientes", expanded=recommended):
        st.markdown(
            "Se crea un proyecto nuevo con el archivo reciente y las mismas decisiones de este "
            "modelo. El modelo actual **sigue en uso** mientras tanto. En el proyecto nuevo "
            "revisas la estrategia, apruebas el entrenamiento, comparas ambos modelos con los "
            "mismos casos y decides si publicar el nuevo."
        )
        service = RetrainingService(repository, storage_root)
        recipe_sources = service.recipe_sources(run["id"])
        strategy_row = repository.get_strategy(str(run["strategy_version_id"]))
        journey_recipe = (
            None if recipe_sources else snapshot_recipe(repository, str(strategy_row["dataset_id"]))
        )
        capability_cycle = CapabilityService(repository).get_cycle_for_project(
            repository.get_run_project_id(run["id"])
        )
        if capability_cycle is not None and journey_recipe is None:
            st.info(
                "La actualización de esta capacidad requiere una receta de recorrido verificable."
            )
            return
        if journey_recipe:
            st.caption(
                "Este modelo se entrenó con una tabla armada desde un recorrido. Sube el archivo "
                "de actividad nuevo (las mismas columnas): Goal Settings repite la misma receta y "
                "arma la tabla nueva sin usar información del futuro."
            )
        if recipe_sources:
            st.caption(
                "Este modelo se entrenó con una tabla preparada desde varios archivos. Sube la "
                "versión nueva de cada uno: Goal Settings repite los mismos pasos de preparación "
                "(la receta) y arma la tabla nueva."
            )
        with st.form(f"retraining-{run['id']}", clear_on_submit=True):
            uploads = {
                name: st.file_uploader(
                    f"Versión nueva de «{name}» (CSV)",
                    type=["csv"],
                    key=f"retraining-file-{run['id']}-{name}",
                )
                for name in recipe_sources or []
            }
            upload = (
                None
                if recipe_sources
                else st.file_uploader(
                    "Archivo reciente con el resultado real (CSV)",
                    type=["csv"],
                    key=f"retraining-file-{run['id']}",
                )
            )
            actor = st.text_input(
                "¿Quién propone el reentrenamiento?",
                key=f"retraining-actor-{run['id']}",
            )
            reason = (
                st.text_input("Motivo de la actualización de esta capacidad")
                if capability_cycle is not None
                else ""
            )
            reference_unintervened = (
                st.checkbox("Confirmo que es una cohorte de referencia sin intervención")
                if capability_cycle is not None
                else False
            )
            reference_basis = (
                st.text_input("Procedencia de la cohorte de referencia")
                if capability_cycle is not None
                else ""
            )
            authorized = st.checkbox(
                "Confirmo que el archivo es sintético, de sandbox o anonimizado y que estoy "
                "autorizado a usarlo, y que las decisiones de este modelo siguen siendo válidas.",
                key=f"retraining-authorized-{run['id']}",
            )
            coverage = coverage_inputs(f"retraining-{run['id']}") if journey_recipe else None
            submitted = st.form_submit_button("Crear proyecto de reentrenamiento")
        if not submitted:
            return
        if recipe_sources:
            if any(item is None for item in uploads.values()):
                st.error("Sube la versión nueva de cada archivo de la receta.")
                return
        elif upload is None:
            st.error("Selecciona el archivo reciente.")
            return
        try:
            if capability_cycle is not None:
                started = CapabilityLearningService(repository, storage_root).start_from_snapshot(
                    capability_cycle.capability_id,
                    source_run_id=run["id"],
                    filename=upload.name,
                    content=upload.getvalue(),
                    coverage=coverage,
                    reason=reason,
                    reference_unintervened=reference_unintervened,
                    reference_basis=reference_basis,
                    actor_label=actor,
                    data_authorized=authorized,
                    operation_id=str(uuid4()),
                )
            elif recipe_sources:
                started = service.start_from_recipe(
                    source_run_id=run["id"],
                    files={
                        name: (item.name, item.getvalue())
                        for name, item in uploads.items()
                        if item is not None
                    },
                    actor_label=actor,
                    data_authorized=authorized,
                )
            elif journey_recipe:
                started = service.start_from_snapshot(
                    source_run_id=run["id"],
                    filename=upload.name,
                    content=upload.getvalue(),
                    coverage=coverage,
                    actor_label=actor,
                    data_authorized=authorized,
                )
            else:
                started = service.start(
                    source_run_id=run["id"],
                    filename=upload.name,
                    content=upload.getvalue(),
                    actor_label=actor,
                    data_authorized=authorized,
                )
        except (
            CapabilityLearningError,
            RunBlockedError,
            DataValidationError,
            ValueError,
            SnapshotError,
        ) as error:
            render_error(error, fallback_code="RETRAINING_FAILED")
            return
        message = (
            "Ciclo candidato creado. El modelo anterior sigue vigente; revisa una estrategia "
            "nueva y aprueba su entrenamiento por separado."
            if capability_cycle is not None
            else "Proyecto de reentrenamiento creado con las mismas decisiones. Revisa la "
            "estrategia y aprueba el entrenamiento cuando estés de acuerdo."
        )
        if on_started is not None:
            on_started(started.project_id, message)
        else:
            st.success(message)


def render_model_comparison(repository: Any, storage_root: Path, run: Any) -> None:
    """MON-03: modelo anterior y nuevo puntúan los mismos casos recientes."""

    project_id = repository.get_run_project_id(run["id"])
    source = RetrainingService(repository, storage_root).source_of(project_id)
    if source is None:
        return
    st.markdown("#### Comparar con el modelo anterior")
    st.markdown(
        "Carga casos recientes **con su resultado real** que ninguno de los dos modelos haya "
        "usado para entrenar. Ambos puntúan exactamente los mismos casos. La comparación no "
        "cuenta como uso de ninguno y no publica nada: esa decisión es tuya."
    )
    with st.form(f"compare-{run['id']}", clear_on_submit=True):
        upload = st.file_uploader(
            "Casos recientes con resultado (CSV)", type=["csv"], key=f"compare-file-{run['id']}"
        )
        actor = st.text_input("¿Quién compara?", key=f"compare-actor-{run['id']}")
        submitted = st.form_submit_button("Comparar modelos")
    if not submitted:
        return
    if upload is None:
        st.error("Selecciona un CSV con casos recientes.")
        return
    service = ScoringService(repository, storage_root)
    try:
        _, _, proposal = service.approved_run(run["id"])
        if proposal.task_type != "binary_classification":
            comparison = service.compare_predictions(
                current_run_id=source["source_run_id"],
                candidate_run_id=run["id"],
                filename=upload.name,
                content=upload.getvalue(),
                actor_label=actor,
            )
            _render_prediction_comparison(comparison)
            return
        current, candidate = service.compare(
            current_run_id=source["source_run_id"],
            candidate_run_id=run["id"],
            filename=upload.name,
            content=upload.getvalue(),
            actor_label=actor,
        )
    except (RunBlockedError, DataValidationError, ValueError) as error:
        render_error(error, fallback_code="COMPARISON_FAILED")
        return
    render_comparison_result(current, candidate)


def _render_prediction_comparison(comparison: Any) -> None:
    """MON-04: misma medida para ambos modelos sobre los mismos casos con resultado real."""

    direction = "más alto es mejor" if comparison.higher_is_better else "más bajo es mejor"
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Medida": f"«{comparison.current.metric}» ({direction})",
                    "Modelo anterior": round(comparison.current.value, 4),
                    "Modelo nuevo": round(comparison.candidate.value, 4),
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    verdict = {
        "CANDIDATE": "Con estos casos, el modelo nuevo rinde mejor.",
        "CURRENT": "Con estos casos, el modelo anterior sigue rindiendo mejor.",
        "TIE": "Con estos casos, ambos modelos rinden igual.",
    }[comparison.better]
    st.info(verdict + " Publicar el nuevo sigue siendo tu decisión.")


def render_comparison_result(current: ScoringResultV1, candidate: ScoringResultV1) -> None:
    before, after = current.follow_up, candidate.follow_up
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Medida": "Detecta",
                    "Modelo anterior": _pct(before.recall),
                    "Modelo nuevo": _pct(after.recall),
                },
                {
                    "Medida": "Aciertan sus avisos",
                    "Modelo anterior": _pct(before.precision),
                    "Modelo nuevo": _pct(after.precision),
                },
                {
                    "Medida": "Calidad al ordenar",
                    "Modelo anterior": _pct(before.average_precision),
                    "Modelo nuevo": _pct(after.average_precision),
                },
                {
                    "Medida": "Alertas",
                    "Modelo anterior": str(current.alerts),
                    "Modelo nuevo": str(candidate.alerts),
                },
            ]
        ),
        hide_index=True,
    )
    better = after.recall >= before.recall and after.precision >= before.precision
    worse = after.recall <= before.recall and after.precision <= before.precision
    if better and not worse:
        verdict = "el modelo nuevo detecta más y sus avisos aciertan más."
    elif worse and not better:
        verdict = "el modelo nuevo no mejora al anterior; no conviene publicarlo."
    else:
        verdict = "cada modelo gana en una medida distinta; decide según qué error cuesta más."
    st.info(
        f"Con {after.rows_with_outcome} casos recientes, {verdict} Si decides usar el nuevo, "
        "apruébalo y publícalo en su reporte; el anterior sigue disponible en su proyecto."
    )


EXPECTED_USE_STATES = {
    "MODEL_NOT_APPROVED_FOR_USE",
    "RUN_HAS_NO_MODEL",
    "MODEL_NOT_CURRENT",
    "MODEL_NOT_ACTIVE",
}


def render_model_scoring(
    repository: Any,
    storage_root: Path,
    run: Any,
    on_retraining_started: Callable[[str, str], None] | None = None,
) -> None:
    """Muestra EXT-A y consulta el gate del dominio antes de aceptar un archivo."""

    st.subheader("Usar el modelo con casos nuevos")
    render_field_card("new_cases")
    service = ScoringService(repository, storage_root)
    try:
        _, _, proposal = service.approved_run(run["id"])
    except RunBlockedError as error:
        if error.code in EXPECTED_USE_STATES:
            # Un paso pendiente del flujo, no una falla: se explica sin código de incidente.
            common = ERROR_MESSAGES[error.code]
            st.info(f"{common.message} {common.action}")
        else:
            render_error(error, fallback_code=error.code)
        return
    st.download_button(
        "Guardar decisiones para otro proyecto",
        data=export_bundle(repository, storage_root, run["id"]),
        file_name="decisiones-goal_settings.json",
        mime="application/json",
        key=f"bundle-export-{run['id']}",
        help=(
            "Descarga un archivo con tus decisiones para reusarlas en otro proyecto. No incluye "
            "filas ni valores de tus datos."
        ),
    )
    if proposal.task_type != "binary_classification":
        _render_prediction(service, run, proposal)
        # MON-04: salud, reentrenamiento y comparación también para estos tipos.
        health = service.health(run["id"])
        render_model_health(health)
        render_retraining_start(repository, storage_root, run, health, on_retraining_started)
        render_model_comparison(repository, storage_root, run)
        return
    render_decision(run, event_cases_label(proposal.positive_class, proposal.target_column))
    # UX-26: decir qué columnas hacen falta y ofrecer una plantilla.
    st.markdown(
        "El archivo necesita estas columnas: "
        + ", ".join(f"«{column}»" for column in proposal.feature_columns)
        + f". Si conoces el resultado real, agrega también «{proposal.target_column}»."
    )
    st.download_button(
        "Descargar plantilla CSV",
        data=(",".join(proposal.feature_columns) + "\n").encode("utf-8"),
        file_name="plantilla-casos-nuevos.csv",
        mime="text/csv",
        key=f"scoring-template-{run['id']}",
    )

    scored_key = f"scored-{run['id']}"
    with st.form(f"score-cases-{run['id']}", clear_on_submit=False):
        render_field_card("new_cases_file", title="Archivo con casos nuevos (CSV)")
        upload = st.file_uploader(
            "CSV con casos nuevos",
            type=["csv"],
            label_visibility="collapsed",
            key=f"new-cases-file-{run['id']}",
        )
        render_field_card("scoring_actor", title="¿Quién usa el modelo?")
        actor_label = st.text_input(
            "¿Quién usa el modelo?",
            key=f"scoring-actor-{run['id']}",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Priorizar casos")
    scored = st.session_state.get(scored_key)
    if (
        upload is not None
        and scored is not None
        and hashlib.sha256(upload.getvalue()).hexdigest() != scored.input_sha256
    ):
        st.session_state.pop(scored_key, None)
    if submitted:
        # Nunca se muestra el resultado de un archivo anterior debajo de un intento nuevo.
        st.session_state.pop(scored_key, None)
        if upload is None:
            st.error("Selecciona un CSV con casos nuevos.")
        elif not actor_label.strip():
            st.error("Indica quién es responsable de este uso del modelo.")
        else:
            try:
                result = service.score(
                    run["id"],
                    filename=upload.name,
                    content=upload.getvalue(),
                    actor_label=actor_label.strip(),
                )
            except (RunBlockedError, DataValidationError, ValueError) as error:
                render_error(error, fallback_code="SCORING_FAILED")
            else:
                st.success("Casos priorizados con el modelo aprobado.")
                # V2-17: se conserva para crear la lista de acción en las recargas siguientes.
                st.session_state[scored_key] = result
    scored = st.session_state.get(scored_key)
    capability_cycle = CapabilityService(repository).get_cycle_for_project(
        repository.get_run_project_id(run["id"])
    )
    if scored is not None:
        evaluation = json.loads(run["candidate_metrics_json"])
        _render_scored_result(scored, evaluation.get("holdout") or {})
        if capability_cycle is None:
            render_action_list_offer(repository, storage_root, run, scored)
        else:
            render_prediction_batch_offer(repository, storage_root, run, scored)
    if capability_cycle is None:
        render_action_list_measure(
            repository, storage_root, run, proposal.target_column, proposal.positive_class
        )
    else:
        render_prediction_batch_tracking(repository, storage_root, run)

    st.markdown("#### Historial de usos")
    _render_scoring_history(repository, run["id"])
    health = service.health(run["id"])
    render_model_health(health)
    render_retraining_start(repository, storage_root, run, health, on_retraining_started)
    render_model_comparison(repository, storage_root, run)


def _render_prediction(service: Any, run: Any, proposal: Any) -> None:
    """Varias clases o cantidades (ADR-016): predicción por fila y seguimiento si hay real."""

    st.markdown(
        "El archivo necesita estas columnas: "
        + ", ".join(f"«{column}»" for column in proposal.feature_columns)
        + f". Si conoces el resultado real, agrega también «{proposal.target_column}» para "
        "ver cómo le va al modelo."
    )
    with st.form(f"predict-cases-{run['id']}", clear_on_submit=False):
        upload = st.file_uploader("CSV con casos nuevos", type=["csv"], key=f"pred-{run['id']}")
        actor_label = st.text_input("¿Quién usa el modelo?", key=f"pred-actor-{run['id']}")
        submitted = st.form_submit_button("Estimar casos")
    if not submitted:
        return
    if upload is None or not actor_label.strip():
        st.error("Selecciona un CSV e indica quién usa el modelo.")
        return
    try:
        result = service.predict(
            run["id"],
            filename=upload.name,
            content=upload.getvalue(),
            actor_label=actor_label.strip(),
        )
    except (RunBlockedError, DataValidationError, ValueError) as error:
        render_error(error, fallback_code="SCORING_FAILED")
        return
    st.success(f"Estimé {result.rows_scored} casos con el modelo aprobado.")
    if result.follow_up is not None:
        current = result.follow_up[result.primary_metric]
        message = (
            f"Con los resultados reales del archivo: {current:.3f} en «{result.primary_metric}» "
            f"frente a {result.training_value:.3f} en la prueba final."
        )
        if result.degraded:
            st.warning(message + " El modelo rinde claramente peor: conviene reentrenarlo.")
        else:
            st.info(message)
    rows = []
    for case in result.cases[:200]:
        row: dict[str, object] = {"Fila": case.row_number}
        if case.case_id is not None:
            row["Identificador"] = case.case_id
        row["Estimación"] = case.prediction
        if case.confidence is not None:
            row["Seguridad"] = f"{case.confidence * 100:.0f} %"
            row["Revisar"] = "Sí" if case.review else "No"
        if case.low is not None and case.high is not None:
            row["Rango probable"] = f"{case.low:.2f} a {case.high:.2f}"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    if any(case.review for case in result.cases):
        st.caption(
            "«Revisar = Sí» marca los casos en que el modelo duda (menos de 50 % de seguridad): "
            "conviene que los mire una persona."
        )
    if any(case.low is not None for case in result.cases):
        st.caption(
            "El rango probable suma y resta el error promedio que tuvo el modelo en la prueba "
            "final; es una guía, no una garantía."
        )
    st.download_button(
        "Descargar estimaciones (CSV)",
        data=result.to_csv(),
        file_name="estimaciones.csv",
        mime="text/csv",
        key=f"pred-download-{run['id']}",
    )
