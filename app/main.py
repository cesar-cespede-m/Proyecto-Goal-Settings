"""Interfaz P0 de punta a punta; SQLite conserva el estado y `core/` decide."""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.ui import (
    WorkflowProgress,
    apply_theme,
    build_navigation,
    empty_state,
    render_app_header,
    render_dataset_profile,
    render_field_card,
    render_project_card,
    render_stage_intro,
    render_workflow_navigation,
    scroll_to_new_step,
)
from app.ui.about_view import render_about
from app.ui.advanced_view import render_advanced_toggle, render_full_metrics
from app.ui.ai_mode import render_ai_mode
from app.ui.assistant_view import ACTOR_MAX_CHARS, looks_like_name, render_context_assistant
from app.ui.batch_upload_view import render_batch_upload
from app.ui.catalog_view import render_project_catalog
from app.ui.copy import FIELD_COPY, METRIC_LABELS, decimal, model_name
from app.ui.cycle_view import render_cycle_summary
from app.ui.decisions_view import render_decisions
from app.ui.events_view import render_incoming_events
from app.ui.explore_view import render_explore
from app.ui.extension_views import (
    render_experiment_comparison,
    render_model_scoring,
    render_report_download,
    render_results_questions,
)
from app.ui.formatting import (
    build_results_summary,
    format_local_datetime,
    format_pct,
    format_pct_range,
    importance_table,
    selection_reason,
)
from app.ui.guide_view import guide_for, navigation_from_guide, render_guide_card
from app.ui.messages import (
    render_database_failure,
    render_error,
    render_other_instance_warning,
)
from app.ui.product_view import render_product_workspace
from app.ui.questions_view import render_questions
from app.ui.timeline_view import render_learning_timeline
from app.ui.warnings import render_warnings
from core import __version__
from core.ai_engine import (
    LLMConfigError,
    LLMProvider,
    LLMSettings,
    NarrativeService,
    build_provider,
)
from core.ai_engine.strategy_explainer import StrategyExplainer
from core.data_engine import (
    DataIngestionService,
    DatasetCompositionError,
    DatasetCompositionService,
    DatasetRejectedError,
    DataValidationError,
    analysis_schema,
)
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.errors import (
    ApprovalRequiredError,
    DomainError,
    InvalidStateTransitionError,
    ReportValidationError,
    StrategyValidationError,
)
from core.domain.gates import has_publication_approval, has_training_approval
from core.domain.models import ReportPayload
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService, RunBlockedError
from core.ml_engine.errors import CandidateTrainingError, MLValidationError
from core.operations import record_instance
from core.reporting import ReportService

_LOGGER = logging.getLogger(__name__)


def _runtime_root() -> Path:
    return Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", "runtime"))


def _env_file() -> Path:
    return Path(os.environ.get("GOAL_SETTINGS_ENV_FILE", ".env"))


_ACTIVE_PROJECT = "active-project"
_PENDING_PROJECT = "active-project-pending"
_PROJECT_FLASH = "project-created-flash"


def _render_project_form(repository: WorkflowRepository, *, key: str = "create-project") -> None:
    with st.form(key, clear_on_submit=True):
        render_field_card("project_name", title="¿Cómo quieres llamar este trabajo?")
        name = st.text_input(
            "Nombre del proyecto",
            placeholder="Ej.: Entregas atrasadas de octubre",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Crear proyecto", type="primary")
    if submitted:
        if not name.strip():
            st.error("Escribe un nombre para poder crear el proyecto.")
        else:
            project_id = repository.create_project(name.strip())
            # El selector ya se dibujó en esta ejecución: se aplica en la siguiente.
            st.session_state[_PENDING_PROJECT] = project_id
            st.session_state[_PROJECT_FLASH] = (
                f"Proyecto «{name.strip()}» creado. Ahora sube su archivo de datos."
            )
            st.rerun()


def _render_archive_form(repository: WorkflowRepository, project_id: str, name: str) -> None:
    """«Eliminar» archiva: el proyecto sale de la lista y su auditoría se conserva."""

    st.markdown(
        f"Se quitará **{name}** de tu lista y de tu avance. Por trazabilidad, sus aprobaciones "
        "y su registro de auditoría se conservan; no se borran."
    )
    with st.form(f"archive-project-{project_id}", clear_on_submit=True):
        actor = st.text_input("¿Quién lo elimina?", placeholder="Nombre o rol")
        confirmed = st.checkbox(f"Confirmo que quiero eliminar «{name}» de la lista")
        submitted = st.form_submit_button("Eliminar proyecto", key="archive-project-submit")
    if not submitted:
        return
    if not confirmed:
        st.error("Marca la confirmación para eliminar el proyecto.")
    elif not actor.strip():
        st.error("Escribe quién elimina el proyecto; queda en el registro de auditoría.")
    else:
        try:
            repository.archive_project(project_id, actor_label=actor)
        except (LookupError, ValueError) as error:
            render_error(error, fallback_code="PROJECT_ARCHIVE_FAILED")
            return
        st.session_state.pop(_ACTIVE_PROJECT, None)
        st.session_state[_PROJECT_FLASH] = f"Proyecto «{name}» eliminado de la lista."
        st.rerun()


def _flash_and_rerun(message: str) -> None:
    st.session_state[_UPLOAD_FLASH] = message
    st.rerun()


def _open_retraining_project(project_id: str, message: str) -> None:
    """Abre el proyecto de reentrenamiento recién creado en la siguiente ejecución."""

    st.session_state[_PENDING_PROJECT] = project_id
    st.session_state["pending-workspace-section"] = "Datos y modelos"
    st.session_state[_PROJECT_FLASH] = message
    st.rerun()


_UPLOAD_FLASH = "upload-accepted-flash"
_COMPOSITION_FLASH = "composition-created-flash"
_COMPOSITION_PREVIEW = "dataset-composition-preview"
_TRAIN_FLASH = "training-finished-flash"


def _declared_actor(context: Any) -> str:
    """Responsable declarado en el contexto (puede ser texto antiguo mal ubicado)."""

    try:
        return str(json.loads(context["context_json"])["context"]["decision_actor_label"] or "")
    except (IndexError, KeyError, TypeError, ValueError):
        return ""


def _reviewer_hint(declared: str) -> str:
    """UX-22: el nombre declarado sólo se sugiere; quien aprueba lo escribe explícitamente."""

    return f"Ej.: {declared}" if looks_like_name(declared) else "Ej.: Ana, jefa de logística"


def _actor_for_run(repository: WorkflowRepository, run: Any) -> str:
    try:
        strategy = repository.get_strategy(run["strategy_version_id"])
        return _declared_actor(repository.get_context(strategy["context_version_id"]))
    except (LookupError, TypeError, ValueError):
        return ""


def _render_upload(
    repository: WorkflowRepository,
    storage_root: Path,
    project_id: str,
    *,
    heading: str = "Sube el archivo de datos",
    form_key: str = "dataset-upload",
) -> None:
    st.subheader(heading)
    with st.form(form_key, clear_on_submit=True):
        render_field_card("csv_file", title="Archivo de datos (CSV)")
        upload = st.file_uploader(
            "Archivo de datos (CSV)", type=["csv"], label_visibility="collapsed"
        )
        render_field_card("source", title="¿De dónde salió el archivo?")
        source_name = st.text_input(
            "¿De dónde salió el archivo?",
            label_visibility="collapsed",
            placeholder="Ej.: datos inventados por el equipo para esta prueba",
        )
        render_field_card("license", title="Condiciones de uso (opcional)")
        license_name = st.text_input(
            "Condiciones de uso (opcional)",
            placeholder="Ej.: CC BY 4.0, o «pendiente»",
            label_visibility="collapsed",
        )
        classification_labels = {
            "synthetic": "Sintético",
            "sandbox": "Prueba controlada",
            "anonymized": "Anonimizado",
            "pending_review": "Pendiente de revisión",
        }
        render_field_card("classification", title="¿Qué tipo de datos son?")
        classification = st.selectbox(
            "¿Qué tipo de datos son?",
            list(classification_labels),
            label_visibility="collapsed",
            index=None,
            format_func=lambda value: classification_labels[value],
            placeholder="Elige una opción",
        )
        render_field_card("authorization")
        authorization = st.checkbox(
            "Confirmo que tengo permiso para usar estos datos en esta prueba"
        )
        render_field_card("anonymization")
        anonymization = st.checkbox(
            "Confirmo que el archivo no tiene datos que identifiquen a personas reales"
        )
        submitted = st.form_submit_button("Revisar y subir el archivo", type="primary")
    if not submitted:
        return
    missing = [
        label
        for label, ok in (
            ("elige el archivo", upload is not None),
            ("cuenta de dónde salió", bool(source_name.strip())),
            ("elige qué tipo de datos son", classification is not None),
        )
        if not ok
    ]
    if missing:
        st.error("Para continuar, " + ", ".join(missing) + ".")
        return
    try:
        result = DataIngestionService(repository, storage_root).ingest_stream(
            project_id=project_id,
            request={
                "original_filename_display": upload.name,
                "source_name": source_name,
                "license": license_name or None,
                "classification": classification,
                "authorization_status": "approved" if authorization else "pending",
                "anonymization_declared": anonymization,
            },
            stream=upload,
        )
    except sqlite3.IntegrityError as error:
        render_error(error, fallback_code="DUPLICATE_DATASET")
        return
    except (DataValidationError, DatasetRejectedError, ValidationError, ValueError) as error:
        render_error(error, fallback_code="INVALID_DATASET")
        return
    if result.metadata.admission_status == "ACCEPTED":
        # El estado del proyecto se leyó antes de la carga: se vuelve a dibujar la página para
        # mostrar el archivo y habilitar el paso siguiente.
        st.session_state[_UPLOAD_FLASH] = (
            "Archivo aceptado. Abajo ves un resumen y el paso siguiente."
        )
        st.rerun()
    else:
        if not authorization:
            reason = "no marcaste que tienes permiso para usar estos datos"
        elif classification == "pending_review":
            reason = "el tipo de datos quedó como «Pendiente de revisión»"
        else:
            reason = "son datos anonimizados y no confirmaste que no identifican a personas"
        st.warning(
            f"El archivo quedó apartado y no se aceptó porque {reason}. Revisa esa respuesta "
            "y vuelve a subirlo."
        )


def _dataset_columns(dataset: Any) -> list[str]:
    profile = json.loads(dataset["profile_json"])
    return [str(item["name"]) for item in profile["columns"]]


def _render_dataset_composition(
    repository: WorkflowRepository,
    storage_root: Path,
    project_id: str,
    active_dataset: Any,
) -> None:
    sources = repository.list_datasets(
        project_id,
        admission_status="ACCEPTED",
        dataset_kind="SOURCE",
    )
    if len(sources) < 2:
        return

    labels = {
        str(row["id"]): f"{row['original_filename_display']} · {str(row['sha256'])[:8]}"
        for row in sources
    }
    source_ids = list(labels)
    if active_dataset["dataset_kind"] == "SOURCE" and active_dataset["id"] in source_ids:
        source_ids.remove(active_dataset["id"])
        source_ids.insert(0, active_dataset["id"])

    st.markdown("### Cruzar archivos")
    st.caption(
        "La fuente principal conserva una fila por caso y debe contener el resultado a anticipar. "
        "Las auxiliares sólo agregan columnas."
    )
    primary_id = st.selectbox(
        "Fuente principal",
        source_ids,
        format_func=lambda value: labels[value],
        key="composition-primary",
    )
    auxiliary_ids = st.multiselect(
        "Fuentes auxiliares (máximo 4)",
        [value for value in source_ids if value != primary_id],
        max_selections=4,
        format_func=lambda value: labels[value],
        key="composition-auxiliaries",
    )
    if not auxiliary_ids:
        return

    rows = {str(row["id"]): row for row in sources}
    primary_columns = _dataset_columns(rows[primary_id])
    joins: list[dict[str, str]] = []
    for index, auxiliary_id in enumerate(auxiliary_ids, start=1):
        auxiliary_columns = _dataset_columns(rows[auxiliary_id])
        st.markdown(f"**Auxiliar {index}: {rows[auxiliary_id]['original_filename_display']}**")
        left_key = st.selectbox(
            "Columna de relación en la principal",
            primary_columns,
            key=f"composition-left-key-{auxiliary_id}",
        )
        right_key = st.selectbox(
            "Columna equivalente en esta auxiliar",
            auxiliary_columns,
            key=f"composition-right-key-{auxiliary_id}",
        )
        cardinality = st.selectbox(
            "Relación esperada",
            ["many_to_one", "one_to_one"],
            format_func=lambda value: (
                "Varias filas principales pueden compartir una auxiliar"
                if value == "many_to_one"
                else "Cada clave aparece una sola vez en ambos archivos"
            ),
            key=f"composition-cardinality-{auxiliary_id}",
        )
        joins.append(
            {
                "dataset_id": auxiliary_id,
                "alias": f"aux{index}",
                "primary_key": left_key,
                "auxiliary_key": right_key,
                "cardinality": cardinality,
            }
        )
    actor = st.text_input(
        "Responsable de confirmar el cruce",
        placeholder="Nombre o rol responsable",
        max_chars=100,
        key="composition-actor",
    )
    request = {
        "primary_dataset_id": primary_id,
        "joins": joins,
        "actor_label": actor,
    }
    if st.button("Previsualizar cruce", key="preview-composition"):
        if not actor.strip():
            st.error(
                "Escribe el nombre o rol de quien confirma el cruce; queda en el registro de "
                "auditoría."
            )
            return
        try:
            preview = DatasetCompositionService(repository, storage_root).preview(
                project_id=project_id,
                request=request,
            )
        except (DatasetCompositionError, ValidationError, ValueError) as error:
            render_error(error, fallback_code="INVALID_DATASET_COMPOSITION")
        else:
            st.session_state[_COMPOSITION_PREVIEW] = {
                "project_id": project_id,
                "request": request,
                "summary": preview.summary.model_dump(mode="json"),
            }

    preview_state = st.session_state.get(_COMPOSITION_PREVIEW)
    if not preview_state or preview_state.get("project_id") != project_id:
        return
    if preview_state.get("request") != request:
        st.info("La configuración cambió. Vuelve a previsualizar antes de confirmar.")
        return
    summary = preview_state["summary"]
    st.success(
        f"Previsualización lista: {summary['source_count']} fuentes y "
        f"{summary['joins'][-1]['output_rows']} filas resultantes."
    )
    st.dataframe(
        [
            {
                "Auxiliar": item["alias"],
                "Coinciden": item["matched_primary_rows"],
                "Sin coincidencia": item["unmatched_primary_rows"],
                "Claves auxiliares sin uso": item["unmatched_auxiliary_rows"],
            }
            for item in summary["joins"]
        ],
        hide_index=True,
        width="stretch",
    )
    confirmed = st.checkbox(
        "Revisé los conteos y confirmo este cruce",
        key="confirm-composition",
    )
    if st.button(
        "Crear y activar dataset cruzado",
        type="primary",
        disabled=not confirmed,
        key="create-composition",
    ):
        try:
            stored = DatasetCompositionService(repository, storage_root).create(
                project_id=project_id,
                request=request,
            )
        except (
            DatasetCompositionError,
            ValidationError,
            ValueError,
            sqlite3.IntegrityError,
        ) as error:
            render_error(error, fallback_code="INVALID_DATASET_COMPOSITION")
        else:
            st.session_state.pop(_COMPOSITION_PREVIEW, None)
            st.session_state[_COMPOSITION_FLASH] = (
                f"Cruce creado con {stored.summary.source_count} fuentes. "
                "Ahora revisa el contexto del nuevo snapshot."
            )
            st.rerun()


def _render_time_split_option(
    repository: WorkflowRepository, project_id: str, strategy: Any, proposal: dict[str, Any]
) -> None:
    """GA-07e: dividir por fechas crea una versión nueva del plan, que se vuelve a aprobar."""

    dataset = repository.get_dataset(str(strategy["dataset_id"]))
    dates = [
        name
        for name, kind in analysis_schema(json.loads(dataset["profile_json"])).items()
        if kind == "datetime"
    ]
    current = proposal["split"].get("time_column")
    if not dates and current is None:
        return
    with st.expander("Validar con fechas (opcional)", expanded=current is not None):
        st.caption(
            "Si tus casos tienen fecha, conviene aprender con los más antiguos y probar con los "
            "más recientes: así se mide cómo le iría al modelo con casos futuros. Cambiarlo crea "
            "una versión nueva del plan que tienes que volver a aprobar."
        )
        options = [None, *dates]
        chosen = st.selectbox(
            "Dividir por",
            options,
            index=options.index(current) if current in options else 0,
            format_func=lambda value: "Al azar (sin fechas)" if value is None else f"«{value}»",
            key=f"time-split-{strategy['id']}",
        )
        current_availability = proposal["split"].get("label_available_at_column")
        availability = None
        if chosen is not None and proposal.get("task_type") == "binary_classification":
            availability = st.selectbox(
                "Fecha en que se conoció el resultado",
                options,
                index=options.index(current_availability) if current_availability in options else 0,
                format_func=lambda value: "Sin declarar" if value is None else f"«{value}»",
                key=f"label-availability-{strategy['id']}",
                help=(
                    "Debe ser una columna excluida de las variables del modelo. Se omiten del "
                    "aprendizaje los casos cuyo resultado aún no se conocía al comenzar "
                    "el período siguiente."
                ),
            )
            if availability is None:
                st.caption(
                    "Se ordenarán los casos por fecha, pero aún no se puede comprobar cuándo "
                    "estaban disponibles sus resultados."
                )
        if chosen == current and availability == current_availability:
            return
        if not st.button("Usar esta división", key=f"time-split-apply-{strategy['id']}"):
            return
        updated = dict(proposal)
        updated["split"] = {
            **proposal["split"],
            "method": "stratified_random" if chosen is None else "temporal",
            "time_column": chosen,
        }
        if availability is None:
            updated["split"].pop("label_available_at_column", None)
        else:
            updated["split"]["label_available_at_column"] = availability
        try:
            StrategyService(repository).create_version(
                project_id=project_id, proposal=StrategyProposalV1.model_validate(updated)
            )
        except (StrategyValidationError, ValidationError, ValueError) as error:
            render_error(error, fallback_code="INVALID_STRATEGY")
            return
        st.rerun()


def _render_strategy_review(
    repository: WorkflowRepository,
    project_id: str,
    context: Any,
) -> Any | None:
    st.subheader("Cómo se entrenará (primera aprobación)")
    service = StrategyService(repository)
    strategy = repository.get_latest_strategy(context["id"])
    if strategy is None:
        render_field_card("strategy_origin")
        if st.button(
            "Preparar el plan de entrenamiento", key=f"strategy-{context['id']}", type="primary"
        ):
            try:
                created = service.create_template_version(
                    project_id=project_id,
                    context_version_id=context["id"],
                )
            except (StrategyValidationError, ValidationError, ValueError) as error:
                render_error(error, fallback_code="INVALID_STRATEGY")
                return None
            strategy = repository.get_strategy(created.strategy_version_id)
            st.success("Plan de entrenamiento listo. Revísalo y decide si lo apruebas.")
    if strategy is None:
        return None

    proposal = json.loads(strategy["strategy_json"])
    st.caption(f"Plan de entrenamiento: versión {strategy['version_no']}")

    explanation = repository.get_latest_strategy_explanation(strategy["id"])
    if explanation is None:
        provider, _ = _llm_provider()
        waiting = (
            "Redactando el plan en palabras simples… puede tardar unos segundos."
            if provider is not None
            else "Preparando la explicación del plan…"
        )
        with st.spinner(waiting):
            explained = StrategyExplainer(provider).explain(
                StrategyProposalV1.model_validate(proposal)
            )
        repository.add_strategy_explanation(
            strategy_version_id=strategy["id"],
            source=explained.source,
            explanation=explained.explanation.model_dump(mode="json"),
            ai_trace=explained.trace,
        )
        explanation = repository.get_latest_strategy_explanation(strategy["id"])
    source_label = "Redactada por IA" if explanation["source"] == "LLM" else "Plantilla sin IA"
    content = json.loads(explanation["explanation_json"])
    with st.container(border=True):
        st.markdown("**Qué se hará, en palabras simples**")
        render_field_card("strategy_explanation", state=source_label)
        st.write(content["summary"])
        for section in content["sections"]:
            st.markdown(f"**{section['title']}.** {section['text']}")

    render_field_card("strategy_warnings")
    review_points = render_warnings(
        proposal["warnings"],
        empty_text="No hay puntos para revisar en esta versión.",
        excluded=set(proposal.get("excluded_columns", [])),
    )

    split = proposal["split"]
    threshold_rule = (
        "un punto de corte fijo definido por la estrategia"
        if proposal["threshold_policy"]["method"] == "fixed"
        else "con los datos de comparación, nunca con la prueba final"
    )
    with st.expander("Ver las reglas fijas del entrenamiento"):
        st.markdown(
            "\n".join(
                [
                    f"- **Datos:** el sistema mirará {len(proposal['feature_columns'])} "
                    f"columnas y dejará fuera {len(proposal['excluded_columns'])}.",
                    f"- **División del archivo:** {split['train']:.0%} para aprender, "
                    f"{split['validation']:.0%} para comparar y {split['test']:.0%} para la "
                    "prueba final"
                    + (
                        f", en orden de «{split['time_column']}»: lo más antiguo aprende y lo "
                        "más reciente es la prueba final."
                        if split.get("time_column")
                        else ", al azar."
                    ),
                    "- **Repetible:** la división se hace siempre igual.",
                    "- **Preparación:** los vacíos se completan y los textos se convierten a "
                    "números sin tocar tu archivo.",
                    "- **Modelos:** una referencia simple y dos alternativas; sólo se recomienda "
                    "una si le gana a la referencia.",
                    f"- **Punto desde el que se avisa:** {threshold_rule}.",
                    f"- **Tiempo máximo:** {proposal['caps']['max_training_seconds']} segundos "
                    "por entrenamiento, uno a la vez.",
                ]
            )
        )
        st.caption("Detalle técnico")
        st.json(proposal, expanded=False)
        st.code(f"fingerprint: {strategy['fingerprint']}")
    _render_time_split_option(repository, project_id, strategy, proposal)

    current_approval = repository.get_latest_training_decision(strategy["id"])
    if current_approval is None:
        st.info("Todavía no decidiste sobre este plan.")
    else:
        decision_label = "Aprobada" if current_approval["decision"] == "APPROVED" else "Rechazada"
        st.write(
            f"Última decisión: **{decision_label}** · "
            f"responsable declarado: {current_approval['actor_label']}"
        )

    with st.form(f"training-decision-{strategy['id']}"):
        render_field_card("training_decision", title="¿Apruebas este plan de entrenamiento?")
        decision = st.radio(
            "Decisión sobre el entrenamiento",
            ["APPROVED", "REJECTED"],
            label_visibility="collapsed",
            index=None,
            horizontal=True,
            format_func=lambda value: "Aprobar" if value == "APPROVED" else "Rechazar",
        )
        render_field_card("strategy_reviewer", title="¿Quién lo revisó?")
        actor_label = st.text_input(
            "Responsable de revisar la estrategia",
            label_visibility="collapsed",
            placeholder=_reviewer_hint(_declared_actor(context)),
            max_chars=ACTOR_MAX_CHARS,
        )
        render_field_card("strategy_comment", title="Comentario")
        comment = st.text_area("Comentario sobre la estrategia", label_visibility="collapsed")
        read_points = (
            st.checkbox("Leí los puntos para revisar de este plan") if review_points else True
        )
        decision_submitted = st.form_submit_button("Registrar decisión")
    if decision_submitted:
        if decision is None:
            st.error("Elige si apruebas o rechazas el plan.")
        elif decision == "APPROVED" and not read_points:
            st.error("Antes de aprobar, lee y marca los puntos para revisar de este plan.")
        elif not actor_label.strip():
            st.error("Indica quién revisó el plan.")
        elif decision == "REJECTED" and not comment.strip():
            st.error("Explica el motivo del rechazo antes de registrarlo.")
        else:
            try:
                repository.decide_training(
                    strategy_version_id=strategy["id"],
                    decision=decision,
                    actor_label=actor_label,
                    comment=comment or None,
                )
            except (InvalidStateTransitionError, ValueError, sqlite3.IntegrityError) as error:
                render_error(error, fallback_code="INVALID_STATE_TRANSITION")
            else:
                st.toast("Decisión registrada.")
                st.rerun()
    if has_training_approval(repository.connection, strategy["id"]):
        st.success("Plan aprobado: ya puedes entrenar.")
    else:
        st.info("Nada se entrena hasta que apruebes este plan.")
    return strategy


_REPORTABLE_RUNS = {"COMPLETED", "COMPLETED_WITHOUT_RECOMMENDATION"}
_RUN_STATUS_LABELS = {
    "PENDING": "Pendiente",
    "RUNNING": "En curso",
    "COMPLETED": "Completada",
    "COMPLETED_WITHOUT_RECOMMENDATION": "Completada sin recomendación",
    "FAILED": "Fallida",
}


def _render_training(
    repository: WorkflowRepository,
    runtime_root: Path,
    strategy: Any,
) -> Any | None:
    flash = st.session_state.pop(_TRAIN_FLASH, None)
    if flash:
        st.success(flash)
    render_field_card("train_action")
    # V2-12: tras un entrenamiento, volver a entrenar es secundario y se explica.
    previous = repository.get_latest_run(strategy["id"])
    if not has_training_approval(repository.connection, strategy["id"]):
        st.caption("Se habilita cuando apruebes el plan de entrenamiento de arriba.")
    elif (
        previous is not None
        and st.button(
            "Entrenar una versión nueva",
            key=f"train-{strategy['id']}",
            type="secondary",
            help="Ya entrenaste este plan. Hazlo de nuevo sólo si cambiaste los datos o el plan.",
        )
        or previous is None
        and st.button(
            "Entrenar y comparar modelos",
            key=f"train-{strategy['id']}",
            type="primary",
        )
    ):
        with st.spinner("Entrenando y comparando los modelos… puede tardar unos minutos."):
            try:
                result = MLRunService(repository, runtime_root).execute(strategy["id"])
            except sqlite3.IntegrityError as error:
                render_error(error, fallback_code="TRAINING_ALREADY_RUNNING")
            except (
                DomainError,
                CandidateTrainingError,
                DataValidationError,
                MLValidationError,
                RunBlockedError,
            ) as error:
                render_error(error, fallback_code="TRAINING_FAILED")
            else:
                status_label = _RUN_STATUS_LABELS.get(result.status, result.status)
                recommendation = (
                    "Hay un modelo recomendado."
                    if result.selected_model
                    else "No hay modelo recomendado."
                )
                # Se vuelve a dibujar para que la barra lateral y el paso siguiente se
                # actualicen con el entrenamiento recién terminado.
                st.session_state[_TRAIN_FLASH] = (
                    f"Entrenamiento terminado: {status_label}. {recommendation}"
                )
                st.rerun()
    run = repository.get_latest_run(strategy["id"])
    if run is not None:
        _render_run_results(run, strategy)
    return run


def _render_task_results(run: Any, evaluation: dict[str, Any]) -> None:
    """Resultados de varias clases o cantidades (ADR-016), con las cifras guardadas."""

    metric = evaluation["primary_metric"]
    label = METRIC_LABELS.get(metric, metric)
    selected = evaluation.get("selected_model")
    if selected is None:
        st.info(
            "Ninguna alternativa superó a la respuesta de referencia simple en "
            f"«{label}». Con estos datos y columnas no se recomienda un modelo."
        )
    else:
        direction = "más alto es mejor" if evaluation["higher_is_better"] else "más bajo es mejor"
        st.success(
            f"Se recomienda la {model_name(selected)}. En la prueba final obtuvo "
            f"{decimal(evaluation['test'][metric])} en «{label}» ({direction})."
        )
    st.dataframe(
        [
            {
                "Alternativa": name,
                **{METRIC_LABELS.get(key, key): round(value, 4) for key, value in values.items()},
            }
            for name, values in evaluation["validation"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Tabla de validación: sirve para elegir. La prueba final se mide una sola vez y sólo "
        "con la alternativa elegida."
    )
    with st.expander("Ver detalle técnico"):
        st.code(f"run_id: {run['id']}\nstatus: {run['status']}")


def _render_run_results(run: Any, strategy: Any | None = None) -> None:
    status_label = _RUN_STATUS_LABELS.get(run["status"], "Estado técnico no reconocido")
    render_field_card("run_status", state=status_label)
    st.caption(f"Último entrenamiento: empezó el {format_local_datetime(run['started_at'])}.")
    if run["status"] not in _REPORTABLE_RUNS:
        with st.expander("Ver detalle técnico"):
            st.code(f"run_id: {run['id']}\nstatus: {run['status']}")
        if run["error_message"]:
            render_error(
                RuntimeError(run["error_message"]),
                fallback_code="TRAINING_FAILED",
            )
        return

    evaluation = json.loads(run["candidate_metrics_json"])
    render_full_metrics(evaluation)
    if evaluation.get("schema_version") == "task-evaluation.v1":
        _render_task_results(run, evaluation)
        return
    selection = evaluation.get("selection", {})
    holdout = evaluation.get("holdout")
    validation = evaluation.get("validation", {})
    baseline_metrics = validation.get("DummyClassifier", {}).get("metrics", {})
    strategy_payload: dict[str, Any] = {}
    if strategy is not None:
        try:
            strategy_payload = json.loads(strategy["strategy_json"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            strategy_payload = {}
    st.info(
        build_results_summary(
            has_recommendation=holdout is not None,
            recall=None if holdout is None else holdout.get("recall"),
            precision=None if holdout is None else holdout.get("precision"),
            baseline_recall=baseline_metrics.get("recall"),
            positive_class=strategy_payload.get("positive_class"),
            target_column=strategy_payload.get("target_column"),
        )
    )
    with st.expander("Ver detalle técnico"):
        st.code(f"run_id: {run['id']}\nstatus: {run['status']}")
    primary = selection.get("primary_metric", "")
    model_labels = {
        "DummyClassifier": "Modelo simple de referencia",
        "LogisticRegression": "Alternativa lineal",
        "RandomForestClassifier": "Alternativa de reglas combinadas",
    }
    rows = [
        {
            "Modelo": model_labels.get(name, "Alternativa"),
            "Detecta": format_pct(candidate["metrics"]["recall"]),
            "Aciertan sus avisos": format_pct(candidate["metrics"]["precision"]),
            "Calidad al ordenar": format_pct(candidate["metrics"]["average_precision"]),
            "Le gana a la referencia": ("Sí" if candidate["beats_baseline"] is True else "No"),
            "Recomendado": "Sí" if name == selection.get("selected_model") else "",
        }
        for name, candidate in validation.items()
    ]

    render_field_card("primary_result", state="Calculado")
    render_field_card("translated_metrics", state="Calculado")
    st.markdown("**Datos para comparar**: aquí se eligen el modelo y su punto de corte.")
    st.caption(f"Lo que elegiste como más importante: «{METRIC_LABELS.get(primary, primary)}».")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    render_field_card("baseline_comparison", state="Calculado")
    if holdout is None:
        st.info(
            "La prueba final no se realizó porque no había una alternativa que mejorara "
            "la referencia simple."
        )
        return

    selected = selection.get("selected_model")
    selected_label = model_labels.get(selected, "la alternativa elegida")
    st.success(
        selection_reason(
            selected_label.lower() if selected in model_labels else selected_label,
            METRIC_LABELS.get(primary, primary),
            validation.get(selected, {}).get("metrics", {}).get(primary),
            [
                (model_labels.get(name, "otra alternativa").lower(), item["metrics"].get(primary))
                for name, item in validation.items()
                if name not in {selected, "DummyClassifier"}
            ],
        )
    )
    st.markdown(f"**Prueba final reservada** con **{selected_label}**, evaluada una sola vez.")
    intervals = {
        item["metric"]: item
        for item in (evaluation.get("holdout_bootstrap") or {}).get("intervals", [])
    }
    metrics = list(dict.fromkeys([primary, "recall", "precision"]))
    render_field_card("uncertainty_range", state="Calculado")
    for column, metric in zip(st.columns(len(metrics)), metrics, strict=True):
        column.metric(METRIC_LABELS.get(metric, metric), format_pct(holdout[metric]))
        interval = intervals.get(metric)
        if interval:
            column.caption(
                "Rango de incertidumbre del 95 %: "
                f"{format_pct_range(interval['lower'], interval['upper'])}"
            )

    render_field_card("confusion_matrix", state="Calculado")
    confusion_matrix = holdout["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Resultado": "Avisó y pasó",
                    "Casos": confusion_matrix[1][1],
                },
                {
                    "Resultado": "Avisó y no pasó (falsa alarma)",
                    "Casos": confusion_matrix[0][1],
                },
                {
                    "Resultado": "No avisó y pasó (se escapó)",
                    "Casos": confusion_matrix[1][0],
                },
                {
                    "Resultado": "No avisó y no pasó",
                    "Casos": confusion_matrix[0][0],
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    importance = evaluation.get("permutation_importance", [])
    render_field_card(
        "feature_importance",
        state="Calculado" if importance else "No disponible",
    )
    if importance:
        st.markdown("**Qué datos pesan más en la predicción** (asociación, no causa)")
        st.dataframe(
            importance_table(importance),
            hide_index=True,
            width="stretch",
            column_config={
                "Peso frente al que más pesa": st.column_config.ProgressColumn(
                    "Peso frente al que más pesa", min_value=0, max_value=100, format="%d %%"
                )
            },
        )
        st.caption(
            "Cada barra compara el dato con el que más pesa: muestra cuánto empeoran las "
            "predicciones si ese dato se desordena. Un dato con 0 % casi no cambia nada."
        )
    render_warnings(evaluation.get("warnings", []))


def _llm_provider() -> tuple[LLMProvider | None, bool]:
    """Proveedor configurado o None; el segundo valor indica un error de configuración."""

    try:
        return build_provider(LLMSettings.from_env(env_file=_env_file())), False
    except LLMConfigError:
        return None, True


def _narrative_service() -> tuple[NarrativeService | None, str]:
    """Construye el proveedor configurado; un error de configuración degrada a plantilla."""

    provider, config_error = _llm_provider()
    if config_error:
        return None, "La redacción con IA no está disponible; se usa una plantilla."
    if provider is None:
        return None, "Modo sin IA generativa: reporte por plantilla."
    return NarrativeService(provider), "Redacción con IA generativa activa."


def _render_generation_mode(payload: dict[str, Any]) -> None:
    trace = payload.get("ai_trace") or {}
    mode = trace.get("mode")
    if mode == "LLM":
        st.success("Redactado por IA generativa y verificado contra las cifras calculadas.")
    elif mode == "TEMPLATE_FALLBACK":
        st.warning("La redacción de IA no pasó los controles; se muestra una plantilla.")
    else:
        st.info("Plantilla sin IA generativa, construida con cifras calculadas.")
    if trace:
        with st.expander("Detalle técnico"):
            st.json(trace, expanded=False)


def _create_report(repository: WorkflowRepository, run_id: str) -> None:
    narrative, _ = _narrative_service()
    waiting = (
        "La IA está redactando el reporte y Goal Settings verifica cada cifra… "
        "puede tardar hasta un minuto."
        if narrative is not None
        else "Armando el reporte con las cifras calculadas…"
    )
    try:
        with st.spinner(waiting):
            ReportService(repository, narrative).create_report(run_id)
    except ReportValidationError as error:
        render_error(error, fallback_code="INVALID_REPORT")
        return
    st.rerun()


def _render_report(repository: WorkflowRepository, project_id: str, run: Any) -> None:
    st.subheader("Reporte (segunda aprobación)")
    if run["status"] not in _REPORTABLE_RUNS:
        st.caption("El reporte estará disponible cuando termine un entrenamiento.")
        return
    narrative, mode_label = _narrative_service()
    st.caption(mode_label)
    report = repository.get_latest_report(run["id"])
    if report is None:
        render_field_card("generate_report")
        if st.button("Generar reporte verificable", key=f"report-{run['id']}", type="primary"):
            _create_report(repository, run["id"])
        return

    payload = json.loads(report["payload_json"])
    status_labels = {
        "DRAFT": "Pendiente de revisión",
        "APPROVED": "Aprobado",
        "REJECTED": "Rechazado",
        "PUBLISHED": "Publicado",
    }
    status_label = status_labels.get(report["status"], "Estado técnico no reconocido")
    st.caption(f"Reporte v{report['version_no']} · {status_label}")
    with st.expander("Ver detalle técnico"):
        st.code(
            f"report_sha256: {report['report_sha256']}\n"
            f"generator: {payload.get('generator', '?')}\n"
            f"status: {report['status']}"
        )

    render_field_card("generation_mode", state="Calculado")
    _render_generation_mode(payload)

    render_field_card("insight_kind", state="Calculado")
    kind_labels = {
        "fact": "Dato comprobado",
        "interpretation": "Lectura",
        "hypothesis": "Hipótesis",
        "recommendation": "Recomendación",
    }
    for insight in payload["insights"]:
        st.markdown(f"- **{kind_labels.get(insight['kind'], 'Hallazgo')}** · {insight['claim']}")

    render_field_card("evidence", state="Calculado")
    with st.expander("Limitaciones y evidencia"):
        for limitation in payload["limitations"]:
            st.markdown(f"- {limitation}")
        refs = [ref for item in payload["insights"] for ref in item["evidence_refs"]]
        st.caption("Referencias técnicas: " + "; ".join(refs))

    provider, _ = _llm_provider()
    render_results_questions(
        repository,
        project_id,
        ReportPayload.model_validate(payload),
        provider,
    )
    render_report_download(repository, report)

    if report["status"] == "PUBLISHED":
        st.success("Reporte publicado en la demo privada.")
        return
    if narrative is not None and st.button(
        "Pedir otra redacción con IA (crea una versión y exige otra aprobación)",
        key=f"rewrite-{report['id']}",
    ):
        _create_report(repository, run["id"])
    if report["status"] == "REJECTED":
        st.warning(
            "El reporte fue rechazado. Puedes pedir otra redacción o revisar las decisiones "
            "y ejecutar nuevamente el experimento."
        )
        return
    if report["status"] == "DRAFT":
        with st.form(f"publication-decision-{report['id']}"):
            render_field_card("report_decision", title="¿Apruebas este reporte?")
            decision = st.radio(
                "Decisión sobre el reporte",
                ["APPROVED", "REJECTED"],
                label_visibility="collapsed",
                index=None,
                horizontal=True,
                format_func=lambda value: "Aprobar" if value == "APPROVED" else "Rechazar",
            )
            render_field_card("report_reviewer", title="¿Quién lo revisó?")
            actor_label = st.text_input(
                "Responsable de revisar el reporte",
                label_visibility="collapsed",
                placeholder=_reviewer_hint(_actor_for_run(repository, run)),
                max_chars=ACTOR_MAX_CHARS,
            )
            render_field_card("report_comment", title="Comentario")
            comment = st.text_area("Comentario sobre el reporte", label_visibility="collapsed")
            submitted = st.form_submit_button("Registrar decisión del reporte")
        if submitted:
            if decision is None:
                st.error("Elige explícitamente aprobar o rechazar el reporte.")
            elif not actor_label.strip():
                st.error("Indica quién revisó el reporte.")
            elif decision == "REJECTED" and not comment.strip():
                st.error("Explica el motivo del rechazo antes de registrarlo.")
            else:
                try:
                    repository.decide_publication(
                        report_id=report["id"],
                        decision=decision,
                        actor_label=actor_label,
                        comment=comment or None,
                    )
                except (InvalidStateTransitionError, ValueError, sqlite3.IntegrityError) as error:
                    render_error(error, fallback_code="INVALID_STATE_TRANSITION")
                else:
                    st.rerun()
        return

    render_field_card("publish_action", state="Falta confirmar")
    confirm_key = f"publish-confirm-{report['id']}"
    if not st.session_state.get(confirm_key):
        if st.button("Publicar reporte aprobado", key=f"publish-{report['id']}", type="primary"):
            st.session_state[confirm_key] = True
            st.rerun()
        return
    with st.container(border=True):
        st.markdown(
            "**¿Publicar este reporte?** Quedará visible sólo dentro de esta demo privada, "
            "para quienes abran Goal Settings en este equipo. No se envía a Internet ni a nadie."
        )
        publish_col, cancel_col = st.columns(2)
        publish = publish_col.button(
            "Sí, publicar en la demo", key=f"publish-yes-{report['id']}", type="primary"
        )
        cancel = cancel_col.button("Cancelar", key=f"publish-no-{report['id']}")
    if cancel:
        st.session_state.pop(confirm_key, None)
        st.rerun()
    if publish:
        st.session_state.pop(confirm_key, None)
        try:
            repository.publish_report(report["id"])
        except (ApprovalRequiredError, InvalidStateTransitionError) as error:
            render_error(error, fallback_code="REPORT_NOT_APPROVED")
        else:
            st.rerun()


def render_app() -> None:
    st.set_page_config(
        page_title="Goal Settings Analytics Studio",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_theme()
    render_app_header(__version__)
    st.title("Goal Settings Analytics Studio")
    render_about()
    render_ai_mode()
    st.warning("Demo privada: usa sólo datos de prueba, inventados o sin datos personales.")

    runtime_root = _runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    if (runtime_root / "REVISIONES_SIMULADAS.txt").is_file():
        st.info(
            "Demo guiada: datos sintéticos y revisiones simuladas. Los modelos se entrenaron "
            "realmente; las aprobaciones del recorrido son ficticias y las cifras económicas "
            "son supuestos, no ingresos ni ahorros observados."
        )
    other = record_instance(runtime_root)
    if other is not None:
        render_other_instance_warning(other.seen_at.strftime("%H:%M:%S UTC"))
    try:
        _render_workflow(runtime_root)
    except sqlite3.DatabaseError as error:
        _LOGGER.exception("Fallo de almacenamiento en %s", runtime_root)
        render_database_failure(error)
    except Exception as error:  # noqa: BLE001 - red de seguridad: nunca mostrar una traza
        _LOGGER.exception("Falla inesperada en la interfaz")
        render_error(error, fallback_code="UNEXPECTED_ERROR")
    st.caption(f"Versión del paquete: {__version__}")


def _render_workflow(runtime_root: Path) -> None:
    connection = get_connection(runtime_root / "goal_settings.sqlite3", use_wal=True)
    try:
        migrate(connection)
        repository = WorkflowRepository(connection)

        from core.domain.capabilities import CapabilityService

        # Streamlit elimina el estado de widgets ocultos entre vistas.
        # Conservamos la selección para poder volver al mismo objetivo y evaluación.
        for key in list(st.session_state):
            if key in {"business-product", _ACTIVE_PROJECT} or str(key).startswith(
                ("business-capability-", "business-cycle-")
            ):
                st.session_state[key] = st.session_state[key]
        # Los trabajos anteriores sin producto conservan su entrada habitual.
        default_section = (
            "Datos y modelos"
            if repository.list_active_projects()
            and not CapabilityService(repository).list_products()
            else "Producto y decisiones"
        )
        pending_section = st.session_state.pop("pending-workspace-section", None)
        if pending_section is not None:
            st.session_state["workspace-section"] = pending_section
        st.session_state.setdefault("workspace-section", default_section)
        with st.sidebar:
            section = st.radio(
                "Espacio de trabajo",
                ["Producto y decisiones", "Datos y modelos"],
                key="workspace-section",
            )
            st.divider()
        if section == "Producto y decisiones":
            render_product_workspace(
                repository,
                on_open=_open_retraining_project,
                provider_factory=_llm_provider,
                storage_root=runtime_root,
            )
            with st.sidebar:
                st.caption(
                    "Empieza por tu producto y el objetivo que quieres mejorar. "
                    "Al abrir una evaluación continuarás con sus datos y su modelo."
                )
            return

        render_stage_intro(
            1,
            "Tus datos",
            description=(
                "Crea un proyecto y sube la tabla de casos pasados con la que vas a trabajar."
            ),
            anchor="proyecto-y-datos",
        )
        projects = repository.list_active_projects()

        project_id: str | None = None
        accepted: list[Any] = []
        latest = None
        latest_strategy = None
        latest_run = None
        latest_report = None

        with st.sidebar:
            st.markdown("## Tus proyectos")
            labels = {row["id"]: row["name"] for row in projects}
            if projects:
                pending = st.session_state.pop(_PENDING_PROJECT, None)
                if pending in labels:
                    st.session_state[_ACTIVE_PROJECT] = pending
                if st.session_state.get(_ACTIVE_PROJECT) not in labels:
                    st.session_state.pop(_ACTIVE_PROJECT, None)
                project_id = st.selectbox(
                    "Proyecto",
                    list(labels),
                    format_func=lambda value: labels[value],
                    key=_ACTIVE_PROJECT,
                    help=FIELD_COPY["active_project"].text,
                )
                accepted = repository.list_datasets(
                    project_id,
                    admission_status="ACCEPTED",
                    active_only=True,
                )
                latest = repository.get_latest_context(project_id)
                if latest is not None:
                    latest_strategy = repository.get_latest_strategy(latest["id"])
                if latest_strategy is not None:
                    latest_run = repository.get_latest_run(latest_strategy["id"])
                if latest_run is not None:
                    latest_report = repository.get_latest_report(latest_run["id"])

            progress = WorkflowProgress(
                has_project=bool(projects),
                has_dataset=bool(accepted),
                has_context=latest is not None,
                has_training_approval=(
                    latest_strategy is not None
                    and has_training_approval(connection, latest_strategy["id"])
                ),
                has_reportable_run=(
                    latest_run is not None and latest_run["status"] in _REPORTABLE_RUNS
                ),
                report_approved=(
                    latest_report is not None and latest_report["status"] == "APPROVED"
                ),
                report_published=(
                    latest_report is not None and latest_report["status"] == "PUBLISHED"
                ),
                model_usable=(
                    latest_report is not None
                    and latest_run is not None
                    and latest_run["status"] == "COMPLETED"
                    and bool(latest_run["artifact_path"])
                    and has_publication_approval(connection, latest_report["id"])
                ),
                has_scoring=(
                    latest_run is not None and bool(repository.list_scorings(latest_run["id"]))
                ),
            )
            navigation = (
                navigation_from_guide(guide_for(repository, project_id))
                if project_id is not None
                else build_navigation(progress)
            )
            if project_id is not None:
                render_project_card(labels[project_id], navigation)
            with st.popover("Nuevo proyecto", width="stretch", key="create-project-popover"):
                _render_project_form(repository, key="create-project-sidebar")
            if project_id is not None:
                with st.popover(
                    "Eliminar proyecto", width="stretch", key="archive-project-popover"
                ):
                    _render_archive_form(repository, project_id, labels[project_id])
            render_advanced_toggle()
            st.markdown("## Tu avance")
            if project_id is None:
                st.caption("Crea o elige un proyecto para habilitar sus pasos.")
            render_workflow_navigation(navigation)
            st.divider()
            st.caption("Los nombres que escribes se registran tal cual; no hay inicio de sesión.")
            st.caption("Demo privada: nada se publica en Internet.")

        project_flash = st.session_state.pop(_PROJECT_FLASH, None)
        if project_flash:
            st.success(project_flash)
        if not projects:
            empty_state(
                "Todavía no hay proyectos",
                "Crea el primero aquí abajo. Nada se entrena en este paso.",
            )
            _render_project_form(repository)
        else:
            assert project_id is not None
            st.markdown(
                f"Estás trabajando en **{labels[project_id]}**. Para cambiar, crear o eliminar "
                "proyectos usa el panel de la izquierda."
            )
            render_guide_card(repository, project_id)
            render_incoming_events(repository, runtime_root, project_id, on_change=_flash_and_rerun)
            if not accepted:
                st.subheader("Sube tus datos")
                render_batch_upload(repository, runtime_root, project_id, on_done=_flash_and_rerun)
            else:
                flash = st.session_state.pop(_UPLOAD_FLASH, None)
                if flash:
                    st.success(flash)
                dataset = accepted[0]
                # V2-04: lo opcional (explorar) va aparte del camino principal.
                main_tab, explore_tab = st.tabs(["Crear y usar el modelo", "Explorar tus datos"])
                with explore_tab:
                    st.caption(
                        "Opcional: entiende tus datos antes o después de crear el modelo. "
                        "Nada de lo que hagas aquí entrena un modelo ni cambia tus archivos."
                    )
                    render_explore(repository, runtime_root, project_id, on_saved=_flash_and_rerun)
                    render_questions(
                        repository, runtime_root, project_id, on_saved=_flash_and_rerun
                    )
                    render_decisions(repository, project_id)
                    render_cycle_summary(repository, runtime_root, project_id)
                    render_learning_timeline(repository, runtime_root, project_id)
                with main_tab:
                    render_project_catalog(
                        repository,
                        project_id,
                        render_upload=lambda **kwargs: render_batch_upload(
                            repository, runtime_root, project_id, on_done=_flash_and_rerun
                        ),
                        on_change=_flash_and_rerun,
                    )
                    render_field_card("active_file", state="Confirmado")
                    st.write(f"Tabla para entrenar: **{dataset['original_filename_display']}**.")
                    render_dataset_profile(json.loads(dataset["profile_json"]))

                    composition_flash = st.session_state.pop(_COMPOSITION_FLASH, None)
                    if composition_flash:
                        st.success(composition_flash)
                    _render_dataset_composition(
                        repository,
                        runtime_root,
                        project_id,
                        dataset,
                    )

                    render_stage_intro(
                        2,
                        "Qué quieres anticipar",
                        description=(
                            "Cuenta tu situación, confirma cada respuesta y revisa cómo se "
                            "entrenará. Nada se entrena hasta que lo apruebes."
                        ),
                        anchor="contexto-y-estrategia",
                    )
                    provider, _ = _llm_provider()
                    if latest is None:
                        render_context_assistant(
                            repository, runtime_root, project_id, dataset, provider
                        )
                    else:
                        with st.expander("Cambiar mis decisiones (se guardará una versión nueva)"):
                            render_context_assistant(
                                repository, runtime_root, project_id, dataset, provider
                            )
                    latest = repository.get_latest_context(project_id)
                    if latest is not None:
                        st.caption(f"Decisiones guardadas: versión {latest['version_no']}.")
                        with st.expander("Ver detalle técnico"):
                            st.code(f"context_sha256: {latest['context_sha256']}")
                        render_stage_intro(
                            3,
                            "Plan y primera aprobación",
                            description=(
                                "Revisa en palabras simples cómo se entrenará y apruébalo. "
                                "Recién después se puede entrenar."
                            ),
                            anchor="plan-y-aprobacion",
                        )
                        strategy = _render_strategy_review(repository, project_id, latest)
                        if strategy is not None:
                            render_stage_intro(
                                4,
                                "Entrenar y ver resultados",
                                description=(
                                    "El sistema entrena tres modelos con tu archivo, los compara y "
                                    "te muestra qué tan bien anticipa el resultado."
                                ),
                                anchor="entrenamiento-y-resultados",
                            )
                            run = _render_training(repository, runtime_root, strategy)
                            if run is not None:
                                render_experiment_comparison(repository, project_id)
                                render_stage_intro(
                                    5,
                                    "Reporte y segunda aprobación",
                                    description=(
                                        "Revisa qué se puede concluir y sus límites. Aprobar el "
                                        "reporte y publicarlo son dos pasos separados."
                                    ),
                                    anchor="insights-y-cierre",
                                )
                                _render_report(repository, project_id, run)
                                render_stage_intro(
                                    None,
                                    "Usar el modelo",
                                    description=(
                                        "Con el modelo aprobado, ordena casos nuevos de mayor a "
                                        "menor riesgo sin volver a entrenar."
                                    ),
                                    anchor="uso-continuo",
                                )
                                render_model_scoring(
                                    repository, runtime_root, run, _open_retraining_project
                                )
        if project_id is not None:
            scroll_to_new_step(project_id, navigation)
    finally:
        connection.close()


if __name__ == "__main__":
    render_app()
