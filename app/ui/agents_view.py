"""Equipo de IA con pausas visibles y trazas persistidas por capacidad."""

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import streamlit as st

from core.ai_engine.provider import LLMProvider
from core.contracts.orchestration import AgentBudgetV1
from core.db.repositories import WorkflowRepository
from core.db.repositories.agent_turns import AgentTurnRepository
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.errors import DomainError

_ROLES = {
    "coordinator": "Coordinador",
    "business": "Negocio",
    "data": "Datos y análisis",
    "modeling": "Modelado",
    "value_evidence": "Valor y evidencia",
}
_STATES = {
    "PENDING": "Preparado",
    "RUNNING": "En curso",
    "WAITING_HUMAN": "Necesita tu revisión",
    "WAITING_DATA": "Espera información o datos",
    "COMPLETED": "Trabajo terminado",
    "FAILED": "Interrumpido por un error",
    "CANCELLED": "Cancelado",
}


def _error(error: Exception) -> None:
    code = getattr(error, "code", "INVALID_REQUEST")
    st.error(f"El equipo se detuvo ({code}). Revisa el estado y la información antes de continuar.")


def _render_data_evidence(summary: dict) -> None:
    """Muestra cálculos del servicio; nunca interpreta cifras escritas por el modelo."""

    kind = summary.get("kind")
    if kind == "label_readiness":
        status = summary.get("status")
        if status == "ready":
            st.success("Los resultados de esta tabla tienen observación comprobada.")
        elif status == "needs_data":
            st.info("Falta completar la observación de los resultados para usar esta tabla.")
        else:
            st.info("Hay definiciones o evidencia de observación que debes revisar.")
        left, right = st.columns(2)
        left.metric("Filas de la tabla", summary.get("table_rows", "—"))
        right.metric("Verificadas en esta revisión", summary.get("verified_table_rows", "—"))
        if summary.get("snapshot_readiness") is not None:
            st.caption(
                f"Casos pendientes en la fuente: {summary.get('pending_source_count', 0)}. "
                "Los pendientes deben permanecer fuera del entrenamiento."
            )
        for step in summary.get("measurement_plan", []):
            st.write(f"• {step}")
        st.caption(
            "Esta comprobación no aprueba un entrenamiento. La evaluación estadística "
            "y la revisión del plan se hacen antes de entrenar."
        )
    elif kind == "data_quality":
        st.caption(
            f"Calidad calculada sobre {summary.get('row_count', '—')} filas "
            f"y {summary.get('column_count', '—')} columnas."
        )
        labels = {
            "EXTREME_OUTLIER": "Valor numérico extremo",
            "DATE_INVALID_OR_OUT_OF_RANGE": "Fecha fuera del rango esperado",
            "POSSIBLE_SIGN_ERROR": "Posible signo incorrecto",
            "NEAR_DUPLICATE_CATEGORY": "Categorías con escritura parecida",
            "KEY_MISMATCH": "Claves sin correspondencia",
        }
        for field, label in (("missing_counts", "Valores faltantes"), ("findings", "Hallazgos")):
            rows = [item for item in summary.get(field, []) if item.get("count", 0)]
            if rows:
                st.write(label)
                st.dataframe(
                    [
                        {
                            "Columna": item["column_alias"],
                            "Casos": item["count"],
                            "Proporción": item["share"],
                            **(
                                {"Hallazgo": labels.get(item["kind"], item["kind"])}
                                if "kind" in item
                                else {}
                            ),
                        }
                        for item in rows
                    ],
                    hide_index=True,
                    width="stretch",
                )
    elif kind == "aggregate_analysis":
        st.write("Resumen calculado de los datos")
        metrics = summary.get("metrics", {})
        labels = {
            "count": "Conteo",
            "mean": "Promedio",
            "sum": "Suma",
            "column_count": "Columnas",
            "missing_cells": "Celdas vacías",
            "duplicate_rows": "Filas repetidas",
        }
        if metrics:
            st.dataframe(
                [
                    {"Medida": labels.get(name, name), "Valor": value}
                    for name, value in metrics.items()
                ],
                hide_index=True,
                width="stretch",
            )
    elif kind == "model_viability":
        if summary.get("status") == "ready":
            st.success("La tabla permite preparar un entrenamiento con separación temporal.")
        else:
            st.info("Faltan datos o definiciones para preparar este entrenamiento.")
        split = summary.get("split") or {}
        rows = [
            {
                "Período": label,
                "Casos": split[name].get("row_count", 0),
                "Con el resultado objetivo": split[name].get("positive_support", 0),
                "Sin el resultado objetivo": split[name].get("negative_support", 0),
            }
            for name, label in (
                ("train", "Aprendizaje"),
                ("validation", "Selección"),
                ("test", "Prueba final"),
            )
            if name in split
        ]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                f"Excluidos por maduración entre períodos: "
                f"{split.get('purged_train_rows', 0)} de aprendizaje y "
                f"{split.get('purged_validation_rows', 0)} de selección."
            )
        for step in summary.get("measurement_plan", []):
            st.write(f"• {step}")
        st.caption("La viabilidad no aprueba el entrenamiento ni anticipa sus métricas.")
    elif kind == "training_strategy":
        st.info(
            f"Plan de entrenamiento preparado: versión {summary.get('version_no', '—')}. "
            "Abre el ciclo y revisa la primera aprobación antes de entrenar."
        )
    elif kind == "model_training":
        if summary.get("selected_model"):
            st.success(f"Modelo seleccionado: {summary['selected_model']}.")
        else:
            st.info("El entrenamiento terminó sin un modelo recomendado.")
        st.caption(
            "El entrenamiento no publica el modelo. Revisa su evaluación y el reporte "
            "antes de autorizar su uso."
        )
        for label, key in (
            ("Selección sobre validación", "validation_metrics"),
            ("Prueba final del entrenamiento", "holdout_metrics"),
        ):
            if summary.get(key):
                st.write(label)
                st.json(summary[key], expanded=False)
    elif kind == "value_policy":
        st.info(
            f"Política versión {summary.get('version_no', '—')} preparada con evidencia "
            "exploratoria de validación. Requiere revisión humana separada."
        )


def render_agents(
    repository: WorkflowRepository,
    capability_id: str,
    *,
    cycle_id: str | None = None,
    provider_factory: Callable[[], tuple[LLMProvider | None, bool]] | None = None,
    storage_root: str | Path | None = None,
) -> None:
    from core.ai_engine.agents import AgentRunner

    with st.expander("Equipo de IA", expanded=False):
        st.write(
            "El coordinador distribuye el trabajo entre especialistas. Consultan evidencia, "
            "proponen decisiones y se detienen cuando necesitan datos o una revisión tuya."
        )
        st.caption(
            "Sus mensajes son propuestas de IA. Revisa las cifras en los resultados calculados "
            "y confirma las decisiones en sus formularios."
        )
        provider, config_error = provider_factory() if provider_factory else (None, False)
        if config_error:
            st.warning("La configuración del proveedor IA necesita revisión.")
        elif provider is None:
            st.info(
                "El proveedor IA está desactivado. Puedes guardar un encargo para después. "
                "Mientras tanto, continúa manualmente en Decisión de negocio o abre una "
                "evaluación para trabajar con tus datos."
            )
        else:
            st.caption(f"Proveedor: {provider.name} · Modelo: {provider.model}")
        runner = AgentRunner(repository, provider, storage_root=storage_root)
        tasks = OrchestrationRepository(repository)
        capability = CapabilityService(repository).get_capability(capability_id)
        roots = [
            task
            for task in reversed(tasks.list_tasks())
            if task.capability_id == capability_id and task.parent_task_id is None
        ]
        with (
            st.expander("Preparar un nuevo encargo", expanded=not roots),
            st.form(f"new-agent-work-{capability_id}"),
        ):
            objective = st.text_area(
                "Qué quieres que resuelva el equipo", value=capability.objective
            )
            actor = st.text_input("Quién solicita el trabajo del equipo")
            submitted = st.form_submit_button("Preparar trabajo del equipo")
        if submitted:
            key = f"agent-create-{capability_id}"
            operation = st.session_state.setdefault(key, str(uuid4()))
            try:
                root = runner.start(
                    capability_id=capability_id,
                    cycle_id=cycle_id,
                    objective=objective,
                    actor_label=actor,
                    operation_id=operation,
                    budget=AgentBudgetV1(max_steps=60, max_calls=24, max_seconds=600),
                )
            except (DomainError, ValueError) as error:
                _error(error)
            else:
                st.session_state[f"agent-work-{capability_id}"] = root.id
                st.session_state.pop(key, None)
                st.rerun()
        if not roots:
            st.caption("Todavía no hay trabajos del equipo para este objetivo.")
            return
        by_id = {task.id: task for task in roots}
        selected = st.selectbox(
            "Trabajo del equipo",
            list(by_id),
            format_func=lambda key: (
                f"{_STATES[by_id[key].state]} · "
                f"{(by_id[key].objective or _ROLES[by_id[key].role])[:90]}"
                f"{'…' if len(by_id[key].objective or '') > 90 else ''}"
            ),
            key=f"agent-work-{capability_id}",
        )
        root = by_id[selected]
        st.caption(root.objective or "Encargo del coordinador")
        tree = tasks.list_tasks(root.id)
        waiting = any(task.state in {"WAITING_HUMAN", "WAITING_DATA"} for task in tree)
        st.write("Necesita información o tu revisión" if waiting else _STATES[root.state])
        if root.state in {"PENDING", "RUNNING"} and root.calls_used and not waiting:
            st.info(
                "El análisis aún no termina. Puedes continuarlo con la evidencia y el "
                "presupuesto que quedan; esto no aprueba ni publica ninguna decisión."
            )
        if root.state == "COMPLETED":
            conclusions = [
                turn.proposal.message
                for turn in AgentTurnRepository(repository).list_turns(root.id)
                if turn.proposal is not None
                and turn.proposal.action == "finish"
                and turn.error_code is None
                and turn.operation.status == "COMPLETED"
            ]
            with st.container(border=True):
                st.success("El equipo terminó este encargo.")
                if conclusions:
                    st.markdown("**Conclusión del equipo (IA)**")
                    st.write(conclusions[-1])
                st.caption(
                    "Siguiente paso: revisa la conclusión junto con la evidencia. "
                    "Las decisiones y aprobaciones se confirman en sus formularios."
                )
        elif root.state == "FAILED":
            st.warning(
                "Este encargo se interrumpió. Revisa Actividad y evidencia del equipo "
                "antes de preparar otro encargo."
            )
        st.dataframe(
            [
                {
                    "Agente": _ROLES[task.role],
                    "Encargo": task.objective or "—",
                    "Estado": _STATES[task.state],
                }
                for task in tree
            ],
            hide_index=True,
            width="stretch",
        )
        readiness = [
            operation.result.summary
            for task in tree
            for operation in tasks.list_operations(task.id)
            if operation.call.tool_name == "inspect_label_readiness"
            and operation.result is not None
            and operation.result.status == "COMPLETED"
        ]
        if readiness:
            _render_data_evidence(readiness[-1])
        with st.expander("Actividad y evidencia del equipo"):
            st.caption(
                f"Llamadas IA: {root.calls_used}/{root.budget.max_calls} · "
                f"Pasos: {root.steps_used}/{root.budget.max_steps} · "
                f"Tiempo de ejecución registrado: {root.elapsed_ms / 1000:.1f} s "
                f"de {root.budget.max_seconds} s. Las pausas humanas no consumen tiempo."
            )
            for task in tree:
                st.write(_ROLES[task.role])
                for turn in AgentTurnRepository(repository).list_turns(task.id):
                    if turn.proposal is not None:
                        st.write(turn.proposal.message)
                    st.caption(
                        f"{turn.actual_provider or turn.provider} · "
                        f"{turn.actual_model or turn.model} · "
                        f"{turn.operation.status}"
                    )
                for operation in tasks.list_operations(task.id):
                    st.caption(f"{operation.call.tool_name} · {operation.status}")
                    if operation.result is not None:
                        if (
                            operation.result.status == "COMPLETED"
                            and operation.result.summary.get("kind") != "label_readiness"
                        ):
                            _render_data_evidence(operation.result.summary)
                        st.json(operation.result.model_dump(mode="json"), expanded=False)

        if root.state in {"COMPLETED", "FAILED", "CANCELLED"}:
            return
        if any(task.lease_owner for task in tree):
            st.info("Hay un trabajo en ejecución. Espera a que termine antes de iniciar otro.")
            if st.button("Revisar operaciones interrumpidas", key=f"agent-recover-{root.id}"):
                try:
                    recovered = runner.supervisor.recover_expired_operations(root_task_id=root.id)
                except OrchestrationError as error:
                    _error(error)
                else:
                    if recovered:
                        st.rerun()
                    st.caption(
                        "El trabajo aún tiene tiempo para terminar. Vuelve a revisar después."
                    )
            return
        actor = st.text_input("Quién continúa el trabajo", key=f"agent-actor-{root.id}")
        if waiting:
            for task in tree:
                if task.state not in {"WAITING_HUMAN", "WAITING_DATA"}:
                    continue
                turns = AgentTurnRepository(repository).list_turns(task.id)
                if turns and turns[-1].proposal is not None:
                    st.write(f"{_ROLES[task.role]}: {turns[-1].proposal.message}")
                    for question in turns[-1].proposal.questions:
                        st.write(f"• {question}")
            st.info(
                "Completa o revisa la información solicitada en las decisiones y en el ciclo. "
                "Reanudar permite al equipo volver a revisarla; no aprueba un entrenamiento."
            )
            if st.button("Revisé la información; reanudar", key=f"agent-resume-{root.id}"):
                try:
                    runner.resume(root.id, expected_revision=root.revision, actor_label=actor)
                except (DomainError, ValueError) as error:
                    _error(error)
                else:
                    st.rerun()
            return
        try:
            preview = runner.preview(root.id)
        except (DomainError, ValueError) as error:
            _error(error)
            return
        with st.expander("Revisar información que recibirá la IA"):
            st.json(preview, expanded=False)
        consent = st.checkbox(
            "Revisé el contexto y autorizo este recorrido con el proveedor IA",
            key="agent-consent-"
            + canonical_hash(
                {"context": preview, "revisions": {task.id: task.revision for task in tree}}
            ),
        )
        st.caption(
            "El recorrido añade los resultados de las herramientas permitidas. "
            "Envía contexto revisado y resúmenes; las filas de los archivos permanecen locales. "
            "Se ejecutan hasta ocho turnos por recorrido y se conservan los límites del trabajo."
        )
        run_label = "Continuar análisis" if root.calls_used else "Ejecutar equipo"
        if st.button(
            run_label,
            key=f"agent-run-{root.id}",
            disabled=provider is None or not consent,
            type="primary",
        ):
            run_key = f"agent-execution-operation-{root.id}"
            operation_id = st.session_state.setdefault(run_key, f"ui-agent-run:{uuid4()}")
            try:
                with st.spinner("El equipo está trabajando…"):
                    runner.run(
                        root.id,
                        expected_revision=root.revision,
                        actor_label=actor,
                        operation_id=operation_id,
                        max_turns=8,
                    )
            except (DomainError, ValueError) as error:
                _error(error)
            else:
                st.session_state.pop(run_key, None)
                st.rerun()
