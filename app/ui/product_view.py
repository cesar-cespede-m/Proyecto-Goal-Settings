"""MV-04: producto y capacidades de negocio sobre los experimentos existentes."""

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from app.ui.capability_view import render_capability_value
from core.ai_engine.provider import LLMProvider
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService


def _operation(key: str) -> str:
    return st.session_state.setdefault(f"operation-{key}", str(uuid4()))


def _finished(key: str) -> None:
    st.session_state.pop(f"operation-{key}", None)
    st.rerun()


def render_product_workspace(
    repository: WorkflowRepository,
    *,
    on_open: Callable[[str, str], None],
    provider_factory: Callable[[], tuple[LLMProvider | None, bool]] | None = None,
    storage_root: str | Path | None = None,
) -> None:
    service = CapabilityService(repository)
    products = service.list_products()
    active_projects = {str(row["id"]) for row in repository.list_active_projects()}
    st.subheader("Tu producto: ingresos, ahorro y aprendizaje")
    with st.container(border=True):
        st.write(
            "Define qué decisión quieres mejorar para generar ingresos o reducir costos. "
            "Cada objetivo conserva sus datos, modelos y evaluaciones. "
            "Puedes comenzar aunque todavía no tengas datos."
        )
        with st.popover("Nuevo producto"):
            with st.form("create-business-product"):
                name = st.text_input("Nombre del producto")
                description = st.text_area("Descripción del producto")
                stage = st.selectbox(
                    "Etapa",
                    ["building", "operating"],
                    format_func=lambda value: (
                        "En construcción" if value == "building" else "En funcionamiento"
                    ),
                )
                actor = st.text_input("Quién registra este producto")
                submitted = st.form_submit_button("Crear producto")
            if submitted:
                try:
                    product = service.create_product(
                        name=name,
                        description=description,
                        stage=stage,
                        actor_label=actor,
                        operation_id=_operation("create-business-product"),
                    )
                except (ValueError, ValidationError) as error:
                    st.error(str(error))
                else:
                    st.session_state["business-product"] = product.id
                    _finished("create-business-product")
        if not products:
            st.info("Crea tu producto para organizar sus objetivos de ingresos y ahorro.")
            return
        by_id = {item.id: item for item in products}
        if st.session_state.get("business-product") not in by_id:
            st.session_state.pop("business-product", None)
        product_id = st.selectbox(
            "Producto", list(by_id), format_func=lambda key: by_id[key].name, key="business-product"
        )
        st.caption(by_id[product_id].description)
        with st.popover("Nuevo objetivo de negocio"):
            with st.form(f"create-capability-{product_id}"):
                name = st.text_input("Nombre del objetivo", placeholder="Mejorar la primera compra")
                objective = st.text_area(
                    "Qué quieres mejorar", placeholder="Qué resultado buscas y por qué importa."
                )
                value_kind = st.selectbox(
                    "Tipo de valor",
                    ["revenue", "savings"],
                    format_func=lambda value: "Ingresos" if value == "revenue" else "Ahorro",
                )
                unit = st.text_input(
                    "Sobre qué tomarás cada decisión",
                    placeholder="Un pedido, un cliente nuevo, una solicitud…",
                )
                actor = st.text_input("Quién registra este objetivo")
                submitted = st.form_submit_button("Crear objetivo")
            if submitted:
                key = f"create-capability-{product_id}"
                try:
                    capability = service.create_capability(
                        product_id,
                        name=name,
                        objective=objective,
                        value_kind=value_kind,
                        decision_unit=unit,
                        actor_label=actor,
                        operation_id=_operation(key),
                    )
                except (ValueError, ValidationError) as error:
                    st.error(str(error))
                else:
                    st.session_state[f"business-capability-{product_id}"] = capability.id
                    _finished(key)
        capabilities = service.list_capabilities(product_id)
        if not capabilities:
            st.info("Siguiente paso: crea un objetivo de ingresos o ahorro para este producto.")
            return
        by_capability = {item.id: item for item in capabilities}
        capability_id = st.selectbox(
            "Objetivo de negocio",
            list(by_capability),
            format_func=lambda key: by_capability[key].name,
            key=f"business-capability-{product_id}",
        )
        capability = by_capability[capability_id]
        st.write(capability.objective)
        st.caption(
            f"{'Ingresos' if capability.value_kind == 'revenue' else 'Ahorro'} "
            f"· Una decisión por {capability.decision_unit}."
        )
        render_capability_value(repository, capability_id, storage_root=storage_root)
        cycles = service.list_cycles(capability_id)
        st.subheader("Evaluaciones y actualizaciones")
        st.caption(
            "Una evaluación reúne los datos y el modelo de este objetivo. "
            "Inicia otra cuando quieras evaluar datos nuevos; las anteriores se conservan."
        )
        with (
            st.expander("Iniciar una evaluación", expanded=not cycles),
            st.form(f"create-cycle-{capability_id}"),
        ):
            source_cycle_id = None
            if cycles:
                sources = {item.id: item for item in cycles}
                source_cycle_id = st.selectbox(
                    "Evaluación de origen",
                    list(sources),
                    index=len(sources) - 1,
                    format_func=lambda key: (
                        f"Evaluación {sources[key].cycle_no}: {sources[key].reason}"
                    ),
                )
            reason = st.text_input(
                "Motivo de la evaluación", placeholder="Primera evaluación del objetivo"
            )
            actor = st.text_input("Quién inicia esta evaluación")
            submitted = st.form_submit_button("Crear evaluación y continuar")
        if submitted:
            key = f"create-cycle-{capability_id}"
            try:
                cycle = service.create_cycle(
                    capability_id,
                    reason=reason,
                    actor_label=actor,
                    operation_id=_operation(key),
                    source_cycle_id=source_cycle_id,
                )
            except (ValueError, ValidationError) as error:
                st.error(str(error))
            else:
                st.session_state.pop(f"operation-{key}", None)
                on_open(cycle.project_id, "Evaluación creada. Continúa cargando sus datos.")
        if cycles:
            by_cycle = {item.id: item for item in cycles}
            cycle_id = st.selectbox(
                "Evaluación",
                list(by_cycle),
                format_func=lambda key: (
                    f"Evaluación {by_cycle[key].cycle_no}: {by_cycle[key].reason}"
                ),
                key=f"business-cycle-{capability_id}",
            )
            archived = by_cycle[cycle_id].project_id not in active_projects
            if archived:
                st.info(
                    "El trabajo de esta evaluación está archivado. Se conserva como antecedente."
                )
            if st.button(
                "Continuar con datos y modelo",
                key=f"open-cycle-{capability_id}",
                disabled=archived,
                type="primary",
            ):
                on_open(
                    by_cycle[cycle_id].project_id, "Evaluación abierta para continuar el trabajo."
                )
        else:
            st.caption(
                "Tu objetivo queda guardado. Crea una evaluación cuando tengas datos para analizar."
            )
        from app.ui.agents_view import render_agents

        render_agents(
            repository,
            capability_id,
            cycle_id=cycle_id if cycles else None,
            provider_factory=provider_factory,
            storage_root=storage_root,
        )
