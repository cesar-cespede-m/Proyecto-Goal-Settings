"""Componentes visuales atómicos y compuestos para Streamlit."""

import json
from html import escape
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from app.ui.copy import FIELD_COPY
from app.ui.navigation import NavigationItem, StepStatus
from app.ui.warnings import render_warnings, split_warnings


def section_anchor(anchor: str) -> None:
    st.markdown(f'<div id="{escape(anchor)}" class="gsa-anchor"></div>', unsafe_allow_html=True)


def render_app_header(version: str) -> None:
    st.markdown(
        f"""
        <a class="gsa-skip-link" href="#main-content">Saltar al contenido principal</a>
        <div class="gsa-hero" role="banner">
          <div class="gsa-eyebrow">Demo privada · versión {escape(version)}</div>
          <div class="gsa-meta">
            Decisiones de producto con datos, IA y revisión humana.
          </div>
        </div>
        <div id="main-content" class="gsa-anchor"></div>
        """,
        unsafe_allow_html=True,
    )


PREPARATION_STEPS = 5


def stage_eyebrow(number: int | None) -> str:
    """«Paso N de 5» sólo para la preparación; el uso continuo va aparte (V2-02)."""

    if number is None or number > PREPARATION_STEPS:
        return "Cuando el modelo ya está aprobado"
    return f"Paso {number} de {PREPARATION_STEPS}"


def render_stage_intro(
    number: int | None,
    title: str,
    *,
    description: str,
    anchor: str,
) -> None:
    section_anchor(anchor)
    st.markdown(
        f"""
        <section class="gsa-stage" aria-labelledby="stage-{escape(anchor)}">
          <div class="gsa-eyebrow">{escape(stage_eyebrow(number))}</div>
          <h2 id="stage-{escape(anchor)}">{escape(title)}</h2>
          <p class="gsa-meta">{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


_STATUS_LABELS = {
    StepStatus.COMPLETE: "Completado",
    StepStatus.CURRENT: "Paso actual",
}


def render_workflow_navigation(items: tuple[NavigationItem, ...]) -> None:
    """Stepper: el paso actual se destaca y los bloqueados dicen qué falta para abrirse."""

    blocks: list[str] = []
    for item in items:
        if item.number == PREPARATION_STEPS + 1:
            # Recorrido 22/10: con 6 tarjetas y «Paso N de 5» parecía faltar un paso.
            blocks.append('<p class="gsa-nav-divider">Cuando el modelo ya está aprobado</p>')
        locked = item.status == StepStatus.LOCKED
        label = item.requirement if locked else _STATUS_LABELS[item.status]
        tag = "div" if locked else "a"
        attributes = f' href="#{escape(item.anchor)}"' if not locked else ' aria-disabled="true"'
        if item.status == StepStatus.CURRENT:
            attributes += ' aria-current="step"'
        if item.status == StepStatus.COMPLETE:
            badge = "✓"
        elif locked:
            badge = "🔒"
        else:
            badge = str(item.number) if item.number <= PREPARATION_STEPS else "→"
        state = "Bloqueado: " if locked else ""
        blocks.append(
            f'<{tag} class="gsa-nav-item" data-status="{item.status.value}"{attributes}>'
            f'<span class="gsa-nav-number" aria-hidden="true">{badge}</span>'
            "<span>"
            f'<span class="gsa-nav-title">{escape(item.title)}</span><br>'
            f'<span class="gsa-nav-help">{escape(state + label)}</span>'
            f"</span></{tag}>"
        )
    st.markdown(
        '<nav class="gsa-nav" aria-label="Progreso del proyecto">' + "".join(blocks) + "</nav>",
        unsafe_allow_html=True,
    )


def render_project_card(name: str, items: tuple[NavigationItem, ...]) -> None:
    """Tarjeta del proyecto seleccionado con el paso en que está."""

    current = next((item for item in items if item.status == StepStatus.CURRENT), None)
    if current is None:
        where = "Todos los pasos completos"
    elif current.number <= PREPARATION_STEPS:
        where = f"Paso {current.number} de {PREPARATION_STEPS} · {current.title}"
    else:
        where = current.title
    st.markdown(
        '<div class="gsa-project-card" aria-current="true">'
        '<span class="gsa-project-eyebrow">Proyecto seleccionado</span>'
        f'<strong class="gsa-project-name">{escape(name)}</strong>'
        f'<span class="gsa-project-step">{escape(where)}</span></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, detail: str) -> None:
    st.markdown(
        f'<div class="gsa-empty" role="status"><strong>{escape(title)}</strong><br>'
        f'<span class="gsa-meta">{escape(detail)}</span></div>',
        unsafe_allow_html=True,
    )


_SILENT_STATES = {"", "Falta responder", "Calculado"}


def render_field_card(
    field_key: str, *, state: str = "Falta responder", title: str | None = None
) -> None:
    """Ayuda del campo en prosa natural, pegada al control que sigue (DU-1).

    Con ``title`` se muestra primero la pregunta y luego la explicación; el control que
    sigue debe ocultar su propia etiqueta. El estado sólo aparece si aporta información.
    """

    content = FIELD_COPY[field_key]
    badge = (
        "" if state in _SILENT_STATES else f'<span class="gsa-field-state">{escape(state)}</span>'
    )
    heading = f'<span class="gsa-field-title">{escape(title)}</span>' if title else ""
    st.markdown(
        f'<p class="gsa-field-help">{heading}{badge}{escape(content.text)}</p>',
        unsafe_allow_html=True,
    )


def _count(value: Any) -> str:
    """Conteo con punto de miles («3.500»); lo que no es número se muestra tal cual."""

    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}".replace(",", ".")
    return str(value)


def render_dataset_profile(profile: dict[str, Any]) -> None:
    """Muestra resumen humano y conserva el contrato completo en un acordeón."""

    cards = (
        (profile.get("row_count", "—"), "Filas"),
        (profile.get("column_count", "—"), "Columnas"),
        (profile.get("duplicate_row_count", "—"), "Filas duplicadas"),
        (len(split_warnings(profile.get("warnings", []))[0]), "Puntos para revisar"),
    )
    markup = "".join(
        '<div class="gsa-profile-card">'
        f'<div class="gsa-profile-value">{escape(_count(value))}</div>'
        f'<div class="gsa-profile-label">{escape(label)}</div></div>'
        for value, label in cards
    )
    st.markdown(f'<div class="gsa-profile">{markup}</div>', unsafe_allow_html=True)
    # V2-33: los puntos principales ya están en la Guía; aquí, el detalle a pedido.
    if profile.get("warnings"):
        with st.expander("Ver el detalle de los puntos para revisar"):
            render_warnings(profile.get("warnings", []))
    with st.expander("Ver perfil técnico completo"):
        st.json(profile, expanded=False)


def scroll_to_new_step(project_id: str, items: tuple[NavigationItem, ...]) -> bool:
    """Recorrido 22/10: al avanzar, la página quedaba arriba en el paso anterior.

    Sólo desplaza cuando el paso actual cambió respecto de la ejecución anterior; al abrir un
    proyecto o al recargar la página no mueve la vista.
    """

    current = next((item for item in items if item.status == StepStatus.CURRENT), None)
    if current is None:
        return False
    key = f"gsa-last-step-{project_id}"
    previous = st.session_state.get(key)
    st.session_state[key] = current.anchor
    if previous is None or previous == current.anchor:
        return False
    target = json.dumps(current.anchor)
    components.html(
        "<script>const target = window.parent.document.getElementById("
        f"{target});if (target) target.scrollIntoView({{behavior: 'smooth', block: 'start'}});"
        "</script>",
        height=0,
    )
    return True
