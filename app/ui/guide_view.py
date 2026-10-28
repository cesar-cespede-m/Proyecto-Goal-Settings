"""V2-02/V2-03: la Guía arriba de la página y los pasos de la barra lateral.

Ambos salen de `core.domain.guide.project_guide`, así la página y la barra lateral nunca
muestran pasos distintos.
"""

from html import escape
from typing import Any

import streamlit as st

from app.ui.incidents import configured_provider
from app.ui.navigation import NavigationItem, StepStatus
from core.domain.guide import GuideStateV1, project_guide

# Ancla de la página para cada paso de la Guía.
STEP_ANCHORS = {
    "data": "proyecto-y-datos",
    "decisions": "contexto-y-estrategia",
    "plan": "plan-y-aprobacion",
    "train": "entrenamiento-y-resultados",
    "report": "insights-y-cierre",
    "use": "uso-continuo",
}
_STATUS = {"done": StepStatus.COMPLETE, "current": StepStatus.CURRENT, "locked": StepStatus.LOCKED}


def guide_for(repository: Any, project_id: str) -> GuideStateV1:
    return project_guide(repository, project_id, ai_enabled=configured_provider() is not None)


def navigation_from_guide(guide: GuideStateV1) -> tuple[NavigationItem, ...]:
    return tuple(
        NavigationItem(
            number=index,
            title=step.title,
            anchor=STEP_ANCHORS.get(step.id, "proyecto-y-datos"),
            status=_STATUS[step.status],
            requirement=step.locked_reason or step.summary,
        )
        for index, step in enumerate(guide.steps, start=1)
    )


def render_guide_card(repository: Any, project_id: str) -> None:
    guide = guide_for(repository, project_id)
    current = next(step for step in guide.steps if step.id == guide.current_step)
    action = guide.next_action
    title = "Guía con IA" if guide.ai_mode == "llm" else "Guía de Goal Settings"
    anchor = STEP_ANCHORS.get(action.step_id, "proyecto-y-datos")
    with st.container(border=True):
        st.markdown(f"**{title}** · Estás en: **{escape(current.title)}**")
        st.caption(current.summary)
        st.markdown(f"**Siguiente paso: [{action.label} →](#{anchor})**")
        st.caption(f"Por qué: {action.reason}")
        for alert in guide.alerts:
            text = f"{alert.text} {alert.action_label}."
            if alert.severity == "warning":
                st.warning(text)
            else:
                st.info(text)
