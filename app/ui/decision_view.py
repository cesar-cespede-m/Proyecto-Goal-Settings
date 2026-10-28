"""V2-14/V2-15: «Decidir con el modelo» (sí/no): capacidad o costo, qué significaría y niveles.

La elección queda en la sesión y la usa la lista de casos priorizados para marcar a quién
atender y su nivel de riesgo.
"""

import json
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.messages import ERROR_MESSAGES
from core.ml_engine import MLValidationError
from core.ml_engine.decision import DecisionPlanV1, RiskLevelsV1, by_capacity, by_cost, risk_levels
from core.ml_engine.models import EvaluationSummaryV1

_PLAN_KEY = "decision-plan-{run_id}"


def chosen_plan(run_id: str) -> tuple[DecisionPlanV1, RiskLevelsV1] | None:
    return st.session_state.get(_PLAN_KEY.format(run_id=run_id))


def _pct(value: float) -> str:
    return f"{value * 100:.0f} %"


def render_decision(run: Any, event_cases: str = "casos con el resultado") -> None:
    st.markdown("#### Decidir con el modelo")
    st.caption(
        "Un modelo sirve para decidir a quién atender primero. Cuéntanos tu capacidad o qué error "
        "te cuesta más y exploramos una regla con los datos de validación. Esa misma evidencia "
        "se usa para elegirla; todavía necesita una evaluación independiente."
    )
    try:
        evaluation = EvaluationSummaryV1.model_validate(json.loads(run["candidate_metrics_json"]))
        levels = risk_levels(evaluation)
    except MLValidationError as error:
        st.session_state.pop(_PLAN_KEY.format(run_id=run["id"]), None)
        common = ERROR_MESSAGES.get(error.code)
        st.info(f"{common.message} {common.action}" if common else str(error))
        return
    basis = st.radio(
        "¿Cómo quieres decidir?",
        ["capacity", "cost"],
        format_func=lambda value: (
            "Según cuántos casos puedo atender"
            if value == "capacity"
            else "Según qué error me cuesta más"
        ),
        horizontal=True,
        key=f"decision-basis-{run['id']}",
    )
    columns = st.columns(2)
    expected = columns[0].number_input(
        "¿Cuántos casos llegan por periodo (semana, mes)?",
        min_value=1,
        value=1000,
        step=50,
        key=f"decision-expected-{run['id']}",
    )
    try:
        if basis == "capacity":
            capacity = columns[1].number_input(
                "¿Cuántos puedes atender en ese periodo?",
                min_value=0,
                value=100,
                step=10,
                key=f"decision-capacity-{run['id']}",
            )
            plan = by_capacity(
                evaluation, cases_per_period=int(capacity), expected_cases_per_period=int(expected)
            )
        else:
            weight = columns[1].slider(
                "¿Cuántas veces más grave es dejar pasar un caso que avisar de más?",
                min_value=1,
                max_value=20,
                value=5,
                key=f"decision-weight-{run['id']}",
            )
            plan = by_cost(
                evaluation,
                cost_false_alarm=1,
                cost_missed=float(weight),
                expected_cases_per_period=int(expected),
            )
    except MLValidationError as error:
        st.session_state.pop(_PLAN_KEY.format(run_id=run["id"]), None)
        common = ERROR_MESSAGES.get(error.code)
        st.error(
            f"{common.message}\n\n**Qué puedes hacer:** {common.action}" if common else str(error)
        )
        return
    st.session_state[_PLAN_KEY.format(run_id=run["id"])] = (plan, levels)
    metrics = st.columns(4)
    metrics[0].metric("Atender", plan.to_attend)
    metrics[1].metric("Aciertos esperados", plan.expected_true_alerts)
    metrics[2].metric("Falsas alarmas", plan.expected_false_alarms)
    metrics[3].metric("Se escaparían", plan.expected_missed)
    if plan.to_attend:
        st.markdown(
            f"En validación, entre los casos seleccionados, **{round(plan.precision * 100)} "
            f"de cada 100** fueron {event_cases}. Se cubrió **{_pct(plan.recall)}** de los "
            "casos con ese resultado. Su comportamiento en casos nuevos puede cambiar."
        )
    else:
        st.info("Esta regla propone no atender casos. No se creará una lista de intervención.")
    st.caption(
        plan.explanation + " Cada lote representa un período; no acumula cuotas entre archivos."
    )
    selected = evaluation.validation.get(evaluation.selection.selected_model or "")
    general = selected.metrics.precision if selected is not None else None
    note = precision_gap_note(plan.precision, general, plan.to_attend) if plan.to_attend else None
    if note:
        st.info(note)
    st.markdown("**Niveles de riesgo**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Nivel": level.level.capitalize(),
                    "Parte de los casos": _pct(level.share_of_cases),
                    "Con el resultado": _pct(level.event_rate),
                }
                for level in levels.levels
            ]
        ),
        hide_index=True,
    )
    st.caption(
        f"En general, el {_pct(levels.base_rate)} fueron {event_cases}. Al priorizar "
        "casos nuevos verás su nivel y si entran en tu capacidad."
    )


def precision_gap_note(chosen: float, general: float | None, to_attend: int) -> str | None:
    """Compara dos reglas sobre validación, sin inferir mejoras futuras ni causalidad."""

    if general is None or abs(round(chosen * 100) - round(general * 100)) < 5:
        return None
    return (
        f"El umbral original obtuvo cerca de {round(general * 10)} de cada 10 en validación. "
        f"Esta regla prioriza un conjunto distinto y propone atender {to_attend} casos por "
        "período. La diferencia observada no garantiza que una regla sea mejor en casos nuevos."
    )
