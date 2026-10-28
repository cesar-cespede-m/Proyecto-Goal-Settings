"""V2-29/V2-30: modo avanzado. Editar a mano el plan de cálculo con las mismas validaciones que
usa la IA: un plan inválido se explica y no se ejecuta."""

import json
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from app.ui.analysis_view import render_analysis_result
from app.ui.messages import render_error
from app.ui.table_save_view import remember_result
from core.analysis.contracts import AnalysisPlanV1
from core.analysis.engine import AnalysisEngine, load_project_source
from core.analysis.validation import AnalysisPlanError, validate_plan

ADVANCED_KEY = "advanced-mode"


def advanced_enabled() -> bool:
    return bool(st.session_state.get(ADVANCED_KEY))


def render_advanced_toggle() -> None:
    st.toggle(
        "Modo avanzado (analistas)",
        key=ADVANCED_KEY,
        help="Muestra controles manuales. Las validaciones y aprobaciones son las mismas.",
    )


def _example(profiles: dict[str, dict[str, str]]) -> dict[str, Any]:
    source, schema = next(iter(profiles.items()))
    group = next((c for c, k in schema.items() if k == "category"), next(iter(schema)))
    return {
        "sources": [source],
        "steps": [
            {
                "op": "group_by",
                "by": [group],
                "aggregations": [{"function": "count", "output_column": "casos"}],
            },
            {"op": "sort", "by": ["casos"], "descending": True},
        ],
        "output": {"chart": "bar", "x": group, "y": "casos"},
    }


def render_plan_editor(
    repository: Any,
    storage_root: Path,
    project_id: str,
    profiles: dict[str, dict[str, str]],
    names: dict[str, str],
    table_key: str,
) -> None:
    if not advanced_enabled() or not profiles:
        return
    with st.expander("Editor de cálculo (modo avanzado)"):
        st.caption(
            "Escribe el plan en JSON (`AnalysisPlanV1`). Se valida igual que los que arma la IA: "
            "operaciones permitidas, columnas que existen y tipos compatibles. No se ejecuta "
            "código. Fuentes disponibles: " + ", ".join(f"«{name}»" for name in profiles) + "."
        )
        remembered = st.session_state.get(table_key)
        initial = remembered["plan"] if remembered else _example(profiles)
        text = st.text_area(
            "Plan (JSON)",
            value=json.dumps(initial, ensure_ascii=False, indent=1),
            height=260,
            key=f"advanced-plan-{project_id}",
        )
        if not st.button("Validar y ejecutar", key=f"advanced-run-{project_id}"):
            return
        try:
            plan = AnalysisPlanV1.model_validate(json.loads(text))
            validate_plan(plan, profiles)
        except json.JSONDecodeError as error:
            st.error(
                f"El texto no es JSON válido (línea {error.lineno}, columna {error.colno})."
                "\n\n**Qué puedes hacer:** revisa comas, comillas y llaves."
            )
            return
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}"
                for item in error.errors()[:5]
            )
            st.error(
                f"El plan no cumple el formato: {details}.\n\n**Qué puedes hacer:** corrige "
                "esos campos; el ejemplo inicial muestra la forma esperada."
            )
            return
        except AnalysisPlanError as error:
            render_error(error, fallback_code="ANALYSIS_FAILED")
            return
        ids = {name: dataset_id for dataset_id, name in names.items()}
        try:
            sources = {
                name: load_project_source(repository, storage_root, ids[name])
                for name in plan.sources
            }
            result = AnalysisEngine().run(plan, sources)
        except AnalysisPlanError as error:
            render_error(error, fallback_code="ANALYSIS_FAILED")
            return
        render_analysis_result(result, plan.output)
        remember_result(table_key, plan, {name: ids[name] for name in plan.sources})
        st.caption("Puedes guardarlo como tabla en «Guardar este cálculo como tabla».")


def render_full_metrics(evaluation: dict[str, Any]) -> None:
    """V2-32: todas las métricas guardadas, para analistas; no cambia la capa guiada."""

    if not advanced_enabled():
        return
    import pandas as pd

    with st.expander("Métricas completas (modo avanzado)"):
        validation = evaluation.get("validation") or {}
        rows = []
        for name, values in validation.items():
            metrics = values.get("metrics", values) if isinstance(values, dict) else {}
            rows.append(
                {
                    "Alternativa": name,
                    **{
                        key: round(value, 4)
                        for key, value in metrics.items()
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    },
                }
            )
        if rows:
            st.markdown("**Validación (todas las alternativas)**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        final = evaluation.get("holdout") or evaluation.get("test") or {}
        if final:
            st.markdown("**Prueba final (alternativa elegida)**")
            st.json(final, expanded=False)
        points = evaluation.get("holdout_operating_points") or []
        if points:
            st.markdown(
                "**Puntos descriptivos de la prueba final; no se usan para elegir políticas**"
            )
            st.dataframe(pd.DataFrame(points), hide_index=True, width="stretch")
        policy = evaluation.get("policy_selection")
        if policy:
            st.markdown("**Selección de política sobre validación**")
            st.caption(
                "Exploración sobre datos usados para seleccionar. Cambiar costos o capacidad "
                "no hereda una evaluación independiente de la política anterior."
            )
            st.json(policy, expanded=False)
        diagnostics = evaluation.get("probability_diagnostics") or []
        if diagnostics:
            st.markdown("**Diagnóstico de probabilidades**")
            st.caption(
                "Estos diagnósticos no certifican calibración. El puntaje permite priorizar; "
                "no se multiplica por dinero como probabilidad validada."
            )
            st.json(diagnostics, expanded=False)
        importance = evaluation.get("permutation_importance") or []
        if importance:
            st.markdown("**Importancia por permutación (validación)**")
            st.dataframe(pd.DataFrame(importance), hide_index=True, width="stretch")
