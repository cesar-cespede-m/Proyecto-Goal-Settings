"""MV-05: exposición y beneficio supuesto con tasas externas del grupo seleccionado.

Un escenario no acredita ingresos, ahorro observado ni causalidad. El riesgo externo y la
reducción relativa son supuestos confirmados; un score ML no es una entrada de esta función.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from numbers import Integral
from typing import Any

from core.contracts.decision_value import EconomicScenarioVersionV1
from core.contracts.economics import EconomicEvaluationV1, EconomicScenarioV1
from core.domain.canonical import canonical_hash

_WARNINGS = (
    "SYNTHETIC_SCENARIO_NOT_OBSERVED_IMPACT",
    "RELATIVE_EFFECT_IS_AN_ASSUMPTION_NOT_CAUSAL_EVIDENCE",
    "EXTERNAL_RISK_APPLIES_TO_SELECTED_CASES",
    "MONEY_ROUNDED_HALF_EVEN_BY_COMPONENT",
)


class EconomicCalculationError(ValueError):
    """Una selección o referencia no corresponde al escenario declarado."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _selected_quantity(scenario: EconomicScenarioV1, selected_cases: int | None) -> int | None:
    if selected_cases is not None and (
        not isinstance(selected_cases, Integral)
        or isinstance(selected_cases, bool)
        or selected_cases < 0
    ):
        raise EconomicCalculationError(
            "INVALID_SELECTED_CASES", "Los casos seleccionados deben ser un entero no negativo"
        )
    limits = [
        value for value in (scenario.planned_cases, scenario.capacity_cases) if value is not None
    ]
    if selected_cases is not None and any(selected_cases > limit for limit in limits):
        raise EconomicCalculationError(
            "INVALID_SELECTED_CASES", "La selección supera el volumen o la capacidad declarados"
        )
    if scenario.planned_cases is None:
        return None
    return min(limits) if selected_cases is None else int(selected_cases)


def evaluate_scenario(
    scenario: EconomicScenarioV1 | dict[str, Any],
    *,
    selected_cases: int | None = None,
    confirmed_human: bool = False,
    scenario_sha256: str | None = None,
) -> EconomicEvaluationV1:
    """Calcula sólo un escenario completo y confirmado; no persiste ni decide acciones.

    ``planned_cases`` es el volumen del período. Por defecto se limita a la capacidad;
    una política puede proporcionar una cantidad menor. La tasa externa debe describir
    ese grupo seleccionado, no la población general por sustitución implícita.

    La exposición bruta se conserva hasta calcular el beneficio. Sólo al final se
    redondean componentes monetarios con HALF_EVEN; costos y neto suman los componentes
    visibles para evitar discrepancias de centavos. Si no se actúa, tampoco hay costo fijo.
    """

    # Volver a validar también instancias evita que model_copy/update salte las restricciones.
    payload = (
        scenario.model_dump(mode="python") if isinstance(scenario, EconomicScenarioV1) else scenario
    )
    model = EconomicScenarioV1.model_validate(payload)
    if not isinstance(confirmed_human, bool):
        raise EconomicCalculationError(
            "INVALID_CONFIRMATION", "La confirmación humana debe ser explícitamente booleana"
        )
    expected_hash = canonical_hash(model)
    if scenario_sha256 is not None and (
        not isinstance(scenario_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", scenario_sha256) is None
        or scenario_sha256 != expected_hash
    ):
        raise EconomicCalculationError(
            "INVALID_SCENARIO_HASH", "La huella no corresponde a los supuestos de este escenario"
        )
    quantity = _selected_quantity(model, selected_cases)
    missing = model.missing_fields()
    metadata = {
        "scenario_sha256": expected_hash,
        "currency": model.currency,
        "currency_minor_units": model.currency_minor_units,
        "period": model.period,
        "planned_cases": model.planned_cases,
    }
    if missing or not confirmed_human:
        return EconomicEvaluationV1(
            status="blocked",
            missing_fields=missing,
            warnings=[
                *_WARNINGS,
                *(["ECONOMIC_SCENARIO_INCOMPLETE"] if missing else []),
                *(["HUMAN_CONFIRMATION_REQUIRED"] if not confirmed_human else []),
            ],
            **metadata,
        )

    # El contrato acota volumen, importes y decimales: precisión 50 cubre sus productos
    # exactos. Un Context nuevo evita heredar precisión, redondeo o traps del llamador.
    assert quantity is not None and model.currency_minor_units is not None
    assert model.external_risk_rate is not None and model.relative_effect is not None
    assert model.value_per_case is not None and model.action_cost_per_case is not None
    assert model.fixed_cost is not None
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        count = Decimal(quantity)
        quantum = Decimal(1).scaleb(-model.currency_minor_units)
        avoided_events = count * model.external_risk_rate * model.relative_effect
        raw_exposure = count * model.external_risk_rate * model.value_per_case
        exposure = raw_exposure.quantize(quantum)
        gross = (raw_exposure * model.relative_effect).quantize(quantum)
        action_cost = (count * model.action_cost_per_case).quantize(quantum)
        fixed_cost = (model.fixed_cost if quantity else Decimal(0)).quantize(quantum)
        total_cost = action_cost + fixed_cost
        net_value = gross - total_cost
    return EconomicEvaluationV1(
        status="scenario_estimate",
        missing_fields=[],
        selected_cases=quantity,
        expected_avoided_events=avoided_events,
        exposure_value=exposure,
        gross_value=gross,
        action_cost=action_cost,
        fixed_cost_applied=fixed_cost,
        total_cost=total_cost,
        net_value=net_value,
        warnings=list(_WARNINGS),
        **metadata,
    )


def evaluate_scenario_version(
    version: EconomicScenarioVersionV1, *, selected_cases: int | None = None
) -> EconomicEvaluationV1:
    """Adapta la versión persistida sin sustituir su confirmación ni su huella."""

    if not isinstance(version, EconomicScenarioVersionV1):
        raise EconomicCalculationError(
            "INVALID_SCENARIO_VERSION", "Se requiere una versión económica declarada"
        )
    return evaluate_scenario(
        version.scenario,
        selected_cases=selected_cases,
        confirmed_human=version.confirmed_human,
        scenario_sha256=version.sha256,
    )
