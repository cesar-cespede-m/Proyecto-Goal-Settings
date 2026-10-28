"""MV-05: aritmética de escenarios sintéticos, sin atribuir efecto real ni monetizar scores."""

from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal, Inexact, getcontext, localcontext

import pytest
from pydantic import ValidationError

from core.contracts.decision_value import EconomicScenarioVersionV1
from core.contracts.economics import EconomicEvaluationV1, EconomicScenarioV1
from core.domain.canonical import canonical_hash
from core.economics import (
    EconomicCalculationError,
    evaluate_scenario,
    evaluate_scenario_version,
)

MONEY_FIELDS = (
    "exposure_value",
    "gross_value",
    "action_cost",
    "fixed_cost_applied",
    "total_cost",
    "net_value",
)


def _scenario(**changes):
    return {
        "currency": "USD",
        "currency_minor_units": 2,
        "period": "mes sintético",
        "planned_cases": 1000,
        "capacity_cases": 200,
        "external_risk_rate": "0.40",
        "risk_rate_basis": "Tasa externa supuesta del grupo seleccionado, no score ML",
        "relative_effect": "0.25",
        "relative_effect_basis": "Reducción relativa hipotética; no efecto demostrado",
        "value_per_case": "35.50",
        "action_cost_per_case": "1.20",
        "fixed_cost": "18.00",
        "value_basis": "Margen o costo evitado unitario sintético, no ingreso bruto",
        **changes,
    }


def _evaluate(**changes):
    return evaluate_scenario(_scenario(**changes), confirmed_human=True)


def _version(scenario=None, confirmed=True):
    model = EconomicScenarioV1.model_validate(_scenario() if scenario is None else scenario)
    return EconomicScenarioVersionV1(
        id="scenario-fixture",
        capability_id="capability-fixture",
        version_no=1,
        sha256=canonical_hash(model),
        confirmed_human=confirmed,
        actor_label="Ana",
        created_at=datetime(2026, 10, 23, tzinfo=UTC),
        missing_fields=model.missing_fields(),
        scenario=model,
    )


def test_revenue_scenario_uses_at_risk_nonconversions_and_margin_not_total_revenue():
    result = _evaluate()

    assert result.status == "scenario_estimate"
    assert result.planned_cases == 1000 and result.selected_cases == 200
    assert result.expected_avoided_events == Decimal("20")
    assert result.exposure_value == Decimal("2840.00")
    assert result.gross_value == Decimal("710.00")
    assert result.action_cost == Decimal("240.00")
    assert result.fixed_cost_applied == Decimal("18.00")
    assert result.total_cost == Decimal("258.00")
    assert result.net_value == Decimal("452.00")
    assert result.evidence_type == "synthetic_scenario"
    assert "SYNTHETIC_SCENARIO_NOT_OBSERVED_IMPACT" in result.warnings
    assert "RELATIVE_EFFECT_IS_AN_ASSUMPTION_NOT_CAUSAL_EVIDENCE" in result.warnings


def test_savings_scenario_applies_relative_reduction_to_external_risk():
    result = _evaluate(
        planned_cases=100,
        capacity_cases=None,
        external_risk_rate="0.20",
        relative_effect="0.50",
        value_per_case="120",
        action_cost_per_case="2",
        fixed_cost="10",
    )

    assert result.selected_cases == 100
    assert result.expected_avoided_events == Decimal("10")
    assert result.exposure_value == Decimal("2400.00")
    assert result.gross_value == Decimal("1200.00")
    assert result.total_cost == Decimal("210.00")
    assert result.net_value == Decimal("990.00")


def test_policy_may_select_fewer_cases_and_uses_risk_declared_for_that_group():
    result = evaluate_scenario(_scenario(), selected_cases=50, confirmed_human=True)

    assert result.selected_cases == 50
    assert result.expected_avoided_events == Decimal("5")
    assert result.exposure_value == Decimal("710.00")
    assert result.gross_value == Decimal("177.50")
    assert result.action_cost == Decimal("60.00")
    assert result.net_value == Decimal("99.50")


@pytest.mark.parametrize(
    "changes,selected", [({"planned_cases": 0}, None), ({"capacity_cases": 0}, None), ({}, 0)]
)
def test_no_action_has_zero_exposure_benefit_and_all_costs_including_fixed(changes, selected):
    result = evaluate_scenario(_scenario(**changes), selected_cases=selected, confirmed_human=True)

    assert result.selected_cases == 0
    assert result.expected_avoided_events == 0
    assert all(getattr(result, field) == 0 for field in MONEY_FIELDS)
    assert all(str(getattr(result, field)) == "0.00" for field in MONEY_FIELDS)


def test_capacity_can_exceed_volume_without_inventing_extra_cases():
    assert _evaluate(planned_cases=10, capacity_cases=20).selected_cases == 10


@pytest.mark.parametrize("selected", [-1, 201, 1001, True, 1.0, "1", Decimal("1")])
def test_invalid_policy_quantity_is_rejected(selected):
    with pytest.raises(EconomicCalculationError) as error:
        evaluate_scenario(_scenario(), selected_cases=selected, confirmed_human=True)
    assert error.value.code == "INVALID_SELECTED_CASES"


def test_selected_quantity_cannot_exceed_volume_even_without_capacity_limit():
    with pytest.raises(EconomicCalculationError) as error:
        evaluate_scenario(
            _scenario(planned_cases=10, capacity_cases=None),
            selected_cases=11,
            confirmed_human=True,
        )
    assert error.value.code == "INVALID_SELECTED_CASES"


def test_exposure_is_distinct_from_recoverable_benefit_and_zero_effect_can_lose_money():
    result = _evaluate(relative_effect="0")

    assert result.exposure_value == Decimal("2840.00")
    assert result.gross_value == 0
    assert result.net_value == Decimal("-258.00")


def test_unconfirmed_or_incomplete_scenarios_produce_no_numeric_estimates():
    unconfirmed = evaluate_scenario(_scenario())
    incomplete = evaluate_scenario(
        _scenario(external_risk_rate=None, relative_effect_basis=None), confirmed_human=True
    )
    for result in (unconfirmed, incomplete):
        assert result.status == "blocked"
        assert result.selected_cases is None
        assert result.expected_avoided_events is None
        assert all(getattr(result, field) is None for field in MONEY_FIELDS)
    assert "HUMAN_CONFIRMATION_REQUIRED" in unconfirmed.warnings
    assert set(incomplete.missing_fields) == {"external_risk_rate", "relative_effect_basis"}


def test_zero_volume_does_not_supply_missing_assumptions_or_confirmation():
    result = evaluate_scenario({"planned_cases": 0}, confirmed_human=True)
    assert result.status == "blocked"
    assert "external_risk_rate" in result.missing_fields
    assert result.net_value is None


def test_known_zero_rates_and_costs_are_not_missing_data():
    result = _evaluate(
        external_risk_rate="0",
        relative_effect="0",
        value_per_case="0",
        action_cost_per_case="0",
        fixed_cost="0",
    )
    assert result.status == "scenario_estimate"
    assert result.missing_fields == []
    assert result.net_value == 0


@pytest.mark.parametrize(
    "field",
    [
        "external_risk_rate",
        "relative_effect",
        "value_per_case",
        "action_cost_per_case",
        "fixed_cost",
    ],
)
@pytest.mark.parametrize(
    "invalid", [0.1, Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1"), True]
)
def test_float_nonfinite_negative_and_boolean_values_cannot_enter_arithmetic(field, invalid):
    with pytest.raises(ValidationError):
        _evaluate(**{field: invalid})


@pytest.mark.parametrize(
    "extra",
    [
        {"score": "0.5"},
        {"probability": "0.5"},
        {"scenario_kind": "observed"},
        {"risk_rate_scope": "whole_population"},
    ],
)
def test_scores_observed_claims_and_population_rate_substitution_are_rejected(extra):
    with pytest.raises(ValidationError):
        _evaluate(**extra)


def test_gross_benefit_uses_unrounded_exposure_and_final_components_are_consistent():
    result = _evaluate(
        planned_cases=1,
        external_risk_rate="1",
        relative_effect="0.25",
        value_per_case="0.025",
        action_cost_per_case="0",
        fixed_cost="0",
    )
    # 0.025 × 0.25 = 0.00625 -> 0.01. Redondear exposición antes daría 0.005 -> 0.00.
    assert result.exposure_value == Decimal("0.02")
    assert result.gross_value == Decimal("0.01")
    assert result.net_value == result.gross_value - result.total_cost
    assert result.total_cost == result.action_cost + result.fixed_cost_applied


def test_money_components_round_half_even_at_declared_currency_precision():
    result = _evaluate(
        planned_cases=1,
        external_risk_rate="1",
        relative_effect="1",
        value_per_case="0.005",
        action_cost_per_case="0.015",
        fixed_cost="0.025",
    )
    assert result.gross_value == Decimal("0.00")
    assert result.action_cost == result.fixed_cost_applied == Decimal("0.02")
    assert result.total_cost == Decimal("0.04")
    assert result.net_value == Decimal("-0.04")


@pytest.mark.parametrize("minor_units,expected", [(0, "0"), (2, "0.12"), (4, "0.1235")])
def test_currency_precision_is_explicit_and_does_not_depend_on_locale(minor_units, expected):
    result = _evaluate(
        planned_cases=1,
        external_risk_rate="1",
        relative_effect="1",
        value_per_case="0.123456",
        action_cost_per_case="0",
        fixed_cost="0",
        currency_minor_units=minor_units,
    )
    assert str(result.net_value) == expected


def test_results_and_hash_do_not_depend_on_the_callers_decimal_context():
    expected = _evaluate()
    with localcontext() as hostile:
        hostile.prec = 2
        hostile.rounding = ROUND_UP
        hostile.traps[Inexact] = True
        actual = _evaluate()
        assert getcontext().prec == 2
        assert getcontext().rounding == ROUND_UP
        assert getcontext().traps[Inexact]
    assert actual == expected
    assert EconomicEvaluationV1.model_validate_json(actual.model_dump_json()) == expected
    assert all(isinstance(getattr(actual, field), Decimal) for field in MONEY_FIELDS)


def test_contract_maxima_remain_exact_without_overflow_or_float_conversion():
    result = _evaluate(
        planned_cases=1_000_000_000,
        capacity_cases=None,
        external_risk_rate="1",
        relative_effect="1",
        value_per_case="1000000000000",
        action_cost_per_case="1000000000000",
        fixed_cost="1000000000000",
    )
    assert result.gross_value == Decimal("1000000000000000000000.00")
    assert result.net_value == Decimal("-1000000000000.00")


def test_equivalent_decimal_representations_have_identical_scenario_hashes():
    text = _evaluate()
    decimal = _evaluate(value_per_case=Decimal("35.500000"), external_risk_rate=Decimal("0.4"))
    assert text == decimal


@pytest.mark.parametrize("hash_value", ["invalid", "0" * 64, 123])
def test_supplied_scenario_hash_must_match_the_exact_declared_assumptions(hash_value):
    with pytest.raises(EconomicCalculationError) as error:
        evaluate_scenario(_scenario(), confirmed_human=True, scenario_sha256=hash_value)
    assert error.value.code == "INVALID_SCENARIO_HASH"


def test_version_adapter_preserves_confirmation_and_rejects_changed_assumptions():
    version = _version()
    actual = evaluate_scenario_version(version, selected_cases=50)
    assert actual.scenario_sha256 == version.sha256
    assert actual.net_value == Decimal("99.50")
    assert evaluate_scenario_version(_version(confirmed=False)).status == "blocked"
    stale = version.model_copy(
        update={"scenario": EconomicScenarioV1.model_validate(_scenario(fixed_cost="19"))}
    )
    with pytest.raises(EconomicCalculationError) as error:
        evaluate_scenario_version(stale)
    assert error.value.code == "INVALID_SCENARIO_HASH"


def test_mutated_model_cannot_bypass_input_validation_or_truthy_confirmation():
    model = EconomicScenarioV1.model_validate(_scenario()).model_copy(update={"fixed_cost": 0.1})
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        with pytest.raises(ValidationError):
            evaluate_scenario(model, confirmed_human=True)
    with pytest.raises(EconomicCalculationError) as error:
        evaluate_scenario(_scenario(), confirmed_human="yes")
    assert error.value.code == "INVALID_CONFIRMATION"
