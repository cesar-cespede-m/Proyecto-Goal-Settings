"""MV-03: política reproducible en validation, sin seleccionar sobre holdout."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from core.ml_engine import MLValidationError
from core.ml_engine.decision import (
    DecisionPlanV1,
    by_capacity,
    by_cost,
    require_probability_valuation,
    risk_levels,
    selected_row_numbers,
)
from core.ml_engine.evaluation import operating_points, probability_diagnostics
from core.ml_engine.models import (
    EvaluationSummaryV1,
    OperatingPointV1,
    PolicySelectionArtifactV1,
)


def _artifact(truth=None, scores=None, **changes):
    truth = np.asarray([0, 1] * 5 if truth is None else truth)
    scores = np.asarray(np.linspace(0.95, 0.05, len(truth)) if scores is None else scores)
    return PolicySelectionArtifactV1.build(
        **{
            "dataset_sha256": "a" * 64,
            "partition_sha256": "b" * 64,
            "selected_model": "LogisticRegression",
            "positive_class": 1,
            "points": operating_points(truth, scores),
            **changes,
        }
    )


def _evaluation(artifact=None, **changes):
    validation = {}
    if artifact is not None:
        support, positives = artifact.points[0].support, artifact.points[0].positives
        validation[artifact.selected_model] = {
            "candidate": artifact.selected_model,
            "metrics": {
                "threshold": 0.5,
                "threshold_source": "fixed",
                "positive_class": 1,
                "negative_class": 0,
                "support": support,
                "positive_support": positives,
                "negative_support": support - positives,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "balanced_accuracy": 0.5,
                "average_precision": positives / support,
                "roc_auc": 0.5,
                "confusion_matrix": [[support - positives, 0], [positives, 0]],
            },
            "primary_metric_value": 0,
            "guardrails_passed": True,
            "failed_guardrails": [],
        }
    return EvaluationSummaryV1.model_validate(
        {
            "validation": validation,
            "selection": {
                "primary_metric": "f1",
                "baseline_value": 0.1,
                "eligible_candidates": ["LogisticRegression"],
                "selected_model": "LogisticRegression",
                "outcome": "RECOMMENDED",
                "reason": "Fixture sintético para comprobar políticas",
                "secondary_metric": "balanced_accuracy",
                "threshold_policy": "fixed",
                "selected_threshold": 0.5,
            },
            "holdout": None,
            "holdout_evaluations": 0,
            "permutation_importance_partition": None,
            "permutation_importance": [],
            "warnings": [],
            "policy_selection": artifact,
            **changes,
        }
    )


def test_capacity_zero_means_no_action_even_when_every_score_is_one():
    evaluation = _evaluation(_artifact(scores=np.ones(10)))
    plan = by_capacity(evaluation, cases_per_period=0, expected_cases_per_period=10)

    assert plan.to_attend == 0
    assert plan.expected_true_alerts == plan.expected_false_alarms == 0
    assert selected_row_numbers(plan, scores=[1.0] * 10, row_numbers=list(range(1, 11))) == set()


def test_zero_planned_cases_yields_zero_actions_without_dividing_by_zero():
    plan = by_capacity(_evaluation(_artifact()), cases_per_period=0, expected_cases_per_period=0)
    assert plan.to_attend == 0
    assert plan.expected_cases == 0
    assert plan.selection_fraction == 0


def test_tied_scores_take_exact_capacity_using_original_row_order():
    evaluation = _evaluation(_artifact(scores=np.full(10, 0.8)))
    plan = by_capacity(evaluation, cases_per_period=2, expected_cases_per_period=10)
    chosen = selected_row_numbers(plan, scores=[0.8] * 10, row_numbers=list(range(1, 11)))

    assert chosen == {1, 2}
    # La vista puede reordenar casos; el desempate sigue usando su posición original.
    assert (
        selected_row_numbers(plan, scores=[0.8] * 10, row_numbers=[10, 5, 3, 9, 2, 6, 1, 4, 8, 7])
        == chosen
    )


def test_capacity_below_validation_resolution_does_not_promote_zero_count_fraction():
    evaluation = _evaluation(_artifact())
    plan = by_capacity(evaluation, cases_per_period=1, expected_cases_per_period=1000)
    chosen = selected_row_numbers(
        plan, scores=np.linspace(1.0, 0.0, 1000), row_numbers=list(range(1, 1001))
    )

    assert plan.to_attend <= 1
    assert len(chosen) <= 1
    assert plan.to_attend == len(chosen)


def test_validation_operating_counts_match_actual_top_fraction_action():
    truth = np.asarray([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    scores = np.asarray([0.9, 0.9, 0.9, 0.6, 0.6, 0.5, 0.4, 0.4, 0.2, 0.1])
    evaluation = _evaluation(_artifact(truth, scores))
    plan = by_capacity(evaluation, cases_per_period=3, expected_cases_per_period=10)
    rows = selected_row_numbers(plan, scores=scores, row_numbers=list(range(1, 11)))

    assert rows == {1, 2, 3}
    positions = [row - 1 for row in rows]
    assert plan.to_attend == len(rows)
    assert plan.expected_true_alerts == sum(truth[positions])
    assert plan.expected_false_alarms == len(rows) - sum(truth[positions])
    assert plan.expected_missed == sum(truth) - sum(truth[positions])


def test_decimal_fraction_does_not_lose_a_case_to_binary_float_roundoff():
    points = operating_points(np.asarray([0, 1] * 45), np.linspace(0.99, 0.01, 90))
    point = next(item for item in points if item.selection_fraction == 0.7)
    assert point.flagged == 63


def test_cost_policy_can_choose_no_action_instead_of_a_forced_alert():
    evaluation = _evaluation(_artifact(truth=[0] * 5 + [1] * 5))
    plan = by_cost(
        evaluation,
        cost_false_alarm=1000,
        cost_missed=1,
        expected_cases_per_period=10,
    )

    assert plan.to_attend == 0
    assert plan.expected_false_alarms == 0
    assert plan.expected_missed == 5


@pytest.mark.parametrize(
    "truth,true_alerts,false_alarms,missed",
    [([1, 0] * 5, 15, 0, 35), ([1] + [0] * 9, 10, 5, 0)],
)
def test_scaled_counts_conserve_projected_positives_despite_fraction_rounding(
    truth, true_alerts, false_alarms, missed
):
    evaluation = _evaluation(_artifact(truth=truth))
    plan = by_capacity(evaluation, cases_per_period=15, expected_cases_per_period=100)

    assert plan.to_attend == 15
    assert plan.expected_true_alerts == true_alerts
    assert plan.expected_false_alarms == false_alarms
    assert plan.expected_missed == missed
    assert plan.expected_true_alerts + plan.expected_false_alarms == plan.to_attend
    assert plan.expected_true_alerts + plan.expected_missed == sum(truth) * 10


def test_policy_artifact_is_self_verifying_and_binds_dataset_partition_and_model():
    initial = _artifact()
    assert initial.content_sha256 == initial.compute_hash()
    assert PolicySelectionArtifactV1.model_validate_json(initial.model_dump_json()) == initial
    for change in (
        {"dataset_sha256": "c" * 64},
        {"partition_sha256": "d" * 64},
        {"selected_model": "RandomForestClassifier"},
    ):
        assert _artifact(**change).content_sha256 != initial.content_sha256
        with pytest.raises(ValidationError):
            PolicySelectionArtifactV1.model_validate({**initial.model_dump(), **change})


def test_artifact_hash_changes_with_validation_predictions_or_results():
    initial = _artifact()
    assert _artifact(scores=np.linspace(0.8, 0.2, 10)).content_sha256 != initial.content_sha256
    assert _artifact(truth=[1, 0] * 5).content_sha256 != initial.content_sha256


@pytest.mark.parametrize("positive_class", [0, True])
def test_valid_hash_cannot_substitute_another_positive_class(positive_class):
    evaluation = _evaluation(_artifact(positive_class=positive_class))
    with pytest.raises(MLValidationError) as error:
        by_capacity(evaluation, cases_per_period=2, expected_cases_per_period=10)
    assert error.value.code == "INVALID_POLICY_ARTIFACT"


def test_valid_hash_cannot_substitute_a_different_validation_cohort_support():
    evaluation = _evaluation(_artifact()).model_copy(
        update={"policy_selection": _artifact(truth=[0, 1] * 6)}
    )
    with pytest.raises(MLValidationError) as error:
        by_capacity(evaluation, cases_per_period=2, expected_cases_per_period=10)
    assert error.value.code == "INVALID_POLICY_ARTIFACT"


def _legacy_points():
    return [
        OperatingPointV1(
            flagged_share=0.5,
            threshold=0.5,
            flagged=5,
            true_alerts=5,
            false_alarms=0,
            missed=0,
            support=10,
            positives=5,
        )
    ]


@pytest.mark.parametrize("operation", ["capacity", "cost", "risk_levels"])
def test_legacy_holdout_points_never_substitute_missing_validation_artifact(operation):
    legacy = _evaluation(holdout_operating_points=_legacy_points())
    with pytest.raises(MLValidationError) as error:
        if operation == "capacity":
            by_capacity(legacy, cases_per_period=5, expected_cases_per_period=10)
        elif operation == "cost":
            by_cost(legacy, cost_false_alarm=1, cost_missed=1, expected_cases_per_period=10)
        else:
            risk_levels(legacy)
    assert error.value.code == "NO_OPERATING_POINTS"


def test_changing_legacy_holdout_points_cannot_change_a_validation_policy():
    evaluation = _evaluation(_artifact())
    changed = _evaluation(evaluation.policy_selection, holdout_operating_points=_legacy_points())
    assert by_capacity(evaluation, cases_per_period=5, expected_cases_per_period=10) == by_capacity(
        changed, cases_per_period=5, expected_cases_per_period=10
    )
    assert by_cost(
        evaluation, cost_false_alarm=2, cost_missed=5, expected_cases_per_period=10
    ) == by_cost(changed, cost_false_alarm=2, cost_missed=5, expected_cases_per_period=10)


def test_old_threshold_plan_cannot_be_applied_as_a_top_fraction_policy():
    old_fields = {
        "basis",
        "flagged_share",
        "threshold",
        "expected_cases",
        "to_attend",
        "expected_true_alerts",
        "expected_false_alarms",
        "expected_missed",
        "precision",
        "recall",
        "uncertainty",
        "explanation",
    }
    plan = by_capacity(_evaluation(_artifact()), cases_per_period=2, expected_cases_per_period=10)
    old = DecisionPlanV1.model_validate(
        {name: value for name, value in plan.model_dump().items() if name in old_fields}
    )
    with pytest.raises(MLValidationError) as error:
        selected_row_numbers(old, scores=[0.8] * 10, row_numbers=list(range(1, 11)))
    assert error.value.code == "LEGACY_DECISION_PLAN"


@pytest.mark.parametrize(
    "repetitions,status", [(10, "insufficient_support"), (100, "diagnostic_only")]
)
def test_probability_diagnostics_do_not_approve_monetization_for_perfect_ranking(
    repetitions, status
):
    truth = np.asarray([0, 1] * repetitions)
    scores = np.where(truth == 1, 0.9, 0.1)
    diagnostics = probability_diagnostics(
        truth, scores, partition="validation", baseline_probability=0.5
    )
    # Diagnosticar una muestra no demuestra calibración transportable ni valor monetario.
    evaluation = _evaluation(_artifact(truth, scores), probability_diagnostics=[diagnostics])
    with pytest.raises(MLValidationError) as error:
        require_probability_valuation(evaluation)
    assert error.value.code == "PROBABILITY_VALUATION_NOT_VERIFIED"
    assert diagnostics.allowed_use == "ranking_only"
    assert diagnostics.calibration_fit == "none"
    assert diagnostics.status == status
    assert diagnostics.brier_score == pytest.approx(0.01)
    assert diagnostics.log_loss == pytest.approx(-math.log(0.9))
    assert diagnostics.prevalence_baseline_brier == pytest.approx(0.25)
    assert diagnostics.prevalence_baseline_log_loss == pytest.approx(-math.log(0.5))
    assert sum(item.support for item in diagnostics.reliability_bins) == len(truth)
    assert [item.observed_rate for item in diagnostics.reliability_bins] == [0, 1]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_predictions_cannot_create_diagnostics_or_policy_evidence(invalid):
    scores = np.asarray([0.1, invalid])
    truth = np.asarray([0, 1])
    with pytest.raises(MLValidationError):
        operating_points(truth, scores)
    with pytest.raises(MLValidationError):
        probability_diagnostics(truth, scores, partition="validation", baseline_probability=0.5)
