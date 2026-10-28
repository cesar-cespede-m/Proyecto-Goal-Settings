"""Pruebas ML-02 sobre fixtures sintéticos; no representan resultados de negocio."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.domain.strategy import StrategyProposalV1
from core.ml_engine import evaluate_candidates, train_candidates
from core.ml_engine.decision import by_capacity, by_cost
from core.ml_engine.models import EvaluationSummaryV1


def _proposal(
    *,
    positive_class: str | int = "yes",
    primary_metric: str = "f1",
    guardrails: dict[str, object] | None = None,
) -> StrategyProposalV1:
    return StrategyProposalV1(
        dataset_id="synthetic-fixture",
        dataset_sha256="a" * 64,
        context_version_id="synthetic-context",
        context_sha256="b" * 64,
        target_column="target",
        positive_class=positive_class,
        feature_columns=["signal", "noise", "category"],
        excluded_columns=["fixture_id"],
        primary_metric=primary_metric,
        guardrails=guardrails or {"mode": "no_quantitative_guardrail", "rules": []},
        split={"train": 0.6, "validation": 0.2, "test": 0.2},
        preprocessing={},
        candidates=[
            "DummyClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
        ],
        model_params={"random_forest": {"n_estimators": 10}},
        seed=42,
        caps={"max_training_seconds": 60},
        warnings=["PYTEST_SYNTHETIC_FIXTURE"],
    )


def _signal_frame(row_count: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(20261016)
    target = np.asarray(["no", "yes"] * (row_count // 2))
    target_binary = (target == "yes").astype(float)
    return pd.DataFrame(
        {
            "fixture_id": [f"fixture-{index}" for index in range(row_count)],
            "signal": target_binary + rng.normal(0, 0.02, row_count),
            "noise": rng.normal(0, 1, row_count),
            "category": np.where(target_binary == 1, "positive", "negative"),
            "target": target,
        }
    )


def test_validation_selects_deterministically_then_evaluates_holdout_once() -> None:
    proposal = _proposal()
    training = train_candidates(_signal_frame(), proposal)
    result = evaluate_candidates(training, proposal, permutation_repeats=3)

    assert result.summary.selection.selected_model == "LogisticRegression"
    assert result.summary.selection.outcome == "RECOMMENDED"
    assert result.summary.selection.secondary_metric == "balanced_accuracy"
    assert result.summary.holdout is not None
    assert result.summary.holdout_evaluations == 1
    assert result.summary.holdout.positive_class == "yes"
    assert result.summary.holdout.support == len(training.splits.X_test)
    assert result.summary.holdout.confusion_matrix == [[20, 0], [0, 20]]
    assert result.summary.permutation_importance_partition == "validation"
    assert result.summary.permutation_importance_repeats == 3
    assert {item.feature for item in result.summary.permutation_importance} == {
        "signal",
        "noise",
        "category",
    }


def test_holdout_values_do_not_change_validation_selection() -> None:
    proposal = _proposal()
    training = train_candidates(_signal_frame(), proposal)
    first = evaluate_candidates(training, proposal, permutation_repeats=2)
    training.splits.X_test.loc[:, "signal"] = 999_999.0
    training.splits.X_test.loc[:, "category"] = "opposite"
    second = evaluate_candidates(training, proposal, permutation_repeats=2)

    assert first.summary.validation == second.summary.validation
    assert first.summary.selection == second.summary.selection
    assert first.summary.selection.selected_model == second.summary.selection.selected_model


def test_holdout_labels_cannot_change_policy_artifact_or_decisions() -> None:
    proposal = _proposal()
    training = train_candidates(_signal_frame(), proposal)
    first = evaluate_candidates(training, proposal, permutation_repeats=1, bootstrap_resamples=20)
    # Mutar exclusivamente las etiquetas finales conserva el soporte de ambas clases.
    training.splits.y_test[:] = training.splits.y_test.map({"yes": "no", "no": "yes"})
    second = evaluate_candidates(training, proposal, permutation_repeats=1, bootstrap_resamples=20)

    assert first.summary.policy_selection == second.summary.policy_selection
    assert (
        first.summary.policy_selection.content_sha256
        == second.summary.policy_selection.compute_hash()
    )
    assert first.summary.holdout.f1 != second.summary.holdout.f1
    assert by_capacity(
        first.summary, cases_per_period=10, expected_cases_per_period=100
    ) == by_capacity(second.summary, cases_per_period=10, expected_cases_per_period=100)
    assert by_cost(
        first.summary, cost_false_alarm=2, cost_missed=5, expected_cases_per_period=100
    ) == by_cost(second.summary, cost_false_alarm=2, cost_missed=5, expected_cases_per_period=100)
    restored = EvaluationSummaryV1.model_validate_json(first.summary.model_dump_json())
    assert restored.policy_selection == first.summary.policy_selection
    assert restored.holdout_operating_points == []
    diagnostics = restored.probability_diagnostics
    assert [item.partition for item in diagnostics] == ["validation", "holdout"]
    assert all(item.allowed_use == "ranking_only" for item in diagnostics)
    assert all(item.calibration_fit == "none" for item in diagnostics)
    assert diagnostics[0] == second.summary.probability_diagnostics[0]
    assert diagnostics[1].brier_score != second.summary.probability_diagnostics[1].brier_score


def test_equal_to_baseline_is_valid_without_recommendation_and_skips_holdout() -> None:
    frame = pd.DataFrame(
        {
            "fixture_id": [f"fixture-{index}" for index in range(120)],
            "signal": [1.0] * 120,
            "noise": [0.0] * 120,
            "category": ["constant"] * 120,
            "target": [index % 2 for index in range(120)],
        }
    )
    proposal = _proposal(positive_class=1, primary_metric="balanced_accuracy")
    result = evaluate_candidates(train_candidates(frame, proposal), proposal)

    assert result.summary.selection.outcome == "WITHOUT_RECOMMENDATION"
    assert result.summary.selection.selected_model is None
    assert result.summary.holdout is None
    assert result.summary.holdout_evaluations == 0
    assert result.selected_pipeline is None
    assert "NO_MODEL_RECOMMENDED" in result.summary.warnings


def test_quantitative_guardrails_filter_candidates_without_inventing_thresholds() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "fixture_id": [f"fixture-{index}" for index in range(200)],
            "signal": rng.normal(size=200),
            "noise": rng.normal(size=200),
            "category": rng.choice(["a", "b", "c"], size=200),
            "target": ["no", "yes"] * 100,
        }
    )
    proposal = _proposal(
        guardrails={
            "mode": "quantitative",
            "rules": [
                {
                    "metric": "recall",
                    "operator": ">=",
                    "threshold": 1.0,
                    "rationale": "Umbral sintético definido por el fixture",
                }
            ],
        }
    )
    result = evaluate_candidates(train_candidates(frame, proposal), proposal)

    assert result.summary.selection.outcome == "WITHOUT_RECOMMENDATION"
    for candidate in ("LogisticRegression", "RandomForestClassifier"):
        assert not result.summary.validation[candidate].guardrails_passed
        assert result.summary.validation[candidate].failed_guardrails == ["recall>=1.0"]


def _imbalanced_frame(row_count: int = 2000, positive_rate: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    target = (rng.random(row_count) < positive_rate).astype(int)
    return pd.DataFrame(
        {
            "fixture_id": [f"fixture-{index}" for index in range(row_count)],
            "signal": target * 1.2 + rng.normal(0, 1, row_count),
            "noise": rng.normal(0, 1, row_count),
            "category": rng.choice(["a", "b"], row_count),
            "target": target,
        }
    )


def test_threshold_is_tuned_on_validation_and_frozen_for_holdout() -> None:
    frame = _imbalanced_frame()
    tuned = _proposal(positive_class=1)
    fixed = StrategyProposalV1.model_validate(
        {
            **tuned.model_dump(mode="json"),
            "threshold_policy": {"method": "fixed", "fixed_value": 0.5},
        }
    )
    tuned_result = evaluate_candidates(
        train_candidates(frame, tuned), tuned, permutation_repeats=2, bootstrap_resamples=50
    )
    fixed_result = evaluate_candidates(
        train_candidates(frame, fixed), fixed, permutation_repeats=2, bootstrap_resamples=50
    )

    tuned_lr = tuned_result.summary.validation["LogisticRegression"].metrics
    fixed_lr = fixed_result.summary.validation["LogisticRegression"].metrics
    assert tuned_lr.threshold_source == "validation_max_f1"
    assert fixed_lr.threshold_source == "fixed"
    assert fixed_lr.threshold == 0.5
    # Con 5 % de positivos, 0.5 fijo deja casi sin recall a la regresión logística.
    assert tuned_lr.recall > fixed_lr.recall
    assert tuned_lr.f1 >= fixed_lr.f1
    baseline = tuned_result.summary.validation["DummyClassifier"].metrics
    assert baseline.threshold_source == "baseline_default"
    assert baseline.threshold == 0.5

    selection = tuned_result.summary.selection
    holdout = tuned_result.summary.holdout
    assert selection.threshold_policy == "max_f1_on_validation"
    assert holdout is not None and selection.selected_model is not None
    frozen = tuned_result.summary.validation[selection.selected_model].metrics.threshold
    assert selection.selected_threshold == frozen == holdout.threshold
    assert "THRESHOLD_TUNED_ON_VALIDATION_HOLDOUT_UNTOUCHED" in tuned_result.summary.warnings


def test_holdout_bootstrap_is_deterministic_and_brackets_point_estimate() -> None:
    frame = _imbalanced_frame()
    proposal = _proposal(positive_class=1, primary_metric="average_precision")
    training = train_candidates(frame, proposal)
    first = evaluate_candidates(training, proposal, permutation_repeats=2, bootstrap_resamples=200)
    second = evaluate_candidates(training, proposal, permutation_repeats=2, bootstrap_resamples=200)

    bootstrap = first.summary.holdout_bootstrap
    assert bootstrap is not None
    assert bootstrap == second.summary.holdout_bootstrap
    assert bootstrap.resamples_requested == 200
    assert 0 < bootstrap.resamples_used <= 200
    assert [interval.metric for interval in bootstrap.intervals] == [
        "average_precision",
        "precision",
        "recall",
    ]
    for interval in bootstrap.intervals:
        assert interval.lower <= interval.point <= interval.upper


def test_low_positive_support_and_importance_scorer_are_declared() -> None:
    frame = _imbalanced_frame()
    proposal = _proposal(positive_class=1)
    result = evaluate_candidates(
        train_candidates(frame, proposal), proposal, permutation_repeats=2, bootstrap_resamples=20
    )

    assert result.summary.permutation_importance_scorer == "average_precision"
    assert "LOW_POSITIVE_SUPPORT_VALIDATION" in result.summary.warnings
    assert "LOW_POSITIVE_SUPPORT_HOLDOUT" in result.summary.warnings


@pytest.mark.parametrize(
    "policy",
    [{"method": "fixed"}, {"method": "max_f1_on_validation", "fixed_value": 0.3}],
)
def test_threshold_policy_rejects_incoherent_configuration(policy: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StrategyProposalV1.model_validate(
            {**_proposal().model_dump(mode="json"), "threshold_policy": policy}
        )
