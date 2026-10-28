"""Pruebas ML-01: split y fit reales, todavía sin evaluación ML-02."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.domain.strategy import StrategyProposalV1
from core.ml_engine import MLValidationError, split_dataset, train_candidates


def _frame(row_count: int = 120) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "fixture_id": [f"row-{index}" for index in range(row_count)],
            "numeric_signal": [float(index) for index in range(row_count)],
            "numeric_with_missing": [
                None if index % 11 == 0 else float(index % 9) for index in range(row_count)
            ],
            "category": ["a" if index % 3 else "b" for index in range(row_count)],
            "category_with_missing": [
                None if index % 13 == 0 else ("x" if index % 2 else "y")
                for index in range(row_count)
            ],
            "target": [index % 2 for index in range(row_count)],
        }
    )
    return frame


def _proposal(*, seed: int = 42) -> StrategyProposalV1:
    return StrategyProposalV1(
        dataset_id="fixture-dataset",
        dataset_sha256="a" * 64,
        context_version_id="fixture-context",
        context_sha256="b" * 64,
        target_column="target",
        positive_class=1,
        feature_columns=[
            "category",
            "category_with_missing",
            "numeric_signal",
            "numeric_with_missing",
        ],
        excluded_columns=["fixture_id"],
        primary_metric="recall",
        guardrails={"mode": "no_quantitative_guardrail", "rules": []},
        split={"train": 0.6, "validation": 0.2, "test": 0.2},
        preprocessing={},
        candidates=[
            "DummyClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
        ],
        model_params={"random_forest": {"n_estimators": 10}},
        seed=seed,
        caps={"max_training_seconds": 60},
        warnings=["PYTEST_SYNTHETIC_FIXTURE"],
    )


def test_split_is_stratified_disjoint_and_reproducible() -> None:
    frame = _frame()
    proposal = _proposal()
    first = split_dataset(frame, proposal)
    second = split_dataset(frame, proposal)

    assert (len(first.X_train), len(first.X_validation), len(first.X_test)) == (72, 24, 24)
    train_indices = set(first.X_train.index)
    validation_indices = set(first.X_validation.index)
    test_indices = set(first.X_test.index)
    assert not train_indices & validation_indices
    assert not train_indices & test_indices
    assert not validation_indices & test_indices
    assert train_indices | validation_indices | test_indices == set(frame.index)
    assert first.summary == second.summary
    assert [item.count for item in first.summary.train.class_distribution] == [36, 36]
    assert [item.count for item in first.summary.validation.class_distribution] == [12, 12]
    assert [item.count for item in first.summary.test.class_distribution] == [12, 12]


def test_different_seed_changes_partition_membership() -> None:
    frame = _frame()
    first = split_dataset(frame, _proposal(seed=42))
    second = split_dataset(frame, _proposal(seed=99))
    assert first.summary.train.index_sha256 != second.summary.train.index_sha256


def test_three_real_candidates_fit_same_training_partition() -> None:
    session = train_candidates(_frame(), _proposal())
    assert set(session.pipelines) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    }
    assert isinstance(
        session.pipelines["DummyClassifier"].named_steps["classifier"], DummyClassifier
    )
    assert isinstance(
        session.pipelines["LogisticRegression"].named_steps["classifier"], LogisticRegression
    )
    assert isinstance(
        session.pipelines["RandomForestClassifier"].named_steps["classifier"],
        RandomForestClassifier,
    )
    for pipeline in session.pipelines.values():
        predictions = pipeline.predict(session.splits.X_validation)
        assert len(predictions) == len(session.splits.X_validation)
        assert list(pipeline.feature_names_in_) == _proposal().feature_columns
    assert session.summary.status == "TRAINED_PENDING_EVALUATION"
    assert [item.candidate for item in session.summary.candidates] == _proposal().candidates
    assert "metrics" not in session.summary.model_dump()
    assert "HOLDOUT_NOT_EVALUATED_ML01" in session.summary.warnings


def test_preprocessors_learn_statistics_and_categories_only_from_train() -> None:
    frame = _frame()
    proposal = _proposal()
    initial_split = split_dataset(frame, proposal)
    outside_train = initial_split.X_validation.index.union(initial_split.X_test.index)
    frame.loc[outside_train, "numeric_signal"] = 1_000_000_000.0
    frame.loc[outside_train, "category"] = "never-seen-in-train"

    session = train_candidates(frame, proposal)
    logistic_preprocessor = session.pipelines["LogisticRegression"].named_steps["preprocessor"]
    numeric_pipeline = logistic_preprocessor.named_transformers_["numeric"]
    categorical_pipeline = logistic_preprocessor.named_transformers_["categorical"]
    numeric_columns = session.summary.numeric_columns
    signal_position = numeric_columns.index("numeric_signal")
    fitted_median = numeric_pipeline.named_steps["imputer"].statistics_[signal_position]
    expected_train_median = session.splits.X_train["numeric_signal"].median()
    full_data_median = frame["numeric_signal"].median()
    assert fitted_median == expected_train_median
    assert fitted_median != full_data_median

    encoder = categorical_pipeline.named_steps["encoder"]
    category_position = session.summary.categorical_columns.index("category")
    assert "never-seen-in-train" not in encoder.categories_[category_position]
    predictions = session.pipelines["LogisticRegression"].predict(session.splits.X_validation)
    assert len(predictions) == 24


def test_seed_and_bounded_parameters_reach_estimators() -> None:
    session = train_candidates(_frame(), _proposal(seed=77))
    logistic = session.pipelines["LogisticRegression"].named_steps["classifier"]
    forest = session.pipelines["RandomForestClassifier"].named_steps["classifier"]
    dummy = session.pipelines["DummyClassifier"].named_steps["classifier"]
    assert logistic.random_state == 77
    assert logistic.max_iter == 1000
    assert forest.random_state == 77
    assert forest.n_estimators == 10
    assert forest.n_jobs == 1
    assert dummy.strategy == "most_frequent"


def test_repeated_fit_is_deterministic_in_same_environment() -> None:
    first = train_candidates(_frame(), _proposal())
    second = train_candidates(_frame(), _proposal())
    for candidate in _proposal().candidates:
        assert np.array_equal(
            first.pipelines[candidate].predict(first.splits.X_validation),
            second.pipelines[candidate].predict(second.splits.X_validation),
        )


def test_training_time_cap_is_checked_between_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 0.0, 0.1, 11.0])
    monkeypatch.setattr("core.ml_engine.training.perf_counter", lambda: next(ticks))
    proposal_value = _proposal().model_dump(mode="json")
    proposal_value["caps"]["max_training_seconds"] = 10
    with pytest.raises(MLValidationError) as captured:
        train_candidates(_frame(), StrategyProposalV1.model_validate(proposal_value))
    assert captured.value.code == "TRAINING_TIME_CAP_EXCEEDED"


@pytest.mark.parametrize(
    ("frame_mutation", "proposal_mutation", "expected_code"),
    [
        (lambda frame: frame.drop(columns=["numeric_signal"]), None, "TRAINING_COLUMN_NOT_FOUND"),
        (
            lambda frame: frame.assign(target=[index % 3 for index in range(len(frame))]),
            None,
            "TARGET_NOT_BINARY",
        ),
        (
            lambda frame: frame.assign(
                target=[None if index == 0 else index % 2 for index in range(len(frame))]
            ),
            None,
            "TARGET_HAS_MISSING",
        ),
        (
            lambda frame: frame,
            lambda value: value.update({"positive_class": "1"}),
            "POSITIVE_CLASS_NOT_FOUND",
        ),
    ],
)
def test_invalid_training_inputs_are_blocked(
    frame_mutation: object,
    proposal_mutation: object,
    expected_code: str,
) -> None:
    frame = frame_mutation(_frame())
    proposal_value = _proposal().model_dump(mode="json")
    if proposal_mutation is not None:
        proposal_mutation(proposal_value)
    proposal = StrategyProposalV1.model_validate(proposal_value)
    with pytest.raises(MLValidationError) as captured:
        split_dataset(frame, proposal)
    assert captured.value.code == expected_code


def test_too_few_minority_examples_cannot_be_split_three_ways() -> None:
    frame = _frame(20)
    frame["target"] = [1, 1, *([0] * 18)]
    with pytest.raises(MLValidationError) as captured:
        split_dataset(frame, _proposal())
    assert captured.value.code == "STRATIFIED_SPLIT_FAILED"
