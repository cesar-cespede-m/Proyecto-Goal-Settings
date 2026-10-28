"""GA-07a: registro de tipos de modelo; sí/no migrado sin cambio de comportamiento."""

import numpy as np
import pytest

from core.ml_engine import MLValidationError, evaluate_candidates, train_candidates
from core.ml_engine.task_kinds import (
    ESTIMATOR_STEP,
    TASK_KINDS,
    UNSUPPORTED_TASK_KIND,
    BinaryClassification,
    get_task_kind,
)
from tests.unit.test_ml_engine import _frame, _proposal


def test_registry_knows_binary_and_explains_unknown_kinds() -> None:
    assert set(TASK_KINDS) == {"binary_classification", "multiclass_classification", "regression"}
    kind = get_task_kind("binary_classification")
    assert isinstance(kind, BinaryClassification)
    assert kind.baseline == "DummyClassifier" and kind.higher_is_better

    with pytest.raises(MLValidationError) as error:
        get_task_kind("forecasting")
    assert error.value.code == UNSUPPORTED_TASK_KIND


def test_training_builds_the_same_candidates_through_the_registry() -> None:
    proposal = _proposal()
    session = train_candidates(_frame(), proposal)

    assert list(session.pipelines) == proposal.candidates
    expected = BinaryClassification().build_estimators(proposal, ["a"], ["b"])
    for name, pipeline in session.pipelines.items():
        estimator = pipeline.named_steps[ESTIMATOR_STEP]
        assert type(estimator) is type(expected[name][1])
        assert estimator.get_params() == expected[name][1].get_params()


def test_evaluation_and_scoring_match_the_previous_engine() -> None:
    proposal = _proposal()
    training = train_candidates(_frame(), proposal)
    kind = get_task_kind(proposal.task_type)

    through_registry = kind.evaluate(training, proposal, permutation_repeats=2)
    direct = evaluate_candidates(training, proposal, permutation_repeats=2)
    assert through_registry.summary == direct.summary

    pipeline = training.pipelines["LogisticRegression"]
    probability = kind.score(pipeline, training.splits.X_test, proposal)
    classes = pipeline.named_steps[ESTIMATOR_STEP].classes_.tolist()
    expected = pipeline.predict_proba(training.splits.X_test)[:, classes.index(1)]
    np.testing.assert_array_equal(probability, expected)
