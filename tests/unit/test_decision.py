"""V2-13: decidir con un modelo sí/no por capacidad o por costo; niveles de riesgo."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.ml_engine import MLValidationError
from core.ml_engine.decision import by_capacity, by_cost, risk_levels
from core.ml_engine.evaluation import operating_points
from core.ml_engine.models import EvaluationSummaryV1
from tests.support.ml_workflow import prepared_strategy


@pytest.fixture
def evaluation(tmp_path: Path) -> EvaluationSummaryV1:
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="pytest"
    )
    result = service.execute(strategy_id)
    row = connection.execute(
        "SELECT candidate_metrics_json FROM runs WHERE id = ?", (result.run_id,)
    ).fetchone()
    connection.close()
    return EvaluationSummaryV1.model_validate(json.loads(row[0]))


def test_points_mark_more_cases_as_the_share_grows() -> None:
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 400)
    probability = np.clip(truth * 0.4 + rng.random(400) * 0.6, 0, 1)
    points = operating_points(truth, probability)

    flagged = [point.flagged for point in points]
    assert flagged == sorted(flagged) and points[-1].flagged == 400
    assert all(p.true_alerts + p.false_alarms == p.flagged for p in points)
    assert points[0].flagged == 0  # no actuar también es una alternativa
    assert points[1].true_alerts / points[1].flagged > truth.mean()  # lo de arriba es más rico


def test_capacity_is_respected(evaluation) -> None:
    plan = by_capacity(evaluation, cases_per_period=20, expected_cases_per_period=200)

    assert plan.to_attend <= 20
    assert plan.expected_true_alerts + plan.expected_false_alarms == pytest.approx(
        plan.to_attend, abs=1
    )


def test_expensive_misses_lower_the_cut(evaluation) -> None:
    cheap = by_cost(evaluation, cost_false_alarm=1, cost_missed=1, expected_cases_per_period=100)
    expensive = by_cost(
        evaluation, cost_false_alarm=1, cost_missed=20, expected_cases_per_period=100
    )

    assert expensive.flagged_share >= cheap.flagged_share
    assert "no dinero" in expensive.explanation


def test_levels_cover_every_case(evaluation) -> None:
    levels = risk_levels(evaluation)

    assert [level.level for level in levels.levels] == ["alto", "medio", "bajo"]
    assert sum(level.share_of_cases for level in levels.levels) == pytest.approx(1.0)
    assert levels.levels[0].event_rate >= levels.base_rate


def test_older_runs_explain_that_they_need_retraining(evaluation) -> None:
    old = evaluation.model_copy(update={"policy_selection": None})
    with pytest.raises(MLValidationError) as error:
        by_capacity(old, cases_per_period=1, expected_cases_per_period=10)
    assert error.value.code == "NO_OPERATING_POINTS"
