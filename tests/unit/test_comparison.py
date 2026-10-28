"""EXT-B: comparar experimentos cuando cambia una decisión de negocio."""

import json
from pathlib import Path

from core.domain.context_service import BusinessContextService
from core.domain.strategy_service import StrategyService
from core.reporting.comparison import context_changes, list_experiments
from tests.support.ml_workflow import prepared_strategy

ACTOR = "pytest-usuario"


def _train(repository, ml, strategy_id: str) -> str:
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label=ACTOR
    )
    return ml.execute(strategy_id).run_id


def test_changing_the_business_priority_is_visible_side_by_side(tmp_path: Path) -> None:
    connection, repository, ml, strategy_id, storage = prepared_strategy(tmp_path)
    first_run = _train(repository, ml, strategy_id)

    strategy = repository.get_strategy(strategy_id)
    first_context = repository.get_context(strategy["context_version_id"])
    payload = json.loads(first_context["context_json"])["context"]
    payload["primary_metric"] = "precision"
    payload["metric_rationale"] = "Ahora importa más que las alertas acierten"
    for decision in payload["feature_decisions"]:
        if decision["column"] == "noise":
            decision["use"] = "exclude"
    project_id = first_context["project_id"]
    second_context = BusinessContextService(repository, storage).create_version(
        project_id=project_id, dataset_id=first_context["dataset_id"], context=payload
    )
    second_strategy = StrategyService(repository).create_template_version(
        project_id=project_id, context_version_id=second_context.context_version_id
    )
    second_run = _train(repository, ml, second_strategy.strategy_version_id)

    experiments = list_experiments(repository, project_id)
    assert [item.run_id for item in experiments] == [first_run, second_run]
    assert [item.is_current for item in experiments] == [False, True]
    assert [item.primary_metric for item in experiments] == ["f1", "precision"]
    assert "noise" in experiments[0].features_used
    assert "noise" not in experiments[1].features_used
    assert all(item.holdout_precision is not None for item in experiments)

    changes = {
        change.field: change for change in context_changes(repository, first_run, second_run)
    }
    assert (changes["primary_metric"].before, changes["primary_metric"].after) == (
        "f1",
        "precision",
    )
    assert changes["features"].before == ["noise"] and changes["features"].after == []
    connection.close()
