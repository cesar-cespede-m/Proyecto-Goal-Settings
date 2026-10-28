"""V2-17: de la lista priorizada a una medición agregada de lo observado."""

import json
import sqlite3

import pytest

from core.domain.action_lists import ActionListService
from core.ml_engine.decision import DecisionPlanV1
from tests.support.ml_workflow import prepared_strategy

ACTOR = "Responsable de prueba"
INPUT_HASH = "d" * 64


@pytest.fixture(scope="module")
def action_context(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("action-lists")
    connection, repository, runner, strategy_id, storage = prepared_strategy(root)
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label=ACTOR,
    )
    run = runner.execute(strategy_id)
    project_id = repository.get_run_project_id(run.run_id)
    yield connection, repository, ActionListService(repository, storage), project_id, run.run_id
    connection.close()


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:03d}" for index in range(count)]


def _create(
    service: ActionListService,
    project_id: str,
    run_id: str,
    *,
    prefix: str,
    cases: int,
    intended: int,
    control_share: float,
    seed: int = 2026,
):
    case_ids = _ids(prefix, cases)
    return service.create(
        project_id,
        run_id,
        scoring_input_sha256=INPUT_HASH,
        case_ids=case_ids,
        attended_ids=case_ids[:intended],
        control_share=control_share,
        seed=seed,
        plan=None,
        actor_label=ACTOR,
    )


def test_control_sizes_are_correct_reproducible_and_linked_to_decision(action_context) -> None:
    _, repository, service, project_id, run_id = action_context
    first = _create(
        service,
        project_id,
        run_id,
        prefix="repro-a",
        cases=100,
        intended=80,
        control_share=0.25,
    )
    second = _create(
        service,
        project_id,
        run_id,
        prefix="repro-a",
        cases=100,
        intended=80,
        control_share=0.25,
    )

    assert (len(first.attended_ids), len(first.control_ids), len(first.not_attended_ids)) == (
        60,
        20,
        20,
    )
    assert first.control_ids == second.control_ids
    assert not set(first.attended_ids) & set(first.control_ids)
    stored = next(
        item
        for item in repository.connection.execute(
            "SELECT record_json FROM decision_records WHERE id = ?", (first.decision_id,)
        )
    )
    decision = json.loads(stored["record_json"])
    assert decision["based_on"] == "prediction" and decision["reference_id"] == run_id
    assert decision["decision"].startswith("Atender 60 casos priorizados por el modelo")


def test_measurement_with_control_has_negative_difference_and_interval(action_context) -> None:
    _, _, service, project_id, run_id = action_context
    action = _create(
        service,
        project_id,
        run_id,
        prefix="effect",
        cases=120,
        intended=100,
        control_share=0.4,
    )
    outcomes = {
        **{case_id: int(index < 6) for index, case_id in enumerate(action.attended_ids)},
        **{case_id: int(index < 16) for index, case_id in enumerate(action.control_ids)},
        **{case_id: int(index < 4) for index, case_id in enumerate(action.not_attended_ids)},
    }

    measured = service.measure(action.id, outcomes=outcomes, positive_class=1, actor_label=ACTOR)

    assert measured.attended.proportion == pytest.approx(0.1)
    assert measured.control.proportion == pytest.approx(0.4)
    assert measured.attended_minus_control == pytest.approx(-0.3)
    assert measured.approximate_interval is not None
    assert measured.approximate_interval.lower < measured.attended_minus_control
    assert measured.approximate_interval.upper > measured.attended_minus_control
    assert "no prueba" in measured.summary


def test_without_control_is_explicit_and_unknown_ids_are_ignored(action_context) -> None:
    _, _, service, project_id, run_id = action_context
    action = _create(
        service,
        project_id,
        run_id,
        prefix="no-control",
        cases=12,
        intended=8,
        control_share=0,
    )
    outcomes = {case_id: index % 2 for index, case_id in enumerate(action.case_ids)}
    outcomes["not-from-this-list"] = 1

    measured = service.measure(action.id, outcomes=outcomes, positive_class=1, actor_label=ACTOR)

    assert measured.control.cases == 0
    assert measured.attended_minus_control is None and measured.approximate_interval is None
    assert measured.unknown_ids_ignored == 1
    assert "Sin grupo de control no se puede saber si actuar ayudó" in measured.summary
    assert "Se ignoraron 1 identificadores" in measured.summary


def test_small_control_comparison_warns(action_context) -> None:
    _, _, service, project_id, run_id = action_context
    action = _create(
        service,
        project_id,
        run_id,
        prefix="small",
        cases=24,
        intended=20,
        control_share=0.5,
    )
    measured = service.measure(
        action.id,
        outcomes={case_id: index % 2 for index, case_id in enumerate(action.case_ids)},
        positive_class=1,
        actor_label=ACTOR,
    )

    assert any("menos de 30" in warning for warning in measured.warnings)


def test_audit_has_counts_not_case_ids_and_storage_is_append_only(action_context) -> None:
    connection, _, service, project_id, run_id = action_context
    plan = DecisionPlanV1(
        basis="capacity",
        flagged_share=0.5,
        threshold=0.6,
        expected_cases=16,
        to_attend=8,
        expected_true_alerts=5,
        expected_false_alarms=3,
        expected_missed=2,
        precision=0.625,
        recall=0.714,
        uncertainty=0.2,
        explanation="Plan genérico de prueba.",
    )
    case_ids = _ids("private-case", 16)
    action = service.create(
        project_id,
        run_id,
        scoring_input_sha256=INPUT_HASH,
        case_ids=case_ids,
        attended_ids=case_ids[:8],
        control_share=0.25,
        seed=7,
        plan=plan,
        actor_label=ACTOR,
    )
    audit = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'ACTION_LIST_CREATED' "
        "AND entity_id = ?",
        (action.id,),
    ).fetchone()["detail_json"]
    assert json.loads(audit) == {
        "actor_label": ACTOR,
        "attended": 6,
        "cases": 16,
        "control": 2,
        "not_attended": 8,
    }
    assert all(case_id not in audit for case_id in case_ids)

    measured = service.measure(
        action.id,
        outcomes={case_id: index % 2 for index, case_id in enumerate(case_ids)},
        positive_class=1,
        actor_label=ACTOR,
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("UPDATE action_lists SET id = id WHERE id = ?", (action.id,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM action_list_outcomes WHERE id = ?", (measured.id,))
