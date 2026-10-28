"""MV-10: políticas versionadas sobre ML sintético real y supuestos declarados."""

import pytest

from core.domain.gates import has_training_approval
from core.domain.value_policy import ValuePolicyError, ValuePolicyService
from core.orchestration.runtime import Supervisor
from tests.unit.test_decision_value import _economic
from tests.unit.test_modeling_agent import _approve, _propose

pytest_plugins = ["tests.unit.test_modeling_agent"]


def _trained(state):
    item = state.prepare()
    _supervisor, _task, strategy_id = _propose(state, item)
    _approve(state, strategy_id)  # Fixture de prueba, no aprobación humana real.
    result = state.service.execute(strategy_id)
    assert result.status == "COMPLETED"
    return item, strategy_id, result.run_id


def _economy(state, item, *, operation="economy", base=None, **changes):
    return state.values.create_economic_version(
        item.cap.id,
        scenario=_economic(**changes),
        based_on_version_id=base,
        confirmed_human=True,
        actor_label="pytest-synthetic-reviewer",
        operation_id=operation,
    )


def test_policy_is_validation_exploratory_and_new_cost_requires_new_review(state):
    item, strategy_id, run_id = _trained(state)
    first_economy = _economy(state, item)
    service = ValuePolicyService(state.workflow)
    policy = service.propose(
        item.cap.id,
        cycle_id=item.cycle.id,
        run_id=run_id,
        economic_version_id=first_economy.id,
        actor_label="pytest",
        operation_id="policy-1",
    )
    assert policy.content.plan.evidence_partition == "validation"
    assert policy.content.plan.selection_artifact_sha256 == policy.content.selection_artifact_sha256
    assert policy.content.final_policy_evidence is None
    assert policy.content.economic_evaluation.status == "scenario_estimate"
    assert (
        service.propose(
            item.cap.id,
            cycle_id=item.cycle.id,
            run_id=run_id,
            economic_version_id=first_economy.id,
            actor_label="pytest",
            operation_id="policy-1",
        )
        == policy
    )
    review = service.review(
        policy.id,
        approved_hash=policy.sha256,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        operation_id="review-1",
    )
    assert service.is_review_current(policy.id)
    assert (
        service.review(
            policy.id,
            approved_hash=policy.sha256,
            decision="APPROVED",
            actor_label="pytest-reviewer",
            operation_id="review-1",
        )
        == review
    )

    second_economy = _economy(
        state,
        item,
        operation="economy-2",
        base=first_economy.id,
        action_cost_per_case="300",
    )
    assert has_training_approval(state.connection, strategy_id)
    assert state.workflow.get_run(run_id)["status"] == "COMPLETED"
    assert not service.is_review_current(policy.id)
    with pytest.raises(ValuePolicyError, match="Confirma el escenario vigente"):
        service.propose(
            item.cap.id,
            cycle_id=item.cycle.id,
            run_id=run_id,
            economic_version_id=first_economy.id,
            based_on_version_id=policy.id,
            actor_label="pytest",
            operation_id="stale-economy",
        )
    revised = service.propose(
        item.cap.id,
        cycle_id=item.cycle.id,
        run_id=run_id,
        economic_version_id=second_economy.id,
        based_on_version_id=policy.id,
        actor_label="pytest",
        operation_id="policy-2",
    )
    assert revised.sha256 != policy.sha256
    assert service.latest_review(revised.id) is None
    assert revised.content.final_policy_evidence is None
    assert (
        revised.content.economic_evaluation.net_value
        != policy.content.economic_evaluation.net_value
    )


def test_policy_review_rejects_wrong_hash_and_operation_conflict(state):
    item, _strategy, run_id = _trained(state)
    economic = _economy(state, item)
    service = ValuePolicyService(state.workflow)
    policy = service.propose(
        item.cap.id,
        cycle_id=item.cycle.id,
        run_id=run_id,
        economic_version_id=economic.id,
        actor_label="pytest",
        operation_id="policy",
    )
    with pytest.raises(ValuePolicyError):
        service.review(
            policy.id,
            approved_hash="0" * 64,
            decision="APPROVED",
            actor_label="pytest-reviewer",
            operation_id="wrong-hash",
        )
    service.review(
        policy.id,
        approved_hash=policy.sha256,
        decision="REJECTED",
        actor_label="pytest-reviewer",
        operation_id="review",
    )
    assert not service.is_review_current(policy.id)
    with pytest.raises(ValuePolicyError, match="otro efecto"):
        service.review(
            policy.id,
            approved_hash=policy.sha256,
            decision="APPROVED",
            actor_label="pytest-reviewer",
            operation_id="review",
        )


def test_value_agent_proposes_without_review_and_replay_is_exact(state):
    item, _strategy, run_id = _trained(state)
    economic = _economy(state, item)
    supervisor = Supervisor(state.workflow, storage_root=state.storage)
    task = supervisor.create_task(
        capability_id=item.cap.id,
        cycle_id=item.cycle.id,
        role="value_evidence",
        objective="Revisar valor sintético",
        actor_label="pytest-agent",
        operation_id="value-task",
    )
    call = {
        "tool_name": "propose_value_policy",
        "arguments": {
            "run_id": run_id,
            "economic_version_id": economic.id,
            "based_on_version_id": None,
        },
    }
    result = supervisor.execute_tool(
        task.id,
        call,
        operation_id="propose-value",
        expected_revision=task.revision,
        actor_label="pytest-agent",
    )
    assert result.status == "COMPLETED", result
    policy_id = result.output_refs[0].id
    service = ValuePolicyService(state.workflow)
    assert service.latest_review(policy_id) is None
    replay = supervisor.execute_tool(
        task.id,
        call,
        operation_id="propose-value",
        expected_revision=task.revision,
        actor_label="pytest-agent",
    )
    assert replay == result
    assert len(service.list_versions(item.cap.id)) == 1
