"""MV-07: atomicidad entre llamadas, propuestas, reservas y despacho."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from core.contracts.agent_reasoning import AgentProposalV1, AgentTurnResultV1
from core.contracts.orchestration import ArtifactRefV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.db.repositories.agent_turns import AgentTurnRepository
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository
from core.domain.capabilities import CapabilityService

NOW = datetime(2026, 10, 23, 12, tzinfo=UTC)
CONTEXT = "a" * 64


@pytest.fixture
def setup(tmp_path):
    connection = get_connection(tmp_path / "turns.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(name="Sandbox", actor_label="Human", operation_id="p")
    capability = capabilities.create_capability(
        product.id,
        name="Activacion",
        objective="Hipotesis sintetica",
        value_kind="revenue",
        decision_unit="alta",
        actor_label="Human",
        operation_id="cap",
    )
    orchestration = OrchestrationRepository(workflow)
    task = orchestration.create_task(
        capability_id=capability.id,
        role="coordinator",
        context_sha256=CONTEXT,
        actor_label="Human",
        operation_id="task",
        now=NOW,
    )
    yield AgentTurnRepository(workflow), orchestration, task, capability
    connection.close()


def _claim(setup, operation="turn", **changes):
    turns, repo, task, _ = setup
    values = dict(
        operation_id=operation,
        expected_revision=repo.get_task(task.id).revision,
        actor_label="Human",
        owner="worker",
        provider="fake",
        model="fake-model",
        prompt_version="agent.v1",
        prompt_sha256="b" * 64,
        now=NOW,
        verify_current=lambda: CONTEXT,
    )
    values.update(changes)
    return turns.claim(task.id, **values)


def _finish(setup, turn, **changes):
    turns, *_ = setup
    values = dict(
        expected_revision=turn.operation.task_revision,
        owner="worker",
        proposal=AgentProposalV1(
            action="wait_human", message="Falta contexto", questions=["Cual es el objetivo?"]
        ),
        response_sha256="c" * 64,
        provider="fake",
        model="fake-model",
        input_tokens=12,
        output_tokens=8,
        latency_ms=100,
        now=NOW + timedelta(seconds=1),
        verify_current=lambda: CONTEXT,
    )
    values.update(changes)
    return turns.finish(turn.operation_id, **values)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "tool", "message": "Revisar"},
        {"action": "delegate", "message": "Delegar", "delegate_role": "business"},
        {"action": "wait_human", "message": "Falta contexto"},
        {"action": "finish", "message": "Listo", "questions": ["Que falta?"]},
        {"action": "finish", "message": "Listo", "confirmed_human": True},
        {
            "action": "delegate",
            "message": "Delegar",
            "delegate_role": "coordinator",
            "delegate_objective": "Revisar",
        },
    ],
)
def test_proposals_reject_missing_action_fields_and_untrusted_approval(payload):
    with pytest.raises(ValidationError):
        AgentProposalV1.model_validate(payload)


def test_claim_metadata_failure_rolls_back_operation_and_budget(setup):
    turns, repo, task, _ = setup
    with pytest.raises(ValidationError):
        _claim(setup, provider="invalid provider")
    assert turns.list_turns(task.id) == []
    assert repo.list_operations(task.id) == []
    assert repo.get_task(task.id).calls_used == 0
    assert repo.get_task(task.id).revision == 0


def test_claim_normalizes_key_without_leaving_an_unreachable_reservation(setup):
    turns, repo, task, _ = setup
    turn = _claim(setup, operation="  turn  ")
    assert turn.operation_id == "turn"
    assert turns.get(" turn ") == turn
    assert repo.get_task(task.id).calls_used == 1
    result = _finish(setup, turn)
    assert result.operation.status == "COMPLETED"


def test_finish_scope_failure_rolls_back_response_and_reservation_release(setup):
    turns, repo, task, _ = setup
    turn = _claim(setup)
    proposal = AgentProposalV1(
        action="finish",
        message="Revisado",
        evidence_refs=[ArtifactRefV1(kind="capability", id="foreign")],
    )
    with pytest.raises(OrchestrationError) as error:
        _finish(setup, turn, proposal=proposal)
    assert error.value.code == "ARTIFACT_SCOPE_MISMATCH"
    stored = turns.get("turn")
    assert stored.operation.status == "CLAIMED"
    assert stored.proposal is None and stored.response_sha256 is None
    assert repo.get_task(task.id).reserved_ms == 60000
    assert repo.get_task(task.id).elapsed_ms == 0


def test_one_repair_is_a_second_real_call_and_cannot_extend_to_a_third(setup):
    turns, repo, task, _ = setup
    first = _finish(setup, _claim(setup), proposal=None, error_code="INVALID_AGENT_PROPOSAL")
    repaired = _claim(
        setup, "turn:repair", repair_of_operation_id="turn", now=NOW + timedelta(seconds=2)
    )
    repaired = _finish(setup, repaired, now=NOW + timedelta(seconds=3))
    assert first.operation.status == "FAILED"
    assert turns.get_repair("turn") == repaired
    assert len(turns.list_turns(task.id)) == 2
    assert repo.get_task(task.id).calls_used == 2
    with pytest.raises(OrchestrationError) as error:
        _claim(setup, "extra", repair_of_operation_id="turn", now=NOW + timedelta(seconds=4))
    assert error.value.code == "AGENT_REPAIR_LIMIT"
    assert repo.get_task(task.id).calls_used == 2
    assert repo.get_operation("extra") is None


def test_unapplied_proposal_blocks_new_provider_calls(setup):
    turns, repo, task, _ = setup
    _finish(setup, _claim(setup))
    with pytest.raises(OrchestrationError) as error:
        _claim(setup, "next", now=NOW + timedelta(seconds=2))
    assert error.value.code == "AGENT_DISPATCH_PENDING"
    assert repo.get_task(task.id).calls_used == 1
    assert len(turns.list_turns(task.id)) == 1


def test_response_trace_stores_hashes_and_metadata_only(setup):
    turns, _, _, _ = setup
    turn = _finish(
        setup,
        _claim(setup),
        provider="actual-provider",
        model="actual-model",
        response_id="response-sandbox",
    )
    row = dict(turns.connection.execute("SELECT * FROM agent_turns").fetchone())
    assert "prompt" not in row and "raw_response" not in row and "content" not in row
    assert turn.prompt_sha256 == "b" * 64 and turn.response_sha256 == "c" * 64
    assert (turn.provider, turn.model) == ("fake", "fake-model")
    assert (turn.actual_provider, turn.actual_model) == ("actual-provider", "actual-model")
    assert (turn.input_tokens, turn.output_tokens, turn.latency_ms) == (12, 8, 100)
    assert turn.proposal_sha256 is not None


def test_transition_and_dispatch_roll_back_together_after_failure(setup, monkeypatch):
    turns, repo, task, _ = setup
    _finish(setup, _claim(setup))
    before = repo.get_task(task.id)
    original = turns._persist_dispatch

    def fail_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure before commit")

    with monkeypatch.context() as patch:
        patch.setattr(turns, "_persist_dispatch", fail_after_write)
        with pytest.raises(RuntimeError, match="injected failure"):
            turns.transition_and_dispatch(
                "turn",
                expected_revision=before.revision,
                new_state="WAITING_HUMAN",
                actor_label="Human",
                verify_current=lambda: CONTEXT,
            )
    assert repo.get_task(task.id) == before
    assert turns.get("turn").dispatch_result is None
    final = turns.transition_and_dispatch(
        "turn",
        expected_revision=before.revision,
        new_state="WAITING_HUMAN",
        actor_label="Human",
        verify_current=lambda: CONTEXT,
    )
    assert final.dispatch_result.task.state == "WAITING_HUMAN"
    replay = turns.transition_and_dispatch(
        "turn",
        expected_revision=before.revision,
        new_state="WAITING_HUMAN",
        actor_label="Human",
        verify_current=lambda: "d" * 64,
    )
    assert replay == final


def test_context_changed_since_response_cannot_be_legitimized_at_dispatch(setup):
    turns, repo, task, _ = setup
    _finish(setup, _claim(setup))
    before = repo.get_task(task.id)
    with pytest.raises(OrchestrationError) as error:
        turns.transition_and_dispatch(
            "turn",
            expected_revision=before.revision,
            new_state="WAITING_HUMAN",
            actor_label="Human",
            verify_current=lambda: "d" * 64,
        )
    assert error.value.code == "STALE_CONTEXT"
    assert repo.get_task(task.id) == before
    assert turns.get("turn").dispatch_result is None


def test_expired_response_never_leaves_an_actionable_proposal(setup):
    turns, repo, task, _ = setup
    turn = _claim(setup, lease_seconds=1)
    expired = _finish(setup, turn, now=NOW + timedelta(seconds=2))
    assert expired.operation.status == "FAILED"
    assert expired.error_code == "OPERATION_LEASE_EXPIRED"
    assert expired.proposal is None and expired.proposal_sha256 is None
    assert expired.response_sha256 == "c" * 64
    assert repo.get_task(task.id).state == "WAITING_HUMAN"


def test_failure_dispatch_is_persisted_without_an_invented_proposal(setup):
    turns, repo, task, _ = setup
    failed = _finish(
        setup,
        _claim(setup),
        proposal=None,
        response_sha256=None,
        error_code="PROVIDER_UNAVAILABLE",
        next_state="WAITING_HUMAN",
    )
    result = AgentTurnResultV1(
        operation_id="turn", task=repo.get_task(task.id), error_code=failed.error_code
    )
    stored = turns.mark_dispatched(
        "turn", result=result, expected_proposal_sha256=None, actor_label="Reviewer"
    )
    assert stored.dispatch_result.error_code == "PROVIDER_UNAVAILABLE"
    dispatch = [
        event
        for event in repo.list_events(task.id)
        if event["event_type"] == "AGENT_TURN_DISPATCHED"
    ]
    assert len(dispatch) == 1 and dispatch[0]["actor_label"] == "Reviewer"
    replay = _claim(setup, expected_revision=0)
    assert replay == stored
    assert repo.get_task(task.id).calls_used == 1


def test_recovery_can_be_scoped_to_one_root_without_touching_another(setup):
    turns, repo, task, capability = setup
    _claim(setup, lease_seconds=1)
    other = repo.create_task(
        capability_id=capability.id,
        role="coordinator",
        context_sha256=CONTEXT,
        actor_label="Human",
        operation_id="other-task",
        now=NOW,
    )
    turns.claim(
        other.id,
        operation_id="other-turn",
        expected_revision=0,
        actor_label="Human",
        owner="other",
        provider="fake",
        model="fake",
        prompt_version="agent.v1",
        prompt_sha256="b" * 64,
        lease_seconds=1,
        now=NOW,
    )
    assert repo.recover_expired_claims(now=NOW + timedelta(seconds=2), root_task_id=task.id) == [
        "turn"
    ]
    assert turns.get("turn").error_code == "OPERATION_OUTCOME_UNKNOWN"
    assert turns.get("other-turn").operation.status == "CLAIMED"
    assert repo.get_task(other.id).reserved_ms == 1000


def test_delegation_revalidates_parent_context_in_its_creation_transaction(setup):
    turns, repo, task, capability = setup
    turn = _finish(
        setup,
        _claim(setup),
        proposal=AgentProposalV1(
            action="delegate",
            message="Revisar negocio",
            delegate_role="business",
            delegate_objective="Completar la pregunta de negocio",
        ),
    )
    arguments = dict(
        capability_id=capability.id,
        role="business",
        context_sha256=CONTEXT,
        actor_label="Human",
        operation_id="delegate",
        parent_task_id=task.id,
        parent_expected_revision=turn.operation.task_revision + 1,
    )
    with pytest.raises(OrchestrationError) as error:
        repo.create_task(**arguments, verify_current=lambda: "d" * 64)
    assert error.value.code == "STALE_CONTEXT"
    assert repo.list_children(task.id) == []
    checked = []

    def verify():
        checked.append(repo.connection.in_transaction)
        return CONTEXT

    child = repo.create_task(**arguments, verify_current=verify)
    assert checked == [True]
    replay = repo.create_task(
        **{**arguments, "parent_expected_revision": -1}, verify_current=lambda: "d" * 64
    )
    assert replay.id == child.id
    assert len(repo.list_children(task.id)) == 1
