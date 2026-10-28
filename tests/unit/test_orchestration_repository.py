"""MV-06: presupuesto global, exclusión, recuperación y referencias persistidas."""

from datetime import UTC, datetime, timedelta

import pytest

from core.contracts.orchestration import AgentBudgetV1, ArtifactRefV1, ToolCallV1, ToolResultV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository
from core.domain.capabilities import CapabilityService

NOW = datetime(2026, 10, 23, 12, tzinfo=UTC)
CONTEXT = "a" * 64
CALL = ToolCallV1(tool_name="read_capability_context")


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "orchestration.sqlite3"
    connection = get_connection(path)
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(name="Sandbox", actor_label="Test", operation_id="p")
    capability = capabilities.create_capability(
        product.id,
        name="Activacion",
        objective="Hipotesis sintetica",
        value_kind="revenue",
        decision_unit="alta",
        actor_label="Test",
        operation_id="c",
    )
    cycle = capabilities.create_cycle(
        capability.id,
        reason="Inicial sintetico",
        actor_label="Test",
        operation_id="cy",
    )
    repo = OrchestrationRepository(workflow)
    yield repo, capability, cycle, path
    connection.close()


def _task(setup, *, name="root", parent=None, budget=None, cycle=True):
    repo, capability, selected_cycle, _ = setup
    return repo.create_task(
        capability_id=capability.id,
        cycle_id=selected_cycle.id if cycle else None,
        role="coordinator" if parent is None else "business",
        context_sha256=CONTEXT,
        actor_label="Test",
        operation_id=name,
        parent_task_id=None if parent is None else parent.id,
        budget=budget,
        now=NOW,
        objective="Inspeccionar evidencia sintetica",
    )


def _claim(repo, task, *, operation="op", now=NOW, **changes):
    values = dict(
        operation_id=operation,
        expected_revision=repo.get_task(task.id).revision,
        owner="worker",
        actor_label="Test",
        lease_seconds=5,
        now=now,
        verify_current=lambda: CONTEXT,
    )
    values.update(changes)
    return repo.claim_operation(task.id, CALL, **values)


def _finish(repo, operation, *, seconds=1, **changes):
    values = dict(
        expected_revision=operation.task_revision,
        owner="worker",
        result=ToolResultV1(status="COMPLETED"),
        now=NOW + timedelta(seconds=seconds),
    )
    values.update(changes)
    return repo.finish_operation(operation.operation_id, **values)


def test_global_calls_are_shared_and_blocked_claim_spends_nothing(setup):
    repo, *_ = setup
    root = _task(setup, budget=AgentBudgetV1(max_calls=1))
    child = _task(setup, name="child", parent=root)
    _finish(repo, _claim(repo, root, calls_cost=1))
    before = repo.get_task(child.id)
    with pytest.raises(OrchestrationError) as error:
        _claim(repo, child, operation="child-op", calls_cost=1)
    assert error.value.code == "BUDGET_EXHAUSTED"
    after = repo.get_task(child.id)
    assert (after.steps_used, after.calls_used, after.elapsed_ms) == (1, 1, 1000)
    assert after == before
    assert repo.get_operation("child-op") is None


def test_global_steps_prevent_child_bypass(setup):
    repo, *_ = setup
    root = _task(setup, budget=AgentBudgetV1(max_steps=1))
    child = _task(setup, name="child", parent=root)
    _finish(repo, _claim(repo, child))
    with pytest.raises(OrchestrationError) as error:
        _claim(repo, root, operation="second")
    assert error.value.code == "BUDGET_EXHAUSTED"
    assert repo.get_task(root.id).steps_used == 1


def test_revision_owner_and_live_root_claim_prevent_competing_effects(setup):
    repo, *_ = setup
    root = _task(setup)
    child = _task(setup, name="child", parent=root)
    operation = _claim(repo, root)
    with pytest.raises(OrchestrationError, match="cambió") as error:
        _finish(repo, operation, expected_revision=0)
    assert error.value.code == "STALE_TASK_REVISION"
    with pytest.raises(OrchestrationError) as error:
        _finish(repo, operation, owner="other")
    assert error.value.code == "LEASE_NOT_OWNED"
    with pytest.raises(OrchestrationError) as error:
        _claim(repo, child, operation="second")
    assert error.value.code == "ROOT_BUSY"
    assert repo.get_operation("op").status == "CLAIMED"
    assert repo.get_task(root.id).steps_used == 1


def test_expired_finish_preserves_effect_refs_and_requires_review(setup):
    repo, capability, *_ = setup
    root = _task(setup)
    operation = _claim(repo, root, lease_seconds=1)
    outcome = _finish(
        repo,
        operation,
        seconds=2,
        result=ToolResultV1(
            status="COMPLETED",
            output_refs=[ArtifactRefV1(kind="capability", id=capability.id)],
        ),
    )
    assert outcome.status == "FAILED"
    assert outcome.result.error_code == "OPERATION_LEASE_EXPIRED"
    assert outcome.result.output_refs[0].id == capability.id
    task = repo.get_task(root.id)
    assert task.state == "WAITING_HUMAN"
    assert (task.elapsed_ms, task.reserved_ms, task.steps_used) == (2000, 0, 1)


@pytest.mark.parametrize("reconstruct", [False, True])
def test_recovery_charges_only_reservation_and_never_reexecutes(setup, reconstruct):
    repo, capability, *_ = setup
    root = _task(setup)
    _claim(repo, root, lease_seconds=1)
    reads = []

    def recover(operation):
        reads.append((operation.operation_id, repo.connection.in_transaction))
        return (
            ToolResultV1(
                status="COMPLETED",
                output_refs=[
                    ArtifactRefV1(kind="capability", id=capability.id),
                ],
            )
            if reconstruct
            else None
        )

    result = repo.recover_expired_claims(now=NOW + timedelta(days=30), recover_result=recover)
    assert result == ["op"]
    assert reads == [("op", True)]
    task = repo.get_task(root.id)
    assert (task.elapsed_ms, task.reserved_ms, task.state) == (1000, 0, "WAITING_HUMAN")
    operation = repo.get_operation("op")
    assert operation.status == ("COMPLETED" if reconstruct else "FAILED")
    if not reconstruct:
        assert operation.result.error_code == "OPERATION_OUTCOME_UNKNOWN"
    replay = _claim(repo, root, expected_revision=0)
    assert replay == operation
    assert repo.get_task(root.id).steps_used == 1
    assert repo.recover_expired_claims(now=NOW + timedelta(days=31)) == []


def test_stale_context_before_claim_has_no_effect_or_spend(setup):
    repo, *_ = setup
    root = _task(setup)
    with pytest.raises(OrchestrationError) as error:
        _claim(repo, root, verify_current=lambda: "b" * 64)
    assert error.value.code == "STALE_CONTEXT"
    assert repo.get_task(root.id) == root
    assert repo.list_operations(root.id) == []


def test_stale_context_after_effect_can_record_failure_without_losing_refs(setup):
    repo, capability, *_ = setup
    root = _task(setup)
    operation = _claim(repo, root)
    with pytest.raises(OrchestrationError) as error:
        _finish(repo, operation, verify_current=lambda: "b" * 64)
    assert error.value.code == "STALE_CONTEXT"
    assert repo.get_operation("op").status == "CLAIMED"
    outcome = _finish(
        repo,
        operation,
        next_state="WAITING_HUMAN",
        result=ToolResultV1(
            status="FAILED",
            error_code="EFFECT_RECORDED_CONTEXT_CHANGED",
            output_refs=[ArtifactRefV1(kind="capability", id=capability.id)],
        ),
    )
    assert outcome.result.output_refs[0].id == capability.id
    assert repo.get_task(root.id).state == "WAITING_HUMAN"


def test_own_write_refresh_requires_verified_post_effect_context(setup):
    repo, *_ = setup
    root = _task(setup)
    operation = _claim(repo, root)
    with pytest.raises(OrchestrationError) as error:
        _finish(repo, operation, expected_context_sha256="b" * 64)
    assert error.value.code == "CONTEXT_VERIFICATION_REQUIRED"
    _finish(repo, operation, expected_context_sha256="b" * 64, verify_current=lambda: "b" * 64)
    assert repo.get_task(root.id).context_sha256 == "b" * 64


def test_task_and_historical_replay_survive_new_connection(setup):
    repo, _, _, path = setup
    root = _task(setup, cycle=False)
    operation = _finish(repo, _claim(repo, root))
    other_connection = get_connection(path)
    try:
        other = OrchestrationRepository(WorkflowRepository(other_connection))
        replay = _claim(other, root, expected_revision=0)
        assert replay == operation
        assert other.get_task(root.id).objective == "Inspeccionar evidencia sintetica"
        with pytest.raises(OrchestrationError) as error:
            other.claim_operation(
                root.id,
                ToolCallV1(tool_name="read_run_summary"),
                operation_id="op",
                expected_revision=0,
                owner="worker",
                actor_label="Test",
            )
        assert error.value.code == "IDEMPOTENCY_CONFLICT"
    finally:
        other_connection.close()


def test_refresh_context_uses_cas_and_does_not_reset_or_spend_budget(setup):
    repo, *_ = setup
    root = _task(setup)
    _finish(repo, _claim(repo, root))
    task = repo.get_task(root.id)
    refreshed = repo.refresh_task_context(
        root.id,
        expected_revision=task.revision,
        context_sha256="b" * 64,
        actor_label="Test",
        verify_current=lambda: "b" * 64,
        now=NOW + timedelta(days=10),
    )
    assert refreshed.context_sha256 == "b" * 64
    assert (refreshed.steps_used, refreshed.elapsed_ms) == (1, 1000)
    assert refreshed.revision == task.revision + 1
    with pytest.raises(OrchestrationError) as error:
        repo.refresh_task_context(
            root.id,
            expected_revision=task.revision,
            context_sha256=CONTEXT,
            actor_label="Test",
            verify_current=lambda: CONTEXT,
        )
    assert error.value.code == "STALE_TASK_REVISION"


def test_active_time_exhaustion_does_not_reset_on_human_resume(setup):
    repo, *_ = setup
    root = _task(setup, budget=AgentBudgetV1(max_seconds=1))
    _claim(repo, root, lease_seconds=5)
    repo.recover_expired_claims(now=NOW + timedelta(seconds=2))
    task = repo.get_task(root.id)
    resumed = repo.transition_task(
        root.id,
        expected_revision=task.revision,
        new_state="RUNNING",
        actor_label="Test",
        now=NOW + timedelta(days=30),
    )
    with pytest.raises(OrchestrationError) as error:
        _claim(repo, resumed, operation="second", now=NOW + timedelta(days=30))
    assert error.value.code == "BUDGET_EXHAUSTED"
    assert repo.get_task(root.id).elapsed_ms == 1000
