"""MV-06: supervisor local, permisos, aislamiento y recuperación sin repetir efectos."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.contracts.orchestration import AgentBudgetV1, ToolCallV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.orchestration.runtime import OrchestrationError, Supervisor
from core.orchestration.tools import domain_operation_id
from tests.unit.test_capabilities import _seed_approved_run
from tests.unit.test_decision_value import _economic

ACTOR = "Responsable de prueba sintética"


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "runtime.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(name="Demo", actor_label=ACTOR, operation_id="p")
    caps = [
        capabilities.create_capability(
            product.id,
            name=f"Capacidad {kind}",
            objective=f"Validar {kind}",
            value_kind=kind,
            decision_unit="caso",
            actor_label=ACTOR,
            operation_id=kind,
        )
        for kind in ("revenue", "savings")
    ]
    cycles = [
        capabilities.create_cycle(
            cap.id, reason="Cohorte sintética", actor_label=ACTOR, operation_id=f"cycle-{cap.id}"
        )
        for cap in caps
    ]
    refs = [_seed_approved_run(capabilities, cycle) for cycle in cycles]
    values = DecisionValueService(workflow)
    scenarios = [
        values.create_economic_version(
            cap.id,
            scenario=_economic(),
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id=f"economic-{cap.id}",
        )
        for cap in caps
    ]
    clock = SimpleNamespace(now=datetime(2030, 1, 1, tzinfo=UTC))
    supervisor = Supervisor(workflow, clock=lambda: clock.now, storage_root=tmp_path)
    yield SimpleNamespace(
        connection=connection,
        workflow=workflow,
        capabilities=capabilities,
        caps=caps,
        cycles=cycles,
        refs=refs,
        values=values,
        scenarios=scenarios,
        clock=clock,
        supervisor=supervisor,
        storage_root=tmp_path,
    )
    connection.close()


def _task(state, role="business", *, name="task", cycle=True, **kwargs):
    return state.supervisor.create_task(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id if cycle else None,
        role=role,
        actor_label=ACTOR,
        operation_id=name,
        **kwargs,
    )


def _execute(state, task, name, arguments=None, *, operation="tool", **kwargs):
    return state.supervisor.execute_tool(
        task.id,
        ToolCallV1(tool_name=name, arguments=arguments or {}),
        operation_id=operation,
        expected_revision=task.revision,
        actor_label=ACTOR,
        **kwargs,
    )


def _reload(state, task):
    return state.supervisor.repository.get_task(task.id)


@pytest.mark.parametrize("role", ["coordinator", "business", "data", "modeling", "value_evidence"])
def test_every_role_can_read_its_capability_without_a_cycle(state, role):
    task = _task(state, role, cycle=False)
    result = _execute(state, task, "read_capability_context")
    assert result.status == "COMPLETED"
    assert _reload(state, task).steps_used == 1
    assert _reload(state, task).calls_used == 0
    assert state.values.list_decision_versions(state.caps[0].id) == []


@pytest.mark.parametrize(
    ("role", "tool", "arguments"),
    [
        ("coordinator", "save_decision_draft", {"spec": {}}),
        ("data", "save_decision_draft", {"spec": {}}),
        ("modeling", "save_decision_draft", {"spec": {}}),
        ("value_evidence", "save_decision_draft", {"spec": {}}),
        ("business", "read_dataset_profile", {"dataset_id": "placeholder"}),
        ("modeling", "read_dataset_profile", {"dataset_id": "placeholder"}),
        ("data", "read_run_summary", {"run_id": "placeholder"}),
        ("coordinator", "evaluate_economic_scenario", {"economic_version_id": "placeholder"}),
        ("data", "evaluate_economic_scenario", {"economic_version_id": "placeholder"}),
    ],
)
def test_tool_permissions_are_enforced_before_any_effect(state, role, tool, arguments):
    task = _task(state, role)
    with pytest.raises(OrchestrationError) as error:
        _execute(state, task, tool, arguments)
    assert error.value.code == "TOOL_ROLE_DENIED"
    assert _reload(state, task).steps_used == 0
    assert state.values.list_decision_versions(state.caps[0].id) == []


@pytest.mark.parametrize(
    "tool", ["train_model", "execute_python", "publish_report", "approve_training"]
)
def test_unknown_or_future_tools_cannot_be_executed(state, tool):
    task = _task(state)
    with pytest.raises(OrchestrationError) as error:
        _execute(state, task, tool)
    assert error.value.code == ("TOOL_ROLE_DENIED" if tool == "train_model" else "TOOL_NOT_ALLOWED")
    assert _reload(state, task).steps_used == 0


@pytest.mark.parametrize(
    "injected",
    [
        {"capability_id": "foreign"},
        {"cycle_id": "foreign"},
        {"confirmed_human": True},
        {"python": "__import__('os').system('echo forbidden')"},
    ],
)
def test_extra_arguments_cannot_override_scope_or_human_confirmation(state, injected):
    task = _task(state)
    with pytest.raises(OrchestrationError) as error:
        _execute(state, task, "save_decision_draft", {"spec": {}, **injected})
    assert error.value.code == "INVALID_TOOL_ARGUMENTS"
    assert state.values.list_decision_versions(state.caps[0].id) == []
    assert _reload(state, task).steps_used == 0


def test_agent_cannot_invent_a_human_owner_label(state):
    task = _task(state)
    with pytest.raises(OrchestrationError) as error:
        _execute(
            state,
            task,
            "save_decision_draft",
            {"spec": {"question": "¿Qué revisar?", "owner_label": "Invented human"}},
        )
    assert error.value.code == "TOOL_HUMAN_FIELD_FORBIDDEN"
    assert state.values.list_decision_versions(state.caps[0].id) == []


def test_agent_draft_preserves_previously_declared_human_owner(state):
    original = state.values.create_decision_version(
        state.caps[0].id,
        spec={"question": "Pregunta inicial", "owner_label": "Responsable declarado por persona"},
        actor_label=ACTOR,
        operation_id="human-decision",
    )
    task = _task(state)
    result = _execute(
        state,
        task,
        "save_decision_draft",
        {"spec": {"question": "Pregunta refinada"}, "based_on_version_id": original.id},
    )
    assert result.status == "COMPLETED"
    saved = state.values.list_decision_versions(state.caps[0].id)[-1]
    assert saved.spec.owner_label == original.spec.owner_label
    assert saved.confirmed_human is False


@pytest.mark.parametrize("role", ["business", "value_evidence"])
def test_allowed_economic_tool_reuses_decimal_calculation_without_training(state, role):
    task = _task(state, role)
    result = _execute(
        state,
        task,
        "evaluate_economic_scenario",
        {"economic_version_id": state.scenarios[0].id},
    )
    assert result.status == "COMPLETED"
    assert result.summary["status"] == "scenario_estimate"
    assert result.summary["selected_cases"] == 10
    assert result.summary["net_value"] == "875"
    assert result.summary["evidence_type"] == "synthetic_scenario"
    assert _reload(state, task).calls_used == 0
    assert state.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_safe_profile_omits_raw_rows_values_paths_and_redacts_column_identifiers(state):
    name = "fixture@example.test"
    profile = {
        "schema_version": "dataset-profile.v1",
        "data_classification": "synthetic",
        "row_count": 20,
        "column_count": 1,
        "duplicate_row_count": 0,
        "columns": [
            {
                "name": name,
                "inferred_type": "categorical",
                "missing_count": 0,
                "missing_rate": 0,
                "unique_count": 2,
                "constant": False,
                "high_cardinality": False,
                "possible_identifier": False,
                "sample_values": ["DO_NOT_EXPORT_RAW_CELL"],
            }
        ],
        "numeric_columns": [],
        "categorical_columns": [name],
        "high_cardinality_columns": [],
        "constant_columns": [],
        "warnings": ["DO_NOT_EXPORT_FREE_WARNING"],
        "rows": [{name: "DO_NOT_EXPORT_RAW_CELL"}],
    }
    content = (name + "\n" + "a\nb\n" * 10).encode()
    dataset = state.workflow.add_dataset(
        project_id=state.cycles[0].project_id,
        sha256=hashlib.sha256(content).hexdigest(),
        original_filename_display="DO_NOT_EXPORT_FILENAME.csv",
        source="Synthetic test",
        license_name="test-only",
        anonymization_declared=True,
        admission_status="ACCEPTED",
        rejection_reason=None,
        n_rows=20,
        n_cols=1,
        byte_count=len(content),
        profile=profile,
    )
    stored_path = state.storage_root / Path(state.workflow.get_dataset(dataset)["stored_path"])
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)
    task = _task(state, "data")
    result = _execute(state, task, "read_dataset_profile", {"dataset_id": dataset})
    assert result.status == "COMPLETED"
    serialized = result.model_dump_json()
    for forbidden in (name, "DO_NOT_EXPORT", "stored_path", "sample_values", '"rows"'):
        assert forbidden not in serialized
    assert result.summary["row_count"] == 20
    assert result.summary["integrity_verified"] is True
    assert result.summary["profile"]["columns"][0]["name"] == "C001"


def test_run_summary_only_exposes_whitelisted_numeric_metrics(state):
    state.connection.execute(
        "UPDATE runs SET candidate_metrics_json=? WHERE id=?",
        (
            json.dumps(
                {
                    "holdout": {"f1": 0.6, "recall": "SECRET_RAW_FIELD", "raw_rows": ["PRIVATE"]},
                    "validation": {
                        "LogisticRegression": {"metrics": {"f1": 0.5, "raw": "PRIVATE"}},
                        "SECRET_CUSTOM_MODEL": {"metrics": {"f1": 1.0}},
                    },
                    "dataset_rows": ["PRIVATE"],
                }
            ),
            state.refs[0]["run"],
        ),
    )
    task = _task(state, "modeling")
    result = _execute(state, task, "read_run_summary", {"run_id": state.refs[0]["run"]})
    assert result.status == "COMPLETED"
    assert result.summary["holdout_metrics"] == {"f1": 0.6}
    assert result.summary["validation_metrics"] == {"LogisticRegression": {"f1": 0.5}}
    assert "PRIVATE" not in result.model_dump_json()
    assert "SECRET" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("role", "tool", "argument", "refkey"),
    [
        ("data", "read_dataset_profile", "dataset_id", "dataset"),
        ("modeling", "read_run_summary", "run_id", "run"),
    ],
)
def test_tools_reject_artifacts_from_another_capability(state, role, tool, argument, refkey):
    task = _task(state, role)
    with pytest.raises(OrchestrationError):
        _execute(state, task, tool, {argument: state.refs[1][refkey]})
    assert _reload(state, task).steps_used == 0


def test_economic_reference_and_embedded_decision_refs_cannot_cross_capabilities(state):
    task = _task(state)
    with pytest.raises(OrchestrationError):
        _execute(
            state,
            task,
            "evaluate_economic_scenario",
            {"economic_version_id": state.scenarios[1].id},
        )
    with pytest.raises(OrchestrationError):
        _execute(
            state,
            task,
            "save_decision_draft",
            {"spec": {"economic_version_id": state.scenarios[1].id}},
            operation="cross-embedded",
        )
    assert state.values.list_decision_versions(state.caps[0].id) == []


def test_same_capability_does_not_allow_run_from_another_cycle(state):
    later = state.capabilities.create_cycle(
        state.caps[0].id,
        reason="Otra cohorte",
        source_cycle_id=state.cycles[0].id,
        actor_label=ACTOR,
        operation_id="later-cycle",
    )
    later_refs = _seed_approved_run(state.capabilities, later)
    task = _task(state, "modeling")
    with pytest.raises(OrchestrationError):
        _execute(state, task, "read_run_summary", {"run_id": later_refs["run"]})
    assert _reload(state, task).steps_used == 0


def test_draft_write_is_unconfirmed_and_replay_creates_no_second_version(state):
    task = _task(state)
    call = {"spec": {"question": "¿Qué casos revisar?", "action": "Revisión humana"}}
    first = _execute(state, task, "save_decision_draft", call)
    assert first.status == "COMPLETED"
    versions = state.values.list_decision_versions(state.caps[0].id)
    assert len(versions) == 1
    assert versions[0].confirmed_human is False
    assert versions[0].missing_fields
    replay = _execute(state, task, "save_decision_draft", call)
    assert replay == first
    assert state.values.list_decision_versions(state.caps[0].id) == versions
    assert _reload(state, task).steps_used == 1


def test_operation_whitespace_is_normalized_for_domain_write_and_replay(state):
    task = _task(state)
    call = {"spec": {"question": "Pregunta sintética"}}
    first = _execute(state, task, "save_decision_draft", call, operation="  spaced-id  ")
    assert first.status == "COMPLETED"
    assert _execute(state, task, "save_decision_draft", call, operation="spaced-id") == first
    assert len(state.values.list_decision_versions(state.caps[0].id)) == 1
    assert state.supervisor.repository.get_operation("spaced-id").status == "COMPLETED"


def test_recovery_cannot_claim_success_from_a_colliding_domain_operation(state):
    existing = state.values.create_decision_version(
        state.caps[0].id,
        spec={"question": "Contenido de otra solicitud"},
        actor_label=ACTOR,
        operation_id=domain_operation_id("collision"),
    )
    task = _task(state)
    result = _execute(
        state,
        task,
        "save_decision_draft",
        {"spec": {"question": "Contenido de esta solicitud"}},
        operation="collision",
    )
    assert result.status == "FAILED"
    assert result.error_code == "OPERATION_CONFLICT"
    assert result.output_refs == []
    assert state.values.list_decision_versions(state.caps[0].id) == [existing]


def test_operation_id_is_bound_to_tool_arguments_and_task(state):
    task = _task(state)
    _execute(state, task, "read_capability_context")
    with pytest.raises(OrchestrationError):
        _execute(state, task, "save_decision_draft", {"spec": {}})
    other = _task(state, name="other-task")
    with pytest.raises(OrchestrationError):
        _execute(state, other, "read_capability_context")
    assert state.values.list_decision_versions(state.caps[0].id) == []


def test_inputs_and_parent_tasks_must_belong_to_same_scope(state):
    with pytest.raises(OrchestrationError):
        _task(state, input_refs=[{"kind": "run", "id": state.refs[1]["run"]}])
    parent = _task(state, "coordinator")
    with pytest.raises(OrchestrationError):
        state.supervisor.create_task(
            capability_id=state.caps[1].id,
            cycle_id=state.cycles[1].id,
            role="business",
            parent_task_id=parent.id,
            actor_label=ACTOR,
            operation_id="foreign-child",
        )


@pytest.mark.parametrize("parent_role", ["business", "data", "modeling", "value_evidence"])
def test_specialists_cannot_spawn_other_specialists(state, parent_role):
    parent = _task(state, parent_role)
    with pytest.raises(OrchestrationError):
        _task(state, "business", name="child", parent_task_id=parent.id)


def test_coordinator_cannot_spawn_recursive_coordinators(state):
    parent = _task(state, "coordinator")
    with pytest.raises(OrchestrationError):
        _task(state, "coordinator", name="child", parent_task_id=parent.id)


@pytest.mark.parametrize("kind", ["dataset", "context", "strategy", "capability", "cycle"])
def test_declared_input_hash_must_match_actual_artifact(state, kind):
    refid = {
        **state.refs[0],
        "capability": state.caps[0].id,
        "cycle": state.cycles[0].id,
    }[kind]
    with pytest.raises(OrchestrationError) as error:
        _task(state, input_refs=[{"kind": kind, "id": refid, "sha256": "f" * 64}])
    assert error.value.code == "ARTIFACT_HASH_MISMATCH"


def test_root_step_budget_is_shared_with_children(state):
    parent = _task(state, "coordinator", budget=AgentBudgetV1(max_steps=1))
    child = _task(state, "business", name="child", parent_task_id=parent.id)
    _execute(state, child, "read_capability_context", operation="child-read")
    with pytest.raises(OrchestrationError):
        _execute(state, _reload(state, parent), "read_capability_context", operation="parent-read")
    assert _reload(state, parent).steps_used == 1


@pytest.mark.parametrize("waiting", ["WAITING_HUMAN", "WAITING_DATA"])
def test_root_wait_state_blocks_descendant_execution(state, waiting):
    parent = _task(state, "coordinator")
    child = _task(state, name="child", parent_task_id=parent.id)
    state.supervisor.transition_task(
        parent.id, expected_revision=parent.revision, new_state=waiting, actor_label=ACTOR
    )
    with pytest.raises(OrchestrationError):
        _execute(state, child, "save_decision_draft", {"spec": {}})
    assert _reload(state, child).steps_used == 0
    assert state.values.list_decision_versions(state.caps[0].id) == []


@pytest.mark.parametrize("waiting", ["WAITING_HUMAN", "WAITING_DATA"])
def test_waiting_blocks_execution_and_does_not_spend_active_time_budget(state, waiting):
    task = _task(state)
    task = state.supervisor.transition_task(
        task.id, expected_revision=task.revision, new_state=waiting, actor_label=ACTOR
    )
    before = _reload(state, task)
    state.clock.now += timedelta(days=30)
    with pytest.raises(OrchestrationError):
        _execute(state, task, "read_capability_context")
    after = _reload(state, task)
    assert (after.steps_used, after.calls_used, after.elapsed_ms) == (
        before.steps_used,
        before.calls_used,
        before.elapsed_ms,
    )
    resumed = state.supervisor.transition_task(
        task.id,
        expected_revision=after.revision,
        new_state="PENDING",
        actor_label=ACTOR,
        refresh_context=True,
    )
    assert (
        _execute(state, resumed, "read_capability_context", operation="resumed").status
        == "COMPLETED"
    )


def test_expired_claim_recovers_committed_draft_without_reexecuting(state, monkeypatch):
    task = _task(state)
    original_finish = state.supervisor.repository.finish_operation

    def simulate_process_exit(*args, **kwargs):
        raise KeyboardInterrupt("Synthetic crash after domain commit")

    monkeypatch.setattr(state.supervisor.repository, "finish_operation", simulate_process_exit)
    with pytest.raises(KeyboardInterrupt, match="after domain commit"):
        _execute(
            state,
            task,
            "save_decision_draft",
            {"spec": {"question": "¿Qué revisamos?"}},
            lease_seconds=5,
        )
    saved = state.values.list_decision_versions(state.caps[0].id)
    assert len(saved) == 1
    monkeypatch.setattr(state.supervisor.repository, "finish_operation", original_finish)
    state.clock.now += timedelta(seconds=6)
    assert state.supervisor.recover_expired_operations() == ["tool"]
    assert state.values.list_decision_versions(state.caps[0].id) == saved
    recovered = _reload(state, task)
    assert recovered.state == "WAITING_HUMAN"
    assert recovered.steps_used == 1
    assert recovered.reserved_ms == 0
    operation = state.supervisor.repository.get_operation("tool")
    assert operation.status == "COMPLETED"
    assert operation.result.output_refs[0].id == saved[0].id
    assert operation.result.summary["recovered_from_persisted_effect"] is True
    assert state.supervisor.recover_expired_operations() == []


def test_expired_claim_without_proven_effect_stays_failed_without_repeating(state):
    task = _task(state)
    call = ToolCallV1(tool_name="read_capability_context")
    state.supervisor.repository.claim_operation(
        task.id,
        call,
        operation_id="unknown-outcome",
        expected_revision=task.revision,
        owner=state.supervisor.owner,
        actor_label=ACTOR,
        lease_seconds=5,
        now=state.clock.now,
    )
    state.clock.now += timedelta(seconds=6)
    assert state.supervisor.recover_expired_operations() == ["unknown-outcome"]
    operation = state.supervisor.repository.get_operation("unknown-outcome")
    assert operation.status == "FAILED"
    assert operation.result.error_code == "OPERATION_OUTCOME_UNKNOWN"
    replay = _execute(state, task, "read_capability_context", operation="unknown-outcome")
    assert replay == operation.result
    assert _reload(state, task).steps_used == 1
    assert _reload(state, task).state == "WAITING_HUMAN"


def test_context_change_after_task_creation_requires_review(state):
    task = _task(state)
    state.values.create_economic_version(
        state.caps[0].id,
        scenario=_economic(value_per_case="2000"),
        based_on_version_id=state.scenarios[0].id,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="human-economic-update",
    )
    with pytest.raises(OrchestrationError):
        _execute(state, task, "save_decision_draft", {"spec": {}})
    assert state.values.list_decision_versions(state.caps[0].id) == []


def test_prepare_turn_refreshes_context_and_invalidates_previous_revision(state):
    task = _task(state, "coordinator")
    state.values.create_economic_version(
        state.caps[0].id,
        scenario=_economic(value_per_case="2000"),
        based_on_version_id=state.scenarios[0].id,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="human-update-before-new-turn",
    )
    prepared = state.supervisor.prepare_turn(
        task.id, expected_revision=task.revision, actor_label=ACTOR
    )
    assert prepared.context_sha256 != task.context_sha256
    assert prepared.revision > task.revision
    assert (prepared.steps_used, prepared.calls_used, prepared.elapsed_ms) == (0, 0, 0)
    with pytest.raises(OrchestrationError) as stale:
        _execute(state, task, "read_capability_context")
    assert stale.value.code == "STALE_TASK_REVISION"
    assert _execute(state, prepared, "read_capability_context").status == "COMPLETED"


@pytest.mark.parametrize("waiting", ["WAITING_HUMAN", "WAITING_DATA"])
def test_prepare_turn_cannot_bypass_waiting_for_data_or_human(state, waiting):
    task = _task(state, "coordinator")
    waiting_task = state.supervisor.transition_task(
        task.id, expected_revision=task.revision, new_state=waiting, actor_label=ACTOR
    )
    with pytest.raises(OrchestrationError) as error:
        state.supervisor.prepare_turn(
            task.id, expected_revision=waiting_task.revision, actor_label=ACTOR
        )
    assert error.value.code == "TASK_NOT_RUNNABLE"
    assert _reload(state, task).state == waiting


def test_changed_context_after_write_retains_effect_and_requires_human_review(state, monkeypatch):
    task = _task(state)
    tool = state.supervisor._tools["save_decision_draft"]

    def domain_write_then_external_change(context, arguments):
        # Una herramienta no conserva la transacción de reserva mientras trabaja.
        assert state.connection.in_transaction is False
        outcome = tool.execute(context, arguments)
        state.values.create_economic_version(
            state.caps[0].id,
            scenario=_economic(value_per_case="2000"),
            based_on_version_id=state.scenarios[0].id,
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id="concurrent-human-economic-update",
        )
        return outcome

    monkeypatch.setitem(
        state.supervisor._tools,
        "save_decision_draft",
        replace(tool, execute=domain_write_then_external_change),
    )
    result = _execute(state, task, "save_decision_draft", {"spec": {}})
    assert result.status == "FAILED"
    assert result.error_code == "EFFECT_RECORDED_CONTEXT_CHANGED"
    saved = state.values.list_decision_versions(state.caps[0].id)
    assert len(saved) == 1
    assert result.output_refs[0].id == saved[0].id
    assert _reload(state, task).state == "WAITING_HUMAN"
    assert _execute(state, task, "save_decision_draft", {"spec": {}}) == result
    assert state.values.list_decision_versions(state.caps[0].id) == saved
