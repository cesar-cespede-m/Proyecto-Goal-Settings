"""MV-09: ML real sobre datos sintéticos y agentes con proveedor falso, sin red.

Las aprobaciones son actos explícitos de fixtures mediante la API de dominio;
no representan aprobaciones ni resultados económicos de personas reales.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from core.ai_engine.agents import AgentRunner
from core.analysis.snapshot import build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.contracts.orchestration import AgentBudgetV1, ToolCallV1
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService
from core.domain.context_service import BusinessContextService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import ApprovalRequiredError
from core.domain.predictive_guard import PredictiveGuardError
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService, RunBlockedError
from core.orchestration.runtime import OrchestrationError, Supervisor
from tests.unit.test_agent_runner import FakeAgentProvider, _tool, _wait, _wire_context
from tests.unit.test_ml_operation_idempotency import _start_inputs

ACTOR = "pytest-modeling-fixture"
COVERAGE = {
    "observed_from": "2026-01-01T00:00:00Z",
    "observed_until": "2026-09-30T00:00:00Z",
    "complete": True,
}


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "modeling-agent.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    caps = CapabilityService(workflow)
    product = caps.create_product(name="Sandbox ML", actor_label=ACTOR, operation_id="product")
    values = DecisionValueService(workflow)
    storage = tmp_path / "private-storage"
    ingest = DataIngestionService(workflow, storage)
    prepared = {}

    def prepare(index=0, *, bind=True, horizon=None, count=120):
        if index in prepared:
            return prepared[index]
        kind, target, stage, default_days = (
            ("revenue", "no_first_purchase", "purchase", 14)
            if index == 0
            else ("savings", "delivery_late", "delivered", 7)
        )
        days = default_days if horizon is None else horizon
        cap = caps.create_capability(
            product.id,
            name=kind,
            objective="Priorizar casos del producto sintético",
            value_kind=kind,
            decision_unit="caso",
            actor_label=ACTOR,
            operation_id=f"cap-{index}",
        )
        cycle = caps.create_cycle(
            cap.id, reason="Cohorte sintética", actor_label=ACTOR, operation_id=f"cycle-{index}"
        )
        rows = []
        for number in range(count):
            first = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=number * 2)
            entity = f"PRIVATE-CASE-{number:03d}"
            adverse = bool((number + index) % 2)
            rows.append((entity, first.isoformat(), "start"))
            for _ in range(1 if number % 2 else 3):
                rows.append((entity, (first + pd.Timedelta(days=1)).isoformat(), "activity"))
            if not adverse:
                rows.append((entity, (first + pd.Timedelta(days=5)).isoformat(), stage))
        events = pd.DataFrame(rows, columns=["entity", "moment", "stage"])
        recipe = {
            "entity": "entity",
            "moment": "moment",
            "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
            "windows_days": [2],
            "features": [
                {"op": "count_events", "window_days": 2, "output_column": "activity_count"}
            ],
            "outcome": [
                {
                    "op": "not_reached_stage",
                    "stage": {"column": "stage", "value": stage},
                    "within_days": days,
                    "output_column": target,
                }
            ],
            "label_available_at_output_column": "label_available_at",
        }
        snapshot = build_snapshot(events, recipe, coverage=COVERAGE)
        frame = snapshot.training_frame()

        def save(data, name):
            return ingest.ingest_bytes(
                project_id=cycle.project_id,
                request={
                    "original_filename_display": name,
                    "source_name": "Fixture sintético",
                    "classification": "synthetic",
                    "authorization_status": "approved",
                },
                content=data.to_csv(index=False).encode(),
            ).metadata.dataset_id

        source_id, dataset_id = save(events, "events.csv"), save(frame, "snapshot.csv")
        record_snapshot_recipe(
            workflow,
            project_id=cycle.project_id,
            table_dataset_id=dataset_id,
            source_dataset_id=source_id,
            spec=recipe,
            actor_label=ACTOR,
            coverage=COVERAGE,
            readiness=snapshot.readiness,
        )
        workflow.set_modeling_table(cycle.project_id, dataset_id, actor_label=ACTOR)
        context_payload = {
            "objective": "Comprobar un objetivo binario del producto sintético",
            "target_column": target,
            "positive_class": True,
            "prediction_moment": "Al corte declarado por la receta",
            "false_positive_consequence": "Intervenir un caso que no lo necesitaba",
            "false_negative_consequence": "Omitir un caso con resultado adverso",
            "primary_metric": "f1",
            "metric_rationale": "Comparar modelos del fixture",
            "decision_actor_label": ACTOR,
            "feature_decisions": [
                {
                    "column": column,
                    "use": "include" if column == "activity_count" else "exclude",
                    "available_at_prediction": column != "label_available_at",
                    "reason": "Actividad previa"
                    if column == "activity_count"
                    else "Metadato excluido",
                }
                for column in frame.columns
                if column != target
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        }
        context = BusinessContextService(workflow, storage).create_version(
            project_id=cycle.project_id, dataset_id=dataset_id, context=context_payload
        )
        draft = values.draft_predictive_from_context(
            cap.id, context_version_id=context.context_version_id
        )
        spec = draft.model_dump(mode="json")
        spec.update(
            population="Casos sintéticos",
            target_definition="Ausencia del resultado en la ventana",
            outcome_window={"kind": "fixed_days", "days": days},
        )
        version = values.create_predictive_version(
            cap.id, spec=spec, confirmed_human=True, actor_label=ACTOR, operation_id=f"spec-{index}"
        )
        if bind:
            values.bind_predictive_to_cycle(
                cap.id,
                cycle_id=cycle.id,
                predictive_version_id=version.id,
                actor_label=ACTOR,
                operation_id=f"bind-{index}",
            )
        result = SimpleNamespace(
            cap=cap,
            cycle=cycle,
            context=context,
            context_payload=context_payload,
            dataset_id=dataset_id,
            source_id=source_id,
            recipe=recipe,
            spec=version,
            frame=frame,
        )
        prepared[index] = result
        return result

    yield SimpleNamespace(
        connection=connection,
        workflow=workflow,
        caps=caps,
        values=values,
        storage=storage,
        prepare=prepare,
        service=MLRunService(workflow, storage),
    )
    connection.close()


def _task(state, item, *, role="modeling", seconds=180, name="task"):
    supervisor = Supervisor(state.workflow, storage_root=state.storage)
    task = supervisor.create_task(
        capability_id=item.cap.id,
        cycle_id=item.cycle.id,
        role=role,
        actor_label=ACTOR,
        operation_id=name,
        objective="Comprobar viabilidad del modelado",
        budget=AgentBudgetV1(max_steps=30, max_calls=12, max_seconds=seconds),
    )
    return supervisor, task


def _execute(supervisor, task, name, arguments, *, operation="tool", **changes):
    current = supervisor.repository.get_task(task.id)
    return supervisor.execute_tool(
        task.id,
        ToolCallV1(tool_name=name, arguments=arguments),
        expected_revision=current.revision,
        operation_id=operation,
        actor_label=ACTOR,
        **changes,
    )


def _propose(state, item, *, name="task"):
    supervisor, task = _task(state, item, name=name)
    result = _execute(
        supervisor,
        task,
        "propose_training_strategy",
        {"dataset_id": item.dataset_id},
        operation=f"propose-{name}",
    )
    assert result.status == "COMPLETED", result
    identifier = next(ref.id for ref in result.output_refs if ref.kind == "strategy")
    return supervisor, task, identifier


def _approve(state, strategy):
    state.workflow.decide_training(
        strategy_version_id=strategy, decision="APPROVED", actor_label=ACTOR
    )


def _counts(state):
    return tuple(
        state.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("runs", "training_operations", "approval_events")
    )


def test_preflight_proposal_has_real_purged_split_and_never_approves(state):
    item = state.prepare()
    supervisor, task = _task(state, item)
    viability = _execute(
        supervisor, task, "inspect_model_viability", {"dataset_id": item.dataset_id}
    )
    assert viability.summary["status"] == "ready", viability
    split = viability.summary["split"]
    assert split["purged_train_rows"] > 0 and split["purged_validation_rows"] > 0
    assert all(
        split[part]["positive_support"] > 0 and split[part]["negative_support"] > 0
        for part in ("train", "validation", "test")
    )
    result = _execute(
        supervisor,
        task,
        "propose_training_strategy",
        {"dataset_id": item.dataset_id},
        operation="propose",
    )
    strategy = next(ref for ref in result.output_refs if ref.kind == "strategy")
    proposal = StrategyProposalV1.model_validate_json(
        state.workflow.get_strategy(strategy.id)["strategy_json"]
    )
    assert proposal.feature_columns == ["activity_count"]
    assert proposal.split.label_available_at_column == "label_available_at"
    assert proposal.caps.max_training_seconds == 45 and proposal.human_approval_required
    assert not result.summary["grants_training_approval"]
    assert (
        _execute(
            supervisor,
            task,
            "propose_training_strategy",
            {"dataset_id": item.dataset_id},
            operation="propose",
        )
        == result
    )
    again = _execute(
        supervisor,
        task,
        "propose_training_strategy",
        {"dataset_id": item.dataset_id},
        operation="propose-again",
    )
    assert next(ref.id for ref in again.output_refs if ref.kind == "strategy") == strategy.id
    assert again.summary["already_exists"]
    assert _counts(state) == (0, 0, 0)


def test_two_objectives_train_real_isolated_models_and_exact_replay_never_refits(
    state, monkeypatch
):
    trained = []
    for index in (0, 1):
        item = state.prepare(index)
        supervisor, task, strategy = _propose(state, item, name=f"task-{index}")
        _approve(state, strategy)
        supervisor.prepare_turn(
            task.id,
            expected_revision=supervisor.repository.get_task(task.id).revision,
            actor_label=ACTOR,
        )
        result = _execute(
            supervisor, task, "train_model", {"strategy_id": strategy}, operation=f"train-{index}"
        )
        assert result.status == "COMPLETED", result
        run_id = next(ref.id for ref in result.output_refs if ref.kind == "run")
        row = state.workflow.get_run(run_id)
        assert row["status"] == "COMPLETED"
        manifest = json.loads(row["manifest_json"])
        binding = manifest["capability_training_binding"]
        assert binding["capability_id"] == item.cap.id and binding["cycle_id"] == item.cycle.id
        assert binding["predictive_version_id"] == item.spec.id
        assert len(manifest["label_readiness_sha256"]) == 64
        metrics = json.loads(row["candidate_metrics_json"])
        assert set(metrics["validation"]) == {
            "DummyClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
        }
        assert metrics["holdout_evaluations"] == 1
        splits = json.loads(row["split_info_json"])
        assert splits["temporal"]["purged_train_rows"] > 0
        model = joblib.load(state.storage / row["artifact_path"])
        predictions = model.predict(pd.DataFrame({"activity_count": [2, 4]})).tolist()
        assert not result.summary["published_by_this_tool"]
        trained.append((supervisor, task, strategy, result, row, predictions))
    assert trained[0][4]["artifact_path"] != trained[1][4]["artifact_path"]
    assert trained[0][5] != trained[1][5]
    assert _counts(state) == (2, 2, 2)

    def forbid(*args, **kwargs):
        raise AssertionError("El replay no puede ejecutar ML ni leer fuentes para entrenar")

    monkeypatch.setattr(MLRunService, "execute", forbid)
    for index, (supervisor, task, strategy, result, _, _) in enumerate(trained):
        assert (
            _execute(
                supervisor,
                task,
                "train_model",
                {"strategy_id": strategy},
                operation=f"train-{index}",
            )
            == result
        )
        duplicate = _execute(
            supervisor,
            task,
            "train_model",
            {"strategy_id": strategy},
            operation=f"repeat-fit-{index}",
        )
        assert duplicate.status == "FAILED" and duplicate.error_code == "MODEL_ALREADY_TRAINED"
    assert _counts(state) == (2, 2, 2)


def test_unapproved_strategy_cannot_train_through_tool_or_legacy_service_path(state):
    item = state.prepare()
    supervisor, task, strategy = _propose(state, item)
    result = _execute(
        supervisor, task, "train_model", {"strategy_id": strategy}, operation="denied"
    )
    assert result.status == "FAILED" and result.error_code == "STRATEGY_NOT_APPROVED"
    with pytest.raises(ApprovalRequiredError, match="STRATEGY_NOT_APPROVED"):
        state.service.execute(strategy)
    assert _counts(state) == (0, 0, 0)


@pytest.mark.parametrize("entry", ["service", "repository"])
def test_unbound_capability_cannot_bypass_guard_via_legacy_training_path(state, entry):
    item = state.prepare(bind=False)
    strategy_service = StrategyService(state.workflow)
    proposal = strategy_service.build_template(
        project_id=item.cycle.project_id, context_version_id=item.context.context_version_id
    )
    strategy = strategy_service.create_version(
        project_id=item.cycle.project_id, proposal=proposal
    ).strategy_version_id
    _approve(state, strategy)
    with pytest.raises(PredictiveGuardError, match="Vincula"):
        if entry == "service":
            state.service.execute(strategy)
        else:
            state.workflow.start_run(**_start_inputs(state.workflow, state.service, strategy))
    assert _counts(state) == (0, 0, 1)


@pytest.mark.parametrize(
    "tool", ["inspect_model_viability", "propose_training_strategy", "train_model"]
)
def test_modeling_tools_reject_cross_capability_before_claim(state, tool):
    first, second = state.prepare(), state.prepare(1)
    supervisor, task = _task(state, first)
    if tool == "train_model":
        _, _, strategy = _propose(state, second, name="second-task")
        arguments = {"strategy_id": strategy}
    else:
        arguments = {"dataset_id": second.dataset_id}
    with pytest.raises(OrchestrationError):
        _execute(supervisor, task, tool, arguments)
    assert supervisor.repository.get_task(task.id).steps_used == 0
    assert _counts(state) == (0, 0, 0)


@pytest.mark.parametrize("seconds,lease", [(74, None), (180, 60)])
def test_training_reserves_whole_cap_before_starting_ml(state, seconds, lease):
    item = state.prepare()
    _, _, strategy = _propose(state, item, name="proposal-task")
    _approve(state, strategy)
    supervisor, task = _task(state, item, seconds=seconds, name="training-task")
    with pytest.raises(OrchestrationError):
        _execute(supervisor, task, "train_model", {"strategy_id": strategy}, lease_seconds=lease)
    assert supervisor.repository.get_task(task.id).steps_used == 0
    assert _counts(state) == (0, 0, 1)


def test_context_change_after_binding_cannot_be_legitimized_by_new_approval(state):
    item = state.prepare()
    changed = deepcopy(item.context_payload)
    changed["prediction_moment"] = "Otro momento predictivo incompatible"
    context = BusinessContextService(state.workflow, state.storage).create_version(
        project_id=item.cycle.project_id, dataset_id=item.dataset_id, context=changed
    )
    strategy_service = StrategyService(state.workflow)
    proposal = strategy_service.build_template(
        project_id=item.cycle.project_id, context_version_id=context.context_version_id
    )
    strategy = strategy_service.create_version(
        project_id=item.cycle.project_id, proposal=proposal
    ).strategy_version_id
    _approve(state, strategy)
    with pytest.raises(PredictiveGuardError) as error:
        state.service.execute(strategy)
    assert error.value.code == "PREDICTIVE_CONTEXT_MISMATCH"
    assert _counts(state) == (0, 0, 1)


def test_changed_source_bytes_block_before_any_run_even_after_approval(state):
    item = state.prepare()
    _, _, strategy = _propose(state, item)
    _approve(state, strategy)
    source = state.workflow.get_dataset(item.source_id)
    path = state.storage / Path(source["stored_path"])
    path.write_bytes(path.read_bytes().replace(b"activity", b"tampered"))
    with pytest.raises(ValueError) as error:
        state.service.execute(strategy)
    assert getattr(error.value, "code", None) == "DATASET_HASH_MISMATCH"
    assert _counts(state) == (0, 0, 1)


def test_context_drift_after_run_creation_blocks_fit_and_preserves_manifest(state, monkeypatch):
    item = state.prepare()
    _, _, strategy = _propose(state, item)
    _approve(state, strategy)
    original = state.service._load_frame

    def load_then_change(dataset, proposal):
        frame = original(dataset, proposal)
        changed = deepcopy(item.context_payload)
        changed["prediction_moment"] = "Momento modificado durante ejecución"
        BusinessContextService(state.workflow, state.storage).create_version(
            project_id=item.cycle.project_id, dataset_id=item.dataset_id, context=changed
        )
        return frame

    monkeypatch.setattr(state.service, "_load_frame", load_then_change)
    with pytest.raises((ApprovalRequiredError, PredictiveGuardError, RunBlockedError)):
        state.service.execute(strategy, operation_id="drifting")
    row = state.connection.execute("SELECT * FROM runs").fetchone()
    assert row["status"] == "BLOCKED" and row["artifact_path"] is None
    assert (
        json.loads(row["manifest_json"])["capability_training_binding"]["cycle_id"] == item.cycle.id
    )
    assert not (state.storage / "artifacts").exists()


def _runner_step(runner, task, operation):
    current = runner.supervisor.repository.get_task(task.id)
    return runner.step(
        task.id, expected_revision=current.revision, actor_label=ACTOR, operation_id=operation
    )


def test_fake_coordinator_delegates_modeling_until_explicit_human_approval(state):
    item = state.prepare()

    def dataset_tool(name):
        def response(request):
            context = _wire_context(request)
            assert context["role"] == "modeling"
            dataset = next(row for row in context["datasets"] if row["is_active"])
            return _tool(name, {"dataset_id": dataset["id"]})

        return response

    def request_review(request):
        context = _wire_context(request)
        current = context["modeling_context"][0]
        assert current["strategy_id"].startswith("R") and current["training_approval"] == "MISSING"
        assert context["tool_evidence"][-1]["summary"]["grants_training_approval"] is False
        return _wait()

    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "Solicito revisar viabilidad y estrategia",
            "delegate_role": "modeling",
            "delegate_objective": "Revisar soporte y proponer una estrategia",
        },
        dataset_tool("inspect_model_viability"),
        dataset_tool("propose_training_strategy"),
        request_review,
    )
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage)
    task = runner.start(
        capability_id=item.cap.id,
        cycle_id=item.cycle.id,
        objective="Revisar el modelo",
        actor_label=ACTOR,
        operation_id="coordinator",
        budget=AgentBudgetV1(max_steps=30, max_calls=8, max_seconds=180),
    )
    outcomes = runner.run(
        task.id,
        expected_revision=task.revision,
        actor_label=ACTOR,
        operation_id="loop",
        max_turns=6,
    )
    assert len(provider.requests) == 4
    assert all(outcome.error_code is None for outcome in outcomes), outcomes
    children = runner.supervisor.repository.list_children(task.id)
    assert len(children) == 1 and children[0].role == "modeling"
    assert children[0].state == "WAITING_HUMAN"
    assert runner.supervisor.repository.get_task(task.id).calls_used == 4
    assert state.workflow.get_latest_strategy(item.context.context_version_id) is not None
    assert _counts(state) == (0, 0, 0)
    wire = " ".join(request.user for request in provider.requests)
    for private in (
        item.cap.id,
        item.cycle.id,
        children[0].id,
        item.dataset_id,
        item.spec.id,
        item.spec.sha256,
        "PRIVATE-CASE",
        "activity_count",
        "no_first_purchase",
        "label_available_at",
        str(state.storage),
    ):
        assert private not in wire


@pytest.mark.parametrize("role", ["coordinator", "business", "data"])
def test_other_roles_cannot_train_or_propose_strategy(state, role):
    item = state.prepare()
    supervisor, task = _task(state, item, role=role)
    for tool, arguments in (
        ("propose_training_strategy", {"dataset_id": item.dataset_id}),
        ("train_model", {"strategy_id": "unseen"}),
    ):
        with pytest.raises(OrchestrationError) as error:
            _execute(supervisor, task, tool, arguments)
        assert error.value.code == "TOOL_ROLE_DENIED"
    assert _counts(state) == (0, 0, 0)


def test_provider_cannot_supply_approval_or_metrics_in_training_arguments(state):
    item = state.prepare()
    _, _, strategy = _propose(state, item, name="proposal")

    def forge(request):
        alias = _wire_context(request)["modeling_context"][0]["strategy_id"]
        return _tool("train_model", {"strategy_id": alias, "approved": True, "accuracy": 1.0})

    provider = FakeAgentProvider(forge, forge)
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage)
    task = runner.supervisor.create_task(
        capability_id=item.cap.id,
        cycle_id=item.cycle.id,
        role="modeling",
        actor_label=ACTOR,
        operation_id="modeling",
        budget=AgentBudgetV1(max_steps=10, max_calls=4, max_seconds=180),
    )
    result = _runner_step(runner, task, "forged")
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"
    assert len(provider.requests) == 2 and result.task.state == "WAITING_HUMAN"
    assert state.workflow.get_latest_training_decision(strategy) is None
    assert _counts(state) == (0, 0, 0)


def test_context_change_during_provider_call_prevents_strategy_effect(state):
    item = state.prepare()

    def change_context(request):
        dataset = next(row for row in _wire_context(request)["datasets"] if row["is_active"])
        changed = deepcopy(item.context_payload)
        changed["prediction_moment"] = "Una definición diferente durante la llamada"
        BusinessContextService(state.workflow, state.storage).create_version(
            project_id=item.cycle.project_id, dataset_id=item.dataset_id, context=changed
        )
        return _tool("propose_training_strategy", {"dataset_id": dataset["id"]})

    provider = FakeAgentProvider(change_context)
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage)
    task = runner.supervisor.create_task(
        capability_id=item.cap.id,
        cycle_id=item.cycle.id,
        role="modeling",
        actor_label=ACTOR,
        operation_id="modeling",
        budget=AgentBudgetV1(max_steps=10, max_calls=4, max_seconds=180),
    )
    result = _runner_step(runner, task, "stale")
    assert result.error_code == "STALE_CONTEXT" and result.proposal is None
    assert result.task.state == "WAITING_HUMAN" and len(provider.requests) == 1
    assert state.connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 0
    assert _counts(state) == (0, 0, 0)


def test_strategy_commit_survives_process_loss_without_creating_second_version(state, monkeypatch):
    item = state.prepare()
    supervisor, task = _task(state, item)
    now = datetime(2026, 10, 23, 12, tzinfo=UTC)
    supervisor.clock = lambda: now
    original = supervisor.repository.finish_operation

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("Simulated process loss after domain commit")

    monkeypatch.setattr(supervisor.repository, "finish_operation", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _execute(
            supervisor,
            task,
            "propose_training_strategy",
            {"dataset_id": item.dataset_id},
            operation="lost-proposal",
        )
    strategy = state.workflow.get_latest_strategy(item.context.context_version_id)
    assert strategy is not None
    monkeypatch.setattr(supervisor.repository, "finish_operation", original)
    supervisor.recover_expired_operations(now=now + timedelta(seconds=61))
    operation = supervisor.repository.get_operation("lost-proposal")
    assert operation.result is not None and operation.result.status == "COMPLETED"
    assert operation.result.summary["recovered_from_persisted_effect"]
    assert operation.result.output_refs[0].id == strategy["id"]
    assert supervisor.repository.get_task(task.id).state == "WAITING_HUMAN"
    assert state.connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 1
    assert _counts(state) == (0, 0, 0)


def test_nonempty_labels_do_not_make_a_purged_empty_partition_viable(state):
    item = state.prepare(count=20, horizon=60)
    assert not item.frame["no_first_purchase"].isna().any()
    supervisor, task = _task(state, item)
    result = _execute(
        supervisor, task, "propose_training_strategy", {"dataset_id": item.dataset_id}
    )
    assert result.status == "COMPLETED" and result.summary["status"] == "needs_data"
    assert result.summary["reason_codes"] and result.summary["measurement_plan"]
    assert not any(ref.kind == "strategy" for ref in result.output_refs)
    assert _counts(state) == (0, 0, 0)


def test_completed_model_is_recovered_after_supervisor_process_loss_without_refit(
    state, monkeypatch
):
    item = state.prepare()
    supervisor, task, strategy = _propose(state, item)
    _approve(state, strategy)
    supervisor.prepare_turn(
        task.id,
        expected_revision=supervisor.repository.get_task(task.id).revision,
        actor_label=ACTOR,
    )
    now = datetime(2026, 10, 23, 12, tzinfo=UTC)
    supervisor.clock = lambda: now
    original = supervisor.repository.finish_operation

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("Simulated process loss after model commit")

    monkeypatch.setattr(supervisor.repository, "finish_operation", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _execute(supervisor, task, "train_model", {"strategy_id": strategy}, operation="lost-model")
    row = state.workflow.get_latest_run(strategy)
    assert row["status"] == "COMPLETED" and (state.storage / row["artifact_path"]).is_file()
    monkeypatch.setattr(supervisor.repository, "finish_operation", original)

    def forbid(*args, **kwargs):
        raise AssertionError("La recuperación debe leer el resultado sin entrenar")

    monkeypatch.setattr(MLRunService, "execute", forbid)
    supervisor.recover_expired_operations(now=now + timedelta(seconds=76))
    operation = supervisor.repository.get_operation("lost-model")
    assert operation.result is not None and operation.result.status == "COMPLETED"
    assert operation.result.summary["recovered_from_persisted_effect"]
    assert operation.result.output_refs[0].id == row["id"]
    assert supervisor.repository.get_task(task.id).state == "WAITING_HUMAN"
    assert (
        _execute(supervisor, task, "train_model", {"strategy_id": strategy}, operation="lost-model")
        == operation.result
    )
    assert _counts(state) == (1, 1, 1)


def test_strategy_preflight_race_is_checked_inside_creation_transaction(state, monkeypatch):
    item = state.prepare()
    supervisor, task = _task(state, item)
    original = state.workflow.add_strategy

    def change_before_insert(*args, **kwargs):
        record_snapshot_recipe(
            state.workflow,
            project_id=item.cycle.project_id,
            table_dataset_id=item.dataset_id,
            source_dataset_id=item.source_id,
            spec=item.recipe,
            actor_label=ACTOR,
            coverage={**COVERAGE, "observed_until": "2026-10-15T00:00:00Z"},
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(state.workflow, "add_strategy", change_before_insert)
    result = _execute(
        supervisor, task, "propose_training_strategy", {"dataset_id": item.dataset_id}
    )
    assert result.status == "FAILED" and result.error_code == "STALE_CONTEXT"
    assert state.connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 0
    assert _counts(state) == (0, 0, 0)
