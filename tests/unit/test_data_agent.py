"""MV-08: análisis local con datos sintéticos y proveedor falso, sin red ni entrenamiento."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.ai_engine.agents import AgentRunner
from core.analysis.snapshot import build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.contracts.orchestration import AgentBudgetV1, ToolCallV1
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.connection import transaction
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService
from core.domain.data_readiness import DataReadinessService
from core.domain.decision_value import DecisionValueService
from core.orchestration.runtime import OrchestrationError, Supervisor
from tests.unit.test_agent_runner import FakeAgentProvider, _step, _tool, _wait, _wire_context

ACTOR = "Persona responsable de datos sintéticos"
PRIVATE_CATEGORY = "Private Segment"


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "data-agent.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(
        name="Producto de prueba", actor_label=ACTOR, operation_id="product"
    )
    caps = [
        capabilities.create_capability(
            product.id,
            name=kind,
            objective="Entender la calidad antes de decidir",
            value_kind=kind,
            decision_unit="caso",
            actor_label=ACTOR,
            operation_id=kind,
        )
        for kind in ("revenue", "savings")
    ]
    cycles = [
        capabilities.create_cycle(
            cap.id, reason="Cohorte sintética", actor_label=ACTOR, operation_id=f"cycle-{index}"
        )
        for index, cap in enumerate(caps)
    ]
    frame = pd.DataFrame(
        {
            "record_key": [f"RAW-ROW-{index:03d}" for index in range(10)],
            "signal": [10.0] * 8 + [1000.0, None],
            "category": [PRIVATE_CATEGORY, "private segment "] * 5,
            "outcome": [0, 1] * 5,
        }
    )
    storage_root = tmp_path / "private-storage"
    ingest = DataIngestionService(workflow, storage_root)
    datasets = [
        ingest.ingest_bytes(
            project_id=cycle.project_id,
            request={
                "original_filename_display": "synthetic.csv",
                "source_name": "Prueba",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=frame.to_csv(index=False).encode(),
        ).metadata.dataset_id
        for cycle in cycles
    ]
    yield SimpleNamespace(
        connection=connection,
        workflow=workflow,
        capabilities=capabilities,
        caps=caps,
        cycles=cycles,
        datasets=datasets,
        frame=frame,
        storage_root=storage_root,
        ingest=ingest,
        values=DecisionValueService(workflow),
    )
    connection.close()


def _data_task(state, *, role="data", name="task"):
    supervisor = Supervisor(state.workflow, storage_root=state.storage_root)
    task = supervisor.create_task(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id,
        role=role,
        actor_label=ACTOR,
        operation_id=name,
        objective="Revisar calidad y maduración",
        budget=AgentBudgetV1(max_steps=30, max_calls=10, max_seconds=120),
    )
    return supervisor, task


def _execute(supervisor, task, tool, arguments, *, operation="tool"):
    task = supervisor.repository.get_task(task.id)
    return supervisor.execute_tool(
        task.id,
        ToolCallV1(tool_name=tool, arguments=arguments),
        expected_revision=task.revision,
        operation_id=operation,
        actor_label=ACTOR,
    )


def _arguments(tool, dataset_id):
    return {
        "dataset_id": dataset_id,
        **(
            {"template": "numeric_summary", "column_alias": "C002"}
            if tool == "aggregate_analysis"
            else {}
        ),
    }


DATA_TOOLS = [
    "read_dataset_profile",
    "inspect_data_quality",
    "inspect_label_readiness",
    "aggregate_analysis",
]


def _assert_no_training(state):
    assert state.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert state.connection.execute("SELECT COUNT(*) FROM approval_events").fetchone()[0] == 0


@pytest.mark.parametrize("tool", DATA_TOOLS)
def test_data_tools_refuse_foreign_capability_before_loading_source(state, tool):
    supervisor, task = _data_task(state)
    with pytest.raises(OrchestrationError):
        _execute(supervisor, task, tool, _arguments(tool, state.datasets[1]))
    assert supervisor.repository.get_task(task.id).steps_used == 0
    _assert_no_training(state)


@pytest.mark.parametrize("tool", DATA_TOOLS)
def test_data_tools_detect_modified_accepted_file(state, tool):
    dataset = state.workflow.get_dataset(state.datasets[0])
    path = state.storage_root / Path(dataset["stored_path"])
    path.write_bytes(path.read_bytes().replace(b"1000.0", b"9999.0"))
    supervisor, task = _data_task(state)
    result = _execute(supervisor, task, tool, _arguments(tool, state.datasets[0]))
    assert result.status == "FAILED"
    assert result.error_code == "DATASET_HASH_MISMATCH"
    assert "9999" not in json.dumps(result.model_dump(mode="json"))
    _assert_no_training(state)


@pytest.mark.parametrize("tool", DATA_TOOLS)
def test_quarantined_data_never_reaches_inspection_or_provider(state, tool):
    pending = state.ingest.ingest_bytes(
        project_id=state.cycles[0].project_id,
        request={
            "original_filename_display": "pending.csv",
            "source_name": "Pendiente",
            "classification": "synthetic",
            "authorization_status": "pending",
        },
        content=state.frame.iloc[:9].to_csv(index=False).encode(),
    )
    assert pending.metadata.admission_status == "QUARANTINED"
    supervisor, task = _data_task(state)
    result = _execute(supervisor, task, tool, _arguments(tool, pending.metadata.dataset_id))
    assert result.status == "FAILED"
    assert result.error_code == "DATASET_NOT_ACCEPTED"
    assert PRIVATE_CATEGORY not in json.dumps(result.model_dump(mode="json"))


def test_numeric_aggregate_is_computed_locally_without_individual_values(state):
    supervisor, task = _data_task(state)
    result = _execute(
        supervisor, task, "aggregate_analysis", _arguments("aggregate_analysis", state.datasets[0])
    )
    assert result.status == "COMPLETED"
    assert result.summary["metrics"]["count"] == 9
    assert result.summary["metrics"]["mean"] == pytest.approx(120)
    assert result.summary["metrics"]["sum"] == pytest.approx(1080)
    encoded = json.dumps(result.summary)
    assert "rows" not in result.summary
    assert PRIVATE_CATEGORY not in encoded
    assert "RAW-ROW-" not in encoded
    assert "1000" not in encoded
    assert "minimum" not in encoded and "maximum" not in encoded
    _assert_no_training(state)


def test_quality_returns_counts_without_sensitive_category_variants_or_detail(state):
    supervisor, task = _data_task(state)
    result = _execute(supervisor, task, "inspect_data_quality", {"dataset_id": state.datasets[0]})
    assert result.status == "COMPLETED"
    findings = result.summary["findings"]
    assert any(item["kind"] == "EXTREME_OUTLIER" and item["count"] == 1 for item in findings)
    assert any(item["kind"] == "NEAR_DUPLICATE_CATEGORY" for item in findings)
    assert all("detail" not in item for item in findings)
    encoded = json.dumps(result.summary)
    assert PRIVATE_CATEGORY not in encoded
    assert "private segment" not in encoded
    assert "RAW-ROW-" not in encoded
    assert "data/accepted/" not in encoded


@pytest.mark.parametrize(
    "arguments",
    [
        {"template": "select", "columns": ["record_key"]},
        {"template": "numeric_summary", "column_alias": "signal"},
        {"template": "numeric_summary", "column_alias": "C999"},
        {"template": "numeric_summary", "column_alias": "C002", "sql": "SELECT * FROM source"},
        {"template": "numeric_summary", "column_alias": "C002", "path": "private.csv"},
    ],
)
def test_analysis_cannot_request_rows_free_sql_paths_or_unpublished_columns(state, arguments):
    supervisor, task = _data_task(state)
    try:
        result = _execute(
            supervisor, task, "aggregate_analysis", {"dataset_id": state.datasets[0], **arguments}
        )
    except OrchestrationError:
        pass
    else:
        assert result.status == "FAILED"
    _assert_no_training(state)


def test_data_tools_require_explicit_private_storage_configuration(state):
    supervisor = Supervisor(state.workflow)
    task = supervisor.create_task(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id,
        role="data",
        actor_label=ACTOR,
        operation_id="no-storage",
    )
    result = _execute(supervisor, task, "inspect_data_quality", {"dataset_id": state.datasets[0]})
    assert result.status == "FAILED"
    assert result.error_code == "MISSING_STORAGE"


def _snapshot_case(state, *, observed_until="2026-01-25T00:00:00Z", deadline=False):
    events = pd.DataFrame(
        {
            "case": ["PRIVATE-OLD", "PRIVATE-OLD", "PRIVATE-NEW", "PRIVATE-NEW"],
            "moment": ["2026-01-01", "2026-01-04", "2026-01-20", "2026-01-21"],
            "stage": ["signup", "purchase", "signup", "purchase"],
        }
    )
    recipe = {
        "entity": "case",
        "moment": "moment",
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
        "windows_days": [2],
        "features": [{"op": "count_events", "window_days": 2, "output_column": "activity_count"}],
        "outcome": [
            {
                "op": "not_reached_stage",
                "stage": {"column": "stage", "value": "purchase"},
                "within_days": 14,
                "output_column": "no_purchase",
            }
        ],
    }
    coverage = {
        "observed_from": "2026-01-01T00:00:00Z",
        "observed_until": observed_until,
        "complete": True,
    }
    snapshot = build_snapshot(events, recipe, coverage=coverage)
    project = state.cycles[0].project_id

    def ingest(frame, filename):
        return state.ingest.ingest_bytes(
            project_id=project,
            request={
                "original_filename_display": filename,
                "source_name": "Prueba sintética",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=frame.to_csv(index=False).encode(),
        ).metadata.dataset_id

    source = ingest(events, "events.csv")
    table = ingest(
        snapshot.training_frame() if snapshot.readiness.eligible_rows else snapshot.dataframe,
        "snapshot.csv",
    )
    record_snapshot_recipe(
        state.workflow,
        project_id=project,
        table_dataset_id=table,
        source_dataset_id=source,
        spec=recipe,
        coverage=coverage,
        readiness=snapshot.readiness,
        actor_label=ACTOR,
    )
    predictive = state.values.create_predictive_version(
        state.caps[0].id,
        spec={
            "population": "Casos sintéticos elegibles",
            "target_definition": "No compra tras la ventana",
            "target_column": "no_purchase",
            "positive_class": True,
            "prediction_moment": "Al día dos de la primera actividad",
            "primary_metric": "f1",
            "outcome_window": {"kind": "deadline_column", "column": "declared_deadline"}
            if deadline
            else {"kind": "fixed_days", "days": 14},
            "feature_decisions": [
                {
                    "column": "activity_count",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Anterior al corte",
                }
            ],
            "snapshot_recipe": recipe,
        },
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="predictive",
    )
    state.values.bind_predictive_to_cycle(
        state.caps[0].id,
        cycle_id=state.cycles[0].id,
        predictive_version_id=predictive.id,
        actor_label=ACTOR,
        operation_id="bind",
    )
    return SimpleNamespace(table=table, source=source, snapshot=snapshot, predictive=predictive)


def _readiness(state, dataset):
    return DataReadinessService(state.workflow, state.storage_root).inspect_label_readiness(
        state.caps[0].id,
        dataset_id=dataset,
        cycle_id=state.cycles[0].id,
    )


def test_filled_tabular_labels_do_not_prove_observation_coverage(state):
    result = _readiness(state, state.datasets[0])
    assert result.status == "needs_human"
    assert "SNAPSHOT_PROVENANCE_MISSING" in result.reason_codes
    assert result.verified_table_rows == 0
    assert result.measurement_plan
    assert result.grants_training_approval is False


def test_unmatured_snapshot_requests_data_instead_of_inventing_negative_labels(state):
    cohort = _snapshot_case(state, observed_until="2026-01-10T00:00:00Z")
    result = _readiness(state, cohort.table)
    assert result.status == "needs_data"
    assert "NO_MATURE_ROWS" in result.reason_codes
    assert result.snapshot_readiness.eligible_rows == 0
    assert result.pending_source_count == 2
    assert result.reason_counts["OUTCOME_WINDOW_NOT_MATURE"] == 2
    assert result.measurement_plan
    _assert_no_training(state)


def test_mature_subset_is_ready_only_with_exclusion_evidence_and_measurement_plan(state):
    cohort = _snapshot_case(state)
    result = _readiness(state, cohort.table)
    assert result.status == "ready"
    assert result.verified_table_rows == 1
    assert result.pending_source_count == 1
    assert "PENDING_SOURCE_ROWS_EXCLUDED" in result.reason_codes
    assert result.measurement_plan
    assert result.grants_training_approval is False
    assert result.statistical_support_checked is False
    _assert_no_training(state)


def test_deadline_definition_is_not_silently_replaced_by_recipe_days(state):
    cohort = _snapshot_case(state, observed_until="2026-02-10T00:00:00Z", deadline=True)
    result = _readiness(state, cohort.table)
    assert result.status == "needs_human"
    assert "PREDICTIVE_WINDOW_MISMATCH" in result.reason_codes
    assert result.grants_training_approval is False


def test_small_numeric_aggregate_suppresses_value_statistics(state):
    small = state.ingest.ingest_bytes(
        project_id=state.cycles[0].project_id,
        request={
            "original_filename_display": "small.csv",
            "source_name": "Prueba",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=b"signal\n10\n10\n20\n20\n",
    ).metadata.dataset_id
    supervisor, task = _data_task(state)
    result = _execute(
        supervisor,
        task,
        "aggregate_analysis",
        {
            "dataset_id": small,
            "template": "numeric_summary",
            "column_alias": "C001",
        },
    )
    assert result.status == "COMPLETED"
    assert result.summary["metrics"] == {"count": 4, "mean": None, "sum": None}
    assert "AGGREGATE_SUPPORT_TOO_SMALL" in json.dumps(result.summary)


def test_coordinator_delegates_data_and_immature_results_change_route_to_measurement(state):
    cohort = _snapshot_case(state, observed_until="2026-01-10T00:00:00Z")
    state.workflow.set_modeling_table(state.cycles[0].project_id, cohort.table, actor_label=ACTOR)

    def inspect_last_dataset(tool):
        def response(request):
            context = _wire_context(request)
            active = next(item for item in context["datasets"] if item["is_active"])
            return _tool(tool, {"dataset_id": active["id"]})

        return response

    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "El especialista de datos revisará la evidencia.",
            "delegate_role": "data",
            "delegate_objective": "Verificar calidad y maduración del resultado",
        },
        inspect_last_dataset("inspect_data_quality"),
        inspect_last_dataset("inspect_label_readiness"),
        {
            "action": "wait_data",
            "message": "La observación debe continuar antes de usar estos resultados.",
            "questions": ["¿Cuándo estará disponible la cobertura completa del resultado?"],
            "reason_code": "OUTCOME_WINDOW_NOT_MATURE",
        },
    )
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage_root)
    root = runner.start(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id,
        objective="Revisar los datos antes de elegir una intervención",
        actor_label=ACTOR,
        operation_id="team",
        budget=AgentBudgetV1(max_calls=4, max_steps=15, max_seconds=120),
    )
    results = runner.run(
        root.id,
        expected_revision=root.revision,
        actor_label=ACTOR,
        operation_id="run",
        max_turns=8,
    )
    children = runner.supervisor.repository.list_children(root.id)
    assert len(children) == 1
    assert children[0].role == "data"
    assert children[0].state == "WAITING_DATA"
    assert children[0].root_task_id == root.id
    assert len(provider.requests) == 4
    assert runner.supervisor.repository.get_task(root.id).calls_used == 4
    readiness = next(
        result.tool_result
        for result in results
        if result.tool_result is not None
        and result.tool_result.summary.get("kind") == "label_readiness"
    )
    assert readiness.summary["status"] == "needs_data"
    assert readiness.summary["measurement_plan"]
    assert readiness.output_refs[0].id == cohort.table
    outgoing = "\n".join(request.user for request in provider.requests)
    for private in (
        state.caps[0].id,
        state.cycles[0].id,
        root.id,
        children[0].id,
        *state.datasets,
        cohort.source,
        cohort.table,
        "PRIVATE-OLD",
        "PRIVATE-NEW",
        "RAW-ROW-",
        PRIVATE_CATEGORY,
        "private segment",
        str(state.storage_root),
        "data/accepted/",
        '"detail"',
    ):
        assert private not in outgoing
    assert "OUTCOME_WINDOW_NOT_MATURE" in provider.requests[-1].user
    assert "REGISTERED_PROFILE_NOT_CURRENT_INSPECTION" in outgoing
    for key in ("evidence_sha256", "recipe_sha256", "binding_sha256"):
        assert len(readiness.summary[key]) == 64
        assert readiness.summary[key] not in outgoing
    _assert_no_training(state)


def test_data_aggregate_alias_roundtrip_keeps_ids_local_and_passes_only_computed_evidence(state):
    def calculate(request):
        context = _wire_context(request)
        dataset_alias = context["datasets"][0]["id"]
        ref = next(ref for ref in context["available_refs"] if ref["id"] == dataset_alias)
        return {
            **_tool(
                "aggregate_analysis",
                {
                    "dataset_id": dataset_alias,
                    "template": "numeric_summary",
                    "column_alias": "C002",
                },
            ),
            "evidence_refs": [ref],
        }

    provider = FakeAgentProvider(calculate, _wait("wait_data"))
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage_root)
    task = runner.supervisor.create_task(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id,
        role="data",
        actor_label=ACTOR,
        operation_id="data-task",
        objective="Revisar agregados seguros",
    )
    result = _step(runner, task, "aggregate")
    assert result.error_code is None
    assert result.proposal.tool_call.arguments["dataset_id"] == state.datasets[0]
    assert result.proposal.tool_call.arguments["column_alias"] == "C002"
    assert result.proposal.evidence_refs[0].id == state.datasets[0]
    assert result.tool_result.summary["metrics"]["mean"] == 120
    _step(runner, task, "wait")
    outgoing = "\n".join(request.user for request in provider.requests)
    assert state.datasets[0] not in outgoing
    assert state.workflow.get_dataset(state.datasets[0])["sha256"] not in outgoing
    assert "signal" not in outgoing
    assert '"mean": 120.0' in outgoing or '"mean": 120' in outgoing
    assert PRIVATE_CATEGORY not in outgoing
    _assert_no_training(state)


@pytest.mark.parametrize("semantic_type", ["personal", "identifier"])
def test_numeric_storage_does_not_authorize_aggregating_personal_or_identifier_columns(
    state, semantic_type
):
    dataset = state.workflow.get_dataset(state.datasets[0])
    profile = json.loads(dataset["profile_json"])
    profile["columns"][1]["semantic_type"] = semantic_type
    profile["columns"][1]["possible_identifier"] = False
    state.connection.execute(
        "UPDATE datasets SET profile_json=? WHERE id=?",
        (json.dumps(profile), state.datasets[0]),
    )
    supervisor, task = _data_task(state)
    result = _execute(
        supervisor, task, "aggregate_analysis", _arguments("aggregate_analysis", state.datasets[0])
    )
    assert result.status == "FAILED"
    assert result.error_code == "AGGREGATE_COLUMN_NOT_ALLOWED"
    assert "mean" not in result.summary


def test_changed_snapshot_metadata_during_provider_call_invalidates_finish(state):
    def change_metadata(_request):
        assert state.connection.in_transaction is False
        with transaction(state.connection):
            state.workflow._audit(
                project_id=state.cycles[0].project_id,
                event_type="SNAPSHOT_TABLE_SAVED",
                entity_type="DATASET",
                entity_id=state.datasets[0],
                detail={"new_declaration": "Metadata sintética pendiente de revisión"},
            )
        return {"action": "finish", "message": "La revisión de datos está terminada."}

    provider = FakeAgentProvider(change_metadata)
    runner = AgentRunner(state.workflow, provider, storage_root=state.storage_root)
    task = runner.supervisor.create_task(
        capability_id=state.caps[0].id,
        cycle_id=state.cycles[0].id,
        role="data",
        actor_label=ACTOR,
        operation_id="data-task",
        objective="Verificar evidencia disponible",
    )
    result = _step(runner, task)
    assert result.error_code == "STALE_CONTEXT"
    assert result.task.state == "WAITING_HUMAN"
    assert result.proposal is None
    assert len(provider.requests) == 1
    _assert_no_training(state)
