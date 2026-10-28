"""MV-09: un binding confirmado no permite entrenar con otra semántica o receta."""

from types import SimpleNamespace

import pandas as pd
import pytest

from core.analysis.snapshot import SnapshotSpecV1, build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.context_service import BusinessContextService
from core.domain.data_readiness import DataReadinessService
from core.domain.decision_value import DecisionValueService
from core.domain.predictive_guard import (
    PredictiveGuardError,
    require_capability_predictive_context,
    require_capability_training_binding,
)
from core.domain.strategy_service import StrategyService


@pytest.fixture
def bound_cycle(tmp_path):
    connection = get_connection(tmp_path / "guard.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    capabilities = CapabilityService(repository)
    product = capabilities.create_product(name="Sandbox", actor_label="Human", operation_id="p")
    capability = capabilities.create_capability(
        product.id,
        name="Activación",
        objective="Objetivo sintético",
        value_kind="revenue",
        decision_unit="alta",
        actor_label="Human",
        operation_id="cap",
    )
    cycle = capabilities.create_cycle(
        capability.id, reason="Inicial", actor_label="Human", operation_id="cycle"
    )
    events = []
    for index in range(80):
        first = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index * 3)
        entity = f"SYN-{index:04d}"
        events.append((entity, first.isoformat(), "signup"))
        if index % 2:
            events.append((entity, (first + pd.Timedelta(days=4)).isoformat(), "purchase"))
        else:
            events.append((entity, (first + pd.Timedelta(days=1)).isoformat(), "view"))
    events = pd.DataFrame(events, columns=["entity", "moment", "stage"])
    recipe = SnapshotSpecV1.model_validate(
        {
            "entity": "entity",
            "moment": "moment",
            "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
            "windows_days": [3],
            "features": [{"op": "count_events", "window_days": 3}],
            "outcome": [
                {
                    "op": "not_reached_stage",
                    "stage": {"column": "stage", "value": "purchase"},
                    "within_days": 7,
                    "output_column": "adverse",
                }
            ],
            "label_available_at_output_column": "label_available_at",
        }
    )
    coverage = {
        "observed_from": "2025-01-01T00:00:00Z",
        "observed_until": "2026-01-01T00:00:00Z",
        "complete": True,
    }
    built = build_snapshot(events, recipe, coverage=coverage)
    table = built.training_frame()
    ingestion = DataIngestionService(repository, tmp_path)

    def ingest(frame, name):
        return ingestion.ingest_bytes(
            project_id=cycle.project_id,
            request={
                "original_filename_display": name,
                "source_name": "Demo sintética",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=frame.to_csv(index=False).encode("utf-8"),
        ).metadata.dataset_id

    source_id = ingest(events, "events.csv")
    dataset_id = ingest(table, "table.csv")
    record_snapshot_recipe(
        repository,
        project_id=cycle.project_id,
        table_dataset_id=dataset_id,
        source_dataset_id=source_id,
        spec=recipe,
        actor_label="Human",
        coverage=coverage,
        readiness=built.readiness,
    )
    repository.set_modeling_table(cycle.project_id, dataset_id, actor_label="Human")
    context_input = {
        "objective": "Prevenir un evento adverso sintético",
        "target_column": "adverse",
        "positive_class": True,
        "prediction_moment": "Al corte de la receta",
        "false_positive_consequence": "Contacto innecesario",
        "false_negative_consequence": "Caso sin ayuda",
        "primary_metric": "f1",
        "metric_rationale": "Equilibrio del escenario",
        "decision_actor_label": "Human",
        "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        "feature_decisions": [
            {
                "column": name,
                "use": "include" if name == "count_events_3d" else "exclude",
                "available_at_prediction": name != "label_available_at",
                "reason": "Papel declarado en la receta",
            }
            for name in table.columns
            if name != "adverse"
        ],
    }
    context = BusinessContextService(repository, tmp_path).create_version(
        project_id=cycle.project_id, dataset_id=dataset_id, context=context_input
    )
    values = DecisionValueService(repository)
    draft = values.draft_predictive_from_context(
        capability.id, context_version_id=context.context_version_id
    )
    spec = {
        **draft.model_dump(mode="json"),
        "population": "Casos sintéticos",
        "target_definition": "No alcanzar el evento de la receta",
        "outcome_window": {"kind": "fixed_days", "days": 7},
    }
    version = values.create_predictive_version(
        capability.id, spec=spec, confirmed_human=True, actor_label="Human", operation_id="spec"
    )
    binding = values.bind_predictive_to_cycle(
        capability.id,
        cycle_id=cycle.id,
        predictive_version_id=version.id,
        actor_label="Human",
        operation_id="bind",
    )
    strategies = StrategyService(repository)
    proposal = strategies.build_template(
        project_id=cycle.project_id, context_version_id=context.context_version_id
    )
    proposal = {
        **proposal.model_dump(mode="json"),
        "split": {
            "method": "temporal",
            "train": 0.6,
            "validation": 0.2,
            "test": 0.2,
            "time_column": "snapshot_cutoff",
            "label_available_at_column": "label_available_at",
        },
    }
    strategy = strategies.create_version(project_id=cycle.project_id, proposal=proposal)
    yield SimpleNamespace(
        repository=repository,
        capabilities=capabilities,
        capability=capability,
        cycle=cycle,
        values=values,
        binding=binding,
        version=version,
        strategies=strategies,
        strategy=strategy,
        proposal=proposal,
        recipe=recipe,
        source_id=source_id,
        dataset_id=dataset_id,
        context=context,
        context_input=context_input,
        root=tmp_path,
        coverage=coverage,
        readiness=built.readiness,
        table=table,
    )
    connection.close()


def test_guard_binds_exact_definition_and_does_not_require_or_create_an_approval(bound_cycle):
    state = bound_cycle
    guard = require_capability_training_binding(
        state.repository, state.strategy.strategy_version_id
    )
    assert guard.binding_id == state.binding.id
    assert guard.predictive_version_id == state.version.id
    assert guard.label_available_at_column == "label_available_at"
    assert guard.recipe_sha256 == canonical_hash(state.recipe)
    assert guard.fingerprint == canonical_hash(guard)
    assert (
        state.repository.connection.execute("SELECT count(*) FROM approval_events").fetchone()[0]
        == 0
    )
    ready = DataReadinessService(state.repository, state.root).inspect_label_readiness(
        state.capability.id, dataset_id=state.dataset_id, cycle_id=state.cycle.id
    )
    assert ready.status == "ready" and ready.verified_table_rows == 80


def test_new_context_after_binding_cannot_reuse_confirmation(bound_cycle):
    state = bound_cycle
    BusinessContextService(state.repository, state.root).create_version(
        project_id=state.cycle.project_id,
        dataset_id=state.dataset_id,
        context={**state.context_input, "prediction_moment": "Otro momento declarado"},
    )
    with pytest.raises(PredictiveGuardError) as error:
        require_capability_training_binding(state.repository, state.strategy.strategy_version_id)
    assert error.value.code == "PREDICTIVE_CONTEXT_MISMATCH"


@pytest.mark.parametrize(
    "split",
    [
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        {
            "method": "temporal",
            "train": 0.6,
            "validation": 0.2,
            "test": 0.2,
            "time_column": "snapshot_cutoff",
        },
        {
            "method": "temporal",
            "train": 0.6,
            "validation": 0.2,
            "test": 0.2,
            "time_column": "another_date",
            "label_available_at_column": "label_available_at",
        },
    ],
)
def test_cycle_training_rejects_unverified_or_wrong_temporal_columns(bound_cycle, split):
    state = bound_cycle
    changed = state.strategies.create_version(
        project_id=state.cycle.project_id, proposal={**state.proposal, "split": split}
    )
    with pytest.raises(PredictiveGuardError) as error:
        require_capability_training_binding(state.repository, changed.strategy_version_id)
    assert error.value.code == "TEMPORAL_LABEL_AVAILABILITY_REQUIRED"


def test_changed_recipe_is_rejected_without_editing_the_binding(bound_cycle):
    state = bound_cycle
    recipe = state.recipe.model_dump(mode="json")
    recipe["outcome"][0]["within_days"] = 8
    record_snapshot_recipe(
        state.repository,
        project_id=state.cycle.project_id,
        table_dataset_id=state.dataset_id,
        source_dataset_id=state.source_id,
        spec=recipe,
        actor_label="Human",
        coverage=state.coverage,
        readiness=state.readiness,
    )
    with pytest.raises(PredictiveGuardError) as error:
        require_capability_training_binding(state.repository, state.strategy.strategy_version_id)
    assert error.value.code == "PREDICTIVE_RECIPE_MISMATCH"


def test_new_coverage_changes_db_evidence_fingerprint(bound_cycle):
    state = bound_cycle
    before = require_capability_training_binding(
        state.repository, state.strategy.strategy_version_id
    )
    coverage = {**state.coverage, "observed_until": "2026-02-01T00:00:00Z"}
    record_snapshot_recipe(
        state.repository,
        project_id=state.cycle.project_id,
        table_dataset_id=state.dataset_id,
        source_dataset_id=state.source_id,
        spec=state.recipe,
        actor_label="Human",
        coverage=coverage,
    )
    after = require_capability_training_binding(
        state.repository, state.strategy.strategy_version_id
    )
    assert before.recipe_sha256 == after.recipe_sha256
    assert before.snapshot_evidence_sha256 != after.snapshot_evidence_sha256
    assert before.fingerprint != after.fingerprint


def test_base_helper_rejects_cycle_without_confirmed_binding(bound_cycle):
    state = bound_cycle
    other = state.capabilities.create_cycle(
        state.capability.id,
        source_cycle_id=state.cycle.id,
        reason="Nuevo",
        actor_label="Human",
        operation_id="other-cycle",
    )
    with pytest.raises(PredictiveGuardError) as error:
        require_capability_predictive_context(
            state.repository, capability_id=state.capability.id, cycle_id=other.id
        )
    assert error.value.code == "PREDICTIVE_BINDING_REQUIRED"


def test_archived_cycle_cannot_use_its_binding(bound_cycle):
    state = bound_cycle
    state.repository.archive_project(state.cycle.project_id, actor_label="Human")
    with pytest.raises(PredictiveGuardError) as error:
        require_capability_training_binding(state.repository, state.strategy.strategy_version_id)
    assert error.value.code == "CAPABILITY_NOT_ACTIVE"


def test_guard_only_reads_db_even_inside_existing_transaction(bound_cycle, monkeypatch):
    from pathlib import Path

    from core.db.connection import transaction

    state = bound_cycle

    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem not allowed")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with transaction(state.repository.connection):
        guard = require_capability_training_binding(
            state.repository, state.strategy.strategy_version_id
        )
        assert guard.dataset_id == state.dataset_id


def test_legacy_project_does_not_require_capability_binding(bound_cycle):
    state = bound_cycle
    legacy = state.repository.create_project("Legacy")
    admitted = DataIngestionService(state.repository, state.root).ingest_bytes(
        project_id=legacy,
        request={
            "original_filename_display": "legacy.csv",
            "source_name": "Legacy sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=state.table.to_csv(index=False).encode("utf-8"),
    )
    context = BusinessContextService(state.repository, state.root).create_version(
        project_id=legacy,
        dataset_id=admitted.metadata.dataset_id,
        context=state.context_input,
    )
    strategy = state.strategies.create_template_version(
        project_id=legacy, context_version_id=context.context_version_id
    )
    assert (
        require_capability_training_binding(state.repository, strategy.strategy_version_id) is None
    )
