"""MV-08: cobertura verificable de etiquetas y evidencia segura de la cohorte madura."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from core.analysis.snapshot import build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe, snapshot_recipe
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityError, CapabilityService
from core.domain.data_readiness import DataReadinessError, DataReadinessService
from core.domain.decision_value import DecisionValueService

COVERAGE = {
    "observed_from": "2026-01-01T00:00:00Z",
    "observed_until": "2026-01-31T00:00:00Z",
    "complete": True,
}
RECIPE = {
    "entity": "entity",
    "moment": "moment",
    "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
    "windows_days": [3],
    "features": [{"op": "count_events", "window_days": 3}],
    "outcome": [
        {
            "op": "not_reached_stage",
            "stage": {"column": "stage", "value": "purchase_private"},
            "within_days": 7,
            "output_column": "adverse",
        }
    ],
}


def predictive_spec(**changes):
    return {
        "population": "Altas sintéticas",
        "target_definition": "No avanzar en la ventana declarada",
        "target_column": "adverse",
        "positive_class": True,
        "prediction_moment": "Al corte definido por la receta",
        "primary_metric": "f1",
        "outcome_window": {"kind": "fixed_days", "days": 7},
        "feature_decisions": [
            {
                "column": "count_events_3d",
                "use": "include",
                "available_at_prediction": True,
                "reason": "Sólo actividad previa al corte",
            }
        ],
        "snapshot_recipe": deepcopy(RECIPE),
        **changes,
    }


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "readiness.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(name="Sandbox", actor_label="Human", operation_id="p")
    capability = capabilities.create_capability(
        product.id,
        name="Activacion",
        objective="Demo sintética",
        value_kind="revenue",
        decision_unit="alta",
        actor_label="Human",
        operation_id="cap",
    )
    cycle = capabilities.create_cycle(
        capability.id, reason="Demo", actor_label="Human", operation_id="cycle"
    )
    values = DecisionValueService(workflow)
    ingest = DataIngestionService(workflow, tmp_path)

    def save(frame, name="data.csv", *, project_id=None, **changes):
        return ingest.ingest_bytes(
            project_id=project_id or cycle.project_id,
            request={
                "original_filename_display": name,
                "source_name": "Prueba sintética",
                "classification": "synthetic",
                "authorization_status": "approved",
                **changes,
            },
            content=frame.to_csv(index=False).encode("utf-8"),
        ).metadata.dataset_id

    events = pd.DataFrame(
        [
            ("PRIVATE_A", "2026-01-01", "signup"),
            ("PRIVATE_B", "2026-01-10", "signup"),
            ("PRIVATE_B", "2026-01-15", "purchase_private"),
            ("PRIVATE_C", "2026-01-29", "signup"),
        ],
        columns=["entity", "moment", "stage"],
    )
    source_id = save(events, "events.csv")

    def build(*, coverage=COVERAGE, export_all=False, recipe=RECIPE, record=True):
        snapshot = build_snapshot(events, recipe, coverage=coverage)
        table = snapshot.dataframe if export_all else snapshot.training_frame()
        table_id = save(table, "table.csv")
        if record:
            record_snapshot_recipe(
                workflow,
                project_id=cycle.project_id,
                table_dataset_id=table_id,
                source_dataset_id=source_id,
                spec=recipe,
                actor_label="Human",
                coverage=coverage,
                readiness=snapshot.readiness,
            )
        return table_id, snapshot

    def spec(*, confirmed=True, operation="spec", **changes):
        previous = values.list_predictive_versions(capability.id)
        return values.create_predictive_version(
            capability.id,
            spec=predictive_spec(**changes),
            based_on_version_id=previous[-1].id if previous else None,
            confirmed_human=confirmed,
            actor_label="Human",
            operation_id=operation,
        )

    yield SimpleNamespace(
        workflow=workflow,
        capabilities=capabilities,
        capability=capability,
        cycle=cycle,
        values=values,
        root=tmp_path,
        save=save,
        build=build,
        spec=spec,
        source_id=source_id,
        service=DataReadinessService(workflow, tmp_path),
    )
    connection.close()


def inspect(state, table_id, **changes):
    return state.service.inspect_label_readiness(
        state.capability.id, dataset_id=table_id, cycle_id=state.cycle.id, **changes
    )


def test_mature_table_is_ready_while_recent_source_cases_remain_excluded(state):
    table_id, snapshot = state.build()
    version = state.spec()
    result = inspect(state, table_id)
    assert result.status == "ready"
    assert result.table_rows == result.verified_table_rows == 2
    assert result.pending_source_count == 1
    assert result.snapshot_readiness == snapshot.readiness
    assert result.reason_counts == {"OUTCOME_WINDOW_NOT_MATURE": 1}
    assert result.measurement_plan
    assert not result.grants_training_approval and not result.statistical_support_checked
    assert {ref.id for ref in result.output_refs} == {
        table_id,
        state.source_id,
        state.cycle.id,
        version.id,
    }
    assert result.evidence_sha256 == canonical_hash(
        result.model_dump(mode="json", exclude={"evidence_sha256"})
    )
    assert inspect(state, table_id) == result
    wire = str(result.safe_summary())
    assert "PRIVATE_" not in wire and "purchase_private" not in wire
    assert "actor_label" not in wire and "binding_id" not in wire


def test_all_mature_cases_are_ready_without_a_pending_measurement_plan(state):
    table_id, _ = state.build(coverage={**COVERAGE, "observed_until": "2026-02-28T00:00:00Z"})
    state.spec()
    result = inspect(state, table_id)
    assert result.status == "ready" and result.verified_table_rows == 3
    assert result.measurement_plan == [] and result.pending_source_count == 0


@pytest.mark.parametrize("coverage", [None, {**COVERAGE, "complete": False}])
def test_coverage_is_never_inferred_from_event_times_or_filled_labels(state, coverage):
    table_id, _ = state.build(coverage=coverage, export_all=True)
    state.spec()
    result = inspect(state, table_id)
    assert result.status == "needs_data" and result.verified_table_rows == 0
    assert "NO_MATURE_ROWS" in result.reason_codes and result.pending_source_count == 3
    assert result.measurement_plan


def test_table_that_includes_recent_cases_cannot_be_used_as_mature(state):
    table_id, _ = state.build(export_all=True)
    state.spec()
    result = inspect(state, table_id)
    assert result.status == "needs_data" and "TABLE_REBUILD_MISMATCH" in result.reason_codes
    assert result.verified_table_rows == 0


def test_filled_tabular_target_without_recipe_is_explicitly_unknown(state):
    table_id = state.save(pd.DataFrame({"signal": [1, 2], "adverse": [True, False]}))
    state.spec()
    result = inspect(state, table_id)
    assert result.status == "needs_human"
    assert result.reason_codes == ["SNAPSHOT_PROVENANCE_MISSING"]
    assert result.snapshot_readiness is None and result.measurement_plan


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        (
            {"outcome_window": {"kind": "deadline_column", "column": "deadline"}},
            "PREDICTIVE_WINDOW_MISMATCH",
        ),
        ({"outcome_window": {"kind": "fixed_days", "days": 30}}, "PREDICTIVE_WINDOW_MISMATCH"),
        ({"target_column": "other_target"}, "PREDICTIVE_TARGET_MISMATCH"),
        ({"snapshot_recipe": None}, "PREDICTIVE_RECIPE_NOT_DECLARED"),
        (
            {
                "snapshot_recipe": {
                    **RECIPE,
                    "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 1},
                }
            },
            "PREDICTIVE_RECIPE_MISMATCH",
        ),
        (
            {
                "feature_decisions": [
                    {
                        "column": "snapshot_cutoff",
                        "use": "include",
                        "available_at_prediction": True,
                        "reason": "Metadata",
                    }
                ]
            },
            "PREDICTIVE_FEATURES_NOT_VERIFIED",
        ),
    ],
)
def test_incompatible_predictive_definition_cannot_claim_ready(state, changes, code):
    table_id, _ = state.build()
    state.spec(**changes)
    result = inspect(state, table_id)
    assert result.status == "needs_human" and code in result.reason_codes


def test_unconfirmed_predictive_draft_cannot_claim_ready(state):
    table_id, _ = state.build()
    state.spec(confirmed=False)
    result = inspect(state, table_id)
    assert result.status == "needs_human"
    assert "PREDICTIVE_SPEC_UNCONFIRMED_OR_INCOMPLETE" in result.reason_codes


def test_bound_definition_is_used_instead_of_unrelated_new_draft(state):
    table_id, _ = state.build()
    version = state.spec()
    binding = state.values.bind_predictive_to_cycle(
        state.capability.id,
        cycle_id=state.cycle.id,
        predictive_version_id=version.id,
        actor_label="Human",
        operation_id="bind",
    )
    latest = state.spec(operation="new-draft", confirmed=False, target_column="different")
    result = inspect(state, table_id)
    assert result.status == "ready"
    assert result.binding_id == binding.id and result.binding_sha256 == canonical_hash(binding)
    assert version.id in {ref.id for ref in result.output_refs}
    with pytest.raises(DataReadinessError) as error:
        inspect(state, table_id, predictive_version_id=latest.id)
    assert error.value.code == "PREDICTIVE_BINDING_MISMATCH"


@pytest.mark.parametrize("which", ["table", "source"])
def test_changed_private_bytes_are_rejected_before_any_ready_claim(state, which):
    table_id, _ = state.build()
    state.spec()
    dataset = state.workflow.get_dataset(table_id if which == "table" else state.source_id)
    (state.root / dataset["stored_path"]).write_bytes(b"tampered_private\n1\n")
    with pytest.raises(DataReadinessError) as error:
        inspect(state, table_id)
    assert error.value.code == "DATASET_HASH_MISMATCH"
    assert "tampered_private" not in str(error.value)


def test_dataset_and_predictive_scope_are_enforced(state):
    table_id, _ = state.build()
    other = state.capabilities.create_capability(
        state.capability.product_id,
        name="Otra",
        objective="Otra demo",
        value_kind="savings",
        decision_unit="caso",
        actor_label="Human",
        operation_id="other-cap",
    )
    with pytest.raises(CapabilityError) as error:
        state.service.inspect_label_readiness(other.id, dataset_id=table_id)
    assert error.value.code == "CAPABILITY_SCOPE_MISMATCH"
    version = state.values.create_predictive_version(
        other.id,
        spec=predictive_spec(),
        confirmed_human=True,
        actor_label="Human",
        operation_id="other-spec",
    )
    with pytest.raises(DataReadinessError) as error:
        inspect(state, table_id, predictive_version_id=version.id)
    assert error.value.code == "SPEC_SCOPE_MISMATCH"


def test_saved_aggregate_is_checked_against_source_instead_of_trusted(state):
    table_id, _ = state.build()
    state.spec()
    recipe = snapshot_recipe(state.workflow, table_id)
    fake = {**recipe["readiness"], "eligible_rows": 3, "pending_rows": 0, "reason_counts": {}}
    record_snapshot_recipe(
        state.workflow,
        project_id=state.cycle.project_id,
        table_dataset_id=table_id,
        source_dataset_id=state.source_id,
        spec=RECIPE,
        actor_label="Human",
        coverage=COVERAGE,
        readiness=fake,
    )
    result = inspect(state, table_id)
    assert result.status == "needs_human"
    assert result.reason_codes == ["SNAPSHOT_READINESS_MISMATCH"]
    assert result.pending_source_count == 1


def test_quarantined_source_is_not_inspected(state):
    table_id = state.save(
        pd.DataFrame({"signal": [1, 2], "adverse": [True, False]}),
        authorization_status="pending",
    )
    with pytest.raises(DataReadinessError) as error:
        inspect(state, table_id)
    assert error.value.code == "DATASET_NOT_ACCEPTED"
