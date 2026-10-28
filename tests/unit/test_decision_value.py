"""MV-05: versiones sin datos, confirmación, alcance y conservación de gates predictivos."""

import json
import sqlite3
from decimal import Decimal, Overflow, localcontext

import pytest
from pydantic import ValidationError

from core.analysis.snapshot import SnapshotSpecV1
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.contracts.decision_value import PredictiveSpecV1
from core.contracts.economics import EconomicScenarioV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueError, DecisionValueService
from core.domain.gates import has_training_approval, is_strategy_current
from tests.unit.test_capabilities import _seed_approved_run

ACTOR = "Responsable sintético"


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "specs.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    capabilities = CapabilityService(repository)
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
    yield DecisionValueService(repository), capabilities, caps
    connection.close()


def _economic(**changes):
    return {
        "currency": "CLP",
        "currency_minor_units": 0,
        "period": "mes",
        "planned_cases": 100,
        "capacity_cases": 10,
        "external_risk_rate": "0.20",
        "risk_rate_basis": "Tasa supuesta del grupo seleccionado, demo sintética",
        "relative_effect": "0.50",
        "relative_effect_basis": "Supuesto de intervención demo",
        "value_per_case": "1000",
        "action_cost_per_case": "10",
        "fixed_cost": "25",
        "value_basis": "Margen o costo evitable constante supuesto de la demo",
        **changes,
    }


def _predictive(**changes):
    return {
        "population": "Casos sintéticos elegibles",
        "target_definition": "Evento adverso a 14 días",
        "target_column": "target",
        "positive_class": 1,
        "prediction_moment": "Al día dos",
        "primary_metric": "f1",
        "outcome_window": {"kind": "fixed_days", "days": 14},
        "feature_decisions": [
            {
                "column": "signal",
                "use": "include",
                "available_at_prediction": True,
                "reason": "Disponible al corte",
            }
        ],
        **changes,
    }


def _save_predictive(service, capability, operation="predictive", **changes):
    return service.create_predictive_version(
        capability.id,
        spec=_predictive(**changes),
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id=operation,
    )


def test_incomplete_case_can_be_saved_without_a_dataset_and_cannot_be_confirmed(state):
    service, _caps, (cap, _) = state
    decision = service.create_decision_version(
        cap.id, spec={"question": "¿Qué conviene hacer?"}, actor_label=ACTOR, operation_id="draft"
    )
    predictive = service.create_predictive_version(
        cap.id, spec={}, actor_label=ACTOR, operation_id="pred-draft"
    )
    economic = service.create_economic_version(
        cap.id, scenario={}, actor_label=ACTOR, operation_id="econ-draft"
    )
    assert not decision.confirmed_human
    assert "economic_version_id" in decision.missing_fields
    assert "outcome_window" in predictive.missing_fields
    assert "value_per_case" in economic.missing_fields
    assert service.connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
    with pytest.raises(DecisionValueError) as error:
        service.create_decision_version(
            cap.id,
            spec={},
            based_on_version_id=decision.id,
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id="invalid-confirm",
        )
    assert error.value.code == "SPEC_INCOMPLETE"
    assert len(service.list_decision_versions(cap.id)) == 1


def test_decimal_hash_replay_and_optimistic_versions_are_stable(state):
    service, _, (cap, _) = state
    first = service.create_economic_version(
        cap.id,
        scenario=_economic(value_per_case="10.00"),
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="economic",
    )
    replay = service.create_economic_version(
        cap.id,
        scenario=_economic(value_per_case=Decimal("10.0")),
        confirmed_human=True,
        actor_label=f" {ACTOR} ",
        operation_id=" economic ",
    )
    assert replay == first
    second = service.create_economic_version(
        cap.id,
        scenario=_economic(value_per_case="20"),
        based_on_version_id=first.id,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="economic-2",
    )
    assert second.sha256 != first.sha256
    assert [item.version_no for item in service.list_economic_versions(cap.id)] == [1, 2]
    for operation, base, code in (
        ("economic", None, "OPERATION_CONFLICT"),
        ("stale-edit", first.id, "SPEC_VERSION_CONFLICT"),
    ):
        with pytest.raises(DecisionValueError) as error:
            service.create_economic_version(
                cap.id,
                scenario=_economic(value_per_case="30"),
                based_on_version_id=base,
                confirmed_human=True,
                actor_label=ACTOR,
                operation_id=operation,
            )
        assert error.value.code == code
    assert len(service.list_economic_versions(cap.id)) == 2


def test_money_currency_change_requires_another_capability_but_drafts_can_be_corrected(state):
    service, _, (cap, _) = state
    draft = service.create_economic_version(
        cap.id, scenario=_economic(currency="USD"), actor_label=ACTOR, operation_id="wrong-draft"
    )
    confirmed = service.create_economic_version(
        cap.id,
        scenario=_economic(),
        based_on_version_id=draft.id,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="confirmed-currency",
    )
    with pytest.raises(DecisionValueError) as error:
        service.create_economic_version(
            cap.id,
            scenario=_economic(currency="USD"),
            based_on_version_id=confirmed.id,
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id="change-currency",
        )
    assert error.value.code == "CURRENCY_MISMATCH"


def test_decision_references_require_same_capability_and_explicit_confirmation(state):
    service, _, (cap, other) = state
    prediction = _save_predictive(service, cap)
    economy = service.create_economic_version(
        cap.id, scenario=_economic(), actor_label=ACTOR, operation_id="unconfirmed-economy"
    )
    payload = {
        "question": "¿A quién asistir?",
        "action": "Asistencia",
        "alternative": "No asistir",
        "kpi": "Primera compra",
        "owner_label": ACTOR,
        "predictive_version_id": prediction.id,
        "economic_version_id": economy.id,
    }
    draft = service.create_decision_version(
        cap.id, spec=payload, actor_label=ACTOR, operation_id="decision"
    )
    assert draft.missing_fields == ["economic_version_confirmed"]
    with pytest.raises(DecisionValueError) as error:
        service.create_decision_version(
            other.id, spec=payload, actor_label=ACTOR, operation_id="cross-cap"
        )
    assert error.value.code == "SPEC_SCOPE_MISMATCH"
    with pytest.raises(DecisionValueError) as error:
        service.create_decision_version(
            cap.id,
            spec=payload,
            based_on_version_id=draft.id,
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id="unconfirmed-dependency",
        )
    assert error.value.code == "SPEC_INCOMPLETE"


def test_binding_is_immutable_and_cannot_cross_capabilities(state):
    service, capabilities, (cap, other) = state
    cycle = capabilities.create_cycle(
        cap.id, reason="Inicial", actor_label=ACTOR, operation_id="cy"
    )
    version = _save_predictive(service, cap)
    binding = service.bind_predictive_to_cycle(
        cap.id,
        cycle_id=cycle.id,
        predictive_version_id=version.id,
        actor_label=ACTOR,
        operation_id="bind",
    )
    assert (
        service.bind_predictive_to_cycle(
            cap.id,
            cycle_id=cycle.id,
            predictive_version_id=version.id,
            actor_label=ACTOR,
            operation_id="bind",
        )
        == binding
    )
    assert capabilities.get_cycle(cycle.id).predictive_spec_id is None
    assert service.get_cycle_binding(cycle.id) == binding
    with pytest.raises(DecisionValueError) as error:
        service.bind_predictive_to_cycle(
            cap.id,
            cycle_id=cycle.id,
            predictive_version_id=version.id,
            actor_label=ACTOR,
            operation_id="rebind",
        )
    assert error.value.code == "CYCLE_ALREADY_BOUND"
    other_version = _save_predictive(service, other, "other-predictive")
    with pytest.raises(DecisionValueError) as error:
        service.bind_predictive_to_cycle(
            cap.id,
            cycle_id=cycle.id,
            predictive_version_id=other_version.id,
            actor_label=ACTOR,
            operation_id="cross-binding",
        )
    assert error.value.code == "SPEC_SCOPE_MISMATCH"


def test_import_preserves_known_fields_and_economic_edit_does_not_invalidate_model(state):
    service, capabilities, (cap, _) = state
    cycle = capabilities.create_cycle(
        cap.id, reason="Inicial", actor_label=ACTOR, operation_id="cy"
    )
    references = _seed_approved_run(capabilities, cycle)
    source = service.repository.get_latest_context(cycle.project_id)
    imported = service.draft_predictive_from_context(cap.id, context_version_id=source["id"])
    assert imported.target_column == "outcome"
    assert imported.positive_class == "yes"
    assert imported.source_context_sha256 == source["context_sha256"]
    assert imported.population is None and imported.outcome_window is None
    assert "feature_decisions" in imported.missing_fields()  # legacy no declara variables
    snapshot = {
        table: [tuple(row) for row in service.connection.execute(f"SELECT * FROM {table}")]
        for table in ("context_versions", "strategy_versions", "approval_events", "runs", "reports")
    }
    economic = service.create_economic_version(
        cap.id, scenario=_economic(), confirmed_human=True, actor_label=ACTOR, operation_id="econ"
    )
    service.create_economic_version(
        cap.id,
        scenario=_economic(action_cost_per_case="20"),
        based_on_version_id=economic.id,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="econ-2",
    )
    for table, before in snapshot.items():
        assert [
            tuple(row) for row in service.connection.execute(f"SELECT * FROM {table}")
        ] == before
    assert has_training_approval(service.connection, references["strategy"])
    assert is_strategy_current(service.connection, references["strategy"])
    mismatch = _save_predictive(service, cap)
    with pytest.raises(DecisionValueError) as error:
        service.bind_predictive_to_cycle(
            cap.id,
            cycle_id=cycle.id,
            predictive_version_id=mismatch.id,
            actor_label=ACTOR,
            operation_id="mismatched-context",
        )
    assert error.value.code == "CYCLE_CONTEXT_MISMATCH"


def test_predictive_definition_fingerprint_excludes_cycle_specific_origin():
    first = PredictiveSpecV1(
        **_predictive(source_context_version_id="context-a", source_context_sha256="a" * 64)
    )
    second = PredictiveSpecV1(
        **_predictive(source_context_version_id="context-b", source_context_sha256="b" * 64)
    )
    assert canonical_hash(first) != canonical_hash(second)
    assert first.predictive_fingerprint == second.predictive_fingerprint
    changed = PredictiveSpecV1(**_predictive(outcome_window={"kind": "fixed_days", "days": 30}))
    assert first.predictive_fingerprint != changed.predictive_fingerprint


def test_import_carries_recorded_recipe_without_confirming_or_inventing_coverage(state):
    service, capabilities, (cap, _) = state
    cycle = capabilities.create_cycle(
        cap.id, reason="Inicial", actor_label=ACTOR, operation_id="cy"
    )
    _seed_approved_run(capabilities, cycle)
    source = service.repository.get_latest_context(cycle.project_id)
    imported = service.draft_predictive_from_context(cap.id, context_version_id=source["id"])
    assert imported.snapshot_recipe is None
    from tests.unit.test_data_readiness import RECIPE

    record_snapshot_recipe(
        service.repository,
        project_id=cycle.project_id,
        table_dataset_id=source["dataset_id"],
        source_dataset_id=source["dataset_id"],
        spec=RECIPE,
        actor_label=ACTOR,
    )
    draft = service.draft_predictive_from_context(cap.id, context_version_id=source["id"])
    assert draft.snapshot_recipe == SnapshotSpecV1.model_validate(RECIPE)
    assert draft.outcome_window is None
    assert service.list_predictive_versions(cap.id) == []


def test_decimal_canonical_hash_does_not_inherit_hostile_context():
    scenario = EconomicScenarioV1(**_economic(value_per_case="12345.6789"))
    expected = canonical_hash(scenario)
    with localcontext() as context:
        context.prec = 4
        context.Emax = 3
        context.traps[Overflow] = False
        assert canonical_hash(scenario) == expected
        assert scenario.model_dump(mode="json")["value_per_case"] == "12345.6789"


def test_copied_invalid_contract_is_revalidated_before_writing(state):
    service, _, (cap, _) = state
    invalid = EconomicScenarioV1(**_economic()).model_copy(update={"value_per_case": float("nan")})
    with pytest.raises(ValidationError):
        service.create_economic_version(
            cap.id,
            scenario=invalid,
            confirmed_human=True,
            actor_label=ACTOR,
            operation_id="bad-copy",
        )
    assert service.list_economic_versions(cap.id) == []


@pytest.mark.parametrize(
    "table",
    [
        "predictive_spec_versions",
        "economic_scenario_versions",
        "decision_value_spec_versions",
        "cycle_predictive_bindings",
        "value_spec_events",
    ],
)
def test_persisted_specs_and_audit_are_append_only(state, table):
    service, capabilities, (cap, _) = state
    prediction = _save_predictive(service, cap)
    service.create_economic_version(cap.id, scenario={}, actor_label=ACTOR, operation_id="economy")
    service.create_decision_version(cap.id, spec={}, actor_label=ACTOR, operation_id="decision")
    cycle = capabilities.create_cycle(
        cap.id, reason="Inicial", actor_label=ACTOR, operation_id="cy"
    )
    service.bind_predictive_to_cycle(
        cap.id,
        cycle_id=cycle.id,
        predictive_version_id=prediction.id,
        actor_label=ACTOR,
        operation_id="bind",
    )
    for sql in (f"UPDATE {table} SET actor_label='otro'", f"DELETE FROM {table}"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            service.connection.execute(sql)
    details = service.connection.execute("SELECT detail_json FROM value_spec_events").fetchall()
    assert all("content" not in json.loads(row[0]) for row in details)
