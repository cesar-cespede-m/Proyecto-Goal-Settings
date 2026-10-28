"""MV-04: aislamiento de capacidades, idempotencia y transacciones de ciclos."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.contracts.product import CapabilityCycleV1, CapabilityV1, ProductV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityError, CapabilityService
from core.domain.gates import has_training_approval, is_strategy_current


@pytest.fixture
def service(tmp_path: Path):
    connection = get_connection(tmp_path / "capabilities.sqlite3")
    migrate(connection)
    yield CapabilityService(WorkflowRepository(connection))
    connection.close()


def _product(service: CapabilityService, suffix: str = "") -> ProductV1:
    return service.create_product(
        name="Tienda sandbox",
        description="Producto sintético de prueba",
        stage="operating",
        owner_label="Responsable declarado",
        actor_label="Persona de prueba",
        operation_id=f"product{suffix}",
    )


def _capability(service: CapabilityService, product: ProductV1, suffix: str = "A") -> CapabilityV1:
    return service.create_capability(
        product.id,
        name=f"Capacidad {suffix}",
        objective="Mejorar primera compra" if suffix == "A" else "Reducir entregas tardías",
        value_kind="revenue" if suffix == "A" else "savings",
        decision_unit="cliente" if suffix == "A" else "entrega",
        actor_label="Persona de prueba",
        operation_id=f"capability-{suffix}",
    )


def _cycle(
    service: CapabilityService,
    capability: CapabilityV1,
    operation_id: str = "cycle-A",
    source: CapabilityCycleV1 | None = None,
) -> CapabilityCycleV1:
    return service.create_cycle(
        capability.id,
        reason="Cohorte sintética inicial" if source is None else "Nueva cohorte sintética",
        actor_label="Persona de prueba",
        operation_id=operation_id,
        source_cycle_id=None if source is None else source.id,
    )


def _counts(service: CapabilityService) -> dict[str, int]:
    return {
        table: service.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "products",
            "capabilities",
            "capability_cycles",
            "product_events",
            "projects",
            "audit_events",
        )
    }


def _seed_approved_run(service: CapabilityService, cycle: CapabilityCycleV1) -> dict[str, str]:
    """Artefacto histórico de persistencia; no ejecuta ML ni prueba sus gates.

    Desde MV-09 las capacidades requieren binding y maduración para iniciar un run.
    Este fixture anterior conserva deliberadamente contratos históricos incompletos
    para probar aislamiento/lectura. Las pruebas de entrenamiento usan APIs reales.
    """
    repository = service.repository
    dataset_id = repository.add_dataset(
        project_id=cycle.project_id,
        sha256="a" * 64,
        original_filename_display="synthetic.csv",
        source="synthetic fixture",
        license_name="test-only",
        anonymization_declared=True,
        admission_status="ACCEPTED",
        rejection_reason=None,
        n_rows=20,
        n_cols=3,
        byte_count=128,
        profile={"schema_version": "1.0", "columns": [], "fixture": True},
    )
    context_id = repository.add_context(
        project_id=cycle.project_id,
        dataset_id=dataset_id,
        version_no=1,
        payload={
            "schema_version": "1.0",
            "target_column": "outcome",
            "positive_class": "yes",
            "prediction_moment": "before outcome",
            "excluded_columns": ["fixture_id"],
            "primary_metric": "f1",
            "context": {"fixture": True},
        },
    )
    strategy_id = repository.add_strategy(
        context_version_id=context_id,
        dataset_id=dataset_id,
        version_no=1,
        payload={
            "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "preprocessing": {"numeric": "median+scale"},
            "candidates": ["DummyClassifier", "LogisticRegression"],
            "seed": 42,
            "caps": {"n_jobs": 1},
        },
        schema_version="1.0",
        source="TEMPLATE",
    )
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="Revisor de prueba"
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_sha256": "a" * 64,
        "context_sha256": "b" * 64,
        "strategy_fingerprint": "c" * 64,
        "versions": {"contract": "1.0"},
        "hyperparameters": {"fixture": True},
        "features": ["feature_fixture"],
        "warnings": ["Synthetic persistence fixture, not trained model evidence"],
    }
    run_id = str(uuid4())
    repository.connection.execute(
        "INSERT INTO runs(id,strategy_version_id,status,seed,split_info_json,"
        "candidate_metrics_json,manifest_json) VALUES(?,?,'RUNNING',42,?,'{}',?)",
        (
            run_id,
            strategy_id,
            json.dumps({"historical_persistence_fixture": True}),
            json.dumps(manifest),
        ),
    )
    repository.finish_run(
        run_id,
        status="COMPLETED",
        candidate_metrics={"fixture": True},
        selected_model="DummyClassifier",
        artifact_path=f"artifacts/{run_id}",
        manifest=manifest,
    )
    return {"dataset": dataset_id, "context": context_id, "strategy": strategy_id, "run": run_id}


def test_product_and_capabilities_need_no_data_or_technical_project(service: CapabilityService):
    product = _product(service)
    revenue = _capability(service, product)
    savings = _capability(service, product, "B")
    assert service.list_products() == [product]
    assert service.list_capabilities(product.id) == [revenue, savings]
    assert service.list_cycles(revenue.id) == []
    assert _counts(service)["projects"] == 0
    assert _counts(service)["audit_events"] == 0
    assert _counts(service)["product_events"] == 3
    assert service.connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


def test_replays_return_same_entities_without_duplicate_events(service: CapabilityService):
    product = _product(service)
    capability = _capability(service, product)
    cycle = _cycle(service, capability)
    expected = _counts(service)
    assert _product(service) == product
    assert _capability(service, product) == capability
    assert _cycle(service, capability) == cycle
    assert _counts(service) == expected


@pytest.mark.parametrize("changed", ["payload", "actor", "entity_type"])
def test_operation_id_cannot_be_reused_for_another_request(
    service: CapabilityService, changed: str
):
    product = _product(service)
    expected = _counts(service)
    with pytest.raises(CapabilityError) as error:
        if changed == "entity_type":
            service.create_capability(
                product.id,
                name="A",
                objective="Ingresos",
                value_kind="revenue",
                decision_unit="cliente",
                actor_label="Persona de prueba",
                operation_id="product",
            )
        else:
            service.create_product(
                name="Otro nombre" if changed == "payload" else product.name,
                description=product.description,
                stage=product.stage,
                owner_label=product.owner_label,
                actor_label="Otra persona" if changed == "actor" else "Persona de prueba",
                operation_id="product",
            )
    assert error.value.code == "OPERATION_CONFLICT"
    assert _counts(service) == expected


@pytest.mark.parametrize("kind", ["product", "capability", "cycle"])
@pytest.mark.parametrize("actor", ["", "   "])
def test_every_creation_requires_declared_actor(service: CapabilityService, kind: str, actor: str):
    product = _product(service)
    capability = _capability(service, product)
    expected = _counts(service)
    with pytest.raises(CapabilityError, match="actor_label"):
        if kind == "product":
            service.create_product(name="Otro", actor_label=actor, operation_id="invalid")
        elif kind == "capability":
            service.create_capability(
                product.id,
                name="Otra",
                objective="Ahorrar",
                value_kind="savings",
                decision_unit="entrega",
                actor_label=actor,
                operation_id="invalid",
            )
        else:
            service.create_cycle(
                capability.id, reason="Inicial", actor_label=actor, operation_id="invalid"
            )
    assert _counts(service) == expected


@pytest.mark.parametrize("kind", ["product", "capability", "cycle"])
def test_event_failure_rolls_back_all_effects(
    service: CapabilityService, monkeypatch: pytest.MonkeyPatch, kind: str
):
    product = _product(service)
    capability = _capability(service, product)
    expected = _counts(service)

    def fail_event(**kwargs):
        raise RuntimeError("Injected event failure")

    monkeypatch.setattr(service, "_event", fail_event)
    with pytest.raises(RuntimeError, match="Injected event failure"):
        if kind == "product":
            _product(service, "-new")
        elif kind == "capability":
            _capability(service, product, "B")
        else:
            _cycle(service, capability)
    assert _counts(service) == expected
    assert not service.connection.in_transaction


def test_cycle_projects_and_order_are_distinct_and_source_is_explicit(service: CapabilityService):
    product = _product(service)
    a = _capability(service, product)
    b = _capability(service, product, "B")
    a1 = _cycle(service, a)
    b1 = _cycle(service, b, "cycle-B")
    a2 = _cycle(service, a, "cycle-A2", a1)
    assert len({a1.project_id, b1.project_id, a2.project_id}) == 3
    assert (a1.cycle_no, b1.cycle_no, a2.cycle_no) == (1, 1, 2)
    assert a1.origin == "initial" and a1.source_cycle_id is None
    assert a2.origin == "update" and a2.source_cycle_id == a1.id
    assert a2.predictive_spec_id is None
    assert service.list_cycles(a.id) == [a1, a2]
    assert service.get_cycle_for_project(a1.project_id) == a1
    assert service.get_cycle_for_project("legacy-or-missing") is None
    with pytest.raises(CapabilityError) as missing_source:
        _cycle(service, a, "ambiguous-cycle")
    assert missing_source.value.code == "CYCLE_SOURCE_REQUIRED"
    with pytest.raises(CapabilityError) as crossed_source:
        _cycle(service, a, "crossed-cycle", b1)
    assert crossed_source.value.code == "CAPABILITY_SCOPE_MISMATCH"
    assert len(service.list_cycles(a.id)) == 2


def test_new_cycle_and_other_capability_context_do_not_invalidate_approvals(
    service: CapabilityService,
):
    product = _product(service)
    a = _capability(service, product)
    b = _capability(service, product, "B")
    a1 = _cycle(service, a)
    b1 = _cycle(service, b, "cycle-B")
    a_refs = _seed_approved_run(service, a1)
    b_refs = _seed_approved_run(service, b1)
    original_b = dict(service.repository.get_run(b_refs["run"]))
    _cycle(service, a, "cycle-A2", a1)
    assert has_training_approval(service.connection, a_refs["strategy"])
    assert has_training_approval(service.connection, b_refs["strategy"])
    payload = json.loads(service.repository.get_context(a_refs["context"])["context_json"])
    payload["prediction_moment"] = "Different moment for capability A only"
    service.repository.add_context(
        project_id=a1.project_id, dataset_id=a_refs["dataset"], version_no=2, payload=payload
    )
    assert not is_strategy_current(service.connection, a_refs["strategy"])
    assert has_training_approval(service.connection, b_refs["strategy"])
    assert dict(service.repository.get_run(b_refs["run"])) == original_b


def test_scope_checks_run_dataset_cycle_and_project_together(service: CapabilityService):
    product = _product(service)
    a = _capability(service, product)
    b = _capability(service, product, "B")
    a1 = _cycle(service, a)
    b1 = _cycle(service, b, "cycle-B")
    a2 = _cycle(service, a, "cycle-A2", a1)
    a_refs = _seed_approved_run(service, a1)
    b_refs = _seed_approved_run(service, b1)
    service.require_scope(a.id)
    service.require_scope(
        a.id,
        cycle_id=a1.id,
        project_id=a1.project_id,
        run_id=a_refs["run"],
        dataset_id=a_refs["dataset"],
    )
    mismatches = [
        {"cycle_id": b1.id},
        {"project_id": b1.project_id},
        {"run_id": b_refs["run"]},
        {"dataset_id": b_refs["dataset"]},
        {"cycle_id": a2.id, "run_id": a_refs["run"]},
        {"run_id": a_refs["run"], "dataset_id": b_refs["dataset"]},
        {"project_id": service.repository.create_project("Legacy project")},
    ]
    for references in mismatches:
        with pytest.raises(CapabilityError) as error:
            service.require_scope(a.id, **references)
        assert error.value.code == "CAPABILITY_SCOPE_MISMATCH"
    for field in ("cycle_id", "run_id", "dataset_id"):
        with pytest.raises(CapabilityError):
            service.require_scope(a.id, **{field: "missing"})


@pytest.mark.parametrize(
    "table", ["products", "capabilities", "capability_cycles", "product_events"]
)
def test_hierarchy_and_events_cannot_be_reassigned_or_deleted(
    service: CapabilityService, table: str
):
    product = _product(service)
    capability = _capability(service, product)
    _cycle(service, capability)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        service.connection.execute(f"UPDATE {table} SET id = id")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        service.connection.execute(f"DELETE FROM {table}")


def test_database_rejects_duplicate_project_and_cross_capability_source(service: CapabilityService):
    product = _product(service)
    a = _capability(service, product)
    b = _capability(service, product, "B")
    a1 = _cycle(service, a)
    b1 = _cycle(service, b, "cycle-B")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        service.connection.execute(
            "INSERT INTO capability_cycles SELECT 'duplicate', capability_id, project_id, "
            "2, 'update', id, NULL, 'duplicate fixture', created_at "
            "FROM capability_cycles WHERE id = ?",
            (a1.id,),
        )
    new_project = service.repository.create_project("Fixture SQL")
    with pytest.raises(sqlite3.IntegrityError, match="source capability mismatch"):
        service.connection.execute(
            "INSERT INTO capability_cycles SELECT 'crossed', capability_id, ?, "
            "2, 'update', ?, NULL, 'crossed fixture', created_at "
            "FROM capability_cycles WHERE id = ?",
            (new_project, b1.id, a1.id),
        )


def test_project_insert_helper_requires_transaction(service: CapabilityService):
    with pytest.raises(RuntimeError, match="transacción activa"):
        service.repository._insert_project("No orphan")
    assert _counts(service)["projects"] == 0


def test_reopening_preserves_hierarchy_and_idempotency(tmp_path: Path):
    path = tmp_path / "reopened.sqlite3"
    first = get_connection(path)
    migrate(first)
    service = CapabilityService(WorkflowRepository(first))
    product = _product(service)
    capability = _capability(service, product)
    cycle = _cycle(service, capability)
    expected = _counts(service)
    first.close()
    second = get_connection(path)
    migrate(second)
    migrate(second)
    reopened = CapabilityService(WorkflowRepository(second))
    assert _product(reopened) == product
    assert _capability(reopened, product) == capability
    assert _cycle(reopened, capability) == cycle
    assert _counts(reopened) == expected
    second.close()


def test_contract_validation_leaves_no_partial_product(service: CapabilityService):
    with pytest.raises(ValidationError):
        service.create_product(
            name="   ", stage="unrecognized", actor_label="Persona", operation_id="bad"
        )
    assert _counts(service)["products"] == 0
    assert _counts(service)["product_events"] == 0


def test_product_events_contain_references_not_business_free_text(service: CapabilityService):
    product = _product(service)
    capability = _capability(service, product)
    _cycle(service, capability)
    for row in service.connection.execute("SELECT * FROM product_events"):
        detail = json.loads(row["detail_json"])
        assert detail["identity_verified"] is False
        assert product.description not in row["detail_json"]
        assert capability.objective not in row["detail_json"]
