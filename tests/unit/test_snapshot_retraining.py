"""MV-01: cada reentrenamiento conserva receta y exige cobertura nueva."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from core.analysis.engine import load_project_source
from core.analysis.snapshot import SNAPSHOT_CUTOFF_COLUMN, build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe, snapshot_recipe
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.retraining_service import RetrainingService
from core.domain.strategy_service import StrategyService
from core.ml_engine import RunBlockedError
from core.ml_engine.scoring import ScoringService

EVENTS = Path(__file__).parents[2] / "data" / "synthetic" / "recorridos" / "altas_eventos.csv"
COVERAGE = {
    "observed_from": "2026-01-01T00:00:00Z",
    "observed_until": "2026-12-31T23:59:59Z",
    "complete": True,
}


@pytest.fixture
def journey(tmp_path, monkeypatch):
    connection = get_connection(tmp_path / "db.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Recorrido sintético")
    events = pd.read_csv(EVENTS)
    entity, moment, stage = events.columns[:3]
    selected = set(events[entity].drop_duplicates().iloc[:120])
    events = events[events[entity].isin(selected)]
    entities = pd.read_csv(EVENTS.with_name("altas_entidades.csv"))
    entities = entities[entities[entity].isin(selected)]
    spec = {
        "entity": entity,
        "moment": moment,
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 3},
        "windows_days": [3],
        "features": [{"op": "count_events", "window_days": 3}],
        "entity_attributes": ["indice_interes_inicial"],
        "outcome": [
            {
                "op": "not_reached_stage",
                "stage": {"column": stage, "value": "primera_compra"},
                "within_days": 30,
                "output_column": "no_avanza",
            }
        ],
    }
    snapshot = build_snapshot(events, spec, entities, coverage=COVERAGE)
    ingest = DataIngestionService(repository, tmp_path)

    def save(name, frame):
        return ingest.ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": name,
                "source_name": "Datos sintéticos de prueba",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=frame.to_csv(index=False).encode(),
        ).metadata.dataset_id

    source_id = save("actividad.csv", events)
    entity_id = save("entidades.csv", entities)
    table_id = save("snapshot.csv", snapshot.training_frame())
    repository.set_modeling_table(project_id, table_id, actor_label="Ana")
    record_snapshot_recipe(
        repository,
        project_id=project_id,
        table_dataset_id=table_id,
        source_dataset_id=source_id,
        entity_dataset_id=entity_id,
        spec=spec,
        actor_label="Ana",
        coverage=COVERAGE,
        readiness=snapshot.readiness,
    )
    context = BusinessContextService(repository, tmp_path).create_version(
        project_id=project_id,
        dataset_id=table_id,
        context={
            "objective": "Anticipar quién no avanza",
            "target_column": "no_avanza",
            "positive_class": True,
            "prediction_moment": "A los tres días del primer contacto",
            "false_positive_consequence": "Contacto innecesario",
            "false_negative_consequence": "Caso que no avanza sin ayuda",
            "primary_metric": "f1",
            "metric_rationale": "Equilibrio",
            "decision_actor_label": "Ana",
            "feature_decisions": [
                {
                    "column": column,
                    "use": "exclude" if column in {entity, SNAPSHOT_CUTOFF_COLUMN} else "include",
                    "available_at_prediction": True,
                    "reason": "Disponible al corte; identificador y momento se excluyen",
                }
                for column in snapshot.training_frame().columns
                if column != "no_avanza"
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    approved = {}

    def register(run_id, project, context_id):
        strategies = StrategyService(repository)
        proposal = strategies.build_template(project_id=project, context_version_id=context_id)
        strategy = strategies.create_version(project_id=project, proposal=proposal)
        approved[run_id] = (
            {"strategy_version_id": strategy.strategy_version_id},
            {"id": "report-" + run_id},
            proposal,
        )

    # Las aprobaciones/entrenamiento se prueban en test_retraining; aquí aislamos persistencia.
    monkeypatch.setattr(ScoringService, "approved_run", lambda _self, run_id: approved[run_id])
    register("initial", project_id, context.context_version_id)
    yield repository, RetrainingService(repository, tmp_path), events, register, table_id
    connection.close()


def _start(service, events, **overrides):
    return service.start_from_snapshot(
        **{
            "source_run_id": "initial",
            "filename": "actividad-nueva.csv",
            "content": events.to_csv(index=False).encode(),
            "coverage": COVERAGE,
            "actor_label": "Ana",
            "data_authorized": True,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"actor_label": "pendiente"}, "ACTOR_REQUIRED"),
        ({"data_authorized": False}, "DATA_AUTHORIZATION_REQUIRED"),
        ({"coverage": None}, "SNAPSHOT_OUTCOMES_NOT_READY"),
        ({"coverage": {**COVERAGE, "complete": False}}, "SNAPSHOT_OUTCOMES_NOT_READY"),
        (
            {"coverage": {**COVERAGE, "observed_until": "2026-01-03T00:00:00Z"}},
            "SNAPSHOT_OUTCOMES_NOT_READY",
        ),
    ],
)
def test_unready_or_unauthorized_batch_creates_no_candidate(journey, changes, code):
    repository, service, events, _, _ = journey
    with pytest.raises(RunBlockedError) as error:
        _start(service, events, **changes)
    assert error.value.code == code
    assert len(repository.list_projects()) == 1
    assert repository.list_retraining_links() == []


def test_successive_batches_keep_own_sources_and_require_new_coverage(journey, tmp_path):
    repository, service, events, register, original_table = journey
    first = _start(service, events)
    recipe = snapshot_recipe(repository, first.dataset_id)
    assert recipe is not None
    assert (
        recipe["source_sha256"] == hashlib.sha256(events.to_csv(index=False).encode()).hexdigest()
    )
    assert recipe["readiness"]["eligible_rows"] == events.iloc[:, 0].nunique()
    for source_id in (recipe["source_dataset_id"], recipe["entity_dataset_id"]):
        assert repository.get_dataset(source_id)["project_id"] == first.project_id
    assert (
        recipe["entity_dataset_id"]
        != snapshot_recipe(repository, original_table)["entity_dataset_id"]
    )
    persisted = load_project_source(repository, tmp_path, first.dataset_id).frame
    assert "eligible_for_training" not in persisted
    assert "pending_reasons" not in persisted
    assert repository.get_active_dataset(first.project_id)["id"] == first.dataset_id
    assert repository.get_context(first.context_version_id)["dataset_id"] == first.dataset_id
    assert repository.get_latest_strategy(first.context_version_id) is None

    register("second", first.project_id, first.context_version_id)
    with pytest.raises(RunBlockedError) as error:
        _start(service, events, source_run_id="second", coverage=None)
    assert error.value.code == "SNAPSHOT_OUTCOMES_NOT_READY"
    assert len(repository.list_projects()) == 2
    second = _start(service, events, source_run_id="second")
    assert (
        snapshot_recipe(repository, second.dataset_id)["source_dataset_id"]
        != recipe["source_dataset_id"]
    )
    assert len(repository.list_projects()) == 3


def test_only_mature_rows_are_saved_with_pending_counts(journey, tmp_path):
    repository, service, events, _, _ = journey
    changed = events.copy()
    entity, moment = changed.columns[:2]
    recent_id = changed.iloc[0][entity]
    recent = changed[entity].eq(recent_id)
    changed.loc[recent, moment] = (
        pd.to_datetime(changed.loc[recent, moment]) + pd.Timedelta(days=730)
    ).astype(str)
    started = _start(service, changed)
    persisted = load_project_source(repository, tmp_path, started.dataset_id).frame
    assert recent_id not in set(persisted[entity])
    assert len(persisted) == events[entity].nunique() - 1
    recipe = snapshot_recipe(repository, started.dataset_id)
    assert recipe["readiness"]["pending_rows"] == 1
    assert recipe["readiness"]["eligible_rows"] == len(persisted)


def test_partial_source_failure_is_reported_without_returning_success(journey, monkeypatch):
    repository, service, events, _, _ = journey

    def fail_copy(*_args):
        raise OSError("fallo simulado; no guardar esta explicación en auditoría")

    monkeypatch.setattr(service, "_copy_snapshot_source", fail_copy)
    with pytest.raises(RunBlockedError) as error:
        _start(service, events)
    assert error.value.code == "SNAPSHOT_RETRAINING_INCOMPLETE"
    failure = repository.connection.execute(
        "SELECT project_id, detail_json FROM audit_events "
        "WHERE event_type = 'SNAPSHOT_RETRAINING_INCOMPLETE'"
    ).fetchone()
    assert failure is not None
    detail = json.loads(failure["detail_json"])
    assert detail["error_type"] == "OSError"
    assert "fallo simulado" not in failure["detail_json"]
    assert repository.list_project_runs(failure["project_id"]) == []
    assert failure["project_id"] not in {row["id"] for row in repository.list_active_projects()}
    assert len(repository.list_active_projects()) == 1
