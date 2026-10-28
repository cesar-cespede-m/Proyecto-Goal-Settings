"""La tabla armada desde un recorrido guarda su receta y se rehace con actividad nueva."""

import json
from pathlib import Path

import pandas as pd
import pytest

from core.analysis.snapshot import build_snapshot
from core.analysis.snapshot_recipe import (
    rebuild_from_recipe,
    record_snapshot_readiness,
    record_snapshot_recipe,
    snapshot_recipe,
)
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

EVENTS = Path(__file__).parents[2] / "data" / "synthetic" / "recorridos" / "altas_eventos.csv"
COVERAGE = {
    "observed_from": "2026-01-01T00:00:00Z",
    "observed_until": "2026-12-31T23:59:59Z",
    "complete": True,
}


def test_recipe_is_kept_and_reapplied(tmp_path: Path) -> None:
    connection = get_connection(tmp_path / "db.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Recorridos")
    frame = pd.read_csv(EVENTS)
    entity, moment, stage = frame.columns[:3]
    source = DataIngestionService(repository, tmp_path).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": EVENTS.name,
            "source_name": "sintético",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=EVENTS.read_bytes(),
    )
    spec = {
        "entity": entity,
        "moment": moment,
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 3},
        "windows_days": [3],
        "features": [{"op": "count_events", "window_days": 3}],
        "outcome": [
            {
                "op": "not_reached_stage",
                "stage": {"column": stage, "value": "primera_compra"},
                "within_days": 30,
                "output_column": "no_avanza",
            }
        ],
    }
    initial = build_snapshot(frame, spec, coverage=COVERAGE)
    record_snapshot_recipe(
        repository,
        project_id=project_id,
        table_dataset_id="tabla-1",
        source_dataset_id=source.metadata.dataset_id,
        spec=spec,
        actor_label="Ana",
        coverage=COVERAGE,
        readiness=initial.readiness,
    )

    recipe = snapshot_recipe(repository, "tabla-1")
    assert recipe["source_dataset_id"] == source.metadata.dataset_id
    assert recipe["readiness"]["eligible_rows"] == frame[entity].nunique()
    assert recipe["coverage"]["complete"] is True
    newer = frame[frame[entity] < "INC-01500"]  # «actividad nueva» con las mismas columnas
    # La receta conserva evidencia histórica, pero nunca la declara para un archivo nuevo.
    pending = rebuild_from_recipe(repository, "tabla-1", newer)
    assert pending.training_frame().empty
    assert pending.readiness.reason_counts == {"COVERAGE_NOT_DECLARED": newer[entity].nunique()}
    rebuilt = rebuild_from_recipe(repository, "tabla-1", newer, coverage=COVERAGE)
    assert "no_avanza" in rebuilt.dataframe.columns
    assert len(rebuilt.dataframe) == newer[entity].nunique()
    with pytest.raises(LookupError):
        rebuild_from_recipe(repository, "otra", newer)
    connection.close()


def test_a_model_trained_on_a_journey_table_retrains_with_new_activity(tmp_path: Path) -> None:
    from core.analysis.snapshot import SNAPSHOT_CUTOFF_COLUMN
    from core.domain.context_service import BusinessContextService
    from core.domain.retraining_service import RetrainingService
    from core.domain.strategy import StrategyProposalV1
    from core.domain.strategy_service import StrategyService
    from core.ml_engine import MLRunService
    from core.reporting.service import ReportService

    connection = get_connection(tmp_path / "db.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Recorridos")
    frame = pd.read_csv(EVENTS)
    entity, moment, stage = frame.columns[:3]
    spec = {
        "entity": entity,
        "moment": moment,
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 3},
        "windows_days": [3],
        "features": [
            {"op": "count_events", "window_days": 3},
            {"op": "days_since_last_event"},
            {"op": "last_value", "column": frame.columns[3]},
            {"op": "mean", "column": frame.columns[4], "window_days": 3},
        ],
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
    entities = pd.read_csv(EVENTS.with_name("altas_entidades.csv"))
    result = build_snapshot(frame, spec, entities, coverage=COVERAGE)
    table = result.training_frame()
    ingest = DataIngestionService(repository, tmp_path)
    request = {
        "source_name": "sintético",
        "classification": "synthetic",
        "authorization_status": "approved",
    }
    source = ingest.ingest_bytes(
        project_id=project_id,
        request={**request, "original_filename_display": EVENTS.name},
        content=EVENTS.read_bytes(),
    )
    built = ingest.ingest_bytes(
        project_id=project_id,
        request={**request, "original_filename_display": "no_avanza.csv"},
        content=table.to_csv(index=False).encode(),
    )
    table_id = built.metadata.dataset_id
    repository.set_modeling_table(project_id, table_id, actor_label="Ana")
    record_snapshot_recipe(
        repository,
        project_id=project_id,
        table_dataset_id=table_id,
        source_dataset_id=source.metadata.dataset_id,
        spec=spec,
        actor_label="Ana",
        coverage=COVERAGE,
        readiness=result.readiness,
    )
    features = [c for c in table.columns if c not in {entity, "no_avanza", SNAPSHOT_CUTOFF_COLUMN}]
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
                    "column": c,
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "dato inicial",
                }
                for c in features
            ]
            + [
                {
                    "column": column,
                    "use": "exclude",
                    "available_at_prediction": True,
                    "reason": "identificador o momento de corte",
                }
                for column in (entity, SNAPSHOT_CUTOFF_COLUMN)
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategies = StrategyService(repository)
    template = strategies.build_template(
        project_id=project_id, context_version_id=context.context_version_id
    )
    strategy = strategies.create_version(
        project_id=project_id, proposal=StrategyProposalV1.model_validate(template.model_dump())
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id, decision="APPROVED", actor_label="Ana"
    )
    run = MLRunService(repository, tmp_path).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="Ana")

    newer = frame[frame[entity] < "INC-02000"]
    rebuilt = rebuild_from_recipe(repository, table_id, newer, entities, coverage=COVERAGE)
    started = RetrainingService(repository, tmp_path).start(
        source_run_id=run.run_id,
        filename="actividad-nueva.csv",
        content=rebuilt.training_frame().to_csv(index=False).encode(),
        actor_label="Ana",
        data_authorized=True,
    )
    assert repository.get_dataset(started.dataset_id)["n_rows"] == newer[entity].nunique()
    record_snapshot_readiness(
        repository,
        project_id=started.project_id,
        table_dataset_id=started.dataset_id,
        source_sha256="a" * 64,
        readiness=rebuilt.readiness,
        actor_label="Ana",
    )
    detail = json.loads(
        connection.execute(
            "SELECT detail_json FROM audit_events WHERE event_type = 'SNAPSHOT_READINESS_RECORDED'"
        ).fetchone()[0]
    )
    assert detail["source_sha256"] == "a" * 64
    assert detail["readiness"] == rebuilt.readiness.model_dump(mode="json")
    assert "INC-00001" not in json.dumps(detail)
    assert "pending_reasons" not in detail["readiness"]
    connection.close()
