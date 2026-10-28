"""C2-06: los eventos nuevos alimentan archivos y medición sin repetir trabajo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.integration.measurement as measurement_module
from core.analysis.contracts import AnalysisPlanV1
from core.analysis.tables import AnalysisTableService
from core.contracts.platform import ObservationEventV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.integration.events import EventIngestionService
from core.integration.measurement import MeasurementReport, MeasurementService

ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 10, 21, 15, 0, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path: Path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Medición genérica")
    ingestion = EventIngestionService(repository, tmp_path)
    measurement = MeasurementService(repository, tmp_path)
    yield connection, repository, project_id, ingestion, measurement
    connection.close()


def _event(entity: str, case_id: str, *, value: str = "row-private-value"):
    return ObservationEventV1(
        source="external_system",
        entity_type=entity,
        entity_id=case_id,
        occurred_at=NOW,
        attributes={"generic_measure": 3.5, "opaque_label": value},
    )


def _ingest(ingestion, project_id: str, events) -> None:
    ingestion.ingest(
        project_id,
        events,
        source_label="Sistema externo",
        actor_label="Responsable",
    )


def _observed_event(entity: str, case_id: str, *, outcome: float) -> ObservationEventV1:
    return _event(entity, case_id).model_copy(
        update={"outcome": outcome, "outcome_observed_at": NOW + timedelta(hours=1)}
    )


def _save_two_source_table(repository, tmp_path, project_id, ingestion) -> str:
    _ingest(
        ingestion,
        project_id,
        [
            _observed_event("entity_a", "case-1", outcome=1.0),
            _observed_event("entity_b", "case-1", outcome=2.0),
        ],
    )
    source_ids = {
        "principal": ingestion.materialize(
            project_id,
            entity="entity_a",
            actor_label="Responsable",
            display_name="principal.csv",
            classification="synthetic",
        ),
        "relacionada": ingestion.materialize(
            project_id,
            entity="entity_b",
            actor_label="Responsable",
            display_name="relacionada.csv",
            classification="synthetic",
        ),
    }
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": ["principal", "relacionada"],
            "steps": [
                {
                    "op": "join",
                    "source": "relacionada",
                    "left_on": "id",
                    "right_on": "id",
                    "cardinality": "one_to_one",
                    "right_prefix": "relacionada_",
                }
            ],
            "output": {"chart": "table"},
        }
    )
    saved = AnalysisTableService(repository, tmp_path).save(
        project_id=project_id,
        plan=plan,
        source_ids=source_ids,
        display_name="Tabla de seguimiento",
        actor_label="Responsable",
    )
    return saved.dataset_id


def _fake_published_prepared_model(monkeypatch, repository, dataset_id: str):
    proposal = SimpleNamespace(
        dataset_id=dataset_id,
        task_type="regression",
        primary_metric="mae",
    )

    class FakeScoring:
        calls: list[tuple[str, str, bytes]] = []

        def __init__(self, repository, storage_root) -> None:
            self.repository = repository

        def approved_run(self, run_id):
            return {}, {}, proposal

        def evaluate_prediction(self, run_id, *, filename, content):
            self.calls.append((run_id, filename, content))
            return SimpleNamespace(
                primary_metric="mae",
                follow_up={"mae": 0.75},
                degraded=False,
            )

    monkeypatch.setattr(measurement_module, "ScoringService", FakeScoring)
    monkeypatch.setattr(
        repository,
        "list_project_runs",
        lambda requested: [{"report_status": "PUBLISHED", "run_id": "run-prepared"}],
    )
    return FakeScoring


def test_without_new_events_does_not_create_files(setup) -> None:
    connection, _, project_id, ingestion, measurement = setup
    _ingest(ingestion, project_id, [_event("entity_a", "case-1")])
    ingestion.materialize(
        project_id,
        entity="entity_a",
        actor_label="Responsable",
        display_name="manual.csv",
        classification="synthetic",
    )

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert report.created_files == []
    assert report.unchanged_entities == ["entity_a"]
    assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1


def test_new_events_create_one_file_per_entity_and_second_refresh_does_not_duplicate(setup):
    connection, repository, project_id, ingestion, measurement = setup
    _ingest(
        ingestion,
        project_id,
        [_event("entity_a", "case-1"), _event("entity_b", "case-2")],
    )

    first = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="sandbox",
    )
    second = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="sandbox",
    )

    assert {item.entity for item in first.created_files} == {"entity_a", "entity_b"}
    assert all(item.filename.endswith(".csv") for item in first.created_files)
    assert second.created_files == []
    assert second.unchanged_entities == ["entity_a", "entity_b"]
    assert len(repository.list_datasets(project_id, admission_status="ACCEPTED")) == 2
    audits = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'MEASUREMENT_REFRESHED' "
        "ORDER BY rowid"
    ).fetchall()
    assert json.loads(audits[0][0])["created_files"] == 2
    assert json.loads(audits[1][0])["created_files"] == 0


def test_report_and_refresh_audit_do_not_contain_row_values(setup) -> None:
    connection, _, project_id, ingestion, measurement = setup
    private_value = "unique-row-value-that-must-not-leak"
    _ingest(ingestion, project_id, [_event("entity_a", "case-1", value=private_value)])

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert isinstance(report, MeasurementReport)
    assert private_value not in report.model_dump_json()
    detail = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'MEASUREMENT_REFRESHED'"
    ).fetchone()[0]
    assert private_value not in detail
    assert json.loads(detail) == {
        "actor_label": "Responsable",
        "classification": "synthetic",
        "created_files": 1,
        "measured_models": 0,
        "unchanged_entities": 0,
        "warnings": 0,
    }


def test_published_current_model_is_evaluated_without_registering_use(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection, repository, project_id, ingestion, measurement = setup
    original = _event("entity_a", "case-1")
    _ingest(ingestion, project_id, [original])
    training_dataset_id = ingestion.materialize(
        project_id,
        entity="entity_a",
        actor_label="Responsable",
        display_name="training.csv",
        classification="synthetic",
    )
    observed = original.model_copy(
        update={"outcome": 4.5, "outcome_observed_at": NOW + timedelta(hours=1)}
    )
    _ingest(ingestion, project_id, [observed])

    proposal = SimpleNamespace(
        dataset_id=training_dataset_id,
        task_type="regression",
        primary_metric="mae",
    )

    class FakeScoring:
        calls = 0

        def __init__(self, repository, storage_root) -> None:
            self.repository = repository

        def approved_run(self, run_id):
            return {}, {}, proposal

        def evaluate_prediction(self, run_id, *, filename, content):
            FakeScoring.calls += 1
            return SimpleNamespace(
                primary_metric="mae",
                follow_up={"mae": 1.25},
                degraded=True,
            )

    monkeypatch.setattr(measurement_module, "ScoringService", FakeScoring)
    monkeypatch.setattr(
        repository,
        "list_project_runs",
        lambda requested: [{"report_status": "PUBLISHED", "run_id": "run-1"}],
    )

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert FakeScoring.calls == 1
    assert report.measured_models[0].model_dump() == {
        "run_id": "run-1",
        "entity": "entity_a",
        "task_type": "regression",
        "primary_metric": "mae",
        "metric_value": 1.25,
        "degraded": True,
    }
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'MODEL_USED_FOR_SCORING'"
        ).fetchone()[0]
        == 0
    )


def test_prepared_table_model_is_measured_when_every_event_source_has_a_new_version(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, repository, project_id, ingestion, measurement = setup
    table_dataset_id = _save_two_source_table(repository, tmp_path, project_id, ingestion)
    fake_scoring = _fake_published_prepared_model(monkeypatch, repository, table_dataset_id)
    _ingest(
        ingestion,
        project_id,
        [
            _observed_event("entity_a", "case-1", outcome=3.0),
            _observed_event("entity_b", "case-1", outcome=4.0),
        ],
    )

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert len(fake_scoring.calls) == 1
    assert fake_scoring.calls[0][1] == "Tabla de seguimiento.csv"
    assert b"relacionada_generic_measure" in fake_scoring.calls[0][2]
    assert report.warnings == []
    assert report.measured_models[0].entity == "entity_a, entity_b"


def test_prepared_table_model_warns_and_is_not_measured_when_a_new_source_is_missing(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, repository, project_id, ingestion, measurement = setup
    table_dataset_id = _save_two_source_table(repository, tmp_path, project_id, ingestion)
    fake_scoring = _fake_published_prepared_model(monkeypatch, repository, table_dataset_id)
    _ingest(
        ingestion,
        project_id,
        [_observed_event("entity_a", "case-1", outcome=3.0)],
    )

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert fake_scoring.calls == []
    assert report.measured_models == []
    assert len(report.warnings) == 1
    assert "falta versión nueva de relacionada" in report.warnings[0]


def test_prepared_table_measurement_does_not_persist_another_derived_dataset(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection, repository, project_id, ingestion, measurement = setup
    table_dataset_id = _save_two_source_table(repository, tmp_path, project_id, ingestion)
    _fake_published_prepared_model(monkeypatch, repository, table_dataset_id)
    derived_before = connection.execute(
        "SELECT COUNT(*) FROM datasets WHERE dataset_kind = 'DERIVED'"
    ).fetchone()[0]
    _ingest(
        ingestion,
        project_id,
        [
            _observed_event("entity_a", "case-1", outcome=3.0),
            _observed_event("entity_b", "case-1", outcome=4.0),
        ],
    )

    report = measurement.refresh(
        project_id,
        actor_label="Responsable",
        classification="synthetic",
    )

    assert len(report.measured_models) == 1
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM datasets WHERE dataset_kind = 'DERIVED'"
        ).fetchone()[0]
        == derived_before
    )
    assert connection.execute("SELECT COUNT(*) FROM analysis_tables").fetchone()[0] == 1


def test_refresh_measurements_cli_uses_a_temporary_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("CLI")
    _ingest(EventIngestionService(repository, runtime), project_id, [_event("entity_c", "case-3")])
    connection.close()
    env = os.environ.copy()
    env["GOAL_SETTINGS_RUNTIME_DIR"] = str(runtime)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "refresh_measurements.py"),
            "--project",
            project_id,
            "--actor",
            "Programador local",
            "--classification",
            "sandbox",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert len(output["created_files"]) == 1
    assert output["created_files"][0]["entity"] == "entity_c"
    reopened = get_connection(runtime / "goal_settings.sqlite3")
    assert (
        reopened.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'MEASUREMENT_REFRESHED'"
        ).fetchone()[0]
        == 1
    )
    reopened.close()
