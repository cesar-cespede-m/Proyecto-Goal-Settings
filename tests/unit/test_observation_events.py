"""C2-01: captura agnóstica y materialización de ObservationEventV1."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.analysis.engine import load_project_source
from core.contracts.platform import ObservationEventV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.integration.events import EventIngestionService

ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path: Path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Eventos genéricos")
    service = EventIngestionService(repository, tmp_path)
    yield connection, repository, project_id, service
    connection.close()


def _event(event_id: str = "event-1", **overrides) -> ObservationEventV1:
    values = {
        "source": "system_a",
        "entity_type": "generic_entity",
        "entity_id": event_id,
        "occurred_at": NOW,
        "attributes": {"measure": 3.5, "segment": "a"},
    }
    values.update(overrides)
    return ObservationEventV1(**values)


def test_ingests_valid_event_and_audits_only_counts(setup) -> None:
    connection, _, project_id, service = setup

    result = service.ingest(
        project_id,
        [_event()],
        source_label="Sistema de prueba",
        actor_label="Responsable",
    )

    assert result.accepted == 1 and result.duplicates == 0 and result.rejected == []
    row = connection.execute("SELECT * FROM observation_events").fetchone()
    assert row["entity_id"] == "event-1" and row["entity"] == "generic_entity"
    audit = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'OBSERVATION_EVENTS_INGESTED'"
    ).fetchone()
    assert json.loads(audit[0]) == {
        "accepted": 1,
        "actor_label": "Responsable",
        "duplicates": 0,
        "rejected": 0,
        "source_label": "Sistema de prueba",
    }


def test_identical_resend_is_idempotent_per_project(setup) -> None:
    connection, _, project_id, service = setup
    kwargs = {"source_label": "Origen", "actor_label": "Responsable"}

    first = service.ingest(project_id, [_event()], **kwargs)
    second = service.ingest(project_id, [_event()], **kwargs)

    assert first.accepted == 1
    assert second.accepted == 0 and second.duplicates == 1
    assert connection.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 1


def test_outcome_arriving_later_updates_the_case_without_rewriting(setup, tmp_path) -> None:
    connection, repository, project_id, service = setup
    kwargs = {"source_label": "Origen", "actor_label": "Responsable"}
    service.ingest(project_id, [_event(), _event("event-2")], **kwargs)
    later = _event(outcome="observed", outcome_observed_at=NOW + timedelta(days=2))
    result = service.ingest(project_id, [later], **kwargs)

    assert result.accepted == 1  # versión nueva del mismo caso, guardada aparte
    assert connection.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 3
    dataset_id = service.materialize(
        project_id,
        entity="generic_entity",
        actor_label="Responsable",
        display_name="casos.csv",
        classification="synthetic",
    )
    frame = load_project_source(repository, tmp_path, dataset_id).frame
    assert len(frame) == 2  # una fila por caso
    assert frame.set_index("id").loc["event-1", "outcome"] == "observed"


def test_personal_attribute_is_rejected_without_leaking_its_value(setup) -> None:
    connection, _, project_id, service = setup
    private_value = "persona.unica@example.test"

    result = service.ingest(
        project_id,
        [_event(attributes={"contact": private_value})],
        source_label="Origen",
        actor_label="Responsable",
    )

    assert result.accepted == 0
    assert result.rejected == [(0, "PERSONAL_ATTRIBUTE_DETECTED")]
    assert connection.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 0
    persisted_sql = "\n".join(connection.iterdump())
    assert private_value not in persisted_sql
    audit = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'OBSERVATION_EVENTS_INGESTED'"
    ).fetchone()[0]
    assert json.loads(audit)["rejected"] == 1 and private_value not in audit


def test_observation_events_are_append_only(setup) -> None:
    connection, _, project_id, service = setup
    service.ingest(
        project_id,
        [_event()],
        source_label="Origen",
        actor_label="Responsable",
    )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "UPDATE observation_events SET source = 'changed' WHERE project_id = ?",
            (project_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM observation_events WHERE project_id = ?", (project_id,))
    assert connection.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 1


def test_materialize_registers_an_accepted_readable_source(setup, tmp_path: Path) -> None:
    _, repository, project_id, service = setup
    event = _event(
        outcome="observed",
        outcome_observed_at=NOW + timedelta(hours=1),
    )
    service.ingest(
        project_id,
        [event],
        source_label="Origen",
        actor_label="Responsable",
    )

    dataset_id = service.materialize(
        project_id,
        entity="generic_entity",
        actor_label="Responsable",
        display_name="eventos.csv",
        classification="sandbox",
    )

    dataset = repository.get_dataset(dataset_id)
    assert dataset["admission_status"] == "ACCEPTED"
    assert dataset["dataset_kind"] == "SOURCE"
    assert json.loads(dataset["profile_json"])["data_classification"] == "sandbox"
    loaded = load_project_source(repository, tmp_path, dataset_id)
    assert loaded.frame.loc[0, "id"] == "event-1"
    assert loaded.frame.loc[0, "measure"] == 3.5
    assert loaded.frame.loc[0, "outcome"] == "observed"


def test_push_events_cli_reads_jsonl_from_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("CLI")
    connection.close()
    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text(
        json.dumps(_event("event-cli").model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["GOAL_SETTINGS_RUNTIME_DIR"] = str(runtime)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "push_events.py"),
            str(jsonl),
            "--project-id",
            project_id,
            "--source-label",
            "CLI",
            "--actor-label",
            "Responsable",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"accepted": 1, "duplicates": 0, "rejected": 0}
    reopened = get_connection(runtime / "goal_settings.sqlite3")
    assert reopened.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0] == 1
    reopened.close()
