"""C2-02: decisiones y resultados observados, sólo de inserción y auditados."""

import sqlite3
from pathlib import Path

import pytest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.decisions import DecisionRecordError, DecisionService


@pytest.fixture
def setup(tmp_path: Path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    yield repository, repository.create_project("Decisiones")
    connection.close()


def test_decision_and_its_outcome_are_kept_and_audited(setup) -> None:
    repository, project_id = setup
    service = DecisionService(repository)
    decision_id = service.record(
        project_id,
        decision="Reforzar la revisión del grupo con más casos",
        based_on="analysis",
        reference_id="casos por grupo",
        actor_label="Analista",
    )
    service.observe(decision_id, observed_result="Bajaron los casos", actor_label="Analista")

    [stored] = service.list(project_id)
    assert stored.record.based_on == "analysis"
    assert stored.outcomes[0]["observed_result"] == "Bajaron los casos"
    events = {
        row["event_type"]
        for row in repository.connection.execute("SELECT event_type FROM audit_events")
    }
    assert {"DECISION_RECORDED", "DECISION_OUTCOME_OBSERVED"} <= events
    with pytest.raises(sqlite3.DatabaseError):
        repository.connection.execute("DELETE FROM decision_records")


def test_incomplete_decisions_are_explained(setup) -> None:
    repository, project_id = setup
    service = DecisionService(repository)

    with pytest.raises(DecisionRecordError) as missing:
        service.record(
            project_id, decision="x", based_on="analysis", reference_id="r", actor_label="A"
        )
    assert missing.value.code == "INVALID_DECISION_RECORD"
    with pytest.raises(DecisionRecordError) as unknown:
        service.observe("no-existe", observed_result="algo", actor_label="A")
    assert unknown.value.code == "DECISION_NOT_FOUND"
