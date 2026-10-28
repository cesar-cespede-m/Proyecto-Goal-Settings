"""GA-13: reutilizar el contexto de otro proyecto como propuestas que la persona confirma."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.domain.assistant import ConfirmationStatus, DraftField, FieldSource
from core.domain.draft_service import ContextDraftService
from tests.support.ml_workflow import csv_bytes, prepared_strategy

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def _second_project(repository, runtime: Path, content: bytes) -> tuple[str, str]:
    project_id = repository.create_project("Proyecto nuevo")
    dataset_id = (
        DataIngestionService(repository, runtime)
        .ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": "otro.csv",
                "source_name": "Fixture sintético pytest",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=content,
        )
        .metadata.dataset_id
    )
    return project_id, dataset_id


def test_saved_context_arrives_as_proposals_never_confirmed(tmp_path: Path) -> None:
    connection, repository, _, _, runtime = prepared_strategy(tmp_path)
    frame = pd.read_csv(pd.io.common.BytesIO(csv_bytes()))
    content = frame.drop(columns=["noise"]).to_csv(index=False).encode()
    project_id, dataset_id = _second_project(repository, runtime, content)
    service = ContextDraftService(repository, runtime)
    stored = service.open(project_id, dataset_id)
    saved = repository.list_reusable_contexts(project_id)
    assert len(saved) == 1

    with pytest.raises(ValueError):
        service.reuse(stored, saved[0]["id"], actor_label=" ")
    proposed = service.reuse(stored, saved[0]["id"], actor_label="Responsable nuevo")

    draft = service.open(project_id, dataset_id).draft
    assert proposed > 0
    target = draft.get(DraftField.TARGET_COLUMN)
    assert target.value == "target"
    assert target.status is ConfirmationStatus.PROPOSED
    assert target.source is FieldSource.SAVED_CONTEXT
    assert draft.get(DraftField.DECISION_ACTOR_LABEL).status is ConfirmationStatus.EMPTY
    columns = {item.column: item for item in draft.features}
    assert "noise" not in columns
    assert columns["signal"].status is ConfirmationStatus.PROPOSED
    assert columns["signal"].note_source is FieldSource.SAVED_CONTEXT
    assert draft.missing_confirmations()  # nada queda listo sin confirmar
    event = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'CONTEXT_REUSED'"
    ).fetchone()
    assert json.loads(event["detail_json"])["actor_label"] == "Responsable nuevo"
    with pytest.raises(LookupError):
        service.reuse(stored, "inexistente", actor_label="Responsable nuevo")
    connection.close()


def test_archived_projects_are_not_offered_for_reuse(tmp_path: Path) -> None:
    connection, repository, _, _, runtime = prepared_strategy(tmp_path)
    source_project = repository.list_projects()[0]["id"]
    project_id, _ = _second_project(repository, runtime, csv_bytes())
    repository.archive_project(source_project, actor_label="Responsable")

    assert repository.list_reusable_contexts(project_id) == []
    connection.close()


def test_assistant_offers_a_saved_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    connection, repository, _, _, _ = prepared_strategy(tmp_path, storage_subdir=".")
    _second_project(repository, tmp_path, csv_bytes())
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    actor = next(item for item in app.text_input if (item.key or "").startswith("actor-input-"))
    actor.input("Responsable nuevo")
    next(item for item in app.button if (item.key or "").endswith("-actor")).click()
    app.run(timeout=60)
    next(item for item in app.button if item.label == "Usar como propuesta").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("Se propusieron" in item.value for item in app.success)
    assert any("De un contexto guardado" in item.value for item in app.caption)
