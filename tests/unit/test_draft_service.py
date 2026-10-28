"""CTX-01.6: borrador persistente, escritura optimista y conversión revalidada."""

import sqlite3
from pathlib import Path

import pytest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.assistant import Availability, DraftField, DraftNotReadyError, DraftUse
from core.domain.draft_service import ContextDraftService
from core.domain.errors import ContextValidationError, InvalidStateTransitionError
from tests.support.ml_workflow import csv_bytes

ACTOR = "pytest-responsable"


@pytest.fixture
def env(tmp_path: Path):
    connection = get_connection(tmp_path / "drafts.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Borradores")
    storage = tmp_path / "private"
    admission = DataIngestionService(repository, storage).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=csv_bytes(),
    )
    yield (
        connection,
        repository,
        ContextDraftService(repository, storage),
        project_id,
        (admission.metadata.dataset_id),
    )
    connection.close()


def _complete(draft) -> None:
    for name, value in (
        (DraftField.OBJECTIVE, "Priorizar revisiones del fixture"),
        (DraftField.TARGET_COLUMN, "target"),
        (DraftField.POSITIVE_CLASS, 1),
        (DraftField.PREDICTION_MOMENT, "Antes del evento sintético"),
        (DraftField.FALSE_POSITIVE_CONSEQUENCE, "Revisión innecesaria"),
        (DraftField.FALSE_NEGATIVE_CONSEQUENCE, "Evento no detectado"),
        (DraftField.PRIMARY_METRIC, "f1"),
        (DraftField.METRIC_RATIONALE, "Equilibrar detectar y acertar"),
        (DraftField.DECISION_ACTOR_LABEL, ACTOR),
        (DraftField.GUARDRAIL_MODE, "no_quantitative_guardrail"),
    ):
        draft.set_by_person(name, value, actor_label=ACTOR)
    for column in ("fixture_id", "noise"):
        draft.set_feature(
            column,
            use=DraftUse.EXCLUDE,
            available_at_prediction=Availability.YES,
            actor_label=ACTOR,
        )
    for column in ("signal", "category"):
        draft.set_feature(
            column,
            use=DraftUse.INCLUDE,
            available_at_prediction=Availability.YES,
            actor_label=ACTOR,
        )


def test_open_resumes_the_same_active_draft(env) -> None:
    _, _, service, project_id, dataset_id = env
    first = service.open(project_id, dataset_id)
    first.draft.set_by_person(DraftField.OBJECTIVE, "Priorizar", actor_label=ACTOR)
    service.save(first)

    again = service.open(project_id, dataset_id)
    assert again.draft_id == first.draft_id
    assert again.draft.get(DraftField.OBJECTIVE).value == "Priorizar"


def test_stale_tab_cannot_overwrite_a_newer_save(env) -> None:
    _, _, service, project_id, dataset_id = env
    tab_a = service.open(project_id, dataset_id)
    tab_b = service.open(project_id, dataset_id)
    tab_a.draft.set_by_person(DraftField.OBJECTIVE, "Desde A", actor_label=ACTOR)
    service.save(tab_a)
    tab_b.draft.set_by_person(DraftField.OBJECTIVE, "Desde B", actor_label=ACTOR)
    with pytest.raises(InvalidStateTransitionError, match="DRAFT_CHANGED_ELSEWHERE"):
        service.save(tab_b)


def test_finalize_requires_every_confirmation_then_creates_context(env) -> None:
    connection, repository, service, project_id, dataset_id = env
    stored = service.open(project_id, dataset_id)
    with pytest.raises(DraftNotReadyError):
        service.finalize(stored)
    assert repository.get_latest_context(project_id) is None

    _complete(stored.draft)
    result = service.finalize(stored)
    assert sorted(result.context.allowed_features) == ["category", "signal"]
    row = connection.execute(
        "SELECT status, context_version_id FROM context_drafts WHERE id = ?",
        (stored.draft_id,),
    ).fetchone()
    assert (row["status"], row["context_version_id"]) == ("FINALIZED", result.context_version_id)
    with pytest.raises(sqlite3.IntegrityError, match="closed draft"):
        connection.execute(
            "UPDATE context_drafts SET draft_json = '{}' WHERE id = ?", (stored.draft_id,)
        )
    # Un nuevo borrador parte de las decisiones vigentes para editarlas.
    fresh = service.open(project_id, dataset_id)
    assert fresh.draft_id != stored.draft_id
    assert fresh.draft.get(DraftField.OBJECTIVE).value == "Priorizar revisiones del fixture"
    assert fresh.draft.missing_confirmations() == []

    def canonical(context):
        data = context.model_dump(mode="json")
        data["feature_decisions"].sort(key=lambda item: item["column"])
        return data

    assert canonical(fresh.draft.to_business_context()) == canonical(result.context)


def test_finalize_revalidates_against_the_real_data(env) -> None:
    _, _, service, project_id, dataset_id = env
    stored = service.open(project_id, dataset_id)
    _complete(stored.draft)
    stored.draft.set_by_person(DraftField.POSITIVE_CLASS, "no-existe", actor_label=ACTOR)
    with pytest.raises(ContextValidationError):
        service.finalize(stored)


def test_discard_closes_the_draft(env) -> None:
    _, repository, service, project_id, dataset_id = env
    stored = service.open(project_id, dataset_id)
    service.discard(stored)
    with pytest.raises(InvalidStateTransitionError, match="DRAFT_NOT_ACTIVE"):
        service.discard(stored)
    assert service.open(project_id, dataset_id).draft_id != stored.draft_id


def test_draft_rejects_a_dataset_from_another_project(env) -> None:
    _, repository, service, _, dataset_id = env
    other = repository.create_project("Otro proyecto")
    with pytest.raises(ContextValidationError, match="archivo aceptado"):
        service.open(other, dataset_id)
