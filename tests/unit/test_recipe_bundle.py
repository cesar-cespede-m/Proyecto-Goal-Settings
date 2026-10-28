"""C2-04: exportar e importar receta y decisiones sin datos; la persona confirma."""

import json

import pytest

from core.data_engine import DataIngestionService
from core.domain.assistant import ConfirmationStatus, DraftField
from core.domain.draft_service import ContextDraftService
from core.domain.recipe_bundle import RecipeBundleError, export_bundle, parse_bundle
from tests.unit.test_recipe_retraining import _files, trained  # noqa: F401


def test_bundle_carries_recipe_and_decisions_but_no_rows(trained) -> None:  # noqa: F811
    repository, storage, run_id = trained
    content = export_bundle(repository, storage, run_id)
    bundle, context = parse_bundle(content)

    assert bundle.recipe.analysis_plan is not None and bundle.analysis_plan is not None
    assert context.target_column == "marcado"
    text = content.decode()
    first_value = _files(1)["casos"][1].decode().splitlines()[1].split(",")[1]
    assert first_value not in text  # ningún valor de los archivos viaja en el paquete


def test_tampered_or_foreign_files_are_rejected(trained) -> None:  # noqa: F811
    repository, storage, run_id = trained
    raw = json.loads(export_bundle(repository, storage, run_id))
    raw["context"]["objective"] = "Otro objetivo cambiado a mano"
    with pytest.raises(RecipeBundleError) as tampered:
        parse_bundle(json.dumps(raw).encode())
    assert tampered.value.code == "BUNDLE_TAMPERED"
    with pytest.raises(RecipeBundleError) as foreign:
        parse_bundle(b"{}")
    assert foreign.value.code == "BUNDLE_INVALID"


def test_import_proposes_decisions_in_another_project(trained) -> None:  # noqa: F811
    repository, storage, run_id = trained
    content = export_bundle(repository, storage, run_id)
    project_id = repository.create_project("Otro proyecto")
    filename, data = _files(3)["casos"]
    dataset_id = (
        DataIngestionService(repository, storage)
        .ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": filename,
                "source_name": "Fixture sintético pytest",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=data,
        )
        .metadata.dataset_id
    )
    service = ContextDraftService(repository, storage)
    stored = service.open(project_id, dataset_id)

    proposed = service.reuse_from_bundle(
        stored, content, filename="receta.json", actor_label="Analista"
    )

    assert proposed > 0
    objective = stored.draft.get(DraftField.OBJECTIVE)
    assert objective.status is not ConfirmationStatus.CONFIRMED  # sólo propuesta
