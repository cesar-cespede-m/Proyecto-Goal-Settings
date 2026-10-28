"""GA-01: tipos de columna genéricos y tabla para entrenar elegida desde el catálogo."""

import json
from pathlib import Path

import pytest

from app.ui.catalog_view import column_types_summary
from core.data_engine import DataIngestionService, analysis_schema, validate_and_profile
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS

KINDS = {"number", "category", "datetime", "text", "identifier", "personal"}


def _schema(path: Path) -> dict[str, str]:
    return analysis_schema(validate_and_profile(path.name, path.read_bytes()).model_dump())


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_domain_gets_generic_column_types(domain) -> None:
    kinds = {kind for path in domain.files for kind in _schema(path).values()}

    assert kinds <= KINDS
    assert ("datetime" in kinds) == domain.has_datetimes
    assert "personal" not in kinds  # los negocios de prueba no traen datos personales


def test_personal_data_is_flagged_by_name_or_by_value_shape() -> None:
    rows = [
        f"{index},persona{index}@example.test,+56 9 0000 00{index:02d},A,"
        f"2026-01-{index % 28 + 1:02d}"
        for index in range(60)
    ]
    content = ("codigo,contacto,telefono,grupo,fecha\n" + "\n".join(rows) + "\n").encode()
    profile = validate_and_profile("prueba.csv", content)

    assert analysis_schema(profile.model_dump()) == {
        "codigo": "identifier",
        "contacto": "personal",
        "telefono": "personal",
        "grupo": "category",
        "fecha": "datetime",
    }
    assert "POSSIBLE_PERSONAL_DATA_COLUMNS:contacto,telefono" in profile.warnings


def test_profiles_saved_before_ga01_still_get_a_schema() -> None:
    legacy = {
        "columns": [
            {"name": "a", "inferred_type": "numeric", "possible_identifier": False},
            {"name": "b", "inferred_type": "string", "possible_identifier": True},
            {"name": "c", "inferred_type": "boolean", "possible_identifier": False},
        ]
    }

    assert analysis_schema(legacy) == {"a": "number", "b": "identifier", "c": "category"}
    assert column_types_summary(legacy) == "1 número · 1 categoría · 1 identificador"


def ingest(repository, runtime: Path, project_id: str, path: Path) -> str:
    return (
        DataIngestionService(repository, runtime)
        .ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": path.name,
                "source_name": "Negocio de prueba sintético",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=path.read_bytes(),
        )
        .metadata.dataset_id
    )


def test_catalog_keeps_several_files_and_one_modeling_table(tmp_path: Path) -> None:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Catálogo")
    multi = next(domain for domain in DOMAINS if domain.multi_file and domain.has_outcome)
    ids = [ingest(repository, tmp_path, project_id, path) for path in multi.files]

    assert len(repository.list_datasets(project_id, admission_status="ACCEPTED")) == len(ids)
    assert repository.get_active_dataset(project_id)["id"] == ids[0]

    with pytest.raises(ValueError):
        repository.set_modeling_table(project_id, ids[2], actor_label=" ")
    repository.set_modeling_table(project_id, ids[2], actor_label="Responsable")

    assert repository.get_active_dataset(project_id)["id"] == ids[2]
    event = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'MODELING_TABLE_SELECTED'"
    ).fetchone()
    assert json.loads(event["detail_json"])["actor_label"] == "Responsable"
    other = repository.create_project("Otro")
    with pytest.raises(LookupError):
        repository.set_modeling_table(other, ids[0], actor_label="Responsable")
    connection.close()
