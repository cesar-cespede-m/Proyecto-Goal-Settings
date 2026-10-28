"""V2-08/V2-09: carga aislada de varias fuentes con declaración común."""

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.data_engine import (
    BatchFileV1,
    BatchIngestionService,
    SharedAdmissionV1,
)
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

ROOT = Path(__file__).parents[2]
TRANSPORT = ROOT / "data" / "synthetic" / "transporte"
TRANSPORT_FILES = (
    "transporte_buses.csv",
    "transporte_encuestas.csv",
    "transporte_estaciones.csv",
    "transporte_transacciones.csv",
)


@pytest.fixture
def batch_context(
    tmp_path: Path,
) -> tuple[sqlite3.Connection, WorkflowRepository, BatchIngestionService, str, Path]:
    connection = get_connection(tmp_path / "batch.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Lote sintético")
    storage_root = tmp_path / "private-runtime"
    service = BatchIngestionService(repository, storage_root)
    yield connection, repository, service, project_id, storage_root
    connection.close()


def _shared(**changes: object) -> SharedAdmissionV1:
    values = {
        "source_name": "Suite sintética de transporte",
        "license": "test-only",
        "classification": "synthetic",
        "authorization_status": "approved",
        "anonymization_declared": False,
        "owner_label": "Responsable de datos",
        "approver_label": "Aprobador de datos",
    }
    values.update(changes)
    return SharedAdmissionV1.model_validate(values)


def _transport_files() -> list[BatchFileV1]:
    files = [
        BatchFileV1(filename=name, content=(TRANSPORT / name).read_bytes())
        for name in TRANSPORT_FILES
    ]
    files[1] = BatchFileV1(
        filename=files[1].filename,
        content=files[1].content,
        overrides={"source_name": "Encuesta sintética autorizada"},
    )
    return files


def test_preview_four_files_uses_shared_data_and_writes_nothing(
    batch_context: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        BatchIngestionService,
        str,
        Path,
    ],
) -> None:
    connection, repository, service, _, storage_root = batch_context
    audit_before = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    preview = service.preview(_transport_files(), _shared())

    assert len(preview.files) == 4
    assert {item.status for item in preview.files} == {"accepted"}
    assert all(item.dataset_id is None for item in preview.files)
    assert all(item.rows and item.columns for item in preview.files)
    assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM project_roles").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == audit_before
    assert repository.get_project_roles("missing-project") is None
    assert not storage_root.exists()


def test_batch_isolates_invalid_duplicate_and_personal_files_and_keeps_going(
    batch_context: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        BatchIngestionService,
        str,
        Path,
    ],
) -> None:
    connection, repository, service, project_id, _ = batch_context
    transport_files = _transport_files()

    first = service.ingest(project_id, transport_files, _shared())

    assert [item.status for item in first.files] == ["accepted"] * 4
    assert all(item.dataset_id for item in first.files)
    overridden = connection.execute(
        "SELECT source FROM datasets WHERE original_filename_display = ?",
        ("transporte_encuestas.csv",),
    ).fetchone()[0]
    common = connection.execute(
        "SELECT source FROM datasets WHERE original_filename_display = ?",
        ("transporte_buses.csv",),
    ).fetchone()[0]
    assert overridden == "Encuesta sintética autorizada"
    assert common == "Suite sintética de transporte"

    personal = b"email,value\npersona1@example.test,1\npersona2@example.test,2\n"
    second = service.ingest(
        project_id,
        [
            BatchFileV1(filename="no-es-csv.txt", content=b"a,b\n1,2\n"),
            BatchFileV1(
                filename="duplicado.csv",
                content=transport_files[0].content,
            ),
            BatchFileV1(filename="personas.csv", content=personal),
            BatchFileV1(filename="quinta-fuente.csv", content=b"a,b\n3,4\n"),
        ],
        _shared(owner_label="No reemplazar", approver_label="No reemplazar"),
    )

    by_name = {item.filename: item for item in second.files}
    assert by_name["no-es-csv.txt"].status == "rejected"
    assert by_name["no-es-csv.txt"].code == "INVALID_EXTENSION"
    assert by_name["duplicado.csv"].status == "duplicate"
    assert by_name["duplicado.csv"].code == "DATASET_ALREADY_REGISTERED"
    assert by_name["personas.csv"].status == "rejected"
    assert by_name["personas.csv"].code == "POSSIBLE_PERSONAL_DATA"
    assert "POSSIBLE_PERSONAL_DATA_COLUMNS:email" in by_name["personas.csv"].warnings
    assert by_name["quinta-fuente.csv"].status == "accepted"
    assert all(
        item.message and len(item.message.split(". ")) >= 2
        for item in second.files
        if item.status != "accepted"
    )

    roles = repository.get_project_roles(project_id)
    assert roles is not None
    assert roles["version_no"] == 1
    assert roles["owner_label"] == "Responsable de datos"
    assert roles["approver_label"] == "Aprobador de datos"
    assert connection.execute("SELECT COUNT(*) FROM project_roles").fetchone()[0] == 1
    events = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'PROJECT_ROLES_DECLARED'"
    ).fetchall()
    assert len(events) == 1
    assert json.loads(events[0]["detail_json"]) == {
        "approver_label": "Aprobador de datos",
        "identity_verified": False,
        "owner_label": "Responsable de datos",
        "version_no": 1,
    }


def test_project_roles_are_versioned_append_only(
    batch_context: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        BatchIngestionService,
        str,
        Path,
    ],
) -> None:
    connection, repository, _, project_id, _ = batch_context

    assert (
        repository.set_project_roles(
            project_id,
            owner_label="Primera responsable",
            approver_label="Primer aprobador",
        )
        == 1
    )
    assert (
        repository.set_project_roles(
            project_id,
            owner_label="Responsable vigente",
            approver_label="Aprobador vigente",
        )
        == 2
    )
    current = repository.get_project_roles(project_id)
    assert current is not None
    assert current["version_no"] == 2
    assert current["owner_label"] == "Responsable vigente"
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "UPDATE project_roles SET owner_label = 'alterado' WHERE project_id = ?",
            (project_id,),
        )


def test_batch_contracts_forbid_unknown_fields_and_blank_shared_values() -> None:
    with pytest.raises(ValidationError):
        SharedAdmissionV1.model_validate(
            {
                **_shared().model_dump(),
                "source_name": " ",
            }
        )
    with pytest.raises(ValidationError):
        BatchFileV1(filename="x.csv", content=b"a\n1\n", unknown=True)
