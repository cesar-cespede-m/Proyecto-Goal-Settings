"""Pruebas de DATA-01 con contenido sintético etiquetado como fixture."""

import json
import sqlite3
from io import BytesIO
from pathlib import Path

import pytest

from core.data_engine import (
    DataIngestionService,
    DatasetAdmissionRequest,
    DatasetRejectedError,
    IngestionLimits,
    validate_and_profile,
)
from core.data_engine.errors import DataValidationError
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository


@pytest.fixture
def data_context(
    tmp_path: Path,
) -> tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path]:
    connection = get_connection(tmp_path / "data-engine.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Synthetic DATA-01 fixture")
    storage_root = tmp_path / "private-runtime"
    service = DataIngestionService(repository, storage_root)
    yield connection, repository, service, project_id, storage_root
    connection.close()


def _request(
    filename: str = "synthetic-fixture.csv",
    *,
    classification: str = "synthetic",
    authorization_status: str = "approved",
    anonymization_declared: bool = False,
) -> DatasetAdmissionRequest:
    return DatasetAdmissionRequest(
        original_filename_display=filename,
        source_name="Synthetic pytest fixture",
        source_url="https://example.invalid/synthetic-fixture",
        license="test-only",
        classification=classification,
        authorization_status=authorization_status,
        anonymization_declared=anonymization_declared,
    )


def test_valid_utf8_bom_is_profiled_and_stored_without_rows_in_sqlite(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, _, service, project_id, storage_root = data_context
    content = ("id,value,segment,constant\n1,10.5,A,same\n2,,B,same\n2,,B,same\n").encode(
        "utf-8-sig"
    )

    result = service.ingest_bytes(
        project_id=project_id,
        request=_request("../../user-controlled.csv"),
        content=content,
    )

    assert result.metadata.admission_status == "ACCEPTED"
    assert result.metadata.row_count == 3
    assert result.metadata.column_count == 4
    assert result.metadata.storage_key_private == (f"data/accepted/{result.metadata.dataset_id}")
    assert (storage_root / result.metadata.storage_key_private).read_bytes() == content
    assert result.profile is not None
    assert result.profile.duplicate_row_count == 1
    assert "constant" in result.profile.constant_columns
    value_profile = next(column for column in result.profile.columns if column.name == "value")
    assert value_profile.missing_count == 2
    assert value_profile.numeric_summary is not None
    assert value_profile.numeric_summary.mean == 10.5

    stored_profile = connection.execute(
        "SELECT profile_json FROM datasets WHERE id = ?",
        (result.metadata.dataset_id,),
    ).fetchone()[0]
    assert "10.5" in stored_profile  # agregado numérico, no una fila serializada
    assert "user-controlled" not in result.metadata.storage_key_private
    assert json.loads(stored_profile)["schema_version"] == "dataset-profile.v1"


@pytest.mark.parametrize(
    ("filename", "content", "limits", "expected_code"),
    [
        ("fixture.txt", b"a,b\n1,2\n", None, "INVALID_EXTENSION"),
        ("fixture.csv", b"", None, "EMPTY_FILE"),
        ("fixture.csv", b"PK\x03\x04fixture", None, "COMPRESSED_FILE"),
        ("fixture.csv", b"name\n\xff\n", None, "INVALID_ENCODING"),
        ("fixture.csv", b"a,A\n1,2\n", None, "DUPLICATE_HEADERS"),
        ("fixture.csv", b"a,b\n1\n", None, "MALFORMED_ROW"),
        (
            "fixture.csv",
            b"a,b,c\n1,2,3\n",
            IngestionLimits(max_columns=2),
            "TOO_MANY_COLUMNS",
        ),
        (
            "fixture.csv",
            b"a\n1\n2\n3\n",
            IngestionLimits(max_rows=2),
            "TOO_MANY_ROWS",
        ),
        (
            "fixture.csv",
            b"a\n123456789\n",
            IngestionLimits(max_upload_bytes=5),
            "FILE_TOO_LARGE",
        ),
    ],
)
def test_invalid_csv_is_rejected_with_actionable_code(
    filename: str,
    content: bytes,
    limits: IngestionLimits | None,
    expected_code: str,
) -> None:
    with pytest.raises(DataValidationError) as captured:
        validate_and_profile(filename, content, limits=limits)
    assert captured.value.code == expected_code


def test_pending_authorization_stays_quarantined_and_is_not_profiled(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, _, service, project_id, storage_root = data_context
    content = b"private_text,value\nDO_NOT_PROFILE,1\n"

    result = service.ingest_bytes(
        project_id=project_id,
        request=_request(authorization_status="pending"),
        content=content,
    )

    assert result.metadata.admission_status == "QUARANTINED"
    assert result.profile is None
    assert result.metadata.row_count is None
    assert (storage_root / result.metadata.storage_key_private).read_bytes() == content
    stored_profile = connection.execute(
        "SELECT profile_json FROM datasets WHERE id = ?",
        (result.metadata.dataset_id,),
    ).fetchone()[0]
    assert "DO_NOT_PROFILE" not in stored_profile
    assert json.loads(stored_profile)["status"] == "NOT_PROFILED"


def test_anonymized_without_declaration_remains_quarantined(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    _, _, service, project_id, _ = data_context
    result = service.ingest_bytes(
        project_id=project_id,
        request=_request(classification="anonymized", anonymization_declared=False),
        content=b"a,b\n1,2\n",
    )
    assert result.metadata.admission_status == "QUARANTINED"
    assert result.profile is None


def test_invalid_upload_is_persisted_as_rejected(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, _, service, project_id, _ = data_context
    with pytest.raises(DatasetRejectedError) as captured:
        service.ingest_bytes(
            project_id=project_id,
            request=_request("fixture.zip"),
            content=b"PK\x03\x04fixture",
        )
    row = connection.execute(
        "SELECT admission_status, rejection_reason FROM datasets WHERE id = ?",
        (captured.value.dataset_id,),
    ).fetchone()
    assert tuple(row) == ("REJECTED", "INVALID_EXTENSION")


def test_explicitly_rejected_authorization_is_not_profiled(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, _, service, project_id, _ = data_context
    with pytest.raises(DatasetRejectedError) as captured:
        service.ingest_bytes(
            project_id=project_id,
            request=_request(authorization_status="rejected"),
            content=b"text,value\nSHOULD_NOT_PROFILE,1\n",
        )
    row = connection.execute(
        "SELECT profile_json FROM datasets WHERE id = ?",
        (captured.value.dataset_id,),
    ).fetchone()
    assert "SHOULD_NOT_PROFILE" not in row[0]


def test_profile_warns_about_high_cardinality_and_possible_identifiers() -> None:
    rows = ["fixture_id,category,value"]
    rows.extend(f"{index},category-{index},{index * 2}" for index in range(1, 22))
    content = ("\n".join(rows) + "\n").encode()
    profile = validate_and_profile(
        "synthetic.csv",
        content,
        limits=IngestionLimits(high_cardinality_min_unique=20),
    )
    assert "category" in profile.high_cardinality_columns
    assert any(warning.startswith("POSSIBLE_IDENTIFIER_COLUMNS:") for warning in profile.warnings)


def test_same_hash_is_allowed_in_another_project_but_not_the_same_project(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, repository, service, first_project, storage_root = data_context
    content = b"a,b\n1,2\n"
    service.ingest_bytes(project_id=first_project, request=_request(), content=content)
    with pytest.raises(DataValidationError) as captured:
        service.ingest_bytes(project_id=first_project, request=_request(), content=content)
    assert captured.value.code == "DATASET_ALREADY_REGISTERED"

    second_project = repository.create_project("Second synthetic fixture")
    second_service = DataIngestionService(repository, storage_root)
    result = second_service.ingest_bytes(
        project_id=second_project,
        request=_request(),
        content=content,
    )
    assert result.metadata.admission_status == "ACCEPTED"
    assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 2


def test_stream_limit_rejects_before_reading_the_entire_upload(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, repository, _, project_id, storage_root = data_context
    limited_service = DataIngestionService(
        repository,
        storage_root,
        limits=IngestionLimits(max_upload_bytes=8),
    )
    stream = BytesIO(b"a,b\n1,2\ntrailing-content-not-read")
    with pytest.raises(DataValidationError, match="supera el límite") as captured:
        limited_service.ingest_stream(
            project_id=project_id,
            request=_request(),
            stream=stream,
        )
    assert captured.value.code == "FILE_TOO_LARGE"
    assert stream.tell() == 9
    assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


def test_project_accepts_multiple_sources_but_only_one_is_active(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, repository, service, project_id, _ = data_context
    pending = service.ingest_bytes(
        project_id=project_id,
        request=_request(authorization_status="pending"),
        content=b"a,b\n1,2\n",
    )
    assert pending.metadata.admission_status == "QUARANTINED"
    service.ingest_bytes(project_id=project_id, request=_request(), content=b"a,b\n3,4\n")

    service.ingest_bytes(project_id=project_id, request=_request(), content=b"a,b\n5,6\n")
    accepted = repository.list_datasets(project_id, admission_status="ACCEPTED")
    assert len(accepted) == 2
    assert sum(int(row["is_active"]) for row in accepted) == 1

    repository_row = accepted[0]
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO datasets(
                id, project_id, sha256, original_filename_display, stored_path, source,
                anonymization_declared, admission_status, bytes, profile_json, is_active
            ) VALUES ('bypass', ?, ?, 'x.csv', 'bypass/path', 'pytest', 0,
                      'ACCEPTED', 1, '{}', 1)
            """,
            (project_id, "f" * 64),
        )
    assert repository_row["admission_status"] == "ACCEPTED"


def test_project_rejects_a_sixth_accepted_source(
    data_context: tuple[sqlite3.Connection, WorkflowRepository, DataIngestionService, str, Path],
) -> None:
    connection, repository, service, project_id, _ = data_context
    for index in range(5):
        result = service.ingest_bytes(
            project_id=project_id,
            request=_request(f"source-{index}.csv"),
            content=f"id,value\n{index},{index + 1}\n".encode(),
        )
        assert result.metadata.admission_status == "ACCEPTED"

    with pytest.raises(DataValidationError) as captured:
        service.ingest_bytes(
            project_id=project_id,
            request=_request("source-5.csv"),
            content=b"id,value\n5,6\n",
        )

    assert captured.value.code == "PROJECT_SOURCE_LIMIT_REACHED"
    assert (
        len(
            repository.list_datasets(
                project_id,
                admission_status="ACCEPTED",
                dataset_kind="SOURCE",
            )
        )
        == 5
    )
    assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 5
