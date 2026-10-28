"""V2-11: protección previa al cambio de tabla para entrenar."""

import hashlib

import pytest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.table_guard import compare_tables, dataset_origin


def _repository(tmp_path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    return connection, WorkflowRepository(connection)


def _add_table(
    repository: WorkflowRepository,
    project_id: str,
    dataset_id: str,
    *,
    rows: int,
    columns: list[str],
    active: bool = False,
) -> str:
    return repository.add_dataset(
        project_id=project_id,
        dataset_id=dataset_id,
        sha256=hashlib.sha256(dataset_id.encode()).hexdigest(),
        original_filename_display=f"{dataset_id}.csv",
        source="Prueba sintética",
        license_name=None,
        anonymization_declared=True,
        admission_status="ACCEPTED",
        rejection_reason=None,
        n_rows=rows,
        n_cols=len(columns),
        byte_count=100,
        profile={
            "schema_version": "dataset-profile.v1",
            "row_count": rows,
            "column_count": len(columns),
            "columns": [{"name": name} for name in columns],
        },
        make_active=active,
    )


def test_three_row_summary_has_clear_explanation_and_is_blocked(tmp_path) -> None:
    connection, repository = _repository(tmp_path)
    project_id = repository.create_project("Resumen")
    _add_table(repository, project_id, "active", rows=100, columns=["case", "result"], active=True)
    _add_table(repository, project_id, "summary", rows=3, columns=["group", "count"])

    review = compare_tables(repository, project_id, "summary")

    assert review.active_rows == 100
    assert review.candidate_rows == 3
    assert review.looks_like_summary
    assert "resumen por grupos, no una fila por caso" in review.summary_explanation
    assert review.summary_explanation in review.warnings
    assert review.level == "block"
    connection.close()


def test_missing_result_column_warns_that_decisions_must_be_redone(tmp_path) -> None:
    connection, repository = _repository(tmp_path)
    project_id = repository.create_project("Resultado")
    active_id = _add_table(
        repository,
        project_id,
        "active",
        rows=80,
        columns=["case", "measure", "result"],
        active=True,
    )
    _add_table(repository, project_id, "candidate", rows=60, columns=["case", "measure"])
    repository.add_context(
        project_id=project_id,
        dataset_id=active_id,
        version_no=None,
        payload={
            "schema_version": "context.v1",
            "target_column": "result",
            "positive_class": "yes",
            "prediction_moment": "Antes de decidir",
            "excluded_columns": [],
            "primary_metric": "recall",
        },
    )

    review = compare_tables(repository, project_id, "candidate")

    assert review.missing_result_column == "result"
    assert any(
        "Tus decisiones usan «result» como resultado" in warning
        and "tendrás que definirlas de nuevo" in warning
        for warning in review.warnings
    )
    assert review.invalidated == ["context"]
    assert review.level == "warning"
    connection.close()


def test_suitable_table_is_ok(tmp_path) -> None:
    connection, repository = _repository(tmp_path)
    project_id = repository.create_project("Adecuada")
    _add_table(repository, project_id, "active", rows=100, columns=["case", "result"], active=True)
    _add_table(
        repository,
        project_id,
        "candidate",
        rows=80,
        columns=["case", "measure", "result"],
    )

    review = compare_tables(repository, project_id, "candidate")

    assert review.level == "ok"
    assert review.warnings == []
    assert review.invalidated == []
    connection.close()


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (9, ["case", "result"]),
        (50, ["result"]),
    ],
)
def test_too_few_rows_or_columns_is_blocked(tmp_path, rows, columns) -> None:
    connection, repository = _repository(tmp_path)
    project_id = repository.create_project("Insuficiente")
    _add_table(repository, project_id, "active", rows=60, columns=["case", "result"], active=True)
    _add_table(repository, project_id, "candidate", rows=rows, columns=columns)

    review = compare_tables(repository, project_id, "candidate")

    assert review.level == "block"
    connection.close()


class _OriginRepository:
    def __init__(self, *, computed=(), compositions=()):
        self.computed = set(computed)
        self.compositions = set(compositions)

    def get_analysis_table(self, dataset_id):
        return {"output_dataset_id": dataset_id} if dataset_id in self.computed else None

    def get_dataset_composition(self, dataset_id):
        return {"output_dataset_id": dataset_id} if dataset_id in self.compositions else None


def test_origin_distinguishes_computed_composition_and_source() -> None:
    repository = _OriginRepository(computed={"calculated"}, compositions={"joined"})

    assert dataset_origin(repository, "calculated") == "computed"
    assert dataset_origin(repository, "joined") == "composition"
    assert dataset_origin(repository, "uploaded") == "source"
