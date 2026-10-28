"""DATA-02.2: linaje, snapshot activo e invalidación al componer CSV."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.data_engine import (
    DataIngestionService,
    DatasetAdmissionRequest,
    DatasetCompositionError,
    DatasetCompositionService,
)
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.errors import ContextValidationError
from core.domain.gates import has_training_approval


def _request(filename: str) -> DatasetAdmissionRequest:
    return DatasetAdmissionRequest(
        original_filename_display=filename,
        source_name="Synthetic DATA-02 pytest fixture",
        license="test-only",
        classification="synthetic",
        authorization_status="approved",
    )


def _ingest(service: DataIngestionService, project_id: str, filename: str, content: bytes) -> str:
    return service.ingest_bytes(
        project_id=project_id,
        request=_request(filename),
        content=content,
    ).metadata.dataset_id


def _old_approved_strategy(repository: WorkflowRepository, project_id: str, dataset_id: str) -> str:
    context_id = repository.add_context(
        project_id=project_id,
        dataset_id=dataset_id,
        version_no=None,
        payload={
            "schema_version": "fixture.v1",
            "target_column": "target",
            "positive_class": 1,
            "prediction_moment": "before outcome",
            "excluded_columns": ["case_id"],
            "primary_metric": "f1",
        },
    )
    strategy_id = repository.add_strategy(
        context_version_id=context_id,
        dataset_id=dataset_id,
        version_no=None,
        payload={
            "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "preprocessing": {},
            "candidates": [
                "DummyClassifier",
                "LogisticRegression",
                "RandomForestClassifier",
            ],
            "seed": 42,
            "caps": {},
        },
        schema_version="fixture.v1",
        source="TEMPLATE",
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest reviewer",
    )
    return strategy_id


def _business_context(target: str, columns: list[str], *, include: set[str]) -> dict[str, object]:
    return {
        "schema_version": "business-context.v1",
        "objective": "Clasificar el resultado del fixture compuesto",
        "target_column": target,
        "positive_class": 1 if target == "target" else "A",
        "task_type": "binary_classification",
        "prediction_moment": "Antes de observar el resultado",
        "false_positive_consequence": "Revisión innecesaria",
        "false_negative_consequence": "No detectar un caso positivo",
        "primary_metric": "recall",
        "metric_rationale": "Priorizar detección de positivos",
        "decision_actor_label": "pytest reviewer",
        "feature_decisions": [
            {
                "column": column,
                "use": "include" if column in include else "exclude",
                "available_at_prediction": True,
                "reason": "Decisión humana del fixture",
            }
            for column in columns
            if column != target
        ],
        "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
    }


def test_persists_lineage_activates_derived_snapshot_and_invalidates_old_approval(
    tmp_path: Path,
) -> None:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    runtime = tmp_path / "runtime"
    ingestion = DataIngestionService(repository, runtime)
    project_id = repository.create_project("Composición sintética")
    primary_id = _ingest(
        ingestion,
        project_id,
        "primary.csv",
        b"case_id,target,value\n1,0,10\n2,1,20\n3,0,30\n4,1,40\n5,0,50\n6,1,60\n",
    )
    segment_id = _ingest(
        ingestion,
        project_id,
        "segments.csv",
        b"key,segment\n1,A\n2,B\n3,A\n4,B\n5,A\n6,B\n",
    )
    region_id = _ingest(
        ingestion,
        project_id,
        "regions.csv",
        b"key,region\n1,N\n2,S\n3,N\n4,S\n5,N\n6,S\n",
    )
    strategy_id = _old_approved_strategy(repository, project_id, primary_id)
    assert has_training_approval(connection, strategy_id)

    stored = DatasetCompositionService(repository, runtime).create(
        project_id=project_id,
        request={
            "primary_dataset_id": primary_id,
            "actor_label": "Data steward pytest",
            "joins": [
                {
                    "dataset_id": segment_id,
                    "alias": "segment",
                    "primary_key": "case_id",
                    "auxiliary_key": "key",
                },
                {
                    "dataset_id": region_id,
                    "alias": "region",
                    "primary_key": "case_id",
                    "auxiliary_key": "key",
                },
            ],
        },
    )

    active = repository.get_active_dataset(project_id)
    assert active is not None
    assert active["id"] == stored.output_dataset_id
    assert active["dataset_kind"] == "DERIVED"
    assert repository.get_latest_context(project_id) is None
    assert not has_training_approval(connection, strategy_id)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM dataset_composition_sources WHERE composition_id = ?",
            (stored.composition_id,),
        ).fetchone()[0]
        == 3
    )
    composition = repository.get_dataset_composition(stored.output_dataset_id)
    assert composition is not None
    assert json.loads(composition["summary_json"])["source_count"] == 3

    output = runtime / active["stored_path"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == active["sha256"]
    frame = pd.read_csv(output)
    assert list(frame.columns) == [
        "case_id",
        "target",
        "value",
        "segment__segment",
        "region__region",
    ]
    audit = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'DATASET_COMPOSITION_CREATED'"
    ).fetchone()[0]
    assert "segment__segment" not in audit
    assert "A" not in json.loads(audit).values()

    context_service = BusinessContextService(repository, runtime)
    columns = list(frame.columns)
    with pytest.raises(ContextValidationError) as join_key_error:
        context_service.create_version(
            project_id=project_id,
            dataset_id=stored.output_dataset_id,
            context=_business_context(
                "target",
                columns,
                include={"case_id", "value", "segment__segment"},
            ),
        )
    assert join_key_error.value.code == "JOIN_KEY_INCLUDED"

    with pytest.raises(ContextValidationError) as target_error:
        context_service.create_version(
            project_id=project_id,
            dataset_id=stored.output_dataset_id,
            context=_business_context(
                "segment__segment",
                columns,
                include={"value", "region__region"},
            ),
        )
    assert target_error.value.code == "TARGET_FROM_AUXILIARY"

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "UPDATE dataset_compositions SET actor_label = 'changed' WHERE id = ?",
            (stored.composition_id,),
        )
    connection.close()


def test_rejects_cross_project_source_and_placeholder_actor(tmp_path: Path) -> None:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    runtime = tmp_path / "runtime"
    ingestion = DataIngestionService(repository, runtime)
    first = repository.create_project("First")
    second = repository.create_project("Second")
    primary_id = _ingest(ingestion, first, "primary.csv", b"id,target\n1,0\n2,1\n")
    foreign_id = _ingest(ingestion, second, "foreign.csv", b"id,value\n1,A\n2,B\n")
    service = DatasetCompositionService(repository, runtime)
    request = {
        "primary_dataset_id": primary_id,
        "actor_label": "PENDIENTE",
        "joins": [
            {
                "dataset_id": foreign_id,
                "alias": "foreign",
                "primary_key": "id",
                "auxiliary_key": "id",
            }
        ],
    }
    with pytest.raises(DatasetCompositionError) as actor_error:
        service.create(project_id=first, request=request)
    assert actor_error.value.code == "COMPOSITION_ACTOR_REQUIRED"

    request["actor_label"] = "Data steward pytest"
    with pytest.raises(DatasetCompositionError) as source_error:
        service.create(project_id=first, request=request)
    assert source_error.value.code == "SOURCE_NOT_AVAILABLE"
    connection.close()
