"""Pruebas STRAT-01 con estrategia determinista y aprobación exacta."""

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.canonical import strategy_fingerprint
from core.domain.context_service import BusinessContextService
from core.domain.errors import (
    InvalidStateTransitionError,
    StrategyValidationError,
)
from core.domain.gates import has_training_approval
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService


@pytest.fixture
def strategy_env(
    tmp_path: Path,
) -> tuple[
    sqlite3.Connection,
    WorkflowRepository,
    StrategyService,
    str,
    str,
    str,
]:
    connection = get_connection(tmp_path / "strategy.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Fixture STRAT-01")
    storage_root = tmp_path / "private"
    rows = ["fixture_id,signal,category,target"]
    rows.extend(
        f"id-{index},{index % 5},{'a' if index % 2 else 'b'},{index % 2}" for index in range(30)
    )
    dataset = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic-strategy-fixture.csv",
            "source_name": "pytest generated fixture",
            "license": "test-only",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=("\n".join(rows) + "\n").encode(),
    )
    context = BusinessContextService(repository, storage_root).create_version(
        project_id=project_id,
        dataset_id=dataset.metadata.dataset_id,
        context={
            "schema_version": "business-context.v1",
            "objective": "Clasificar el target del fixture sintético",
            "target_column": "target",
            "positive_class": 1,
            "task_type": "binary_classification",
            "prediction_moment": "Antes de observar el target del fixture",
            "false_positive_consequence": "Revisión innecesaria del fixture",
            "false_negative_consequence": "No detectar el positivo del fixture",
            "primary_metric": "recall",
            "metric_rationale": "El fixture prioriza cobertura de positivos",
            "decision_actor_label": "pytest-context-reviewer",
            "feature_decisions": [
                {
                    "column": "fixture_id",
                    "use": "exclude",
                    "available_at_prediction": True,
                    "reason": "Identificador del fixture",
                },
                {
                    "column": "signal",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Disponible antes del target",
                },
                {
                    "column": "category",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Disponible antes del target",
                },
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    yield (
        connection,
        repository,
        StrategyService(repository),
        project_id,
        dataset.metadata.dataset_id,
        context.context_version_id,
    )
    connection.close()


def test_template_is_deterministic_complete_and_fingerprinted(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
) -> None:
    connection, _, service, project_id, dataset_id, context_id = strategy_env
    proposal = service.build_template(project_id=project_id, context_version_id=context_id)
    result = service.create_version(project_id=project_id, proposal=proposal)
    row = connection.execute(
        "SELECT * FROM strategy_versions WHERE id = ?",
        (result.strategy_version_id,),
    ).fetchone()
    dataset_hash = connection.execute(
        "SELECT sha256 FROM datasets WHERE id = ?", (dataset_id,)
    ).fetchone()[0]
    context_hash = connection.execute(
        "SELECT context_sha256 FROM context_versions WHERE id = ?", (context_id,)
    ).fetchone()[0]

    assert proposal.feature_columns == ["category", "signal"]
    assert proposal.excluded_columns == ["fixture_id"]
    assert proposal.split.model_dump() == {
        "method": "stratified_random",
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
    }
    assert proposal.candidates == [
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    ]
    assert proposal.seed == 42
    assert proposal.source == "TEMPLATE"
    assert "DETERMINISTIC_TEMPLATE_NO_LLM" in proposal.warnings
    assert result.fingerprint == strategy_fingerprint(
        dataset_hash,
        context_hash,
        row["strategy_json"],
        "strategy.v2",
    )


def test_strategy_operation_replay_and_conflicting_request_are_atomic(strategy_env):
    connection, repository, service, project, _dataset, context = strategy_env
    proposal = service.build_template(project_id=project, context_version_id=context)
    first = service.create_version(
        project_id=project, proposal=proposal, operation_id="proposal-1", actor_label="Test actor"
    )
    replay = service.create_version(
        project_id=project, proposal=proposal, operation_id="proposal-1", actor_label="Test actor"
    )
    assert replay == first
    assert repository.get_latest_training_decision(first.strategy_version_id) is None
    assert connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 1
    changed = proposal.model_copy(update={"seed": 43})
    with pytest.raises(InvalidStateTransitionError, match="STRATEGY_OPERATION_CONFLICT"):
        service.create_version(
            project_id=project,
            proposal=changed,
            operation_id="proposal-1",
            actor_label="Test actor",
        )
    with pytest.raises(InvalidStateTransitionError, match="STRATEGY_OPERATION_CONFLICT"):
        service.create_version(
            project_id=project,
            proposal=proposal,
            operation_id="proposal-1",
            actor_label="Another actor",
        )
    assert connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"candidates": ["DummyClassifier", "LogisticRegression", "UnknownModel"]},
        {"preprocessing": {"numeric_imputation": "mean"}},
        {"caps": {"max_parallel_jobs": 8}},
        {"model_params": {"random_forest": {"n_jobs": -1}}},
    ],
)
def test_contract_rejects_models_transformations_or_unsafe_caps(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
    mutation: dict[str, object],
) -> None:
    _, _, service, project_id, _, context_id = strategy_env
    value = service.build_template(project_id=project_id, context_version_id=context_id).model_dump(
        mode="json"
    )
    value.update(mutation)
    with pytest.raises(ValidationError):
        StrategyProposalV1.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_columns", ["signal", "invented"]),
        ("dataset_sha256", "0" * 64),
        ("primary_metric", "precision"),
        ("positive_class", "1"),
    ],
)
def test_strategy_cannot_contradict_the_context(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
    field: str,
    value: object,
) -> None:
    _, _, service, project_id, _, context_id = strategy_env
    proposal = service.build_template(
        project_id=project_id, context_version_id=context_id
    ).model_dump(mode="json")
    proposal[field] = value
    with pytest.raises(StrategyValidationError) as captured:
        service.create_version(project_id=project_id, proposal=proposal)
    assert captured.value.code == "STRATEGY_CONTEXT_MISMATCH"


def test_split_without_class_support_is_rejected(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
) -> None:
    _, _, service, project_id, _, context_id = strategy_env
    proposal = service.build_template(
        project_id=project_id, context_version_id=context_id
    ).model_dump(mode="json")
    proposal["split"] = {
        "method": "stratified_random",
        "train": 0.98,
        "validation": 0.01,
        "test": 0.01,
    }
    with pytest.raises(StrategyValidationError) as captured:
        service.create_version(project_id=project_id, proposal=proposal)
    assert captured.value.code == "INSUFFICIENT_CLASS_SUPPORT"


def test_reordered_identical_strategy_does_not_create_a_version(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
) -> None:
    connection, _, service, project_id, _, context_id = strategy_env
    first = service.create_template_version(
        project_id=project_id,
        context_version_id=context_id,
    )
    proposal = first.proposal.model_dump(mode="json")
    proposal["feature_columns"].reverse()
    proposal["candidates"].reverse()
    with pytest.raises(StrategyValidationError) as captured:
        service.create_version(project_id=project_id, proposal=proposal)
    assert captured.value.code == "STRATEGY_UNCHANGED"
    assert connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 1


def test_new_strategy_invalidates_old_approval_in_domain_and_database(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
) -> None:
    connection, repository, service, project_id, _, context_id = strategy_env
    first = service.create_template_version(
        project_id=project_id,
        context_version_id=context_id,
    )
    repository.decide_training(
        strategy_version_id=first.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-ml-reviewer",
        comment="Aprobación de fixture",
    )
    assert has_training_approval(connection, first.strategy_version_id)

    edited = first.proposal.model_dump(mode="json")
    edited["seed"] = 99
    second = service.create_version(project_id=project_id, proposal=edited)
    assert second.version_no == 2
    assert second.fingerprint != first.fingerprint
    assert not has_training_approval(connection, first.strategy_version_id)
    assert not has_training_approval(connection, second.strategy_version_id)
    with pytest.raises(InvalidStateTransitionError, match="STRATEGY_NOT_CURRENT"):
        repository.decide_training(
            strategy_version_id=first.strategy_version_id,
            decision="APPROVED",
            actor_label="pytest-ml-reviewer",
        )
    with pytest.raises(sqlite3.IntegrityError, match="current context or strategy"):
        connection.execute(
            """
            INSERT INTO runs(
                id, strategy_version_id, status, seed, split_info_json,
                candidate_metrics_json, manifest_json
            ) VALUES ('old-strategy-run', ?, 'RUNNING', 42, '{}', '{}', '{}')
            """,
            (first.strategy_version_id,),
        )

    repository.decide_training(
        strategy_version_id=second.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-ml-reviewer",
    )
    assert has_training_approval(connection, second.strategy_version_id)


def test_latest_rejection_overrides_approval(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
) -> None:
    connection, repository, service, project_id, _, context_id = strategy_env
    strategy = service.create_template_version(
        project_id=project_id,
        context_version_id=context_id,
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-ml-reviewer",
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="REJECTED",
        actor_label="pytest-ml-reviewer",
        comment="Rechazo posterior de fixture",
    )
    assert not has_training_approval(connection, strategy.strategy_version_id)
    latest = repository.get_latest_training_decision(strategy.strategy_version_id)
    assert latest["decision"] == "REJECTED"
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM approval_events WHERE strategy_version_id = ?",
            (strategy.strategy_version_id,),
        ).fetchone()[0]
        == 2
    )


def test_old_context_cannot_receive_a_new_strategy(
    strategy_env: tuple[
        sqlite3.Connection,
        WorkflowRepository,
        StrategyService,
        str,
        str,
        str,
    ],
    tmp_path: Path,
) -> None:
    connection, repository, service, project_id, dataset_id, context_id = strategy_env
    original = service.build_template(project_id=project_id, context_version_id=context_id)
    stored_context = json.loads(repository.get_context(context_id)["context_json"])["context"]
    stored_context["metric_rationale"] = "Nueva justificación humana del fixture"
    storage_root = tmp_path / "private"
    BusinessContextService(repository, storage_root).create_version(
        project_id=project_id,
        dataset_id=dataset_id,
        context=stored_context,
    )
    with pytest.raises(StrategyValidationError) as captured:
        service.create_version(project_id=project_id, proposal=original)
    assert captured.value.code == "CONTEXT_NOT_CURRENT"
    with pytest.raises(sqlite3.IntegrityError, match="current context"):
        repository.add_strategy(
            context_version_id=context_id,
            dataset_id=dataset_id,
            version_no=None,
            payload=original.model_dump(mode="json"),
            schema_version="strategy.v1",
            source="TEMPLATE",
        )
    assert connection.execute("SELECT COUNT(*) FROM strategy_versions").fetchone()[0] == 0
