"""Pruebas CTX-01 con CSV sintéticos etiquetados como fixtures."""

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context import BusinessContextV1
from core.domain.context_service import BusinessContextService
from core.domain.errors import ApprovalRequiredError, ContextValidationError
from core.domain.gates import require_training_approval


@pytest.fixture
def context_env(
    tmp_path: Path,
) -> tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str]:
    connection = get_connection(tmp_path / "context.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Fixture CTX-01")
    storage_root = tmp_path / "private"
    ingestion = DataIngestionService(repository, storage_root)
    rows = ["fixture_id,signal,category,target,alternate_target"]
    rows.extend(
        f"id-{index},{index % 3},{'a' if index % 2 else 'b'},{index % 2},{(index + 1) % 2}"
        for index in range(20)
    )
    result = ingestion.ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic-context-fixture.csv",
            "source_name": "pytest generated fixture",
            "license": "test-only",
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": False,
        },
        content=("\n".join(rows) + "\n").encode(),
    )
    service = BusinessContextService(repository, storage_root)
    yield connection, repository, service, project_id, result.metadata.dataset_id
    connection.close()


def _business_context(
    *,
    target: str = "target",
    positive_class: object = 1,
    exclude: set[str] | None = None,
) -> dict[str, object]:
    columns = {"fixture_id", "signal", "category", "target", "alternate_target"}
    excluded = (exclude or {"fixture_id", "alternate_target"}) - {target}
    decisions = []
    for column in sorted(columns - {target}):
        is_excluded = column in excluded
        decisions.append(
            {
                "column": column,
                "use": "exclude" if is_excluded else "include",
                "available_at_prediction": True,
                "reason": (
                    "Fixture excluida por revisión humana"
                    if is_excluded
                    else "Fixture declarada disponible por revisión humana"
                ),
            }
        )
    return {
        "schema_version": "business-context.v1",
        "objective": "Clasificar el resultado binario de este fixture sintético",
        "target_column": target,
        "positive_class": positive_class,
        "task_type": "binary_classification",
        "prediction_moment": "Antes de observar el resultado del fixture",
        "false_positive_consequence": "Revisión innecesaria del caso sintético",
        "false_negative_consequence": "No detectar el caso sintético positivo",
        "primary_metric": "recall",
        "metric_rationale": "El revisor prioriza detectar los positivos del fixture",
        "decision_actor_label": "pytest-reviewer-no-auth",
        "feature_decisions": decisions,
        "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
    }


def _strategy() -> dict[str, object]:
    return {
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "preprocessing": {"numeric": "median+scale"},
        "candidates": [
            "DummyClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
        ],
        "seed": 42,
        "caps": {"n_jobs": 1},
    }


def _context_service_for_csv(
    tmp_path: Path,
    content: bytes,
) -> tuple[sqlite3.Connection, BusinessContextService, str, str]:
    connection = get_connection(tmp_path / "target-cases.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Fixture target validation")
    storage_root = tmp_path / "private-target-cases"
    dataset = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic-target-case.csv",
            "source_name": "pytest generated fixture",
            "license": "test-only",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=content,
    )
    return (
        connection,
        BusinessContextService(repository, storage_root),
        project_id,
        dataset.metadata.dataset_id,
    )


def _simple_context() -> dict[str, object]:
    context = _business_context()
    context["feature_decisions"] = [
        {
            "column": "signal",
            "use": "include",
            "available_at_prediction": True,
            "reason": "Señal disponible en el fixture",
        }
    ]
    return context


def test_valid_context_is_persisted_with_derived_fields_and_hash(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
) -> None:
    connection, _, service, project_id, dataset_id = context_env
    result = service.create_version(
        project_id=project_id,
        dataset_id=dataset_id,
        context=_business_context(),
    )

    row = connection.execute(
        "SELECT * FROM context_versions WHERE id = ?", (result.context_version_id,)
    ).fetchone()
    assert result.version_no == 1
    assert len(result.context_sha256) == 64
    assert result.context.allowed_features == ["category", "signal"]
    assert result.context.excluded_features == ["alternate_target", "fixture_id"]
    assert row["primary_metric"] == "recall"
    assert row["positive_class"] == "1"
    assert [item.count for item in result.validation.target_distribution] == [10, 10]


def test_same_schema_supports_a_second_domain_fixture(tmp_path: Path) -> None:
    connection = get_connection(tmp_path / "second.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Fixture ecommerce")
    storage_root = tmp_path / "private"
    dataset = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic-ecommerce.csv",
            "source_name": "pytest generated fixture",
            "license": "test-only",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=(
            "session_id,channel,amount,converted\n"
            + "\n".join(
                f"s-{index},{'web' if index % 2 else 'store'},{index % 4},{index % 2}"
                for index in range(20)
            )
            + "\n"
        ).encode(),
    )
    context = _business_context()
    context.update(
        {
            "objective": "Clasificar conversión en un fixture sintético",
            "target_column": "converted",
            "positive_class": 1,
            "feature_decisions": [
                {
                    "column": "session_id",
                    "use": "exclude",
                    "available_at_prediction": True,
                    "reason": "Identificador del fixture",
                },
                {
                    "column": "channel",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Disponible antes del resultado",
                },
                {
                    "column": "amount",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Disponible antes del resultado",
                },
            ],
        }
    )
    result = BusinessContextService(repository, storage_root).create_version(
        project_id=project_id,
        dataset_id=dataset.metadata.dataset_id,
        context=context,
    )
    assert result.context.schema_version == "business-context.v1"
    assert result.context.allowed_features == ["amount", "channel"]
    connection.close()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"target_column": "missing"}, "TARGET_NOT_FOUND"),
        ({"positive_class": "1"}, "POSITIVE_CLASS_NOT_FOUND"),
    ],
)
def test_invalid_target_or_positive_class_is_blocked(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
    mutation: dict[str, object],
    code: str,
) -> None:
    _, _, service, project_id, dataset_id = context_env
    context = _business_context()
    context.update(mutation)
    with pytest.raises(ContextValidationError) as captured:
        service.create_version(project_id=project_id, dataset_id=dataset_id, context=context)
    assert captured.value.code == code


@pytest.mark.parametrize(
    ("rows", "expected_code"),
    [
        (
            [f"{index % 3},{'' if index == 0 else index % 2}" for index in range(12)],
            "TARGET_HAS_MISSING",
        ),
        ([f"{index % 3},{index % 3}" for index in range(18)], "TARGET_NOT_BINARY"),
        (
            [f"{index % 3},{1 if index < 4 else 0}" for index in range(20)],
            "INSUFFICIENT_CLASS_SUPPORT",
        ),
    ],
)
def test_null_non_binary_or_unsplittable_target_is_blocked(
    tmp_path: Path,
    rows: list[str],
    expected_code: str,
) -> None:
    content = ("signal,target\n" + "\n".join(rows) + "\n").encode()
    connection, service, project_id, dataset_id = _context_service_for_csv(tmp_path, content)
    try:
        with pytest.raises(ContextValidationError) as captured:
            service.create_version(
                project_id=project_id,
                dataset_id=dataset_id,
                context=_simple_context(),
            )
        assert captured.value.code == expected_code
    finally:
        connection.close()


def test_every_feature_requires_a_decision_and_unknown_features_are_rejected(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
) -> None:
    _, _, service, project_id, dataset_id = context_env
    incomplete = _business_context()
    incomplete["feature_decisions"] = incomplete["feature_decisions"][:-1]
    with pytest.raises(ContextValidationError) as missing:
        service.create_version(project_id=project_id, dataset_id=dataset_id, context=incomplete)
    assert missing.value.code == "FEATURE_DECISIONS_INCOMPLETE"

    unknown = _business_context()
    unknown["feature_decisions"].append(
        {
            "column": "invented_by_fixture",
            "use": "exclude",
            "available_at_prediction": False,
            "reason": "Columna inexistente de prueba",
        }
    )
    with pytest.raises(ContextValidationError) as extra:
        service.create_version(project_id=project_id, dataset_id=dataset_id, context=unknown)
    assert extra.value.code == "FEATURE_NOT_FOUND"


def test_possible_identifier_cannot_be_silently_included(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
) -> None:
    _, _, service, project_id, dataset_id = context_env
    context = _business_context(exclude={"alternate_target"})
    with pytest.raises(ContextValidationError) as captured:
        service.create_version(project_id=project_id, dataset_id=dataset_id, context=context)
    assert captured.value.code == "UNSAFE_FEATURE_INCLUDED"


def test_contract_rejects_unknown_fields_and_implicit_guardrail() -> None:
    context = _business_context()
    context["chosen_by_llm"] = True
    with pytest.raises(ValidationError):
        BusinessContextV1.model_validate(context)

    context = _business_context()
    context["guardrails"] = {
        "mode": "no_quantitative_guardrail",
        "rules": [
            {
                "metric": "precision",
                "operator": ">=",
                "threshold": 0.5,
                "rationale": "Fixture inválido",
            }
        ],
    }
    with pytest.raises(ValidationError):
        BusinessContextV1.model_validate(context)


def test_new_context_version_changes_hash_and_invalidates_old_strategy_approval(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
) -> None:
    connection, repository, service, project_id, dataset_id = context_env
    first = service.create_version(
        project_id=project_id,
        dataset_id=dataset_id,
        context=_business_context(),
    )
    strategy_id = repository.add_strategy(
        context_version_id=first.context_version_id,
        dataset_id=dataset_id,
        version_no=1,
        payload=_strategy(),
        schema_version="strategy.v1",
        source="TEMPLATE",
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer-no-auth",
    )
    require_training_approval(connection, strategy_id)

    second = service.create_version(
        project_id=project_id,
        dataset_id=dataset_id,
        context=_business_context(
            target="alternate_target",
            exclude={"fixture_id", "target"},
        ),
    )
    assert second.version_no == 2
    assert second.context_sha256 != first.context_sha256
    with pytest.raises(ApprovalRequiredError):
        require_training_approval(connection, strategy_id)
    with pytest.raises(ApprovalRequiredError):
        repository.start_run(
            strategy_version_id=strategy_id,
            seed=42,
            split_info={},
            candidate_metrics={},
            manifest={
                "schema_version": "fixture.v1",
                "dataset_sha256": "a" * 64,
                "context_sha256": first.context_sha256,
                "strategy_fingerprint": "b" * 64,
                "versions": {},
                "hyperparameters": {},
                "features": ["signal"],
                "warnings": ["pytest fixture"],
            },
        )
    with pytest.raises(sqlite3.IntegrityError, match="current context"):
        connection.execute(
            """
            INSERT INTO runs(
                id, strategy_version_id, status, seed, split_info_json,
                candidate_metrics_json, manifest_json
            ) VALUES ('direct-stale-run', ?, 'RUNNING', 42, '{}', '{}', '{}')
            """,
            (strategy_id,),
        )


def test_identical_context_does_not_create_a_redundant_version(
    context_env: tuple[sqlite3.Connection, WorkflowRepository, BusinessContextService, str, str],
) -> None:
    connection, _, service, project_id, dataset_id = context_env
    context = _business_context()
    service.create_version(project_id=project_id, dataset_id=dataset_id, context=context)
    context["feature_decisions"].reverse()
    with pytest.raises(ContextValidationError) as captured:
        service.create_version(project_id=project_id, dataset_id=dataset_id, context=context)
    assert captured.value.code == "CONTEXT_UNCHANGED"
    assert connection.execute("SELECT COUNT(*) FROM context_versions").fetchone()[0] == 1
