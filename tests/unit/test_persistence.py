"""Pruebas de integridad y workflow de la persistencia HITL-01."""

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.db import get_connection, migrate
from core.db.migrations import MIGRATIONS_DIR
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.errors import ApprovalRequiredError


@pytest.fixture
def connection(tmp_path: Path) -> sqlite3.Connection:
    database = tmp_path / "goal_settings.sqlite3"
    value = get_connection(database)
    migrate(value)
    yield value
    value.close()


@pytest.fixture
def repository(connection: sqlite3.Connection) -> WorkflowRepository:
    return WorkflowRepository(connection)


def _profile() -> dict[str, object]:
    return {"schema_version": "1.0", "columns": [], "fixture": True}


def _context(target: str = "target") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "target_column": target,
        "positive_class": "yes",
        "prediction_moment": "before outcome",
        "excluded_columns": ["fixture_id"],
        "primary_metric": "f1",
        "guardrails": None,
        "context": {"fixture": True},
    }


def _strategy(seed: int = 42) -> dict[str, object]:
    return {
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "preprocessing": {"numeric": "median+scale"},
        "candidates": [
            "DummyClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
        ],
        "seed": seed,
        "caps": {"n_jobs": 1},
    }


def _manifest(
    *,
    dataset_sha256: str = "a" * 64,
    context_sha256: str = "b" * 64,
    strategy_fingerprint: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_sha256": dataset_sha256,
        "context_sha256": context_sha256,
        "strategy_fingerprint": strategy_fingerprint,
        "versions": {"contract": "1.0"},
        "hyperparameters": {"fixture": True},
        "features": ["feature_fixture"],
        "warnings": ["fixture only"],
    }


def _seed_strategy(repository: WorkflowRepository, *, seed: int = 42) -> tuple[str, str, str, str]:
    project_id = repository.create_project("Fixture project")
    dataset_id = repository.add_dataset(
        project_id=project_id,
        sha256="a" * 64,
        original_filename_display="fixture.csv",
        source="synthetic fixture",
        license_name="test-only",
        anonymization_declared=True,
        admission_status="ACCEPTED",
        rejection_reason=None,
        n_rows=20,
        n_cols=3,
        byte_count=128,
        profile=_profile(),
    )
    context_id = repository.add_context(
        project_id=project_id,
        dataset_id=dataset_id,
        version_no=1,
        payload=_context(),
    )
    strategy_id = repository.add_strategy(
        context_version_id=context_id,
        dataset_id=dataset_id,
        version_no=1,
        payload=_strategy(seed),
        schema_version="1.0",
        source="TEMPLATE",
    )
    return project_id, dataset_id, context_id, strategy_id


def _approve_and_start(repository: WorkflowRepository, strategy_id: str) -> str:
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="fixture-reviewer",
    )
    return repository.start_run(
        strategy_version_id=strategy_id,
        seed=42,
        split_info={"fixture": True},
        candidate_metrics={},
        manifest=_manifest(),
    )


def _finish_completed(
    repository: WorkflowRepository, run_id: str, *, selected_model: str = "DummyClassifier"
) -> None:
    repository.finish_run(
        run_id,
        status="COMPLETED",
        candidate_metrics={"validation": {"fixture_score": 0.5}},
        selected_model=selected_model,
        artifact_path=f"artifacts/{run_id}",
        manifest=_manifest(),
    )


def _report_payload(claim: str = "Fixture claim") -> dict[str, object]:
    return {
        "metrics": {"fixture_score": 0.5},
        "insights": [
            {
                "kind": "FACT",
                "claim": claim,
                "evidence_refs": ["metrics.fixture_score"],
                "limitations": ["fixture only"],
                "source": "TEMPLATE",
            }
        ],
        "limitations": ["Synthetic test fixture; not a real result."],
    }


def test_foreign_keys_and_dataset_hash_unique_per_project(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    project_a = repository.create_project("A")
    project_b = repository.create_project("B")
    common = {
        "sha256": "d" * 64,
        "original_filename_display": "same.csv",
        "source": "synthetic fixture",
        "license_name": None,
        "anonymization_declared": True,
        "admission_status": "ACCEPTED",
        "rejection_reason": None,
        "n_rows": 2,
        "n_cols": 2,
        "byte_count": 20,
        "profile": _profile(),
    }
    repository.add_dataset(project_id=project_a, **common)
    repository.add_dataset(project_id=project_b, **common)
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_dataset(project_id=project_a, **common)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO datasets(
                id, project_id, sha256, original_filename_display, stored_path,
                source, anonymization_declared, admission_status, bytes, profile_json
            ) VALUES ('bad-fk', 'missing', ?, 'x.csv', 'data/accepted/bad-fk',
                      'fixture', 1, 'ACCEPTED', 1, '{}')
            """,
            ("e" * 64,),
        )


@pytest.mark.parametrize(
    "table",
    ["context_versions", "strategy_versions", "approval_events", "audit_events"],
)
def test_append_only_tables_reject_update_and_delete(
    table: str, connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    project_id, _, context_id, strategy_id = _seed_strategy(repository)
    approval_id = repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="fixture-reviewer",
    )
    ids = {
        "context_versions": context_id,
        "strategy_versions": strategy_id,
        "approval_events": approval_id,
        "audit_events": connection.execute(
            "SELECT id FROM audit_events WHERE project_id = ? LIMIT 1", (project_id,)
        ).fetchone()[0],
    }
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(f"UPDATE {table} SET id = id WHERE id = ?", (ids[table],))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(f"DELETE FROM {table} WHERE id = ?", (ids[table],))


def test_approval_check_rejects_null_double_and_incoherent_targets(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    _, _, _, strategy_id = _seed_strategy(repository)
    fingerprint = connection.execute(
        "SELECT fingerprint FROM strategy_versions WHERE id = ?", (strategy_id,)
    ).fetchone()[0]
    invalid_targets = [
        ("TRAINING", None, None),
        ("TRAINING", strategy_id, "report-fixture"),
        ("PUBLICATION", strategy_id, None),
    ]
    for index, (stage, strategy_target, report_target) in enumerate(invalid_targets):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO approval_events(
                    id, stage, decision, strategy_version_id, report_id,
                    approved_hash, actor_label
                ) VALUES (?, ?, 'APPROVED', ?, ?, ?, 'fixture-reviewer')
                """,
                (f"invalid-{index}", stage, strategy_target, report_target, fingerprint),
            )


def test_invalid_json_allowlist_and_approval_hash_are_rejected(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    project_id, dataset_id, context_id, strategy_id = _seed_strategy(repository)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO datasets(
                id, project_id, sha256, original_filename_display, stored_path,
                source, anonymization_declared, admission_status, bytes, profile_json
            ) VALUES ('invalid-json', ?, ?, 'fixture.csv', 'data/accepted/invalid-json',
                      'fixture', 1, 'ACCEPTED', 1, 'not-json')
            """,
            (project_id, "f" * 64),
        )
    invalid_strategy = _strategy()
    invalid_strategy["candidates"] = ["UnapprovedEstimator"]
    with pytest.raises(ValidationError):
        repository.add_strategy(
            context_version_id=context_id,
            dataset_id=dataset_id,
            version_no=2,
            payload=invalid_strategy,
            schema_version="1.0",
            source="TEMPLATE",
        )
    with pytest.raises(sqlite3.IntegrityError, match="training approval hash mismatch"):
        connection.execute(
            """
            INSERT INTO approval_events(
                id, stage, decision, strategy_version_id, approved_hash, actor_label
            ) VALUES ('bad-hash', 'TRAINING', 'APPROVED', ?, ?, 'fixture-reviewer')
            """,
            (strategy_id, "0" * 64),
        )


def test_run_gate_rejects_missing_and_stale_approval(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    project_id, dataset_id, context_id, strategy_id = _seed_strategy(repository)
    with pytest.raises(ApprovalRequiredError, match="STRATEGY_NOT_APPROVED"):
        repository.start_run(
            strategy_version_id=strategy_id,
            seed=42,
            split_info={},
            candidate_metrics={},
            manifest=_manifest(),
        )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'TRAINING_DENIED'"
        ).fetchone()[0]
        == 1
    )

    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="fixture-reviewer",
    )
    changed_strategy_id = repository.add_strategy(
        context_version_id=context_id,
        dataset_id=dataset_id,
        version_no=2,
        payload=_strategy(seed=99),
        schema_version="1.0",
        source="TEMPLATE",
    )
    with pytest.raises(sqlite3.IntegrityError, match="training approval required"):
        connection.execute(
            """
            INSERT INTO runs(
                id, strategy_version_id, status, seed, split_info_json,
                candidate_metrics_json, manifest_json
            ) VALUES ('stale-run', ?, 'RUNNING', 99, '{}', '{}', '{}')
            """,
            (changed_strategy_id,),
        )
    assert connection.execute("SELECT COUNT(*) FROM runs WHERE id = 'stale-run'").fetchone()[0] == 0
    assert project_id


def test_old_report_approval_cannot_publish_new_version(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    _, _, _, strategy_id = _seed_strategy(repository)
    run_id = _approve_and_start(repository, strategy_id)
    _finish_completed(repository, run_id)
    report_v1 = repository.create_report(run_id=run_id, payload=_report_payload("v1"))
    repository.decide_publication(
        report_id=report_v1,
        decision="APPROVED",
        actor_label="fixture-reviewer",
    )
    report_v2 = repository.create_report(run_id=run_id, payload=_report_payload("v2"))
    connection.execute(
        "UPDATE reports SET status = 'APPROVED' WHERE id = ?",
        (report_v2,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="publication approval required"):
        connection.execute(
            "UPDATE reports SET status = 'PUBLISHED' WHERE id = ?",
            (report_v2,),
        )
    with pytest.raises(ApprovalRequiredError, match="REPORT_NOT_APPROVED"):
        repository.publish_report(report_v2)
    assert (
        connection.execute("SELECT status FROM reports WHERE id = ?", (report_v2,)).fetchone()[0]
        == "APPROVED"
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'PUBLICATION_DENIED'"
        ).fetchone()[0]
        == 1
    )


def test_approved_and_published_are_distinct_states(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    _, _, _, strategy_id = _seed_strategy(repository)
    run_id = _approve_and_start(repository, strategy_id)
    _finish_completed(repository, run_id)
    report_id = repository.create_report(run_id=run_id, payload=_report_payload())
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="fixture-reviewer",
    )
    assert (
        connection.execute("SELECT status FROM reports WHERE id = ?", (report_id,)).fetchone()[0]
        == "APPROVED"
    )
    repository.publish_report(report_id)
    assert (
        connection.execute("SELECT status FROM reports WHERE id = ?", (report_id,)).fetchone()[0]
        == "PUBLISHED"
    )


def test_only_one_running_run_and_rerun_preserves_history(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    _, _, _, strategy_id = _seed_strategy(repository)
    first_run = _approve_and_start(repository, strategy_id)
    with pytest.raises(sqlite3.IntegrityError):
        repository.start_run(
            strategy_version_id=strategy_id,
            seed=42,
            split_info={},
            candidate_metrics={},
            manifest=_manifest(),
        )
    _finish_completed(repository, first_run)
    second_run = repository.start_run(
        strategy_version_id=strategy_id,
        seed=42,
        split_info={},
        candidate_metrics={},
        manifest=_manifest(),
    )
    assert first_run != second_run
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    assert (
        connection.execute("SELECT status FROM runs WHERE id = ?", (first_run,)).fetchone()[0]
        == "COMPLETED"
    )


def test_valid_no_recommendation_and_failed_states_persist(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    _, _, _, strategy_id = _seed_strategy(repository)
    no_recommendation = _approve_and_start(repository, strategy_id)
    repository.finish_run(
        no_recommendation,
        status="COMPLETED_WITHOUT_RECOMMENDATION",
        candidate_metrics={"validation": {"fixture": 0.1}},
        selected_model=None,
        manifest=_manifest(),
    )
    failed = repository.start_run(
        strategy_version_id=strategy_id,
        seed=42,
        split_info={},
        candidate_metrics={},
        manifest=_manifest(),
    )
    repository.finish_run(
        failed,
        status="FAILED",
        candidate_metrics={},
        selected_model=None,
        manifest=_manifest(),
        error_message="fixture failure",
    )
    statuses = {row[0] for row in connection.execute("SELECT status FROM runs").fetchall()}
    assert statuses == {"COMPLETED_WITHOUT_RECOMMENDATION", "FAILED"}


def test_canonical_hash_is_independent_of_key_order() -> None:
    assert canonical_hash({"a": 1, "b": {"x": 2}}) == canonical_hash({"b": {"x": 2}, "a": 1})


def test_failed_audit_rolls_back_run_state(
    connection: sqlite3.Connection, repository: WorkflowRepository
) -> None:
    project_id, _, _, strategy_id = _seed_strategy(repository)
    run_id = _approve_and_start(repository, strategy_id)
    existing_audit_id = connection.execute(
        "SELECT id FROM audit_events WHERE project_id = ? LIMIT 1", (project_id,)
    ).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        repository.finish_run(
            run_id,
            status="COMPLETED",
            candidate_metrics={},
            selected_model="DummyClassifier",
            manifest=_manifest(),
            audit_event_id=existing_audit_id,
        )
    row = connection.execute(
        "SELECT status, finished_at FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert tuple(row) == ("RUNNING", None)


def test_data_persists_after_reopening_and_migration_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "persistent.sqlite3"
    first = get_connection(database)
    migrate(first)
    migrate(first)
    project_id = WorkflowRepository(first).create_project("Persistent fixture")
    first.close()

    second = get_connection(database)
    migrate(second)
    assert (
        second.execute("SELECT name FROM projects WHERE id = ?", (project_id,)).fetchone()[0]
        == "Persistent fixture"
    )
    expected_versions = sorted(
        int(path.name.split("_", maxsplit=1)[0])
        for path in MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql")
    )
    applied_versions = [
        row[0] for row in second.execute("SELECT version FROM schema_migrations ORDER BY version")
    ]
    assert applied_versions == expected_versions
    second.close()
