"""Integración ML-02 con SQLite/storage usando únicamente fixtures sintéticos."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from core.db.repositories import WorkflowRepository
from core.domain.errors import ApprovalRequiredError
from core.domain.strategy import StrategyProposalV1
from core.ml_engine import RunBlockedError
from tests.support.ml_workflow import prepared_strategy


def test_execution_requires_approval_then_persists_metrics_manifest_and_artifact(
    tmp_path: Path,
) -> None:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    with pytest.raises(ApprovalRequiredError, match="STRATEGY_NOT_APPROVED"):
        service.execute(strategy_id)
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'TRAINING_DENIED'"
        ).fetchone()[0]
        == 1
    )

    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
        comment="Aprobación sintética de prueba",
    )
    result = service.execute(strategy_id)
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (result.run_id,)).fetchone()

    assert result.status == "COMPLETED"
    assert result.selected_model == "LogisticRegression"
    assert row["status"] == "COMPLETED"
    assert row["selected_model"] == result.selected_model
    assert row["finished_at"] is not None
    split = json.loads(row["split_info_json"])
    metrics = json.loads(row["candidate_metrics_json"])
    manifest = json.loads(row["manifest_json"])
    assert (
        split["train"]["row_count"],
        split["validation"]["row_count"],
        split["test"]["row_count"],
    ) == (
        120,
        40,
        40,
    )
    assert set(metrics["validation"]) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    }
    assert metrics["holdout_evaluations"] == 1
    assert metrics["selection"]["selected_model"] == result.selected_model
    assert manifest["run_id"] == result.run_id
    assert manifest["execution_status"] == "COMPLETED"
    expected_dataset_hash = connection.execute(
        """
        SELECT dataset.sha256
        FROM runs AS run
        JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
        JOIN datasets AS dataset ON dataset.id = strategy.dataset_id
        WHERE run.id = ?
        """,
        (result.run_id,),
    ).fetchone()[0]
    assert manifest["dataset_sha256"] == expected_dataset_hash
    assert row["artifact_path"] == result.artifact_path
    artifact = storage_root / Path(result.artifact_path)
    assert artifact.is_file()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact_sha256"]
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE entity_id = ? AND event_type = 'RUN_FINISHED'",
            (result.run_id,),
        ).fetchone()[0]
        == 1
    )
    connection.close()


def test_dataset_hash_tampering_blocks_run_and_never_records_success(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    dataset = connection.execute(
        """
        SELECT dataset.stored_path
        FROM strategy_versions AS strategy
        JOIN datasets AS dataset ON dataset.id = strategy.dataset_id
        WHERE strategy.id = ?
        """,
        (strategy_id,),
    ).fetchone()
    dataset_path = storage_root / Path(dataset["stored_path"])
    dataset_path.write_bytes(dataset_path.read_bytes() + b"\n")

    with pytest.raises(RunBlockedError, match="versión aprobada") as captured:
        service.execute(strategy_id)
    assert captured.value.code == "DATASET_HASH_MISMATCH"
    run = connection.execute("SELECT * FROM runs").fetchone()
    assert run["status"] == "BLOCKED"
    assert run["selected_model"] is None
    assert run["artifact_path"] is None
    assert run["error_message"] == "DATASET_HASH_MISMATCH"
    assert run["finished_at"] is not None
    connection.close()


def test_completed_without_recommendation_persists_validation_and_skips_artifact(
    tmp_path: Path,
) -> None:
    guardrails = {
        "mode": "quantitative",
        "rules": [
            {
                "metric": "precision",
                "operator": ">=",
                "threshold": 1.0,
                "rationale": "Umbral sintético del fixture",
            },
            {
                "metric": "recall",
                "operator": ">=",
                "threshold": 1.0,
                "rationale": "Umbral sintético del fixture",
            },
        ],
    }
    connection, repository, service, strategy_id, _ = prepared_strategy(
        tmp_path,
        predictive=False,
        guardrails=guardrails,
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    result = service.execute(strategy_id)
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (result.run_id,)).fetchone()
    metrics = json.loads(row["candidate_metrics_json"])

    assert result.status == "COMPLETED_WITHOUT_RECOMMENDATION"
    assert result.selected_model is None
    assert result.artifact_path is None
    assert row["status"] == "COMPLETED_WITHOUT_RECOMMENDATION"
    assert row["selected_model"] is None
    assert row["artifact_path"] is None
    assert metrics["selection"]["outcome"] == "WITHOUT_RECOMMENDATION"
    assert metrics["holdout"] is None
    assert metrics["holdout_evaluations"] == 0
    assert set(metrics["validation"]) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    }
    connection.close()


def _approve(repository: WorkflowRepository, strategy_id: str) -> None:
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )


def _delete_accepted_file(connection: sqlite3.Connection, storage_root: Path) -> None:
    stored_path = connection.execute("SELECT stored_path FROM datasets").fetchone()[0]
    (storage_root / Path(stored_path)).unlink()


def test_stale_running_run_is_recovered_and_training_unblocked(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    strategy = connection.execute(
        "SELECT strategy_json, fingerprint FROM strategy_versions WHERE id = ?",
        (strategy_id,),
    ).fetchone()
    proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
    stale_id = repository.start_run(
        strategy_version_id=strategy_id,
        seed=proposal.seed,
        split_info={},
        candidate_metrics={},
        manifest=service._manifest(
            proposal,
            strategy_fingerprint=strategy["fingerprint"],
            versions={"strategy": proposal.schema_version},
            warnings=[],
            execution_status="RUNNING",
        ),
    )
    # Un run reciente puede seguir vivo: no se recupera.
    assert repository.recover_stale_runs() == []
    connection.execute(
        "UPDATE runs SET started_at = datetime('now', '-2 hours') WHERE id = ?",
        (stale_id,),
    )

    result = service.execute(strategy_id)

    stale = connection.execute("SELECT * FROM runs WHERE id = ?", (stale_id,)).fetchone()
    assert stale["status"] == "FAILED"
    assert stale["error_message"] == "STALE_RUN_RECOVERED"
    assert json.loads(stale["manifest_json"])["execution_status"] == "FAILED"
    recovered_events = connection.execute(
        "SELECT COUNT(*) FROM audit_events WHERE entity_id = ? AND event_type = ?",
        (stale_id, "RUN_RECOVERED_AS_FAILED"),
    ).fetchone()[0]
    assert recovered_events == 1
    assert result.status == "COMPLETED"
    connection.close()


def test_close_failure_keeps_original_error_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    _delete_accepted_file(connection, storage_root)

    def broken_finish(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repository, "finish_run", broken_finish)
    with pytest.raises(RunBlockedError) as captured:
        service.execute(strategy_id)

    assert captured.value.code == "ACCEPTED_DATASET_MISSING"
    assert any("database is locked" in note for note in captured.value.__notes__)
    connection.close()


def test_blocked_run_manifest_reports_final_status(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    _delete_accepted_file(connection, storage_root)

    with pytest.raises(RunBlockedError):
        service.execute(strategy_id)

    run = connection.execute("SELECT status, manifest_json FROM runs").fetchone()
    assert run["status"] == "BLOCKED"
    assert json.loads(run["manifest_json"])["execution_status"] == "BLOCKED"
    connection.close()
