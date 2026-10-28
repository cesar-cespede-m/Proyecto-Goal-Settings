"""MV-06: operaciones ML idempotentes conservan gates y resultados tras un fallo."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.db import get_connection
from core.db.repositories import WorkflowRepository
from core.domain.errors import ApprovalRequiredError, InvalidStateTransitionError
from core.domain.strategy import StrategyProposalV1
from core.ml_engine import MLRunService, RunBlockedError
from tests.support.ml_workflow import prepared_strategy


class SimulatedProcessExit(BaseException):
    """Interrumpe la llamada sin ejecutar el cierre ordinario de excepciones."""


def _approve(repository: WorkflowRepository, strategy_id: str) -> None:
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="pytest"
    )


def _start_inputs(repository, service, strategy_id):
    strategy = repository.get_strategy(strategy_id)
    proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
    return {
        "strategy_version_id": strategy_id,
        "seed": proposal.seed,
        "split_info": {},
        "candidate_metrics": {},
        "manifest": service._manifest(
            proposal,
            strategy_fingerprint=strategy["fingerprint"],
            versions={"strategy": proposal.schema_version},
            warnings=[*proposal.warnings, "RUN_STARTED_PENDING_RESULTS"],
            execution_status="RUNNING",
        ),
    }


def _forbid_load(*args, **kwargs):
    raise AssertionError("El replay no puede volver a cargar ni entrenar el dataset")


def test_operation_without_approval_creates_neither_run_nor_mapping(tmp_path: Path):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    with pytest.raises(ApprovalRequiredError, match="STRATEGY_NOT_APPROVED"):
        service.execute(strategy_id, operation_id="train-once")
    for table in ("runs", "training_operations"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    connection.close()


def test_run_and_operation_mapping_roll_back_together(tmp_path: Path):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    connection.execute(
        "CREATE TRIGGER fixture_reject_mapping BEFORE INSERT ON training_operations "
        "BEGIN SELECT RAISE(ABORT, 'synthetic mapping failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic mapping failure"):
        repository.start_run_once(
            operation_id="atomic", **_start_inputs(repository, service, strategy_id)
        )
    for table in ("runs", "training_operations"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'RUN_STARTED'"
        ).fetchone()[0]
        == 0
    )
    connection.close()


def test_duplicate_start_returns_same_run_and_rejects_changed_input(tmp_path: Path):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    inputs = _start_inputs(repository, service, strategy_id)
    first = repository.start_run_once(operation_id="one-operation", **inputs)
    assert first[1] is True
    assert repository.start_run_once(operation_id="one-operation", **inputs) == (first[0], False)
    with pytest.raises(InvalidStateTransitionError, match="TRAINING_OPERATION_CONFLICT"):
        repository.start_run_once(operation_id="one-operation", **{**inputs, "seed": 123})
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM training_operations").fetchone()[0] == 1
    connection.close()


def test_two_connections_claim_the_same_operation_atomically(tmp_path: Path):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    inputs = _start_inputs(repository, service, strategy_id)

    def start_from_new_connection():
        other = get_connection(tmp_path / "goal_settings.sqlite3")
        try:
            return WorkflowRepository(other).start_run_once(operation_id="concurrent", **inputs)
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start_from_new_connection(), range(2)))
    assert len({result[0] for result in results}) == 1
    assert sorted(result[1] for result in results) == [False, True]
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM training_operations").fetchone()[0] == 1
    connection.close()


def test_operation_id_cannot_cross_projects_or_strategies(tmp_path: Path):
    first_connection, first_repo, first_service, first_strategy, _ = prepared_strategy(tmp_path)
    _approve(first_repo, first_strategy)
    first = first_repo.start_run_once(
        operation_id="scoped", **_start_inputs(first_repo, first_service, first_strategy)
    )
    second_connection, second_repo, second_service, second_strategy, _ = prepared_strategy(
        tmp_path, storage_subdir="other-private-runtime"
    )
    _approve(second_repo, second_strategy)
    with pytest.raises(InvalidStateTransitionError, match="TRAINING_OPERATION_CONFLICT"):
        second_repo.start_run_once(
            operation_id="scoped", **_start_inputs(second_repo, second_service, second_strategy)
        )
    row = second_connection.execute("SELECT * FROM training_operations").fetchone()
    assert row["strategy_version_id"] == first_strategy
    assert row["run_id"] == first[0]
    assert second_connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    second_connection.close()
    first_connection.close()


def test_completed_replay_survives_reopened_connection_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    original = service.execute(strategy_id, operation_id="persistent-replay")
    connection.close()
    reopened = get_connection(tmp_path / "goal_settings.sqlite3")
    replay_service = MLRunService(WorkflowRepository(reopened), storage_root)
    monkeypatch.setattr(replay_service, "_load_frame", _forbid_load)
    assert replay_service.execute(strategy_id, operation_id="persistent-replay") == original
    assert reopened.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert (
        reopened.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'RUN_FINISHED'"
        ).fetchone()[0]
        == 1
    )
    reopened.close()


def test_fault_after_final_commit_preserves_artifact_and_can_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    real_finish = repository.finish_run

    def commit_then_lose_reply(*args, **kwargs):
        real_finish(*args, **kwargs)
        raise OSError("synthetic response lost after commit")

    monkeypatch.setattr(repository, "finish_run", commit_then_lose_reply)
    with pytest.raises(OSError, match="response lost after commit"):
        service.execute(strategy_id, operation_id="lost-reply")
    row = connection.execute("SELECT * FROM runs").fetchone()
    assert row["status"] == "COMPLETED"
    assert (storage_root / row["artifact_path"]).is_file()
    monkeypatch.setattr(service, "_load_frame", _forbid_load)
    result = service.execute(strategy_id, operation_id="lost-reply")
    assert result.run_id == row["id"]
    assert result.status == "COMPLETED"
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'RUN_FINISHED'"
        ).fetchone()[0]
        == 1
    )
    connection.close()


def test_active_or_recovered_interrupted_operation_never_fits_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)

    def process_exit(*args, **kwargs):
        raise SimulatedProcessExit()

    monkeypatch.setattr(service, "_load_frame", process_exit)
    with pytest.raises(SimulatedProcessExit):
        service.execute(strategy_id, operation_id="interrupted")
    monkeypatch.setattr(service, "_load_frame", _forbid_load)
    with pytest.raises(RunBlockedError) as active:
        service.execute(strategy_id, operation_id="interrupted")
    assert active.value.code == "TRAINING_OPERATION_IN_PROGRESS"
    connection.execute("UPDATE runs SET started_at = datetime('now', '-2 hours')")
    with pytest.raises(RunBlockedError) as recovered:
        service.execute(strategy_id, operation_id="interrupted")
    assert recovered.value.code == "TRAINING_OPERATION_TERMINATED"
    row = connection.execute("SELECT * FROM runs").fetchone()
    assert row["status"] == "FAILED"
    assert row["error_message"] == "STALE_RUN_RECOVERED"
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()


def test_blocked_operation_cannot_be_retried_by_same_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    stored = connection.execute("SELECT stored_path FROM datasets").fetchone()[0]
    (storage_root / stored).unlink()
    with pytest.raises(RunBlockedError) as initial:
        service.execute(strategy_id, operation_id="blocked")
    assert initial.value.code == "ACCEPTED_DATASET_MISSING"
    monkeypatch.setattr(service, "_load_frame", _forbid_load)
    with pytest.raises(RunBlockedError) as replay:
        service.execute(strategy_id, operation_id="blocked")
    assert replay.value.code == "TRAINING_OPERATION_TERMINATED"
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()


def test_completed_replay_still_requires_current_human_approval(tmp_path: Path):
    connection, repository, service, strategy_id, _ = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    service.execute(strategy_id, operation_id="gate-replay")
    repository.decide_training(
        strategy_version_id=strategy_id, decision="REJECTED", actor_label="pytest"
    )
    with pytest.raises(ApprovalRequiredError, match="STRATEGY_NOT_APPROVED"):
        service.execute(strategy_id, operation_id="gate-replay")
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize("tamper", ["delete", "replace"])
def test_replay_rejects_missing_or_changed_model_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
):
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    _approve(repository, strategy_id)
    first = service.execute(strategy_id, operation_id="artifact-integrity")
    artifact = storage_root / first.artifact_path
    if tamper == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(b"synthetic changed artifact; never deserialized")
    monkeypatch.setattr(service, "_load_frame", _forbid_load)
    with pytest.raises(RunBlockedError) as error:
        service.execute(strategy_id, operation_id="artifact-integrity")
    assert error.value.code == (
        "MODEL_ARTIFACT_MISSING" if tamper == "delete" else "MODEL_ARTIFACT_CHANGED"
    )
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()
