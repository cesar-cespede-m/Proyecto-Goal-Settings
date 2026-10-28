"""QA-01: segundo dominio sintético, doble HITL, reinicio y recuperación."""

import json
from pathlib import Path

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.operations import create_runtime_backup, restore_runtime_backup
from core.reporting import ReportService
from scripts.generate_case_b import build_case_b

ROOT = Path(__file__).parents[2]


def test_case_b_uses_same_core_and_survives_restart_and_restore(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    connection = get_connection(runtime / "goal_settings.sqlite3", use_wal=True)
    migrate(connection)
    repository = WorkflowRepository(connection)
    decisions = json.loads(
        (ROOT / "data/metadata/ecommerce_churn_context.recommended.json").read_text(
            encoding="utf-8"
        )
    )
    decisions["context"]["decision_actor_label"] = "pytest-reviewer"
    project_id = repository.create_project("Caso B sintético - pytest")
    csv_content = build_case_b().to_csv(index=False).encode("utf-8")
    admission = DataIngestionService(repository, runtime).ingest_bytes(
        project_id=project_id,
        request=decisions["dataset"],
        content=csv_content,
    )
    context = BusinessContextService(repository, runtime).create_version(
        project_id=project_id,
        dataset_id=admission.metadata.dataset_id,
        context=decisions["context"],
    )
    strategy_service = StrategyService(repository)
    proposal = strategy_service.build_template(
        project_id=project_id,
        context_version_id=context.context_version_id,
    ).model_dump(mode="json")
    proposal["model_params"]["random_forest"]["n_estimators"] = 10
    strategy = strategy_service.create_version(
        project_id=project_id,
        proposal=StrategyProposalV1.model_validate(proposal),
    )

    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        comment="Fixture sintético: no representa una aprobación real",
    )
    run = MLRunService(repository, runtime).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        comment="Fixture sintético: no representa una aprobación real",
    )
    repository.publish_report(report_id)
    dataset_relative = Path(repository.get_dataset(admission.metadata.dataset_id)["stored_path"])
    artifact_relative = Path(repository.get_run(run.run_id)["artifact_path"])
    connection.close()

    reopened = get_connection(runtime / "goal_settings.sqlite3", use_wal=True)
    migrate(reopened)
    assert reopened.execute("SELECT status FROM reports WHERE id = ?", (report_id,)).fetchone()[
        0
    ] == ("PUBLISHED")
    assert reopened.execute("SELECT status FROM runs WHERE id = ?", (run.run_id,)).fetchone()[
        0
    ] == ("COMPLETED")
    assert (runtime / dataset_relative).is_file()
    assert (runtime / artifact_relative).is_file()
    reopened.close()

    backup = create_runtime_backup(runtime, tmp_path / "backup-case-b")
    restored = restore_runtime_backup(backup, tmp_path / "restored-runtime")
    restored_connection = get_connection(restored / "goal_settings.sqlite3")
    assert (
        restored_connection.execute(
            "SELECT status FROM reports WHERE id = ?", (report_id,)
        ).fetchone()[0]
        == "PUBLISHED"
    )
    assert (
        restored_connection.execute(
            "SELECT COUNT(*) FROM approval_events WHERE actor_label = 'pytest-reviewer'"
        ).fetchone()[0]
        == 2
    )
    restored_connection.close()
    assert (restored / dataset_relative).read_bytes() == csv_content
    assert (restored / artifact_relative).is_file()
