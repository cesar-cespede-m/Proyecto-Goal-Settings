"""QA-04: recorrido real de las herramientas de terminal del ciclo 2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.tables import AnalysisTableService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.reporting.service import ReportService

ROOT = Path(__file__).parents[2]
ACTOR = "qa04-integration"
ROW_VALUE = "917263540.125"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def _event(
    entity: str,
    entity_id: str,
    attributes: dict[str, str | int | float],
    *,
    occurred_at: datetime = NOW,
) -> dict[str, object]:
    return {
        "schema_version": "observation-event.v1",
        "source": "qa04_synthetic_source",
        "entity_type": entity,
        "entity_id": entity_id,
        "occurred_at": occurred_at.isoformat(),
        "attributes": attributes,
    }


def _events(*, revision: int) -> list[dict[str, object]]:
    occurred_at = NOW + timedelta(days=revision)
    events = []
    for index in range(240):
        events.append(
            _event(
                "entity_cases",
                f"case-{index:03d}",
                {
                    "join_key": index % 4,
                    "signal": round(((index * 17 + revision * 3) % 101) / 10, 2),
                    "label_basis": float((index * 13 + revision * 7) % 97),
                    "opaque_measure": float(ROW_VALUE) if index == 0 else float(index % 5),
                },
                occurred_at=occurred_at,
            )
        )
    for index, band in enumerate(("a", "b", "a", "c")):
        events.append(
            _event(
                "entity_groups",
                str(index),
                {"band": band, "revision": revision},
                occurred_at=occurred_at,
            )
        )
    return events


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def _run(
    script: str,
    *arguments: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} terminó con {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _run_json(
    script: str,
    *arguments: str,
    env: dict[str, str],
) -> dict[str, object]:
    result = _run(script, *arguments, env=env)
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _prepare_approved_model(runtime: Path, project_id: str) -> str:
    connection = get_connection(runtime / "goal_settings.sqlite3")
    repository = WorkflowRepository(connection)
    created = repository.list_datasets(project_id, admission_status="ACCEPTED")
    source_ids = {
        str(row["source"]).removeprefix("observation_events:"): str(row["id"]) for row in created
    }
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": ["entity_cases", "entity_groups"],
            "steps": [
                {
                    "op": "join",
                    "source": "entity_groups",
                    "left_on": "join_key",
                    "right_on": "id",
                    "how": "left",
                    "right_prefix": "group_",
                },
                {
                    "op": "threshold_label",
                    "column": "label_basis",
                    "output_column": "target",
                    "percentile": 0.5,
                },
                {"op": "select", "columns": ["signal", "group_band", "target"]},
            ],
            "output": {"chart": "table"},
        }
    )
    table = AnalysisTableService(repository, runtime).save(
        project_id=project_id,
        plan=plan,
        source_ids=source_ids,
        display_name="QA-04 prepared table",
        actor_label=ACTOR,
    )
    repository.set_modeling_table(project_id, table.dataset_id, actor_label=ACTOR)
    context = BusinessContextService(repository, runtime).create_version(
        project_id=project_id,
        dataset_id=table.dataset_id,
        context={
            "objective": "Validate the generic cycle 2 terminal workflow",
            "target_column": "target",
            "positive_class": "sí",
            "prediction_moment": "Before the synthetic result is known",
            "false_positive_consequence": "Synthetic test consequence",
            "false_negative_consequence": "Synthetic test consequence",
            "primary_metric": "f1",
            "metric_rationale": "Deterministic integration-test decision",
            "decision_actor_label": ACTOR,
            "feature_decisions": [
                {
                    "column": column,
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "QA-04 synthetic fixture",
                }
                for column in ("signal", "group_band")
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategies = StrategyService(repository)
    template = strategies.build_template(
        project_id=project_id,
        context_version_id=context.context_version_id,
    )
    strategy = strategies.create_version(
        project_id=project_id,
        proposal=StrategyProposalV1.model_validate(template.model_dump()),
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label=ACTOR,
    )
    run = MLRunService(repository, runtime).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label=ACTOR,
    )
    repository.publish_report(report_id)
    connection.close()
    return run.run_id


def _copy_created_source(
    runtime: Path,
    created: list[dict[str, object]],
    entity: str,
    destination: Path,
) -> None:
    dataset_id = next(str(item["dataset_id"]) for item in created if item["entity"] == entity)
    connection = get_connection(runtime / "goal_settings.sqlite3")
    dataset = WorkflowRepository(connection).get_dataset(dataset_id)
    source = runtime / Path(str(dataset["stored_path"]))
    destination.write_bytes(source.read_bytes())
    connection.close()


def _install_git_guards(directory: Path, marker: Path) -> None:
    directory.mkdir()
    shell_body = f"#!/bin/sh\nprintf invoked > '{marker.as_posix()}'\n"
    cmd_body = f'@echo invoked>"{marker}"\n'
    for command in ("git", "gh"):
        executable = directory / command
        executable.write_text(shell_body, encoding="utf-8")
        executable.chmod(0o755)
        (directory / f"{command}.cmd").write_text(cmd_body, encoding="utf-8")


def test_cycle2_terminal_tools_end_to_end(tmp_path: Path) -> None:
    runtime = tmp_path
    missing_env = tmp_path / "intentionally-missing.env"
    env = os.environ.copy()
    env["GOAL_SETTINGS_RUNTIME_DIR"] = str(runtime)
    env["GOAL_SETTINGS_ENV_FILE"] = str(missing_env)
    assert not missing_env.exists()

    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    project_id = WorkflowRepository(connection).create_project("QA-04 terminal tour")
    connection.close()

    initial_jsonl = tmp_path / "initial-events.jsonl"
    _write_jsonl(initial_jsonl, _events(revision=0))
    pushed = _run_json(
        "push_events.py",
        str(initial_jsonl),
        "--project-id",
        project_id,
        "--source-label",
        "QA-04 synthetic JSONL",
        "--actor-label",
        ACTOR,
        env=env,
    )
    assert pushed == {"accepted": 244, "duplicates": 0, "rejected": 0}

    first_refresh = _run_json(
        "refresh_measurements.py",
        "--project",
        project_id,
        "--actor",
        ACTOR,
        "--classification",
        "synthetic",
        env=env,
    )
    assert first_refresh["schema_version"] == "measurement-report.v1"
    assert {item["entity"] for item in first_refresh["created_files"]} == {
        "entity_cases",
        "entity_groups",
    }
    assert first_refresh["measured_models"] == []
    assert first_refresh["warnings"] == []

    run_id = _prepare_approved_model(runtime, project_id)
    prediction_file = tmp_path / "new-cases.csv"
    prediction_file.write_text(
        "signal,group_band\n1.2,a\n4.8,b\n7.1,c\n",
        encoding="utf-8",
    )
    prediction = _run_json(
        "goal_settings_query.py",
        "predict",
        "--run",
        run_id,
        "--file",
        str(prediction_file),
        "--actor",
        "qa04-external-system",
        env=env,
    )
    assert prediction["run_id"] == run_id
    assert prediction["task_type"] == "binary_classification"
    assert len(prediction["cases"]) == 3
    assert all({"row", "id", "score", "alert"} == set(case) for case in prediction["cases"])

    later_jsonl = tmp_path / "later-events.jsonl"
    _write_jsonl(later_jsonl, _events(revision=1))
    later_push = _run_json(
        "push_events.py",
        str(later_jsonl),
        "--project-id",
        project_id,
        "--source-label",
        "QA-04 synthetic JSONL",
        "--actor-label",
        ACTOR,
        env=env,
    )
    assert later_push == {"accepted": 244, "duplicates": 0, "rejected": 0}

    second_refresh = _run_json(
        "refresh_measurements.py",
        "--project",
        project_id,
        "--actor",
        ACTOR,
        "--classification",
        "synthetic",
        env=env,
    )
    assert len(second_refresh["created_files"]) == 2
    assert second_refresh["warnings"] == []
    assert len(second_refresh["measured_models"]) == 1
    assert second_refresh["measured_models"][0]["run_id"] == run_id

    recipe_cases = tmp_path / "recipe-cases.csv"
    recipe_groups = tmp_path / "recipe-groups.csv"
    _copy_created_source(runtime, second_refresh["created_files"], "entity_cases", recipe_cases)
    _copy_created_source(runtime, second_refresh["created_files"], "entity_groups", recipe_groups)
    retraining = _run_json(
        "run_recipe.py",
        "--run",
        run_id,
        "--file",
        f"entity_cases={recipe_cases}",
        "--file",
        f"entity_groups={recipe_groups}",
        "--actor",
        ACTOR,
        "--authorized",
        env=env,
    )
    assert set(retraining) == {"project_id", "dataset_id", "context_version_id", "next"}
    assert retraining["project_id"] != project_id

    marker = tmp_path / "git-or-gh-invoked"
    fake_bin = tmp_path / "guarded-bin"
    _install_git_guards(fake_bin, marker)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    drafts = tmp_path / "drafts"
    proposal = _run_json(
        "propose_changes.py",
        "--project",
        project_id,
        "--out",
        str(drafts),
        "--json",
        env=env,
    )
    assert proposal["evidence"] >= 1
    assert proposal["proposals"] >= 1
    assert len(proposal["drafts"]) == proposal["proposals"]
    markdown = list(drafts.glob("*.md"))
    assert markdown
    assert not marker.exists()

    connection = get_connection(runtime / "goal_settings.sqlite3")
    audit_rows = connection.execute(
        "SELECT event_type, detail_json FROM audit_events ORDER BY rowid"
    ).fetchall()
    audit_types = [str(row["event_type"]) for row in audit_rows]
    assert audit_types.count("OBSERVATION_EVENTS_INGESTED") == 2
    assert audit_types.count("OBSERVATION_EVENTS_MATERIALIZED") == 4
    assert audit_types.count("MEASUREMENT_REFRESHED") == 2
    assert audit_types.count("MODEL_USED_FOR_SCORING") == 1
    assert "REPORT_PUBLISHED" in audit_types
    assert "RETRAINING_STARTED" in audit_types
    assert ROW_VALUE not in "\n".join(str(row["detail_json"]) for row in audit_rows)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM runs AS run "
            "JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id "
            "JOIN context_versions AS context ON context.id = strategy.context_version_id "
            "WHERE context.project_id = ?",
            (retraining["project_id"],),
        ).fetchone()[0]
        == 0
    )
    connection.close()
    assert ROW_VALUE not in "\n".join(path.read_text(encoding="utf-8") for path in markdown)
