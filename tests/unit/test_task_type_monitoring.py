"""MON-04 (ADR-016): monitoreo y comparación para varias clases y cantidades."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.data_engine import DataIngestionService
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.ml_engine.execution import RunBlockedError
from core.ml_engine.scoring import ScoringService
from core.reporting.service import ReportService
from tests.unit.test_task_type_execution import _csv


@pytest.fixture(
    params=[
        ("multiclass_classification", "macro_f1", True),
        ("regression", "mae", False),
    ],
    ids=["varias-clases", "cantidades"],
)
def monitored_models(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    task_type, metric, higher_is_better = request.param
    monkeypatch.setattr(
        "core.domain.strategy_service.FLOW_READY_TASK_KINDS",
        {"binary_classification", "multiclass_classification", "regression"},
    )
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage = tmp_path / "private"
    project_id = repository.create_project("Monitoreo por tipo")
    dataset = DataIngestionService(repository, storage).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "tipos.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=_csv(task_type),
    )
    context = BusinessContextService(repository, storage).create_version(
        project_id=project_id,
        dataset_id=dataset.metadata.dataset_id,
        context={
            "objective": "Validar monitoreo de otro tipo de modelo con datos sintéticos",
            "target_column": "objetivo",
            "task_type": task_type,
            "prediction_moment": "Antes de conocer el objetivo sintético",
            "false_positive_consequence": "Error sintético por un lado",
            "false_negative_consequence": "Error sintético por el otro",
            "primary_metric": metric,
            "metric_rationale": "Decisión del fixture automatizado",
            "decision_actor_label": "pytest-fixture",
            "feature_decisions": [
                {
                    "column": name,
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "fixture",
                }
                for name in ("senal", "grupo")
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategies = StrategyService(repository)
    proposal = StrategyProposalV1.model_validate(
        strategies.build_template(
            project_id=project_id, context_version_id=context.context_version_id
        ).model_dump()
    )
    strategy = strategies.create_version(project_id=project_id, proposal=proposal)
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    runner = MLRunService(repository, storage)
    run_ids = [runner.execute(strategy.strategy_version_id).run_id for _ in range(2)]
    reports = ReportService(repository)
    for run_id in run_ids:
        report_id = reports.create_template_report(run_id)
        repository.decide_publication(
            report_id=report_id, decision="APPROVED", actor_label="pytest-fixture"
        )
    yield (
        repository,
        ScoringService(repository, storage),
        run_ids,
        task_type,
        metric,
        higher_is_better,
    )
    connection.close()


def _with_degraded_outcome(task_type: str) -> bytes:
    frame = pd.read_csv(io.BytesIO(_csv(task_type)))
    if task_type == "multiclass_classification":
        frame["objetivo"] = frame["objetivo"].map(
            {"bajo": "alto", "medio": "bajo", "alto": "medio"}
        )
    else:
        frame["objetivo"] = -frame["objetivo"]
    return frame.to_csv(index=False).encode("utf-8")


def test_health_without_uses_has_insufficient_data(monitored_models) -> None:
    _, scoring, run_ids, _, _, _ = monitored_models

    health = scoring.health(run_ids[0])

    assert health.status == "NO_USES"
    assert health.recommendation is None
    assert health.points == []


def test_repeated_degradation_proposes_retraining(monitored_models) -> None:
    _, scoring, run_ids, task_type, metric, _ = monitored_models
    degraded = _with_degraded_outcome(task_type)
    for content in (degraded, degraded, _csv(task_type)):
        scoring.predict(run_ids[0], filename="uso.csv", content=content, actor_label="pytest")

    health = scoring.health(run_ids[0])

    assert health.status == "REVIEW"
    assert health.recommendation == "PROPOSE_RETRAINING"
    assert health.reference_metric == metric
    assert [point.degraded for point in health.points] == [True, True, False]


def test_large_data_change_proposes_retraining(monitored_models) -> None:
    _, scoring, run_ids, task_type, _, _ = monitored_models
    frame = pd.read_csv(io.BytesIO(_csv(task_type)))
    frame["senal"] = frame["senal"] + 100
    frame["grupo"] = "grupo-nuevo"

    scoring.predict(
        run_ids[0],
        filename="cambio.csv",
        content=frame.to_csv(index=False).encode("utf-8"),
        actor_label="pytest",
    )
    health = scoring.health(run_ids[0])

    assert health.points[-1].drift_level == "REVIEW"
    assert health.recommendation == "PROPOSE_RETRAINING"


def test_compare_predictions_uses_one_audit_event_and_no_model_use(monitored_models) -> None:
    repository, scoring, run_ids, task_type, metric, higher_is_better = monitored_models

    comparison = scoring.compare_predictions(
        current_run_id=run_ids[0],
        candidate_run_id=run_ids[1],
        filename="comparacion.csv",
        content=_csv(task_type),
        actor_label="pytest",
    )

    assert comparison.task_kind == task_type
    assert comparison.current.metric == comparison.candidate.metric == metric
    assert comparison.higher_is_better is higher_is_better
    assert comparison.better in {"CURRENT", "CANDIDATE", "TIE"}
    assert repository.list_scorings(run_ids[0]) == []
    assert repository.list_scorings(run_ids[1]) == []
    events = repository.connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'MODEL_COMPARISON'"
    ).fetchall()
    assert len(events) == 1
    assert json.loads(events[0]["detail_json"])["better"] == comparison.better


def test_compare_predictions_requires_real_outcome(monitored_models) -> None:
    repository, scoring, run_ids, task_type, _, _ = monitored_models
    frame = pd.read_csv(io.BytesIO(_csv(task_type))).drop(columns=["objetivo"])

    with pytest.raises(RunBlockedError) as captured:
        scoring.compare_predictions(
            current_run_id=run_ids[0],
            candidate_run_id=run_ids[1],
            filename="sin-resultado.csv",
            content=frame.to_csv(index=False).encode("utf-8"),
            actor_label="pytest",
        )

    assert captured.value.code == "COMPARISON_NEEDS_OUTCOMES"
    assert repository.list_scorings(run_ids[0]) == []
    assert repository.list_scorings(run_ids[1]) == []


def test_compare_predictions_rejects_different_task_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scoring = ScoringService(object(), tmp_path)
    proposals = {
        "current": SimpleNamespace(task_type="multiclass_classification"),
        "candidate": SimpleNamespace(task_type="regression"),
    }
    monkeypatch.setattr(
        scoring,
        "approved_run",
        lambda run_id: ({}, {}, proposals[run_id]),
    )

    with pytest.raises(RunBlockedError) as captured:
        scoring.compare_predictions(
            current_run_id="current",
            candidate_run_id="candidate",
            filename="casos.csv",
            content=b"irrelevante",
            actor_label="pytest",
        )

    assert captured.value.code == "COMPARISON_TASK_KIND_MISMATCH"
