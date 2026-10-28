"""GA-07d-2 (ADR-016): varias clases y cantidades se entrenan con aprobación y se guardan."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_engine import DataIngestionService
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.errors import ApprovalRequiredError
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.ml_engine.scoring import ScoringService
from core.reporting.service import ReportService


def _csv(task_type: str) -> bytes:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=240)
    target = (
        np.select([signal < -0.4, signal < 0.4], ["bajo", "medio"], "alto")
        if task_type == "multiclass_classification"
        else np.round(10 + 4 * signal + rng.normal(scale=0.3, size=240), 3)
    )
    frame = pd.DataFrame(
        {
            "senal": np.round(signal, 4),
            "grupo": rng.choice(["a", "b", "c"], 240),
            "objetivo": target,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


@pytest.mark.parametrize(
    ("task_type", "metric", "selected_options"),
    [
        ("multiclass_classification", "macro_f1", {"LogisticRegression", "RandomForestClassifier"}),
        ("regression", "mae", {"Ridge", "RandomForestRegressor"}),
    ],
)
def test_new_types_train_after_approval_and_persist_their_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_type, metric, selected_options
) -> None:
    monkeypatch.setattr(
        "core.domain.strategy_service.FLOW_READY_TASK_KINDS",
        {"binary_classification", task_type},
    )
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage = tmp_path / "private"
    project_id = repository.create_project("Tipos de modelo")
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
    decisions = [
        {"column": name, "use": "include", "available_at_prediction": True, "reason": "fixture"}
        for name in ("senal", "grupo")
    ]
    context = BusinessContextService(repository, storage).create_version(
        project_id=project_id,
        dataset_id=dataset.metadata.dataset_id,
        context={
            "objective": "Validar otro tipo de modelo con datos sintéticos",
            "target_column": "objetivo",
            "task_type": task_type,
            "prediction_moment": "Antes de conocer el objetivo sintético",
            "false_positive_consequence": "Error sintético por un lado",
            "false_negative_consequence": "Error sintético por el otro",
            "primary_metric": metric,
            "metric_rationale": "Decisión del fixture automatizado",
            "decision_actor_label": "pytest-fixture",
            "feature_decisions": decisions,
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategies = StrategyService(repository)
    template = strategies.build_template(
        project_id=project_id, context_version_id=context.context_version_id
    )
    strategy = strategies.create_version(
        project_id=project_id, proposal=StrategyProposalV1.model_validate(template.model_dump())
    )
    service = MLRunService(repository, storage)

    with pytest.raises(ApprovalRequiredError):
        service.execute(strategy.strategy_version_id)  # sin aprobación humana no entrena
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    result = service.execute(strategy.strategy_version_id)

    assert result.status == "COMPLETED" and result.selected_model in selected_options
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (result.run_id,)).fetchone()
    metrics = json.loads(row["candidate_metrics_json"])
    assert metrics["schema_version"] == "task-evaluation.v1"
    assert metrics["task_kind"] == task_type and metrics["holdout_evaluations"] == 1
    assert row["artifact_path"] is not None

    report = ReportService(repository).build(result.run_id)  # verifica cada cifra
    assert report.metrics["task_kind"] == task_type
    assert report.metrics["selected_model"] == result.selected_model

    # Uso: sólo con el reporte aprobado; con el resultado real hay seguimiento.
    scoring = ScoringService(repository, storage)
    report_id = ReportService(repository).create_template_report(result.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="pytest")
    used = scoring.predict(
        result.run_id, filename="nuevos.csv", content=_csv(task_type), actor_label="pytest"
    )
    assert used.rows_scored == 240 and used.task_kind == task_type
    assert used.follow_up is not None and metric in used.follow_up
    assert not used.degraded  # mismos datos: no debe parecer degradado
    # V2-15: seguridad por caso en varias categorías; rango probable en cantidades.
    first = used.cases[0]
    if task_type == "regression":
        assert first.low is not None and first.low < first.prediction < first.high
    else:
        assert first.confidence is not None and 0 <= first.confidence <= 1
    connection.close()
