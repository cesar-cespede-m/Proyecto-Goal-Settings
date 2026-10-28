"""MV-02: una nueva garantía temporal queda versionada y necesita otra aprobación."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.gates import has_training_approval
from core.domain.strategy_service import StrategyService

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def test_label_availability_edit_requires_fresh_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project = repository.create_project("Ventanas sintéticas")
    moments = pd.date_range("2026-01-01", periods=120, freq="D")
    frame = pd.DataFrame(
        {
            "decision_at": moments.astype(str),
            "label_at": (moments + pd.Timedelta(days=7)).astype(str),
            "signal": [0, 1] * 60,
            "target": [0, 1] * 60,
        }
    )
    admitted = DataIngestionService(repository, tmp_path).ingest_bytes(
        project_id=project,
        request={
            "original_filename_display": "temporal.csv",
            "source_name": "Fixture sintético pytest",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=frame.to_csv(index=False).encode(),
    )
    context = BusinessContextService(repository, tmp_path).create_version(
        project_id=project,
        dataset_id=admitted.metadata.dataset_id,
        context={
            "objective": "Validar la separación temporal de un fixture sintético",
            "target_column": "target",
            "positive_class": 1,
            "prediction_moment": "Antes del resultado sintético",
            "false_positive_consequence": "Costo sintético",
            "false_negative_consequence": "Incidente sintético",
            "primary_metric": "f1",
            "metric_rationale": "Prueba técnica del flujo",
            "decision_actor_label": "pytest-fixture",
            "feature_decisions": [
                {
                    "column": name,
                    "use": "include" if name == "signal" else "exclude",
                    "available_at_prediction": name != "label_at",
                    "reason": "Señal o metadato temporal del fixture",
                }
                for name in ("decision_at", "label_at", "signal")
            ],
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategy = StrategyService(repository).create_template_version(
        project_id=project, context_version_id=context.context_version_id
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(i for i in app.selectbox if i.label == "Dividir por").set_value("decision_at")
    app.run(timeout=60)
    next(i for i in app.selectbox if i.label == "Fecha en que se conoció el resultado").set_value(
        "label_at"
    )
    next(i for i in app.button if i.label == "Usar esta división").click().run(timeout=60)
    assert not app.exception
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    try:
        repository = WorkflowRepository(connection)
        latest = repository.get_latest_strategy(context.context_version_id)
        assert latest["id"] != strategy.strategy_version_id
        assert (
            json.loads(latest["strategy_json"])["split"]["label_available_at_column"] == "label_at"
        )
        assert not has_training_approval(connection, latest["id"])
    finally:
        connection.close()
