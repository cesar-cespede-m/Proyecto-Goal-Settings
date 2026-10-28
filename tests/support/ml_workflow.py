"""Flujo sintético compartido: dataset aceptado, contexto y estrategia sin aprobar."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_engine import DataIngestionService
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService


def csv_bytes(row_count: int = 200, *, predictive: bool = True) -> bytes:
    rng = np.random.default_rng(20261016)
    target = np.asarray([0, 1] * (row_count // 2))
    signal = target if predictive else rng.integers(0, 8, row_count)
    category = (
        np.where(target == 1, "positive", "negative")
        if predictive
        else rng.choice(["a", "b", "c"], row_count)
    )
    frame = pd.DataFrame(
        {
            "fixture_id": [f"fixture-{index}" for index in range(row_count)],
            "signal": signal,
            "noise": rng.integers(0, 8, row_count),
            "category": category,
            "target": target,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def prepared_strategy(
    tmp_path: Path,
    *,
    predictive: bool = True,
    guardrails: dict[str, object] | None = None,
    storage_subdir: str = "private-runtime",
    content: bytes | None = None,
) -> tuple[sqlite3.Connection, WorkflowRepository, MLRunService, str, Path]:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage_root = tmp_path / storage_subdir
    project_id = repository.create_project("Proyecto fixture ML-02")
    admission = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "synthetic-fixture.csv",
            "source_name": "Fixture sintético pytest",
            "license": "TEST-ONLY",
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": False,
        },
        content=csv_bytes(predictive=predictive) if content is None else content,
    )
    context = BusinessContextService(repository, storage_root).create_version(
        project_id=project_id,
        dataset_id=admission.metadata.dataset_id,
        context={
            "objective": "Validar el flujo técnico con un fixture sintético",
            "target_column": "target",
            "positive_class": 1,
            "prediction_moment": "Antes del evento sintético del fixture",
            "false_positive_consequence": "Error sintético de prueba",
            "false_negative_consequence": "Error sintético de prueba",
            "primary_metric": "f1",
            "metric_rationale": "Decisión exclusiva del fixture automatizado",
            "decision_actor_label": "pytest-fixture",
            "feature_decisions": [
                {
                    "column": "fixture_id",
                    "use": "exclude",
                    "available_at_prediction": True,
                    "reason": "Identificador sintético",
                },
                {
                    "column": "signal",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Señal sintética disponible",
                },
                {
                    "column": "noise",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Ruido sintético disponible",
                },
                {
                    "column": "category",
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "Categoría sintética disponible",
                },
            ],
            "guardrails": guardrails or {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategy_service = StrategyService(repository)
    template = strategy_service.build_template(
        project_id=project_id,
        context_version_id=context.context_version_id,
    ).model_dump(mode="json")
    template["model_params"]["random_forest"]["n_estimators"] = 10
    strategy = strategy_service.create_version(
        project_id=project_id,
        proposal=StrategyProposalV1.model_validate(template),
    )
    return (
        connection,
        repository,
        MLRunService(repository, storage_root),
        strategy.strategy_version_id,
        storage_root,
    )
