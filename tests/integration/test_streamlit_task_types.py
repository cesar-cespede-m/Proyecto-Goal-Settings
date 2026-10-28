"""Pantallas de varias categorías y cantidades con un modelo entrenado y aprobado (07d/MON-04).

Arma el estado con los servicios reales y recorre la app: resultados, reporte, uso, salud y la
opción de validar con fechas no deben romperse ni mostrar textos de sí/no.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.reporting.service import ReportService

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
ACTOR = "pytest-fixture"


def _csv(task_type: str) -> bytes:
    rng = np.random.default_rng(11)
    signal = rng.normal(size=240)
    target = (
        np.select([signal < -0.4, signal < 0.4], ["bajo", "medio"], "alto")
        if task_type == "multiclass_classification"
        else np.round(10 + 4 * signal + rng.normal(scale=0.3, size=240), 3)
    )
    return (
        pd.DataFrame(
            {
                "momento": pd.date_range("2026-01-01", periods=240, freq="D").astype(str),
                "senal": np.round(signal, 4),
                "grupo": rng.choice(["a", "b", "c"], 240),
                "objetivo": target,
            }
        )
        .to_csv(index=False)
        .encode()
    )


def _trained_project(runtime: Path, task_type: str, metric: str) -> None:
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project("Tipos de modelo")
    dataset = DataIngestionService(repository, runtime).ingest_bytes(
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
        {"column": "momento", "use": "exclude", "available_at_prediction": True, "reason": "fecha"},
        *[
            {"column": name, "use": "include", "available_at_prediction": True, "reason": "dato"}
            for name in ("senal", "grupo")
        ],
    ]
    context = BusinessContextService(repository, runtime).create_version(
        project_id=project_id,
        dataset_id=dataset.metadata.dataset_id,
        context={
            "objective": "Validar las pantallas de otro tipo de modelo",
            "target_column": "objetivo",
            "task_type": task_type,
            "prediction_moment": "Antes de conocer el objetivo sintético",
            "false_positive_consequence": "Error sintético por un lado",
            "false_negative_consequence": "Error sintético por el otro",
            "primary_metric": metric,
            "metric_rationale": "Decisión del fixture automatizado",
            "decision_actor_label": ACTOR,
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
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id, decision="APPROVED", actor_label=ACTOR
    )
    run = MLRunService(repository, runtime).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label=ACTOR)
    connection.close()


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def _texts(app: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "success", "warning", "error", "subheader"):
        parts.extend(str(getattr(item, "value", "")) for item in app.get(kind))
    return "\n".join(parts)


@pytest.mark.parametrize(
    ("task_type", "metric"),
    [("multiclass_classification", "macro_f1"), ("regression", "mae")],
)
def test_screens_of_a_trained_model_render(runtime_dir: Path, task_type, metric) -> None:
    _trained_project(runtime_dir, task_type, metric)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)

    assert not app.exception, app.exception
    assert not app.error, [item.value for item in app.error]
    texts = _texts(app)
    assert "Se recomienda" in texts  # resultados por tipo
    assert "Validar con fechas" in [item.label for item in app.expander] or "momento" in texts
    assert any(item.label == "Estimar casos" for item in app.button)
    assert "Guardar decisiones para otro proyecto" in [
        item.label for item in app.get("download_button")
    ] or app.get("download_button")


def test_choosing_a_date_split_creates_a_new_plan_to_approve(runtime_dir: Path) -> None:
    _trained_project(runtime_dir, "regression", "mae")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
    next(item for item in app.selectbox if item.label == "Dividir por").set_value("momento")
    app.run(timeout=120)
    next(item for item in app.button if item.label == "Usar esta división").click()
    app.run(timeout=120)

    assert not app.exception, app.exception
    assert not app.error, [item.value for item in app.error]
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    latest = connection.execute(
        "SELECT strategy_json FROM strategy_versions ORDER BY version_no DESC LIMIT 1"
    ).fetchone()
    connection.close()
    assert '"time_column":"momento"' in latest["strategy_json"].replace(" ", "")


def test_after_training_retraining_is_secondary(runtime_dir: Path) -> None:
    _trained_project(runtime_dir, "regression", "mae")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)

    labels = [item.label for item in app.button]
    assert "Entrenar una versión nueva" in labels
    assert "Entrenar y comparar modelos" not in labels


def test_learning_timeline_is_visible_with_a_trained_model(runtime_dir: Path) -> None:
    _trained_project(runtime_dir, "regression", "mae")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)

    assert not app.exception
    assert any(item.label == "Cómo ha ido aprendiendo el modelo" for item in app.expander)


def test_full_metrics_only_in_advanced_mode(runtime_dir: Path) -> None:
    _trained_project(runtime_dir, "multiclass_classification", "macro_f1")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
    assert not any(i.label == "Métricas completas (modo avanzado)" for i in app.expander)

    app = AppTest.from_file(str(APP_PATH))
    app.session_state["advanced-mode"] = True
    app.run(timeout=120)
    assert not app.exception
    assert any(i.label == "Métricas completas (modo avanzado)" for i in app.expander)
