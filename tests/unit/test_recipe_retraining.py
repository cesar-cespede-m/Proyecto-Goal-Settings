"""GA-08: la receta completa rehace la tabla con archivos nuevos y reabre el flujo aprobado."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.tables import AnalysisTableService
from core.data_engine import DataIngestionService
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.retraining_service import RetrainingService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService, RunBlockedError
from core.reporting.service import ReportService

ACTOR = "pytest-fixture"


def _files(seed: int) -> dict[str, tuple[str, bytes]]:
    rng = np.random.default_rng(seed)
    rows = 300
    senal = rng.normal(size=rows)
    casos = pd.DataFrame(
        {
            "grupo_id": rng.integers(0, 4, rows),
            "senal": np.round(senal, 4),
            "valor": np.round(senal * 3 + rng.normal(scale=0.3, size=rows), 4),
        }
    )
    grupos = pd.DataFrame({"id": [0, 1, 2, 3], "nivel": ["a", "b", "a", "c"]})
    return {
        "casos": ("casos.csv", casos.to_csv(index=False).encode()),
        "grupos": ("grupos.csv", grupos.to_csv(index=False).encode()),
    }


PLAN = AnalysisPlanV1.model_validate(
    {
        "sources": ["casos", "grupos"],
        "steps": [
            {
                "op": "join",
                "source": "grupos",
                "left_on": "grupo_id",
                "right_on": "id",
                "how": "left",
            },
            {
                "op": "threshold_label",
                "column": "valor",
                "output_column": "marcado",
                "percentile": 0.5,
            },
            {"op": "select", "columns": ["senal", "nivel", "marcado"]},
        ],
        "output": {"chart": "table"},
    }
)


@pytest.fixture
def trained(tmp_path: Path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage = tmp_path / "private"
    project_id = repository.create_project("Receta")
    ingest = DataIngestionService(repository, storage)
    ids = {}
    for name, (filename, content) in _files(1).items():
        ids[name] = ingest.ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": filename,
                "source_name": "Fixture sintético pytest",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=content,
        ).metadata.dataset_id
    table = AnalysisTableService(repository, storage).save(
        project_id=project_id,
        plan=PLAN,
        source_ids=ids,
        display_name="Tabla preparada",
        actor_label=ACTOR,
    )
    repository.set_modeling_table(project_id, table.dataset_id, actor_label=ACTOR)
    context = BusinessContextService(repository, storage).create_version(
        project_id=project_id,
        dataset_id=table.dataset_id,
        context={
            "objective": "Validar el reentrenamiento por receta con datos sintéticos",
            "target_column": "marcado",
            "positive_class": "sí",
            "prediction_moment": "Antes de conocer el valor sintético",
            "false_positive_consequence": "Error sintético de prueba",
            "false_negative_consequence": "Error sintético de prueba",
            "primary_metric": "f1",
            "metric_rationale": "Decisión del fixture automatizado",
            "decision_actor_label": ACTOR,
            "feature_decisions": [
                {
                    "column": name,
                    "use": "include",
                    "available_at_prediction": True,
                    "reason": "fixture",
                }
                for name in ("senal", "nivel")
            ],
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
    run = MLRunService(repository, storage).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label=ACTOR)
    yield repository, storage, run.run_id
    connection.close()


def test_recipe_names_every_piece_of_the_model(trained) -> None:
    repository, storage, run_id = trained
    service = RetrainingService(repository, storage)
    recipe = service.recipe(run_id)

    assert recipe.analysis_plan is not None and len(recipe.sources) == 2
    assert recipe.task_kind == "binary_classification"
    assert service.recipe_sources(run_id) == ["casos", "grupos"]


def test_retraining_rebuilds_the_table_and_waits_for_approvals(trained) -> None:
    repository, storage, run_id = trained
    service = RetrainingService(repository, storage)

    with pytest.raises(RunBlockedError) as missing:
        service.start_from_recipe(
            source_run_id=run_id,
            files={"casos": _files(2)["casos"]},
            actor_label="Analista",
            data_authorized=True,
        )
    assert missing.value.code == "RECIPE_FILES_MISSING"

    started = service.start_from_recipe(
        source_run_id=run_id, files=_files(2), actor_label="Analista", data_authorized=True
    )
    table = repository.get_dataset(started.dataset_id)
    assert table["is_active"] and table["dataset_kind"] == "DERIVED"
    recipe = repository.get_analysis_table(started.dataset_id)
    original = repository.get_analysis_table(
        json.loads(
            repository.get_strategy(repository.get_run(run_id)["strategy_version_id"])[
                "strategy_json"
            ]
        )["dataset_id"]
    )
    assert recipe["plan_sha256"] == original["plan_sha256"]  # mismos pasos
    assert service.source_of(started.project_id)["recipe_plan_sha256"] == recipe["plan_sha256"]
    # El modelo nuevo no existe todavía: falta estrategia, aprobación y entrenamiento.
    assert (
        repository.connection.execute(
            "SELECT COUNT(*) FROM runs r JOIN strategy_versions s ON s.id = r.strategy_version_id "
            "JOIN context_versions c ON c.id = s.context_version_id WHERE c.project_id = ?",
            (started.project_id,),
        ).fetchone()[0]
        == 0
    )
