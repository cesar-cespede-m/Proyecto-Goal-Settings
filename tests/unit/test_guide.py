"""V2-01: el núcleo de la Guía explica el siguiente paso desde estado persistido."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.data_engine import DataIngestionService
from core.data_engine.models import DatasetProfileV1
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.assistant import ContextDraftV1, DraftField
from core.domain.guide import GuideStepV1, project_guide
from tests.support.domains import DOMAINS

PROFILE = {
    "schema_version": "dataset-profile.v1",
    "data_classification": "synthetic",
    "row_count": 120,
    "column_count": 5,
    "duplicate_row_count": 0,
    "columns": [
        {
            "name": "resultado",
            "inferred_type": "boolean",
            "missing_count": 0,
            "missing_rate": 0.0,
            "unique_count": 2,
            "constant": False,
            "high_cardinality": False,
            "possible_identifier": False,
            "semantic_type": "category",
        },
        {
            "name": "variable_util",
            "inferred_type": "numeric",
            "missing_count": 0,
            "missing_rate": 0.0,
            "unique_count": 80,
            "constant": False,
            "high_cardinality": False,
            "possible_identifier": False,
            "semantic_type": "number",
        },
        {
            "name": "columna_constante",
            "inferred_type": "categorical",
            "missing_count": 0,
            "missing_rate": 0.0,
            "unique_count": 1,
            "constant": True,
            "high_cardinality": False,
            "possible_identifier": False,
            "semantic_type": "category",
        },
        {
            "name": "identificador",
            "inferred_type": "string",
            "missing_count": 0,
            "missing_rate": 0.0,
            "unique_count": 120,
            "constant": False,
            "high_cardinality": True,
            "possible_identifier": True,
            "semantic_type": "identifier",
        },
        {
            "name": "dato_sensible",
            "inferred_type": "string",
            "missing_count": 45,
            "missing_rate": 0.375,
            "unique_count": 60,
            "constant": False,
            "high_cardinality": False,
            "possible_identifier": False,
            "semantic_type": "personal",
        },
    ],
    "numeric_columns": ["variable_util"],
    "categorical_columns": [
        "resultado",
        "columna_constante",
        "identificador",
        "dato_sensible",
    ],
    "high_cardinality_columns": ["identificador"],
    "constant_columns": ["columna_constante"],
    "warnings": [
        "CONSTANT_COLUMNS:columna_constante",
        "POSSIBLE_IDENTIFIER_COLUMNS:identificador",
        "POSSIBLE_PERSONAL_DATA_COLUMNS:dato_sensible",
    ],
}


def _repository(tmp_path: Path) -> tuple[WorkflowRepository, str]:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    return repository, repository.create_project("Proyecto de prueba")


def _add_dataset(repository: WorkflowRepository, project_id: str) -> str:
    return repository.add_dataset(
        project_id=project_id,
        sha256="a" * 64,
        original_filename_display="tabla.csv",
        source="Fixture sintético pytest",
        license_name=None,
        anonymization_declared=False,
        admission_status="ACCEPTED",
        rejection_reason=None,
        n_rows=120,
        n_cols=5,
        byte_count=100,
        profile=PROFILE,
    )


def _add_context(repository: WorkflowRepository, project_id: str, dataset_id: str) -> str:
    return repository.add_context(
        project_id=project_id,
        dataset_id=dataset_id,
        version_no=None,
        payload={
            "schema_version": "business-context.v1",
            "target_column": "resultado",
            "positive_class": "sí",
            "prediction_moment": "Antes de conocer el resultado",
            "excluded_columns": ["columna_constante", "identificador", "dato_sensible"],
            "primary_metric": "recall",
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
            "context": {
                "objective": "Anticipar un resultado con datos sintéticos",
                "target_column": "resultado",
            },
        },
    )


def _add_strategy(repository: WorkflowRepository, context_id: str, dataset_id: str) -> str:
    return repository.add_strategy(
        context_version_id=context_id,
        dataset_id=dataset_id,
        version_no=None,
        payload={
            "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "preprocessing": {},
            "candidates": ["DummyClassifier", "LogisticRegression"],
            "seed": 7,
            "caps": {"max_training_seconds": 60},
        },
        schema_version="strategy.v1",
        source="TEMPLATE",
    )


def _add_run(repository: WorkflowRepository, strategy_id: str, *, status: str) -> str:
    strategy = repository.get_strategy(strategy_id)
    context = repository.get_context(strategy["context_version_id"])
    dataset = repository.get_dataset(strategy["dataset_id"])
    metrics = {"holdout": {"precision": 0.8, "recall": 0.8}}
    manifest = {
        "schema_version": "run-manifest.v1",
        "dataset_sha256": dataset["sha256"],
        "context_sha256": context["context_sha256"],
        "strategy_fingerprint": strategy["fingerprint"],
        "versions": {},
        "hyperparameters": {},
        "features": ["variable_util"],
        "warnings": [],
    }
    run_id = repository.start_run(
        strategy_version_id=strategy_id,
        seed=7,
        split_info={},
        candidate_metrics=metrics,
        manifest=manifest,
    )
    repository.finish_run(
        run_id,
        status=status,
        candidate_metrics=metrics,
        selected_model="LogisticRegression" if status == "COMPLETED" else None,
        artifact_path="models/model.joblib" if status == "COMPLETED" else None,
        manifest=manifest,
        error_message="El proceso no terminó" if status == "FAILED" else None,
    )
    return run_id


def _add_report(repository: WorkflowRepository, run_id: str) -> str:
    return repository.create_report(
        run_id=run_id,
        payload={
            "metrics": {"recall": 0.8},
            "insights": [
                {
                    "kind": "FACT",
                    "claim": "El resultado fue calculado por la prueba.",
                    "evidence_refs": ["metrics.recall"],
                    "limitations": [],
                    "source": "TEMPLATE",
                }
            ],
            "limitations": [],
        },
    )


def _state(tmp_path: Path, stage: str):
    repository, project_id = _repository(tmp_path)
    if stage == "empty":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    dataset_id = _add_dataset(repository, project_id)
    if stage == "file":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)
    if stage == "draft":
        draft = ContextDraftV1.start(
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_sha256="a" * 64,
            profile=DatasetProfileV1.model_validate(PROFILE),
        )
        draft.set_by_person(
            DraftField.OBJECTIVE,
            "Anticipar un resultado con datos sintéticos",
            actor_label="pytest",
        )
        repository.create_draft(
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_sha256="a" * 64,
            revision=draft.revision,
            draft_json=draft.model_dump_json(),
        )
        expected_missing = len(draft.missing_confirmations())
        guide = project_guide(repository, project_id, ai_enabled=False)
        assert f"faltan {expected_missing} respuestas" in guide.steps[1].summary
        return repository, project_id, guide

    context_id = _add_context(repository, project_id, dataset_id)
    if stage == "decisions":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    strategy_id = _add_strategy(repository, context_id, dataset_id)
    if stage == "plan":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="pytest"
    )
    if stage == "approved":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    run_status = "COMPLETED_WITHOUT_RECOMMENDATION" if stage == "no_recommendation" else "COMPLETED"
    run_id = _add_run(repository, strategy_id, status=run_status)
    if stage == "no_recommendation":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    report_id = _add_report(repository, run_id)
    if stage == "recommendation":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="pytest")
    if stage == "report_approved":
        return repository, project_id, project_guide(repository, project_id, ai_enabled=False)

    repository.record_scoring(
        run_id=run_id,
        actor_label="pytest",
        detail={"rows_scored": 20, "alerts": 4, "drift": None, "follow_up": None},
    )
    return repository, project_id, project_guide(repository, project_id, ai_enabled=False)


@pytest.mark.parametrize(
    ("stage", "action", "statuses"),
    [
        ("empty", "Subir tus datos", ["current", "locked", "locked", "locked", "locked", "locked"]),
        (
            "file",
            "Definir qué quieres anticipar",
            ["done", "current", "locked", "locked", "locked", "locked"],
        ),
        (
            "draft",
            "Definir qué quieres anticipar",
            ["done", "current", "locked", "locked", "locked", "locked"],
        ),
        (
            "decisions",
            "Preparar el plan de entrenamiento",
            ["done", "done", "current", "locked", "locked", "locked"],
        ),
        (
            "plan",
            "Revisar y aprobar el plan",
            ["done", "done", "current", "locked", "locked", "locked"],
        ),
        ("approved", "Entrenar", ["done", "done", "done", "current", "locked", "locked"]),
        (
            "no_recommendation",
            "Revisar qué datos sumar",
            ["done", "done", "done", "current", "locked", "locked"],
        ),
        (
            "recommendation",
            "Revisar y aprobar el reporte",
            ["done", "done", "done", "done", "current", "locked"],
        ),
        (
            "report_approved",
            "Usar el modelo con casos nuevos",
            ["done", "done", "done", "done", "done", "current"],
        ),
        (
            "uses",
            "Decidir con el modelo y medir si ayudó",
            ["done", "done", "done", "done", "done", "current"],
        ),
    ],
)
def test_each_journey_state_has_one_clear_next_action(
    tmp_path: Path, stage: str, action: str, statuses: list[str]
) -> None:
    repository, project_id, guide = _state(tmp_path, stage)

    assert guide.project_id == project_id
    assert guide.next_action.label == action
    assert [step.status for step in guide.steps] == statuses
    assert sum(step.status == "current" for step in guide.steps) == 1
    assert guide.next_action.step_id == guide.current_step
    assert all(step.locked_reason for step in guide.steps if step.status == "locked")
    assert all(step.locked_reason is None for step in guide.steps if step.status != "locked")
    repository.connection.close()


def test_profile_alerts_are_plain_language_and_limited(tmp_path: Path) -> None:
    repository, _, guide = _state(tmp_path, "file")

    assert len(guide.alerts) == 3
    assert all(alert.step_id == "decisions" for alert in guide.alerts)
    assert all(
        alert.action_label == "Se propondrá excluirla en tus decisiones" for alert in guide.alerts
    )
    raw_code = re.compile(r"\b[A-Z]{2,}(?:_[A-Z]{2,})+\b")
    assert not any(raw_code.search(alert.text) for alert in guide.alerts)
    assert not any(raw_code.search(alert.action_label) for alert in guide.alerts)
    repository.connection.close()


def test_profile_alerts_disappear_once_decisions_are_saved(tmp_path: Path) -> None:
    """Recorrido 22/10: seguía avisando «se propondrá excluirla» de columnas ya excluidas."""
    repository, _, guide = _state(tmp_path, "decisions")

    assert guide.alerts == []
    repository.connection.close()


def test_recorded_health_can_send_the_cycle_back_to_retraining(tmp_path: Path) -> None:
    repository, project_id, _ = _state(tmp_path, "uses")
    context = repository.get_latest_context(project_id)
    strategy = repository.get_latest_strategy(context["id"])
    run = repository.get_latest_run(strategy["id"])
    repository.record_scoring(
        run_id=run["id"],
        actor_label="pytest",
        detail={
            "rows_scored": 100,
            "alerts": 50,
            "drift": None,
            "follow_up": {
                "rows_with_outcome": 100,
                "precision": 0.3,
                "recall": 0.3,
            },
        },
    )

    guide = project_guide(repository, project_id, ai_enabled=False)

    assert guide.current_step == "train"
    assert guide.next_action.label == "Reentrenar"
    assert [step.status for step in guide.steps] == [
        "done",
        "done",
        "done",
        "current",
        "done",
        "done",
    ]
    repository.connection.close()


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED"])
def test_failed_or_blocked_training_always_has_a_review_exit(tmp_path: Path, status: str) -> None:
    repository, project_id = _repository(tmp_path)
    dataset_id = _add_dataset(repository, project_id)
    context_id = _add_context(repository, project_id, dataset_id)
    strategy_id = _add_strategy(repository, context_id, dataset_id)
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="pytest"
    )
    _add_run(repository, strategy_id, status=status)

    guide = project_guide(repository, project_id, ai_enabled=False)

    assert guide.current_step == "train"
    assert guide.next_action.label.startswith("Revisar")
    assert guide.next_action.reason
    repository.connection.close()


def test_ai_mode_is_honest_and_contracts_are_strict(tmp_path: Path) -> None:
    repository, project_id = _repository(tmp_path)

    assert project_guide(repository, project_id, ai_enabled=True).ai_mode == "llm"
    assert project_guide(repository, project_id, ai_enabled=False).ai_mode == "rules"
    with pytest.raises(ValidationError):
        GuideStepV1(
            id="data",
            title="Datos",
            status="current",
            summary="Sin archivos.",
            locked_reason=None,
            campo_desconocido=True,
        )
    repository.connection.close()


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_guide_runs_for_every_test_business(tmp_path: Path, domain) -> None:
    repository, project_id = _repository(tmp_path)
    ingestion = DataIngestionService(repository, tmp_path / "private")
    for path in domain.files:
        ingestion.ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": path.name,
                "source_name": "Fixture sintético pytest",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=path.read_bytes(),
        )

    guide = project_guide(repository, project_id, ai_enabled=False)

    assert guide.current_step == "decisions"
    assert guide.next_action.label == "Definir qué quieres anticipar"
    assert guide.steps[0].summary.startswith(f"{len(domain.files)} archivo")
    repository.connection.close()


def test_retrained_model_asks_to_compare_before_use(tmp_path) -> None:
    """V2-26: en un proyecto de reentrenamiento, comparar antes de reemplazar el modelo."""

    from core.domain.guide import project_guide
    from core.domain.retraining_service import RetrainingService
    from core.domain.strategy import StrategyProposalV1
    from core.domain.strategy_service import StrategyService
    from core.ml_engine import MLRunService
    from core.reporting.service import ReportService
    from tests.unit.test_recipe_retraining import _files
    from tests.unit.test_recipe_retraining import trained as trained_fixture

    generator = trained_fixture.__wrapped__(tmp_path)
    repository, storage, run_id = next(generator)
    started = RetrainingService(repository, storage).start_from_recipe(
        source_run_id=run_id, files=_files(2), actor_label="Analista", data_authorized=True
    )
    strategies = StrategyService(repository)
    template = strategies.build_template(
        project_id=started.project_id, context_version_id=started.context_version_id
    )
    strategy = strategies.create_version(
        project_id=started.project_id,
        proposal=StrategyProposalV1.model_validate(template.model_dump()),
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id, decision="APPROVED", actor_label="A"
    )
    new_run = MLRunService(repository, storage).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(new_run.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="A")

    guide = project_guide(repository, started.project_id, ai_enabled=False)
    assert guide.next_action.label == "Comparar con el modelo anterior"
    repository.record_model_comparison(run_id=new_run.run_id, actor_label="A", detail={})
    guide = project_guide(repository, started.project_id, ai_enabled=False)
    assert guide.next_action.label == "Usar el modelo con casos nuevos"
