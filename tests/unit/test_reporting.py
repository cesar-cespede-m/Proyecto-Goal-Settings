"""REPORT-01: reporte por plantilla, verificador de cifras y aprobación #2."""

import json
from pathlib import Path

import pytest

from core.domain.errors import (
    ApprovalRequiredError,
    InvalidStateTransitionError,
    ReportValidationError,
)
from core.domain.models import ReportPayload
from core.domain.strategy_service import StrategyService
from core.ml_engine import RunBlockedError
from core.ml_engine.models import FeatureImportance, TaskEvaluationV1
from core.reporting import ReportService, verify_report
from core.reporting.template import build_task_report, build_template_report
from tests.support.ml_workflow import prepared_strategy


def _completed_run(tmp_path: Path, **kwargs: object) -> tuple:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(
        tmp_path, **kwargs
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    result = service.execute(strategy_id)
    return connection, repository, result, storage_root


def _payload(claim: str, refs: list[str]) -> ReportPayload:
    return ReportPayload(
        metrics={
            "holdout": {"recall": 0.75, "confusion_matrix": [[10, 2], [3, 9]]},
            "features": ["sensor_2"],
            "confidence": 0.95,
        },
        insights=[
            {
                "kind": "FACT",
                "claim": claim,
                "evidence_refs": refs,
                "limitations": [],
                "source": "LLM",
            }
        ],
        limitations=[],
    )


def test_template_report_is_verified_versioned_and_requires_approval_to_publish(
    tmp_path: Path,
) -> None:
    connection, repository, result, _ = _completed_run(tmp_path)
    reports = ReportService(repository)

    report_id = reports.create_template_report(result.run_id)
    report = repository.get_report(report_id)
    payload = ReportPayload.model_validate(json.loads(report["payload_json"]))

    assert report["status"] == "DRAFT"
    assert payload.generator == "TEMPLATE"
    assert verify_report(payload) == []
    assert {insight.source for insight in payload.insights} == {"TEMPLATE"}
    assert payload.metrics["selection"]["selected_model"] == result.selected_model
    assert payload.metrics["holdout"] == result.evaluation.holdout.model_dump(mode="json")
    assert payload.metrics["event_cases_label"] == "casos en que «target» fue «1»"
    assert any("evaluación offline" in item for item in payload.limitations)
    assert any("inventados" in item for item in payload.limitations)
    # UX-24: lenguaje común en la vista; los nombres técnicos quedan en metrics.
    claims = " ".join(insight.claim for insight in payload.insights)
    for jargon in ("Classifier", "recall", "precision", "baseline", "umbral", "validación"):
        assert jargon not in claims
    assert "casos reales del evento" not in claims
    assert "casos en que «target» fue «1»" in claims
    assert "puntaje de riesgo del caso" in claims and "(de 0 a 1)" in claims
    recommendations = [item for item in payload.insights if item.kind == "RECOMMENDATION"]
    assert len(recommendations) == 1
    assert "decisión automática" in recommendations[0].claim
    assert "en paralelo" in recommendations[0].claim
    # Sin cambios de contenido no se crea otra versión ni se invalida la vigente.
    assert reports.create_template_report(result.run_id) == report_id

    with pytest.raises(ApprovalRequiredError):
        repository.publish_report(report_id)
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        comment="Revisión sintética",
    )
    repository.publish_report(report_id)

    assert repository.get_report(report_id)["status"] == "PUBLISHED"
    assert repository.get_latest_publication_decision(report_id)["decision"] == "APPROVED"
    connection.close()


def test_run_without_recommendation_produces_honest_report(tmp_path: Path) -> None:
    guardrails = {
        "mode": "quantitative",
        "rules": [
            {
                "metric": "precision",
                "operator": ">=",
                "threshold": 1.0,
                "rationale": "Umbral sintético del fixture",
            }
        ],
    }
    connection, repository, result, _ = _completed_run(
        tmp_path, predictive=False, guardrails=guardrails
    )
    assert result.status == "COMPLETED_WITHOUT_RECOMMENDATION"

    report_id = ReportService(repository).create_template_report(result.run_id)
    payload = ReportPayload.model_validate(
        json.loads(repository.get_report(report_id)["payload_json"])
    )

    assert payload.metrics["holdout"] is None
    assert "no se recomienda un modelo" in payload.insights[0].claim
    assert verify_report(payload) == []
    connection.close()


@pytest.mark.parametrize("means", [(0.0, 0.0), (0.0, -0.02), (0.0, 0.1)])
def test_template_importance_claim_matches_measured_effect(tmp_path: Path, means) -> None:
    connection, _, result, _ = _completed_run(tmp_path)
    evaluation = result.evaluation.model_copy(
        update={
            "permutation_importance": [
                FeatureImportance(
                    feature=f"feature_{index}", importance_mean=mean, importance_std=0.01
                )
                for index, mean in enumerate(means)
            ]
        }
    )
    payload = build_template_report(
        run_id=result.run_id,
        run_status=result.status,
        evaluation=evaluation,
        dataset_classification="synthetic",
    )
    interpretation = next(item for item in payload.insights if item.kind == "INTERPRETATION")
    if max(means) > 0:
        assert "«feature_1»" in interpretation.claim
        assert "«feature_0»" not in interpretation.claim
        assert "metrics.permutation_importance.1.importance_mean" in interpretation.evidence_refs
    else:
        assert "no encontró una caída promedio" in interpretation.claim
        assert "«feature_" not in interpretation.claim
    assert verify_report(payload) == []
    connection.close()


def test_blocked_run_cannot_be_reported(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, storage_root = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-fixture",
    )
    stored_path = connection.execute("SELECT stored_path FROM datasets").fetchone()[0]
    (storage_root / Path(stored_path)).unlink()
    with pytest.raises(RunBlockedError):
        service.execute(strategy_id)
    run_id = connection.execute("SELECT id FROM runs").fetchone()[0]

    with pytest.raises(ReportValidationError) as captured:
        ReportService(repository).create_template_report(run_id)
    assert captured.value.code == "RUN_NOT_REPORTABLE"
    connection.close()


@pytest.mark.parametrize(
    ("claim", "refs"),
    [
        ("Recall de 0.750 en prueba.", ["metrics.holdout.recall"]),
        ("Recall de 75 % en prueba.", ["metrics.holdout.recall"]),
        ("Detectó 9 de 10 casos.", ["metrics.holdout.confusion_matrix"]),
        ("La variable sensor_2 fue la más influyente.", ["metrics.features.0"]),
        ("Intervalo al 95 %.", ["metrics.confidence"]),
    ],
)
def test_verifier_accepts_numbers_backed_by_evidence(claim: str, refs: list[str]) -> None:
    assert verify_report(_payload(claim, refs)) == []


@pytest.mark.parametrize(
    ("claim", "refs", "expected"),
    [
        ("Recall de 0.99 en prueba.", ["metrics.holdout.recall"], "UNSUPPORTED_NUMBER:0:0.99"),
        ("Recall de 0.750.", ["metrics.holdout.precision"], "UNRESOLVED_EVIDENCE:0"),
        ("Ahorra 30 % de costos.", ["metrics.holdout.recall"], "UNSUPPORTED_NUMBER:0:30 %"),
        ("Recall 0.750.", ["holdout.recall"], "UNRESOLVED_EVIDENCE:0"),
        # 9 está en la matriz, 12 no: basta una cifra sin evidencia para rechazar.
        ("Detectó 9 de 12.", ["metrics.holdout.confusion_matrix"], "UNSUPPORTED_NUMBER:0:12"),
    ],
)
def test_verifier_rejects_invented_or_unbacked_numbers(
    claim: str, refs: list[str], expected: str
) -> None:
    errors = verify_report(_payload(claim, refs))
    assert any(error.startswith(expected) for error in errors), errors


def _supersede_strategy(repository, strategy_id: str) -> None:
    strategy = repository.get_strategy(strategy_id)
    edited = json.loads(strategy["strategy_json"])
    edited["seed"] = edited["seed"] + 1
    project_id = repository.get_context(strategy["context_version_id"])["project_id"]
    StrategyService(repository).create_version(project_id=project_id, proposal=edited)


def test_report_of_superseded_strategy_cannot_be_approved_or_published(
    tmp_path: Path,
) -> None:
    connection, repository, result, _ = _completed_run(tmp_path)
    reports = ReportService(repository)
    approved_id = reports.create_template_report(result.run_id)
    repository.decide_publication(
        report_id=approved_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    strategy_id = repository.get_run(result.run_id)["strategy_version_id"]
    _supersede_strategy(repository, strategy_id)

    # Aprobado antes del cambio: el dominio y la base de datos bloquean la publicación.
    with pytest.raises(ApprovalRequiredError):
        repository.publish_report(approved_id)
    with pytest.raises(Exception, match="current report"):
        connection.execute(
            "UPDATE reports SET status = 'PUBLISHED' WHERE id = ?",
            (approved_id,),
        )
    assert repository.get_report(approved_id)["status"] == "APPROVED"

    # Una nueva aprobación de un reporte superado se rechaza; rechazarlo sigue permitido.
    with pytest.raises(InvalidStateTransitionError, match="REPORT_NOT_CURRENT"):
        repository.decide_publication(
            report_id=approved_id,
            decision="APPROVED",
            actor_label="pytest-reviewer",
        )
    repository.decide_publication(
        report_id=approved_id,
        decision="REJECTED",
        actor_label="pytest-reviewer",
        comment="Versión superada",
    )


def test_limitations_never_show_raw_codes() -> None:
    import re

    from core.reporting.template import limitation_text

    for warning in (
        "DATA_PROFILE:HIGH_CARDINALITY_COLUMNS:momento",
        "DATA_PROFILE:POSSIBLE_IDENTIFIER_COLUMNS:momento,codigo",
        "DATA_PROFILE:DUPLICATE_ROWS:12",
        "SOMETHING_UNKNOWN",
    ):
        text = limitation_text(warning)
        assert not re.search(r"[A-Z]{3,}_[A-Z_]+", text), text


def test_event_specific_limitations_name_the_chosen_result() -> None:
    from core.reporting.template import event_cases_label, limitation_text

    text = limitation_text(
        "LOW_POSITIVE_SUPPORT_HOLDOUT",
        event_cases="casos en que «resultado» fue «sí»",
    )
    assert "casos en que «resultado» fue «sí»" in text
    assert "casos del evento" not in text
    assert event_cases_label("sí", "entregado_tarde") == "casos en que «entregado_tarde» fue «sí»"


def test_task_report_with_recommended_model_has_one_safe_use_recommendation() -> None:
    evaluation = TaskEvaluationV1(
        task_kind="regression",
        primary_metric="mae",
        higher_is_better=False,
        baseline="DummyRegressor",
        validation={"DummyRegressor": {"mae": 4.0}, "Ridge": {"mae": 2.0}},
        selected_model="Ridge",
        test={"mae": 2.5},
        split_method="random",
        split_rows={"train": 60, "validation": 20, "test": 20},
        holdout_evaluations=1,
        warnings=[],
    )

    report = build_task_report(
        run_id="run-task",
        run_status="COMPLETED",
        evaluation=evaluation,
        dataset_classification="synthetic",
    )

    recommendations = [item for item in report.insights if item.kind == "RECOMMENDATION"]
    assert len(recommendations) == 1
    assert "decisión automática" in recommendations[0].claim
    assert "en paralelo" in recommendations[0].claim
    assert verify_report(report) == []
