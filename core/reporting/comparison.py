"""Comparación de experimentos de un proyecto (EXT-B, P1).

Muestra qué cambia en el modelo y sus resultados cuando cambia una decisión de negocio
(medida principal, evento, momento, variables o condición mínima). Sólo lee cifras ya
calculadas por el ML; no recalcula ni reentrena.
"""

import json
from typing import Any

from pydantic import BaseModel

from core.db.repositories import WorkflowRepository
from core.domain.context import BusinessContextV1
from core.domain.gates import is_strategy_current
from core.ml_engine.models import EvaluationSummaryV1

_COMPARED_FIELDS = {
    "objective": "Decisión que se quiere apoyar",
    "target_column": "Resultado que se anticipa",
    "positive_class": "Evento que se detecta",
    "prediction_moment": "Momento de la decisión",
    "primary_metric": "Medida principal",
    "false_positive_consequence": "Consecuencia de una falsa alarma",
    "false_negative_consequence": "Consecuencia de un caso omitido",
}


class ExperimentRow(BaseModel):
    run_id: str
    started_at: str
    context_version_no: int
    strategy_version_no: int
    is_current: bool
    report_status: str | None
    primary_metric: str
    positive_class: str | int | float | bool | None
    features_used: list[str]
    selected_model: str | None
    outcome: str
    threshold: float | None
    holdout_precision: float | None
    holdout_recall: float | None
    holdout_f1: float | None
    holdout_average_precision: float | None
    holdout_prevalence: float | None
    alerts_in_holdout: int | None
    baseline_validation: float


class ContextChange(BaseModel):
    field: str
    label: str
    before: Any
    after: Any


def _context(row: Any) -> BusinessContextV1:
    return BusinessContextV1.model_validate(json.loads(row["context_json"])["context"])


def list_experiments(repository: WorkflowRepository, project_id: str) -> list[ExperimentRow]:
    rows = []
    for row in repository.list_project_runs(project_id):
        context = _context(row)
        stored = json.loads(row["candidate_metrics_json"])
        if stored.get("schema_version") == "task-evaluation.v1":
            rows.append(_task_row(repository, row, context, stored))
            continue
        evaluation = EvaluationSummaryV1.model_validate(stored)
        holdout = evaluation.holdout
        matrix = holdout.confusion_matrix if holdout else None
        rows.append(
            ExperimentRow(
                run_id=row["run_id"],
                started_at=row["started_at"],
                context_version_no=row["context_version_no"],
                strategy_version_no=row["strategy_version_no"],
                is_current=is_strategy_current(repository.connection, row["strategy_version_id"]),
                report_status=row["report_status"],
                primary_metric=str(context.primary_metric),
                positive_class=context.positive_class,
                features_used=context.allowed_features,
                selected_model=row["selected_model"],
                outcome=evaluation.selection.outcome,
                threshold=evaluation.selection.selected_threshold,
                holdout_precision=None if holdout is None else holdout.precision,
                holdout_recall=None if holdout is None else holdout.recall,
                holdout_f1=None if holdout is None else holdout.f1,
                holdout_average_precision=None if holdout is None else holdout.average_precision,
                holdout_prevalence=(
                    None if holdout is None else holdout.positive_support / holdout.support
                ),
                alerts_in_holdout=None if matrix is None else matrix[0][1] + matrix[1][1],
                baseline_validation=evaluation.selection.baseline_value,
            )
        )
    return rows


def _task_row(
    repository: WorkflowRepository, row: Any, context: BusinessContextV1, stored: dict[str, Any]
) -> ExperimentRow:
    """Varias categorías o cantidades (ADR-016): sin umbral ni matriz de alertas."""

    metric = stored["primary_metric"]
    return ExperimentRow(
        run_id=row["run_id"],
        started_at=row["started_at"],
        context_version_no=row["context_version_no"],
        strategy_version_no=row["strategy_version_no"],
        is_current=is_strategy_current(repository.connection, row["strategy_version_id"]),
        report_status=row["report_status"],
        primary_metric=metric,
        positive_class=None,
        features_used=context.allowed_features,
        selected_model=row["selected_model"],
        outcome="RECOMMENDED" if stored.get("selected_model") else "WITHOUT_RECOMMENDATION",
        threshold=None,
        holdout_precision=None,
        holdout_recall=None,
        holdout_f1=None,
        holdout_average_precision=None,
        holdout_prevalence=None,
        alerts_in_holdout=None,
        baseline_validation=stored["validation"][stored["baseline"]][metric],
    )


def context_changes(
    repository: WorkflowRepository, before_run_id: str, after_run_id: str
) -> list[ContextChange]:
    """Decisiones de negocio que cambiaron entre dos ejecuciones del mismo proyecto."""

    project_id = repository.get_run_project_id(before_run_id)
    runs = {row["run_id"]: row for row in repository.list_project_runs(project_id)}
    before, after = runs.get(before_run_id), runs.get(after_run_id)
    if before is None or after is None:
        raise ValueError("Ambas ejecuciones deben pertenecer al mismo proyecto")
    old, new = _context(before), _context(after)
    changes = [
        ContextChange(
            field=field, label=label, before=getattr(old, field), after=getattr(new, field)
        )
        for field, label in _COMPARED_FIELDS.items()
        if getattr(old, field) != getattr(new, field)
    ]
    old_features, new_features = set(old.allowed_features), set(new.allowed_features)
    if old_features != new_features:
        changes.append(
            ContextChange(
                field="features",
                label="Variables usadas",
                before=sorted(old_features - new_features),
                after=sorted(new_features - old_features),
            )
        )
    if old.guardrails != new.guardrails:
        changes.append(
            ContextChange(
                field="guardrails",
                label="Condición mínima",
                before=old.guardrails.model_dump(mode="json"),
                after=new.guardrails.model_dump(mode="json"),
            )
        )
    return [
        change.model_copy(
            update={
                "before": getattr(change.before, "value", change.before),
                "after": getattr(change.after, "value", change.after),
            }
        )
        for change in changes
    ]
