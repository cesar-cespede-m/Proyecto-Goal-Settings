"""Salud del modelo a lo largo de sus usos (MON-02, ADR-013).

Reúne los eventos `MODEL_USED_FOR_SCORING` de una ejecución y aplica reglas fijas y declaradas
para decidir si el modelo está Estable, hay que Vigilarlo o hay que Revisarlo. Nada se ejecuta
automáticamente: la recomendación sólo orienta a una persona.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["NO_USES", "STABLE", "WATCH", "REVIEW"]
Recommendation = Literal["KEEP_USING", "KEEP_WATCHING", "CHECK_DATA_SOURCE", "PROPOSE_RETRAINING"]

MIN_OUTCOMES = 30
PERFORMANCE_DROP_WATCH = 0.08
PERFORMANCE_DROP_REVIEW = 0.15
PERFORMANCE_DROPPED_REASON = "PERFORMANCE_DROPPED"
_ORDER = {"NO_USES": -1, "STABLE": 0, "WATCH": 1, "REVIEW": 2}


class HealthPoint(BaseModel):
    created_at: str
    task_kind: str | None = None
    rows_scored: int | None = None
    alerts: int | None = None
    drift_level: str | None = None
    max_psi: float | None = None
    changed_columns: list[str] = Field(default_factory=list)
    rows_with_outcome: int | None = None
    precision: float | None = None
    recall: float | None = None
    primary_metric: str | None = None
    metric_value: float | None = None
    training_value: float | None = None
    degraded: bool | None = None


class HealthReason(BaseModel):
    code: Literal[
        "DATA_CHANGED_A_LOT",
        "DATA_CHANGING",
        "PERFORMANCE_DROPPED",
        "PERFORMANCE_SLIPPING",
        "NOT_ENOUGH_OUTCOMES",
        "NO_OUTCOMES_YET",
    ]
    level: Literal["INFO", "WATCH", "REVIEW"]
    columns: list[str] = Field(default_factory=list)
    metric: str | None = None
    reference: float | None = None
    observed: float | None = None
    rows_with_outcome: int | None = None


class ModelHealthV1(BaseModel):
    schema_version: Literal["model-health.v1"] = "model-health.v1"
    status: HealthStatus
    recommendation: Recommendation | None
    reasons: list[HealthReason]
    points: list[HealthPoint]
    reference_precision: float | None = None
    reference_recall: float | None = None
    reference_metric: str | None = None
    reference_value: float | None = None


def _point(created_at: str, detail: dict[str, Any]) -> HealthPoint:
    drift = detail.get("drift") or {}
    follow_up = detail.get("follow_up") or {}
    primary_metric = detail.get("primary_metric")
    return HealthPoint(
        created_at=created_at,
        task_kind=detail.get("task_kind"),
        rows_scored=detail.get("rows_scored"),
        alerts=detail.get("alerts"),
        drift_level=drift.get("level"),
        max_psi=drift.get("max_psi"),
        changed_columns=[
            column["column"]
            for column in drift.get("columns", [])
            if column.get("level") in {"WATCH", "REVIEW"}
        ],
        rows_with_outcome=detail.get("rows_with_outcome") or follow_up.get("rows_with_outcome"),
        precision=follow_up.get("precision"),
        recall=follow_up.get("recall"),
        primary_metric=primary_metric,
        metric_value=follow_up.get(primary_metric) if primary_metric else None,
        training_value=detail.get("training_value"),
        degraded=detail.get("degraded") if follow_up else None,
    )


def _data_reasons(points: list[HealthPoint]) -> list[HealthReason]:
    measured = [point for point in points if point.drift_level in {"STABLE", "WATCH", "REVIEW"}]
    if not measured:
        return []
    latest = measured[-1]
    if latest.drift_level == "REVIEW":
        return [
            HealthReason(code="DATA_CHANGED_A_LOT", level="REVIEW", columns=latest.changed_columns)
        ]
    if latest.drift_level == "WATCH":
        return [HealthReason(code="DATA_CHANGING", level="WATCH", columns=latest.changed_columns)]
    return []


def _performance_reasons(
    points: list[HealthPoint], precision: float | None, recall: float | None
) -> list[HealthReason]:
    with_outcome = [point for point in points if point.rows_with_outcome]
    if not with_outcome:
        return [HealthReason(code="NO_OUTCOMES_YET", level="INFO")]
    latest = with_outcome[-1]
    if latest.rows_with_outcome < MIN_OUTCOMES:
        return [
            HealthReason(
                code="NOT_ENOUGH_OUTCOMES",
                level="INFO",
                rows_with_outcome=latest.rows_with_outcome,
            )
        ]
    reasons = []
    for metric, reference, observed in (
        ("recall", recall, latest.recall),
        ("precision", precision, latest.precision),
    ):
        if reference is None or observed is None:
            continue
        drop = reference - observed
        if drop >= PERFORMANCE_DROP_WATCH:
            reasons.append(
                HealthReason(
                    code=(
                        "PERFORMANCE_DROPPED"
                        if drop >= PERFORMANCE_DROP_REVIEW
                        else "PERFORMANCE_SLIPPING"
                    ),
                    level="REVIEW" if drop >= PERFORMANCE_DROP_REVIEW else "WATCH",
                    metric=metric,
                    reference=reference,
                    observed=observed,
                    rows_with_outcome=latest.rows_with_outcome,
                )
            )
    return reasons


def _task_performance_reasons(points: list[HealthPoint]) -> list[HealthReason]:
    """Varias clases/cantidades: exige degradación repetida, no un lote aislado."""

    with_outcome = [point for point in points if point.degraded is not None]
    if not with_outcome:
        return [HealthReason(code="NO_OUTCOMES_YET", level="INFO")]
    recent = with_outcome[-3:]
    if sum(point.degraded is True for point in recent) < 2:
        return []
    latest = recent[-1]
    return [
        HealthReason(
            code=PERFORMANCE_DROPPED_REASON,
            level="REVIEW",
            metric=latest.primary_metric,
            reference=latest.training_value,
            observed=latest.metric_value,
            rows_with_outcome=sum(point.rows_with_outcome or 0 for point in recent),
        )
    ]


def assess_model_health(
    events: list[tuple[str, dict[str, Any]]],
    *,
    reference_precision: float | None,
    reference_recall: float | None,
    reference_metric: str | None = None,
    reference_value: float | None = None,
) -> ModelHealthV1:
    """Evalúa los usos en orden cronológico contra la prueba final del modelo."""

    points = [_point(created_at, detail) for created_at, detail in events]
    if not points:
        return ModelHealthV1(
            status="NO_USES",
            recommendation=None,
            reasons=[],
            points=[],
            reference_precision=reference_precision,
            reference_recall=reference_recall,
            reference_metric=reference_metric,
            reference_value=reference_value,
        )
    task_monitoring = any(
        point.task_kind in {"multiclass_classification", "regression"} for point in points
    )
    data = _data_reasons(points)
    performance = (
        _task_performance_reasons(points)
        if task_monitoring
        else _performance_reasons(points, reference_precision, reference_recall)
    )
    reasons = data + performance
    levels = [reason.level for reason in reasons if reason.level != "INFO"]
    status: HealthStatus = max(levels, key=_ORDER.__getitem__, default="STABLE")
    if any(reason.code == PERFORMANCE_DROPPED_REASON for reason in reasons) or (
        task_monitoring and any(reason.code == "DATA_CHANGED_A_LOT" for reason in reasons)
    ):
        recommendation: Recommendation = "PROPOSE_RETRAINING"
    elif status == "REVIEW":
        recommendation = "CHECK_DATA_SOURCE"
    elif status == "WATCH":
        recommendation = "KEEP_WATCHING"
    else:
        recommendation = "KEEP_USING"
    return ModelHealthV1(
        status=status,
        recommendation=recommendation,
        reasons=reasons,
        points=points,
        reference_precision=reference_precision,
        reference_recall=reference_recall,
        reference_metric=reference_metric,
        reference_value=reference_value,
    )
