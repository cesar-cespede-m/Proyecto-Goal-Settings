"""V2-13: decidir con un modelo sí/no según la capacidad o el costo de cada error.

MV-03: las políticas se eligen sobre validation. Sus estimaciones son exploratorias,
no una prueba independiente ni evidencia causal. Los costos son relativos, no dinero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.ml_engine.errors import MLValidationError
from core.ml_engine.models import (
    EvaluationSummaryV1,
    PolicyOperatingPointV1,
    PolicySelectionArtifactV1,
    top_fraction_count,
)

NO_OPERATING_POINTS = "NO_OPERATING_POINTS"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionPlanV1(_Strict):
    basis: Literal["capacity", "cost"]
    flagged_share: float = Field(ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)  # sólo descriptivo en top_fraction
    selection_mode: Literal["legacy_threshold", "top_fraction"] = "legacy_threshold"
    selection_fraction: float | None = Field(default=None, ge=0, le=1)
    selection_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_partition: Literal["legacy", "validation"] = "legacy"
    evidence_status: Literal["legacy", "exploratory"] = "legacy"
    capacity_limit: int | None = Field(default=None, ge=0)
    expected_cases: int = Field(ge=0)  # casos del periodo
    to_attend: int = Field(ge=0)
    expected_true_alerts: int = Field(ge=0)
    expected_false_alarms: int = Field(ge=0)
    expected_missed: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)  # de los atendidos, cuántos tendrán el resultado
    recall: float = Field(ge=0, le=1)  # de los que tendrán el resultado, cuántos se atienden
    uncertainty: float = Field(ge=0)  # ± aproximado de la precisión por tamaño de la prueba
    explanation: str

    @model_validator(mode="after")
    def policy_has_evidence(self) -> DecisionPlanV1:
        if self.selection_mode == "top_fraction" and (
            self.selection_fraction is None
            or self.selection_artifact_sha256 is None
            or self.evidence_partition != "validation"
            or self.evidence_status != "exploratory"
            or (self.basis == "capacity" and self.capacity_limit is None)
            or (self.capacity_limit is not None and self.to_attend > self.capacity_limit)
        ):
            raise ValueError(
                "La política por fracción requiere evidencia de validación y capacidad"
            )
        return self


class RiskLevelV1(_Strict):
    level: Literal["alto", "medio", "bajo"]
    share_of_cases: float = Field(ge=0, le=1)
    event_rate: float = Field(ge=0, le=1)
    min_score: float | None = Field(default=None, ge=0, le=1)  # descriptivo, no asigna niveles


class RiskLevelsV1(_Strict):
    levels: list[RiskLevelV1]
    base_rate: float = Field(ge=0, le=1)
    evidence_partition: Literal["validation"] = "validation"


def _artifact(evaluation: EvaluationSummaryV1) -> PolicySelectionArtifactV1:
    if evaluation.policy_selection is None:
        raise MLValidationError(
            NO_OPERATING_POINTS,
            "Este modelo se entrenó antes de esta función: reentrénalo para decidir con él",
        )
    try:
        artifact = PolicySelectionArtifactV1.model_validate(
            evaluation.policy_selection.model_dump(mode="json")
        )
    except ValidationError as error:
        raise MLValidationError(
            "INVALID_POLICY_ARTIFACT", "El artefacto de validación cambió"
        ) from error
    selected = evaluation.selection.selected_model
    if selected != artifact.selected_model or selected not in evaluation.validation:
        raise MLValidationError("INVALID_POLICY_ARTIFACT", "El artefacto no corresponde al modelo")
    metrics = evaluation.validation[selected].metrics
    if (
        type(artifact.positive_class) is not type(metrics.positive_class)
        or artifact.positive_class != metrics.positive_class
    ):
        raise MLValidationError("INVALID_POLICY_ARTIFACT", "La clase positiva no coincide")
    if (artifact.points[0].support, artifact.points[0].positives) != (
        metrics.support,
        metrics.positive_support,
    ):
        raise MLValidationError(
            "INVALID_POLICY_ARTIFACT", "El artefacto no corresponde a validation"
        )
    return artifact


def _plan(
    point: PolicyOperatingPointV1,
    *,
    artifact: PolicySelectionArtifactV1,
    basis: Literal["capacity", "cost"],
    expected_cases: int,
    why: str,
    capacity_limit: int | None = None,
) -> DecisionPlanV1:
    scale = expected_cases / point.support
    precision = point.true_alerts / point.flagged if point.flagged else 0.0
    recall = point.true_alerts / point.positives if point.positives else 0.0
    uncertainty = (
        1.96 * math.sqrt(precision * (1 - precision) / point.flagged) if point.flagged else 0.0
    )
    to_attend = top_fraction_count(point.selection_fraction, expected_cases)
    if capacity_limit is not None:
        to_attend = min(to_attend, capacity_limit)
    expected_positives = round(point.positives * scale)
    expected_negatives = expected_cases - expected_positives
    # La grilla nominal y el share observado pueden diferir por floor. Conservar el total
    # proyectado de cada clase en vez de sumar aciertos y omisiones de escalas distintas.
    true_alerts = min(
        max(round(to_attend * precision), to_attend - expected_negatives), expected_positives
    )
    return DecisionPlanV1(
        basis=basis,
        flagged_share=point.flagged_share,
        threshold=point.cutoff_score,
        selection_mode="top_fraction",
        selection_fraction=point.selection_fraction,
        selection_artifact_sha256=artifact.content_sha256,
        evidence_partition="validation",
        evidence_status="exploratory",
        capacity_limit=capacity_limit,
        expected_cases=expected_cases,
        to_attend=to_attend,
        expected_true_alerts=true_alerts,
        expected_false_alarms=to_attend - true_alerts,
        expected_missed=expected_positives - true_alerts,
        precision=precision,
        recall=recall,
        uncertainty=min(uncertainty, 1.0),
        explanation=why,
    )


def by_capacity(
    evaluation: EvaluationSummaryV1, *, cases_per_period: int, expected_cases_per_period: int
) -> DecisionPlanV1:
    """El corte que no supera la capacidad de atención por periodo."""

    if (
        not isinstance(cases_per_period, Integral)
        or isinstance(cases_per_period, bool)
        or cases_per_period < 0
        or not isinstance(expected_cases_per_period, Integral)
        or isinstance(expected_cases_per_period, bool)
        or expected_cases_per_period < 0
    ):
        raise MLValidationError(
            "INVALID_CAPACITY", "La capacidad y los casos esperados deben ser no negativos"
        )
    share = (
        min(1.0, cases_per_period / expected_cases_per_period) if expected_cases_per_period else 0.0
    )
    artifact = _artifact(evaluation)
    fitting = [
        point
        for point in artifact.points
        if point.selection_fraction <= share
        and (point.flagged > 0 or point.selection_fraction == 0)
    ]
    point = max(fitting, key=lambda item: item.selection_fraction)
    return _plan(
        point,
        artifact=artifact,
        basis="capacity",
        expected_cases=expected_cases_per_period,
        capacity_limit=cases_per_period,
        why=(
            f"Con capacidad para {cases_per_period} de {expected_cases_per_period} casos por "
            "periodo, se elige una fracción de la grilla de validación y se prioriza por puntaje. "
            "La capacidad se aplica a un lote por período; empates por orden original."
        ),
    )


def by_cost(
    evaluation: EvaluationSummaryV1,
    *,
    cost_false_alarm: float,
    cost_missed: float,
    expected_cases_per_period: int,
) -> DecisionPlanV1:
    """El corte que minimiza el costo relativo de avisar de más frente a dejar pasar."""

    if (
        not math.isfinite(cost_false_alarm)
        or not math.isfinite(cost_missed)
        or cost_false_alarm < 0
        or cost_missed < 0
        or cost_false_alarm + cost_missed == 0
        or not isinstance(expected_cases_per_period, Integral)
        or isinstance(expected_cases_per_period, bool)
        or expected_cases_per_period <= 0
    ):
        raise MLValidationError("INVALID_COSTS", "Indica cuánto pesa cada error (mayor que cero)")
    artifact = _artifact(evaluation)
    point = min(
        artifact.points,
        key=lambda item: (
            item.false_alarms * cost_false_alarm + item.missed * cost_missed,
            item.selection_fraction,
        ),
    )
    return _plan(
        point,
        artifact=artifact,
        basis="cost",
        expected_cases=expected_cases_per_period,
        why=(
            "Elegido para que el peso total de los errores sea el menor posible, con dejar pasar "
            f"un caso pesando {cost_missed:g} y una falsa alarma {cost_false_alarm:g} (pesos "
            "relativos, no dinero)."
        ),
    )


def risk_levels(evaluation: EvaluationSummaryV1) -> RiskLevelsV1:
    """Alto: el 10 % de mayor puntaje; medio: el 20 % siguiente; bajo: el resto."""

    points = _artifact(evaluation).points

    def nearest(share: float) -> PolicyOperatingPointV1:
        return min(points, key=lambda item: abs(item.selection_fraction - share))

    top, middle = nearest(0.1), nearest(0.3)
    last = points[-1]
    support, positives = last.support, last.positives
    bands = (
        ("alto", top.flagged, top.true_alerts, top.cutoff_score),
        (
            "medio",
            middle.flagged - top.flagged,
            middle.true_alerts - top.true_alerts,
            middle.cutoff_score,
        ),
        ("bajo", support - middle.flagged, positives - middle.true_alerts, 0.0),
    )
    return RiskLevelsV1(
        levels=[
            RiskLevelV1(
                level=name,
                share_of_cases=count / support,
                event_rate=(hits / count) if count else 0.0,
                min_score=threshold,
            )
            for name, count, hits, threshold in bands
        ],
        base_rate=positives / support,
    )


def selected_row_numbers(
    plan: DecisionPlanV1, *, scores: Sequence[float], row_numbers: Sequence[int]
) -> set[int]:
    """Aplica una sola semántica a tabla y lista; row_number es la posición original del lote."""

    if plan.selection_mode != "top_fraction" or plan.selection_fraction is None:
        raise MLValidationError(
            "LEGACY_DECISION_PLAN", "Vuelve a elegir una política con validation"
        )
    if (
        len(scores) != len(row_numbers)
        or len(set(row_numbers)) != len(row_numbers)
        or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
        or any(
            not isinstance(row, Integral) or isinstance(row, bool) or row < 1 for row in row_numbers
        )
    ):
        raise MLValidationError(
            "INVALID_POLICY_SCORES", "El lote no tiene puntajes y posiciones válidos"
        )
    count = top_fraction_count(plan.selection_fraction, len(scores))
    if plan.capacity_limit is not None:
        count = min(count, plan.capacity_limit)
    ordered = sorted(zip(scores, row_numbers, strict=True), key=lambda item: (-item[0], item[1]))
    return {int(row) for _score, row in ordered[:count]}


def risk_level_for_rank(rank: int, total: int) -> str:
    if total < 1 or rank < 1 or rank > total:
        raise ValueError("La prioridad debe pertenecer al lote")
    if rank <= top_fraction_count(0.1, total):
        return "Alto"
    if rank <= top_fraction_count(0.3, total):
        return "Medio"
    return "Bajo"


def require_probability_valuation(evaluation: EvaluationSummaryV1) -> None:
    """MV-03 sólo permite ranking; ningún Brier observado autoriza multiplicar score por dinero."""

    raise MLValidationError(
        "PROBABILITY_VALUATION_NOT_VERIFIED",
        "La probabilidad no está verificada para dinero esperado; usa tasas externas confirmadas",
    )
