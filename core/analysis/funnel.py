"""V2-20: embudo y tiempo entre etapas desde registros de actividad (muchas filas por caso).

Recibe qué columna identifica cada caso, cuál es el momento y cuál la etapa, y el orden de las
etapas. Cuenta cuántos casos llegan a cada etapa, dónde se quedan y cuánto tardan en avanzar.
Sólo devuelve conteos y tiempos agregados, nunca filas.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunnelStepV1(_Strict):
    stage: str
    reached: int = Field(ge=0)
    share_of_start: float = Field(ge=0, le=1)
    stayed_here: int = Field(ge=0)  # llegaron y no pasaron a la siguiente
    advance_rate: float | None = Field(default=None, ge=0, le=1)  # de aquí a la siguiente
    median_hours_to_next: float | None = Field(default=None, ge=0)


class FunnelV1(_Strict):
    entities: int = Field(ge=0)
    steps: list[FunnelStepV1]
    biggest_drop: str | None  # etapa donde más casos se quedan (sin contar la última)
    summary: str


def funnel(
    frame: pd.DataFrame, *, entity: str, moment: str, stage: str, order: list[str]
) -> FunnelV1:
    data = frame[[entity, moment, stage]].dropna()
    data = data[data[stage].astype(str).isin(order)].copy()
    data[moment] = pd.to_datetime(data[moment], errors="coerce", format="mixed")
    data = data.dropna(subset=[moment])
    first = data.groupby([entity, stage])[moment].min().unstack(stage)
    first = first.reindex(columns=order)
    start = int(first[order[0]].notna().sum()) or int(len(first))
    steps: list[FunnelStepV1] = []
    for index, name in enumerate(order):
        reached = first[name].notna()
        following = first[order[index + 1]] if index + 1 < len(order) else None
        advanced = (
            reached & following.notna() if following is not None else pd.Series(False, first.index)
        )
        hours = (
            ((following - first[name]).dt.total_seconds() / 3600)[advanced]
            if following is not None
            else pd.Series(dtype=float)
        )
        count = int(reached.sum())
        steps.append(
            FunnelStepV1(
                stage=name,
                reached=count,
                share_of_start=(count / start) if start else 0.0,
                stayed_here=int((reached & ~advanced).sum()) if following is not None else 0,
                advance_rate=(int(advanced.sum()) / count)
                if following is not None and count
                else None,
                median_hours_to_next=float(hours.clip(lower=0).median()) if len(hours) else None,
            )
        )
    candidates = steps[:-1]
    biggest = max(candidates, key=lambda item: item.stayed_here) if candidates else None
    if biggest is None or biggest.stayed_here == 0:
        summary = "Todos los casos avanzan hasta la última etapa."
    else:
        summary = (
            f"De {start} casos que empiezan, {steps[-1].reached} llegan a «{steps[-1].stage}». "
            f"Donde más se quedan es en «{biggest.stage}»: {biggest.stayed_here} casos "
            f"({biggest.stayed_here / start * 100:.0f} % de los que empezaron)."
        )
    return FunnelV1(
        entities=int(len(first)),
        steps=steps,
        biggest_drop=None if biggest is None or biggest.stayed_here == 0 else biggest.stage,
        summary=summary,
    )


class CohortRowV1(_Strict):
    period: str
    entities: int = Field(ge=0)
    reached_share: dict[str, float]  # etapa → proporción de la cohorte que la alcanzó


class CohortsV1(_Strict):
    rows: list[CohortRowV1]
    summary: str


def cohorts(
    frame: pd.DataFrame, *, entity: str, moment: str, stage: str, order: list[str]
) -> CohortsV1:
    """V2-21: casos agrupados por el mes de su primera actividad y hasta dónde llegó cada grupo."""

    data = frame[[entity, moment, stage]].dropna().copy()
    data[moment] = pd.to_datetime(data[moment], errors="coerce", format="mixed")
    data = data.dropna(subset=[moment])
    start = data.groupby(entity)[moment].min().dt.to_period("M").astype(str)
    reached = (
        data[data[stage].astype(str).isin(order)]
        .assign(**{stage: lambda d: d[stage].astype(str)})
        .groupby([entity, stage])
        .size()
        .unstack(stage)
        .reindex(columns=order)
        .notna()
    )
    reached = reached.reindex(start.index, fill_value=False)
    rows = []
    for period, members in start.groupby(start).groups.items():
        group = reached.loc[list(members)]
        rows.append(
            CohortRowV1(
                period=str(period),
                entities=len(group),
                reached_share={name: float(group[name].mean()) for name in order},
            )
        )
    rows.sort(key=lambda item: item.period)
    last = order[-1]
    if len(rows) < 2:
        summary = "Todos los casos empezaron en el mismo mes: no hay cohortes que comparar."
    else:
        best = max(rows, key=lambda item: item.reached_share[last])
        worst = min(rows, key=lambda item: item.reached_share[last])
        summary = (
            f"Los que empezaron en {best.period} son los que más llegan a «{last}» "
            f"({best.reached_share[last] * 100:.0f} %); los de {worst.period}, los que menos "
            f"({worst.reached_share[last] * 100:.0f} %). Los meses recientes pueden no haber "
            "tenido tiempo de avanzar."
        )
    return CohortsV1(rows=rows, summary=summary)


class OutcomeSuggestionV1(_Strict):
    target_stage: str
    observe_days: int = Field(ge=1)
    within_days: int = Field(ge=1)
    reason: str


def suggest_outcome(result: FunnelV1) -> OutcomeSuggestionV1 | None:
    """Sugiere qué anticipar: no llegar a la etapa siguiente a donde más casos se quedan."""

    names = [step.stage for step in result.steps]
    if result.biggest_drop is None or result.biggest_drop not in names[:-1]:
        return None
    index = names.index(result.biggest_drop)
    step = result.steps[index]
    hours = step.median_hours_to_next or 24.0
    within = max(7, min(365, round(hours * 3 / 24)))
    return OutcomeSuggestionV1(
        target_stage=names[index + 1],
        observe_days=max(1, min(30, round(hours / 24 / 2) or 1)),
        within_days=within,
        reason=(
            f"Donde más casos se quedan es en «{step.stage}»; los que avanzan tardan cerca de "
            f"{hours / 24:.1f} días en llegar a «{names[index + 1]}», así que se propone mirar "
            f"el comienzo y dar {within} días de plazo."
        ),
    )
