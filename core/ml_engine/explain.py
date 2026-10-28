"""V2-16: por qué un caso tiene su puntaje, en 2–3 razones y en palabras.

Método simple y verificable: para cada columna, se reemplaza el valor del caso por uno típico
del archivo de entrenamiento (mediana o valor más frecuente) y se mide cuánto cambia el puntaje.
Las columnas que más lo suben o bajan son las razones. Es una relación del modelo, no una causa.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field

MAX_EXPLAINED_CASES = 500


class CaseReasonV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    value: str
    direction: Literal["sube", "baja"]
    effect: float = Field(ge=0)  # cambio absoluto del puntaje al usar un valor típico


def typical_values(reference: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    typical: dict[str, Any] = {}
    for column in columns:
        series = reference[column].dropna()
        if series.empty:
            typical[column] = np.nan
        elif is_numeric_dtype(series):
            typical[column] = float(series.median())
        else:
            typical[column] = series.mode().iloc[0]
    return typical


def case_reasons(
    score: Any,
    frame: pd.DataFrame,
    columns: list[str],
    typical: dict[str, Any],
    *,
    top: int = 3,
    min_effect: float = 0.01,
) -> list[list[CaseReasonV1]]:
    """`score(frame) -> np.ndarray` es la función de puntaje del modelo ya aprobado."""

    cases = frame.head(MAX_EXPLAINED_CASES).reset_index(drop=True)
    base = np.asarray(score(cases), dtype=float)
    effects: dict[str, np.ndarray] = {}
    for column in columns:
        altered = cases.copy()
        altered[column] = typical[column]
        effects[column] = base - np.asarray(score(altered), dtype=float)
    reasons: list[list[CaseReasonV1]] = []
    for index in range(len(cases)):
        ranked = sorted(columns, key=lambda column: -abs(effects[column][index]))
        reasons.append(
            [
                CaseReasonV1(
                    column=column,
                    value=str(cases.at[index, column]),
                    direction="sube" if effects[column][index] > 0 else "baja",
                    effect=abs(float(effects[column][index])),
                )
                for column in ranked[:top]
                if abs(effects[column][index]) >= min_effect
            ]
        )
    return reasons


def reasons_text(reasons: list[CaseReasonV1]) -> str:
    if not reasons:
        return "Sin una razón que destaque: su puntaje está cerca del típico."
    parts = [
        f"«{item.column}» = {item.value} lo {'sube' if item.direction == 'sube' else 'baja'}"
        for item in reasons
    ]
    return "; ".join(parts) + "."
