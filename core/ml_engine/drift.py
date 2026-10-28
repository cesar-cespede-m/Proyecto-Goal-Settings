"""Cambios en los datos entre el entrenamiento y un lote nuevo (MON-01, ADR-013).

Compara sólo las columnas que usa el modelo y devuelve cifras agregadas por columna: nunca
filas, valores de celdas ni nombres de categorías. Los umbrales del índice de estabilidad
poblacional (PSI) son la convención habitual de la industria (0,10 y 0,25), no un valor
calibrado para cada caso.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

DriftLevel = Literal["STABLE", "WATCH", "REVIEW"]
OverallDriftLevel = Literal["STABLE", "WATCH", "REVIEW", "INSUFFICIENT_DATA"]

PSI_WATCH = 0.10
PSI_REVIEW = 0.25
OUT_OF_RANGE_WATCH = 0.05
OUT_OF_RANGE_REVIEW = 0.20
NEW_CATEGORY_WATCH = 0.01
NEW_CATEGORY_REVIEW = 0.10
MISSING_CHANGE_WATCH = 0.10
MISSING_CHANGE_REVIEW = 0.20
MIN_ROWS_FOR_DRIFT = 50
_EPSILON = 1e-4
_ORDER: dict[str, int] = {"STABLE": 0, "WATCH": 1, "REVIEW": 2}


class ColumnDriftV1(BaseModel):
    column: str
    kind: Literal["numeric", "categorical"]
    psi: float = Field(ge=0)
    out_of_range_rate: float | None = Field(default=None, ge=0, le=1)
    new_category_rate: float | None = Field(default=None, ge=0, le=1)
    reference_missing_rate: float = Field(ge=0, le=1)
    current_missing_rate: float = Field(ge=0, le=1)
    reference_mean: float | None = None
    current_mean: float | None = None
    level: DriftLevel
    reasons: list[str]


class DriftReportV1(BaseModel):
    schema_version: Literal["drift-report.v1"] = "drift-report.v1"
    reference_rows: int = Field(ge=1)
    current_rows: int = Field(ge=1)
    level: OverallDriftLevel
    max_psi: float = Field(ge=0)
    columns: list[ColumnDriftV1]

    def columns_at(self, *levels: DriftLevel) -> list[ColumnDriftV1]:
        return [column for column in self.columns if column.level in levels]


def _level(value: float, watch: float, review: float) -> DriftLevel:
    if value > review:
        return "REVIEW"
    if value > watch:
        return "WATCH"
    return "STABLE"


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.clip(reference, _EPSILON, None)
    current = np.clip(current, _EPSILON, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype=float)


def _numeric(reference: pd.Series, current: pd.Series) -> dict[str, object]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(ref) or not len(cur):
        return {"psi": 0.0, "out_of_range_rate": 0.0, "reference_mean": None, "current_mean": None}
    edges = np.unique(np.quantile(ref, np.linspace(0.1, 0.9, 9)))
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    ref_counts = np.histogram(ref, bins=bins)[0].astype(float)
    cur_counts = np.histogram(cur, bins=bins)[0].astype(float)
    outside = float(np.mean((cur < ref.min()) | (cur > ref.max())))
    return {
        "psi": _psi(_shares(ref_counts), _shares(cur_counts)),
        "out_of_range_rate": outside,
        "reference_mean": float(ref.mean()),
        "current_mean": float(cur.mean()),
    }


def _categorical(reference: pd.Series, current: pd.Series) -> dict[str, object]:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    if not len(ref) or not len(cur):
        return {"psi": 0.0, "new_category_rate": 0.0}
    known = sorted(ref.unique())
    ref_counts = np.asarray([(ref == value).sum() for value in known] + [0], dtype=float)
    cur_counts = np.asarray(
        [*((cur == value).sum() for value in known), (~cur.isin(known)).sum()], dtype=float
    )
    return {
        "psi": _psi(_shares(ref_counts), _shares(cur_counts)),
        "new_category_rate": float(cur_counts[-1] / len(cur)),
    }


def _column(
    name: str, kind: Literal["numeric", "categorical"], reference: pd.Series, current: pd.Series
) -> ColumnDriftV1:
    stats = _numeric(reference, current) if kind == "numeric" else _categorical(reference, current)
    reference_missing = float(reference.isna().mean())
    current_missing = float(current.isna().mean())
    psi = float(stats["psi"])
    checks: list[tuple[str, DriftLevel]] = [
        ("DISTRIBUTION_SHIFT", _level(psi, PSI_WATCH, PSI_REVIEW)),
        (
            "MISSING_RATE_CHANGE",
            _level(
                abs(current_missing - reference_missing),
                MISSING_CHANGE_WATCH,
                MISSING_CHANGE_REVIEW,
            ),
        ),
    ]
    if kind == "numeric":
        checks.append(
            (
                "OUT_OF_TRAINING_RANGE",
                _level(float(stats["out_of_range_rate"]), OUT_OF_RANGE_WATCH, OUT_OF_RANGE_REVIEW),
            )
        )
    else:
        checks.append(
            (
                "NEW_CATEGORIES",
                _level(float(stats["new_category_rate"]), NEW_CATEGORY_WATCH, NEW_CATEGORY_REVIEW),
            )
        )
    level = max((check_level for _, check_level in checks), key=_ORDER.__getitem__)
    return ColumnDriftV1(
        column=name,
        kind=kind,
        reference_missing_rate=reference_missing,
        current_missing_rate=current_missing,
        level=level,
        reasons=[code for code, check_level in checks if check_level != "STABLE"],
        **stats,
    )


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> DriftReportV1:
    """Compara cada columna del modelo entre el entrenamiento y el lote nuevo."""

    columns = [
        _column(name, "numeric", reference[name], current[name]) for name in numeric_columns
    ] + [
        _column(name, "categorical", reference[name], current[name]) for name in categorical_columns
    ]
    if len(current) < MIN_ROWS_FOR_DRIFT:
        level: OverallDriftLevel = "INSUFFICIENT_DATA"
    else:
        level = max((column.level for column in columns), key=_ORDER.__getitem__, default="STABLE")
    return DriftReportV1(
        reference_rows=len(reference),
        current_rows=len(current),
        level=level,
        max_psi=max((column.psi for column in columns), default=0.0),
        columns=columns,
    )
