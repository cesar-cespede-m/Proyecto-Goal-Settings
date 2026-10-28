"""GA-07b/c (núcleo): entrenar y medir varias clases y cantidades; partición por tiempo.

Kernels puros sin UI, sin IA y sin base de datos. Reciben un DataFrame, el objetivo y las
columnas a usar; devuelven métricas en validación, el candidato elegido contra su baseline y
su medición única en test. Todavía no están conectados a las decisiones ni a la estrategia
(eso es GA-07d): por eso no se registran en `TASK_KINDS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from core.ml_engine.errors import MLValidationError
from core.ml_engine.preprocessing import build_preprocessor, infer_feature_types

MIN_ROWS_PER_CLASS = 5
MAX_CLASSES = 20
MIN_REGRESSION_ROWS = 30
_EPSILON = 1e-12


@dataclass(frozen=True)
class Splits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    method: str  # "random", "stratified" o "temporal"


@dataclass(frozen=True)
class KernelResult:
    task_kind: str
    primary_metric: str
    higher_is_better: bool
    validation: dict[str, dict[str, float]]  # candidato → métricas
    selected: str | None  # None si nadie supera al baseline
    test: dict[str, float] = field(default_factory=dict)  # sólo del elegido, una vez
    split_method: str = "random"
    warnings: list[str] = field(default_factory=list)
    baseline: str = ""
    split_rows: dict[str, int] = field(default_factory=dict)
    pipeline: Pipeline | None = None  # el elegido, ya entrenado; None si nadie gana


def split_frame(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    seed: int,
    stratify: bool,
    time_column: str | None = None,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Splits:
    """60/20/20. Con `time_column`, lo más antiguo entrena y lo más reciente se prueba."""

    data = frame.dropna(subset=[target]).reset_index(drop=True)
    X, y = data[features], data[target]
    train_share, validation_share, _ = fractions
    if time_column is not None:
        moments = pd.to_datetime(data[time_column], errors="coerce", format="mixed")
        if moments.isna().any():
            raise MLValidationError(
                "TIME_COLUMN_NOT_DATES", "La columna de fecha tiene valores que no son fechas"
            )
        order = np.argsort(moments.to_numpy(), kind="stable")
        first = int(len(order) * train_share)
        second = int(len(order) * (train_share + validation_share))
        parts = order[:first], order[first:second], order[second:]
        return Splits(
            *(X.iloc[part] for part in parts), *(y.iloc[part] for part in parts), "temporal"
        )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        train_size=train_share,
        random_state=seed,
        stratify=y if stratify else None,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        random_state=seed,
        stratify=y_rest if stratify else None,
    )
    return Splits(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        "stratified" if stratify else "random",
    )


def _pipelines(estimators: dict[str, Any], X_train: pd.DataFrame) -> dict[str, Pipeline]:
    numeric, categorical = infer_feature_types(X_train)
    return {
        name: Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(numeric, categorical, scale_numeric=scale),
                ),
                ("estimator", estimator),
            ]
        )
        for name, (estimator, scale) in estimators.items()
    }


def _classification_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
    }


def _regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def _select(
    validation: dict[str, dict[str, float]],
    baseline: str,
    metric: str,
    higher_is_better: bool,
) -> str | None:
    sign = 1 if higher_is_better else -1
    reference = sign * validation[baseline][metric]
    better = {
        name: sign * values[metric]
        for name, values in validation.items()
        if name != baseline and sign * values[metric] > reference + _EPSILON
    }
    # Ante empate gana el primero declarado, que es el más simple.
    return max(better, key=lambda name: (better[name], -list(validation).index(name)), default=None)


def _run(
    task_kind: str,
    splits: Splits,
    estimators: dict[str, Any],
    baseline: str,
    metric: str,
    higher_is_better: bool,
    measure,
) -> KernelResult:
    pipelines = _pipelines(estimators, splits.X_train)
    validation: dict[str, dict[str, float]] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(splits.X_train, splits.y_train)
        validation[name] = measure(splits.y_validation, pipeline.predict(splits.X_validation))
    selected = _select(validation, baseline, metric, higher_is_better)
    test = (
        measure(splits.y_test, pipelines[selected].predict(splits.X_test))
        if selected is not None
        else {}
    )
    warnings = [] if selected else ["NO_CANDIDATE_BEATS_BASELINE"]
    return KernelResult(
        baseline=baseline,
        split_rows={
            "train": len(splits.X_train),
            "validation": len(splits.X_validation),
            "test": len(splits.X_test),
        },
        pipeline=pipelines[selected] if selected is not None else None,
        task_kind=task_kind,
        primary_metric=metric,
        higher_is_better=higher_is_better,
        validation=validation,
        selected=selected,
        test=test,
        split_method=splits.method,
        warnings=warnings,
    )


def train_multiclass(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    seed: int = 42,
    time_column: str | None = None,
) -> KernelResult:
    """Varias clases: el baseline predice la clase más frecuente; se elige por F1 macro."""

    counts = frame[target].dropna().astype(str).value_counts()
    if len(counts) < 3:
        raise MLValidationError(
            "NOT_ENOUGH_CLASSES", "Para varias clases se necesitan al menos tres valores"
        )
    if len(counts) > MAX_CLASSES:
        raise MLValidationError(
            "TOO_MANY_CLASSES", f"Hay más de {MAX_CLASSES} valores distintos para anticipar"
        )
    if counts.min() < MIN_ROWS_PER_CLASS:
        raise MLValidationError(
            "CLASS_TOO_SMALL",
            f"Cada valor necesita al menos {MIN_ROWS_PER_CLASS} casos; «{counts.idxmin()}» "
            f"tiene {int(counts.min())}",
        )
    # Clases numéricas quedan como números: scikit-learn falla al ponderar clases que son
    # números escritos como texto («1»…«5»). El resto, como texto común de Python.
    data = frame.dropna(subset=[target])
    if not is_numeric_dtype(data[target]):
        data = data.assign(**{target: data[target].astype(str).to_numpy(dtype=object)})
    splits = split_frame(
        data, target, features, seed=seed, stratify=time_column is None, time_column=time_column
    )
    estimators = {
        "DummyClassifier": (DummyClassifier(strategy="most_frequent"), False),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            True,
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed
            ),
            False,
        ),
    }
    return _run(
        "multiclass_classification",
        splits,
        estimators,
        "DummyClassifier",
        "macro_f1",
        True,
        _classification_metrics,
    )


def train_regression(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    seed: int = 42,
    time_column: str | None = None,
) -> KernelResult:
    """Cantidades: el baseline predice la mediana; se elige por error absoluto medio."""

    values = pd.to_numeric(frame[target], errors="coerce")
    if values.notna().sum() < MIN_REGRESSION_ROWS:
        raise MLValidationError(
            "NOT_ENOUGH_ROWS", f"Se necesitan al menos {MIN_REGRESSION_ROWS} casos con valor"
        )
    if values.dropna().nunique() < 3:
        raise MLValidationError(
            "TARGET_NOT_A_QUANTITY", "Lo que se quiere anticipar no varía como una cantidad"
        )
    data = frame.assign(**{target: values})
    splits = split_frame(data, target, features, seed=seed, stratify=False, time_column=time_column)
    estimators = {
        "DummyRegressor": (DummyRegressor(strategy="median"), False),
        "Ridge": (Ridge(alpha=1.0), True),
        "RandomForestRegressor": (
            RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed),
            False,
        ),
    }
    return _run(
        "regression", splits, estimators, "DummyRegressor", "mae", False, _regression_metrics
    )
