"""Preprocesamiento scikit-learn que se ajusta exclusivamente con train."""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.ml_engine.errors import MLValidationError


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric.append(str(column))
        else:
            categorical.append(str(column))
    if not numeric and not categorical:
        raise MLValidationError("NO_FEATURES", "No hay features para entrenar")
    return numeric, categorical


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
