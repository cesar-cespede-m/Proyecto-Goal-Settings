"""Perfilado agregado de un CSV ya validado."""

import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype

from core.data_engine.loading import read_validated_frame
from core.data_engine.models import (
    ColumnProfile,
    DatasetClassification,
    DatasetProfileV1,
    IngestionLimits,
    NumericSummary,
    OutcomeAssociationV1,
)
from core.data_engine.semantics import semantic_type
from core.data_engine.validator import ValidatedCSV


def _finite_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_profile(
    validated: ValidatedCSV,
    *,
    classification: DatasetClassification,
    limits: IngestionLimits,
) -> DatasetProfileV1:
    """Calcula estadísticas agregadas; nunca devuelve filas del CSV."""

    frame = read_validated_frame(validated)
    duplicate_rows = int(frame.duplicated().sum())
    columns: list[ColumnProfile] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    constant_columns: list[str] = []
    possible_identifier_columns: list[str] = []

    for name in validated.header:
        series = frame[name]
        missing_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        constant = non_null_count > 0 and unique_count <= 1
        possible_identifier = (
            non_null_count == validated.row_count
            and unique_count == validated.row_count
            and validated.row_count > 1
            # Una medición con decimales puede no repetirse nunca y no por eso es un código.
            and not _has_fractions(series)
        )
        ratio = unique_count / non_null_count if non_null_count else 0.0
        high_cardinality = (
            not is_numeric_dtype(series)
            and unique_count >= limits.high_cardinality_min_unique
            and ratio >= limits.high_cardinality_ratio
        )

        numeric_summary = None
        if is_bool_dtype(series):
            inferred_type = "boolean"
            categorical_columns.append(name)
        elif is_numeric_dtype(series):
            inferred_type = "numeric"
            numeric_columns.append(name)
            numeric_summary = NumericSummary(
                minimum=_finite_float(series.min(skipna=True)),
                maximum=_finite_float(series.max(skipna=True)),
                mean=_finite_float(series.mean(skipna=True)),
            )
        elif is_string_dtype(series) and unique_count <= 20:
            inferred_type = "categorical"
            categorical_columns.append(name)
        else:
            inferred_type = "string"
            categorical_columns.append(name)

        if constant:
            constant_columns.append(name)
        if high_cardinality:
            high_cardinality_columns.append(name)
        if possible_identifier:
            possible_identifier_columns.append(name)
        columns.append(
            ColumnProfile(
                name=name,
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_rate=missing_count / validated.row_count,
                unique_count=unique_count,
                constant=constant,
                high_cardinality=high_cardinality,
                possible_identifier=possible_identifier,
                numeric_summary=numeric_summary,
                semantic_type=semantic_type(
                    name,
                    series,
                    possible_identifier=possible_identifier,
                    inferred_type=inferred_type,
                ),
            )
        )

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"DUPLICATE_ROWS:{duplicate_rows}")
    if constant_columns:
        warnings.append("CONSTANT_COLUMNS:" + ",".join(constant_columns))
    if high_cardinality_columns:
        warnings.append("HIGH_CARDINALITY_COLUMNS:" + ",".join(high_cardinality_columns))
    if possible_identifier_columns:
        warnings.append("POSSIBLE_IDENTIFIER_COLUMNS:" + ",".join(possible_identifier_columns))
    personal_columns = [column.name for column in columns if column.semantic_type == "personal"]
    if personal_columns:
        warnings.append("POSSIBLE_PERSONAL_DATA_COLUMNS:" + ",".join(personal_columns))

    return DatasetProfileV1(
        data_classification=classification,
        row_count=validated.row_count,
        column_count=validated.column_count,
        duplicate_row_count=duplicate_rows,
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        high_cardinality_columns=high_cardinality_columns,
        constant_columns=constant_columns,
        warnings=warnings,
        outcome_associations=_outcome_associations(frame),
    )


def _has_fractions(series: pd.Series) -> bool:
    """Verdadero si la columna es numérica y algún valor tiene parte decimal."""

    if not is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    values = series.dropna().astype(float)
    return bool((values != values.round()).any())


def _outcome_associations(frame: pd.DataFrame) -> list[OutcomeAssociationV1]:
    """Resume sólo separaciones muy marcadas, sin persistir valores ni filas.

    Se conservan también señales dudosas desde 0.8/0.75 para que la capa de dominio pueda
    pedir contexto humano. Una categoría sólo cuenta como separadora si distintos valores
    dominan resultados distintos; así, un resultado desbalanceado no marca todas las columnas.
    """

    associations: list[OutcomeAssociationV1] = []
    numeric = frame.select_dtypes(include="number")
    if len(numeric.columns) >= 2:
        correlations = numeric.corr()
        for target in numeric.columns:
            target_count = int(frame[target].notna().sum())
            for feature in numeric.columns:
                if feature == target:
                    continue
                strength = correlations.at[target, feature]
                if pd.isna(strength) or abs(float(strength)) <= 0.75:
                    continue
                coverage = int(frame[[target, feature]].notna().all(axis=1).sum()) / target_count
                associations.append(
                    OutcomeAssociationV1(
                        target_column=str(target),
                        feature_column=str(feature),
                        signal="numeric_correlation",
                        strength=abs(float(strength)),
                        coverage=coverage,
                    )
                )

    for target in frame.columns:
        target_values = frame[target]
        if target_values.nunique(dropna=True) != 2:
            continue
        target_count = int(target_values.notna().sum())
        for feature in frame.columns:
            if feature == target or frame[feature].nunique(dropna=True) > 20:
                continue
            valid = frame[[feature, target]].dropna()
            if valid.empty:
                continue
            coverage = len(valid) / target_count
            if coverage <= 0.75:
                continue
            counts = valid.groupby([feature, target], dropna=False).size().unstack(fill_value=0)
            dominant_results = counts.idxmax(axis=1)
            purity = counts.max(axis=1) / counts.sum(axis=1)
            strong_values = purity > 0.95
            strong_coverage = float(counts.loc[strong_values].sum().sum() / target_count)
            if (
                strong_coverage > 0.9
                and dominant_results.loc[strong_values].nunique(dropna=True) >= 2
            ):
                strength = float(purity.loc[strong_values].min())
                signal_coverage = strong_coverage
            else:
                doubtful_values = purity > 0.8
                doubtful_coverage = float(counts.loc[doubtful_values].sum().sum() / target_count)
                if (
                    doubtful_coverage <= 0.75
                    or dominant_results.loc[doubtful_values].nunique(dropna=True) < 2
                ):
                    continue
                strength = float(purity.loc[doubtful_values].min())
                signal_coverage = doubtful_coverage
            if strength <= 0.8:
                continue
            associations.append(
                OutcomeAssociationV1(
                    target_column=str(target),
                    feature_column=str(feature),
                    signal="categorical_separation",
                    strength=strength,
                    coverage=signal_coverage,
                )
            )
    return associations
