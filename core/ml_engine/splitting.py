"""Split train/validation/test antes de cualquier ajuste de preprocessing."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_number, is_numeric_dtype
from sklearn.model_selection import train_test_split

from core.domain.canonical import canonical_hash, canonical_json
from core.domain.context import ClassSupport
from core.domain.strategy import StrategyProposalV1
from core.ml_engine.errors import MLValidationError
from core.ml_engine.models import PartitionSummary, SplitSummary, TemporalSplitEvidenceV1


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    summary: SplitSummary


def _python_scalar(value: Any) -> str | int | float | bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str | bool | int | float):
        return value
    raise MLValidationError("UNSUPPORTED_TARGET_VALUE", "El target contiene un valor no soportado")


def _typed_key(value: str | int | float | bool) -> str:
    return f"{type(value).__name__}:{canonical_json(value)}"


def _partition_summary(y: pd.Series) -> PartitionSummary:
    counts = y.value_counts(dropna=False, sort=False)
    return PartitionSummary(
        row_count=len(y),
        class_distribution=[
            ClassSupport(value=_python_scalar(value), count=int(count))
            for value, count in counts.items()
        ],
        index_sha256=canonical_hash(sorted(str(index) for index in y.index)),
    )


def _validate_partition(name: str, y: pd.Series) -> None:
    if y.empty:
        raise MLValidationError("SPLIT_EMPTY", f"La partición {name} quedó vacía")
    if y.nunique(dropna=False) != 2:
        raise MLValidationError(
            "SPLIT_CLASS_MISSING",
            f"La partición {name} no contiene ambas clases",
        )


def split_dataset(frame: pd.DataFrame, proposal: StrategyProposalV1) -> DatasetSplits:
    required = {proposal.target_column, *proposal.feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MLValidationError(
            "TRAINING_COLUMN_NOT_FOUND",
            "Faltan columnas requeridas: " + ", ".join(missing),
        )
    y = frame[proposal.target_column]
    if y.isna().any():
        raise MLValidationError("TARGET_HAS_MISSING", "El target contiene valores nulos")
    if y.nunique(dropna=False) != 2:
        raise MLValidationError(
            "TARGET_NOT_BINARY", "El target debe contener exactamente dos clases"
        )
    class_values = {_typed_key(_python_scalar(value)) for value in y.unique()}
    if _typed_key(proposal.positive_class) not in class_values:
        raise MLValidationError(
            "POSITIVE_CLASS_NOT_FOUND",
            "La clase positiva no coincide exactamente con el target",
        )

    X = frame.loc[:, proposal.feature_columns].copy()
    temporal_evidence = None
    if proposal.split.method == "temporal":
        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            temporal_evidence,
        ) = _temporal_split(frame, X, y, proposal)
    else:
        temporary_fraction = proposal.split.validation + proposal.split.test
        try:
            X_train, X_temporary, y_train, y_temporary = train_test_split(
                X,
                y,
                train_size=proposal.split.train,
                test_size=temporary_fraction,
                random_state=proposal.seed,
                stratify=y,
            )
            relative_test = proposal.split.test / temporary_fraction
            X_validation, X_test, y_validation, y_test = train_test_split(
                X_temporary,
                y_temporary,
                test_size=relative_test,
                random_state=proposal.seed,
                stratify=y_temporary,
            )
        except ValueError as error:
            raise MLValidationError(
                "STRATIFIED_SPLIT_FAILED",
                "No fue posible construir las tres particiones estratificadas",
            ) from error

    _validate_partition("train", y_train)
    _validate_partition("validation", y_validation)
    _validate_partition("test", y_test)
    summary = SplitSummary(
        method=proposal.split.method,
        seed=proposal.seed,
        train=_partition_summary(y_train),
        validation=_partition_summary(y_validation),
        test=_partition_summary(y_test),
        temporal=temporal_evidence,
    )
    return DatasetSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        summary=summary,
    )


def _dates(frame: pd.DataFrame, column: str, *, availability: bool = False) -> pd.Series:
    prefix = "LABEL_AVAILABILITY" if availability else "TIME_COLUMN"
    if column not in frame.columns:
        raise MLValidationError(f"{prefix}_NOT_FOUND", "Falta una columna temporal elegida")
    if is_numeric_dtype(frame[column]) or frame[column].map(is_number).any():
        raise MLValidationError(f"{prefix}_NOT_DATES", "La columna temporal contiene números")
    moments = pd.to_datetime(frame[column], errors="coerce", format="mixed", utc=True)
    if moments.isna().any():
        raise MLValidationError(
            f"{prefix}_NOT_DATES", "La columna temporal tiene valores ausentes o fechas inválidas"
        )
    return moments


def _index_hash(index: pd.Index) -> str:
    return canonical_hash(sorted(str(value) for value in index))


def _temporal_split(
    frame: pd.DataFrame, X: pd.DataFrame, y: pd.Series, proposal: StrategyProposalV1
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    TemporalSplitEvidenceV1,
]:
    """Ordena por UTC, conserva empates y purga etiquetas aún futuras en cada origen.

    El dataset debe haber acreditado antes que las etiquetas del test son observables.
    Este paso no inventa cobertura: prueba la disponibilidad entre particiones sólo
    cuando se declaró una columna de disponibilidad del resultado.
    """

    column = proposal.split.time_column
    if column is None:
        raise MLValidationError("TIME_COLUMN_NOT_FOUND", "Falta la columna de fecha elegida")
    if not frame.index.is_unique:
        raise MLValidationError(
            "DATASET_INDEX_NOT_UNIQUE", "La partición temporal requiere un índice de filas único"
        )
    moments = _dates(frame, column)
    order = moments.sort_values(kind="stable").index
    first = int(len(order) * proposal.split.train)
    second = int(len(order) * (proposal.split.train + proposal.split.validation))
    if first < 1 or second <= first or second >= len(order):
        raise MLValidationError(
            "SPLIT_EMPTY", "Las fracciones elegidas no dejan filas en las tres particiones"
        )
    validation_origin = moments.loc[order[first]]
    test_origin = moments.loc[order[second]]
    ordered_moments = moments.loc[order]
    # Se mueve el grupo empatado completo a la partición posterior, sin usar etiquetas.
    parts = [
        order[ordered_moments < validation_origin],
        order[(ordered_moments >= validation_origin) & (ordered_moments < test_origin)],
        order[ordered_moments >= test_origin],
    ]
    if any(part.empty for part in parts):
        raise MLValidationError(
            "SPLIT_EMPTY", "Las fechas no permiten tres particiones sin dividir el mismo instante"
        )

    availability_column = proposal.split.label_available_at_column
    availability = None
    purged = [order[:0], order[:0]]
    if availability_column is not None:
        availability = _dates(frame, availability_column, availability=True)
        if (availability < moments).any():
            raise MLValidationError(
                "LABEL_AVAILABLE_BEFORE_PREDICTION",
                "Un resultado figura disponible antes del momento elegido para predecir",
            )
        for index, origin in enumerate((validation_origin, test_origin)):
            eligible = availability.loc[parts[index]] < origin
            purged[index] = parts[index][~eligible]
            parts[index] = parts[index][eligible]
    # Un horizonte extenso puede eliminar una partición o una clase: no se rellena con futuro.
    for name, part in zip(("train", "validation", "test"), parts, strict=True):
        _validate_partition(name, y.loc[part])

    maxima = (
        [availability.loc[part].max().isoformat() for part in parts]
        if availability is not None
        else [None, None, None]
    )
    evidence = TemporalSplitEvidenceV1(
        time_column=column,
        label_available_at_column=availability_column,
        label_availability_verified=availability is not None,
        source_rows=len(frame),
        validation_origin=validation_origin.isoformat(),
        test_origin=test_origin.isoformat(),
        availability_rule="strictly_before_next_partition" if availability is not None else None,
        train_label_available_max=maxima[0],
        validation_label_available_max=maxima[1],
        test_label_available_max=maxima[2],
        purged_train_rows=len(purged[0]),
        purged_validation_rows=len(purged[1]),
        purged_train_index_sha256=_index_hash(purged[0]),
        purged_validation_index_sha256=_index_hash(purged[1]),
    )
    return (
        X.loc[parts[0]],
        X.loc[parts[1]],
        X.loc[parts[2]],
        y.loc[parts[0]],
        y.loc[parts[1]],
        y.loc[parts[2]],
        evidence,
    )
