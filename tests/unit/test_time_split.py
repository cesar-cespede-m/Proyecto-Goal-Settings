"""GA-07e: validar por fechas; lo antiguo entrena y lo reciente prueba, en todos los tipos."""

import pandas as pd
import pytest
from pydantic import ValidationError

from core.domain.canonical import canonical_hash, canonical_json
from core.domain.models import StrategyPayload
from core.domain.strategy import StrategyProposalV1
from core.ml_engine import MLValidationError, split_dataset
from core.ml_engine.models import SplitSummary
from tests.unit.test_ml_engine import _frame, _proposal


def _temporal(proposal: StrategyProposalV1, column: str = "momento") -> StrategyProposalV1:
    value = proposal.model_dump(mode="json")
    value["split"] = {**value["split"], "method": "temporal", "time_column": column}
    return StrategyProposalV1.model_validate(value)


def _dated(rows: int = 120) -> pd.DataFrame:
    frame = _frame(rows)
    frame["momento"] = pd.date_range("2026-01-01", periods=rows, freq="D").astype(str)
    return frame.sample(frac=1, random_state=4)  # desordenado a propósito


def test_binary_temporal_split_keeps_time_order() -> None:
    frame = _dated()
    splits = split_dataset(frame, _temporal(_proposal()))
    moments = pd.to_datetime(frame["momento"])

    assert moments.loc[splits.X_train.index].max() < moments.loc[splits.X_validation.index].min()
    assert moments.loc[splits.X_validation.index].max() < moments.loc[splits.X_test.index].min()
    assert len(splits.X_test) == 24


def test_temporal_requires_a_date_column_and_only_then() -> None:
    value = _proposal().model_dump(mode="json")
    value["split"] = {**value["split"], "method": "temporal"}
    with pytest.raises(ValidationError):
        StrategyProposalV1.model_validate(value)
    with pytest.raises(MLValidationError) as error:
        split_dataset(_dated().assign(momento="no es fecha"), _temporal(_proposal()))
    assert error.value.code == "TIME_COLUMN_NOT_DATES"


def test_existing_strategies_keep_random_split() -> None:
    assert _proposal().split.method == "stratified_random"
    assert _proposal().split.time_column is None


def _with_availability(proposal: StrategyProposalV1 | None = None) -> StrategyProposalV1:
    value = _temporal(proposal or _proposal()).model_dump(mode="json")
    value["split"]["label_available_at_column"] = "resultado_disponible"
    value["excluded_columns"].append("resultado_disponible")
    return StrategyProposalV1.model_validate(value)


def _available(rows: int = 120, delay: int = 7) -> pd.DataFrame:
    frame = _dated(rows)
    frame["resultado_disponible"] = (
        pd.to_datetime(frame["momento"]) + pd.Timedelta(days=delay)
    ).astype(str)
    return frame


def test_purges_labels_unavailable_at_next_prediction_origin_and_records_evidence() -> None:
    frame = _available()
    result = split_dataset(frame, _with_availability())
    evidence = result.summary.temporal
    assert evidence is not None
    labels = pd.to_datetime(frame["resultado_disponible"], utc=True)
    assert labels.loc[result.X_train.index].max() < pd.Timestamp(evidence.validation_origin)
    assert labels.loc[result.X_validation.index].max() < pd.Timestamp(evidence.test_origin)
    assert (len(result.X_train), len(result.X_validation), len(result.X_test)) == (65, 17, 24)
    assert evidence.label_availability_verified
    assert evidence.availability_rule == "strictly_before_next_partition"
    assert evidence.purged_train_rows == evidence.purged_validation_rows == 7
    assert evidence.purged_train_index_sha256 == canonical_hash(
        sorted(str(index) for index in range(65, 72))
    )
    assert evidence.purged_validation_index_sha256 == canonical_hash(
        sorted(str(index) for index in range(89, 96))
    )
    assert evidence.source_rows == (
        len(result.X_train)
        + len(result.X_validation)
        + len(result.X_test)
        + evidence.purged_train_rows
        + evidence.purged_validation_rows
    )
    assert "resultado_disponible" not in result.X_train.columns
    assert result.summary.method == "temporal"
    assert result.summary == split_dataset(frame, _with_availability()).summary
    assert SplitSummary.model_validate(result.summary.model_dump()) == result.summary


def test_legacy_temporal_split_does_not_claim_label_availability() -> None:
    result = split_dataset(_dated(), _temporal(_proposal()))
    assert result.summary.method == "temporal"
    evidence = result.summary.temporal
    assert evidence is not None
    assert not evidence.label_availability_verified
    assert evidence.label_available_at_column is None
    assert evidence.availability_rule is None
    assert evidence.train_label_available_max is None
    assert evidence.purged_train_rows == evidence.purged_validation_rows == 0


def test_time_ties_stay_together_even_when_fraction_cut_crosses_them() -> None:
    frame = _frame()
    frame["momento"] = [
        str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=index // 3))
        for index in range(len(frame))
    ]
    frame["resultado_disponible"] = frame["momento"]
    value = _with_availability().model_dump(mode="json")
    value["split"].update(train=0.59, validation=0.2, test=0.21)
    result = split_dataset(frame, StrategyProposalV1.model_validate(value))
    dates = pd.to_datetime(frame["momento"])
    assert dates.loc[result.X_train.index].max() < dates.loc[result.X_validation.index].min()
    assert dates.loc[result.X_validation.index].max() < dates.loc[result.X_test.index].min()
    assert (len(result.X_train), len(result.X_validation), len(result.X_test)) == (69, 24, 27)


def test_mixed_timezone_representations_use_utc_instants() -> None:
    frame = _available()
    for column in ("momento", "resultado_disponible"):
        parsed = pd.to_datetime(frame[column], utc=True)
        frame[column] = [
            moment.tz_convert("America/Santiago").isoformat() if index % 2 else moment.isoformat()
            for index, moment in enumerate(parsed)
        ]
    expected = split_dataset(_available(), _with_availability()).summary
    actual = split_dataset(frame, _with_availability()).summary
    assert actual == expected


@pytest.mark.parametrize("invalid", [None, "not-a-date", 123])
def test_invalid_or_missing_availability_is_rejected(invalid: object) -> None:
    frame = _available()
    frame["resultado_disponible"] = frame["resultado_disponible"].astype(object)
    frame.loc[0, "resultado_disponible"] = invalid
    with pytest.raises(MLValidationError) as error:
        split_dataset(frame, _with_availability())
    assert error.value.code == "LABEL_AVAILABILITY_NOT_DATES"


def test_availability_before_prediction_is_rejected() -> None:
    with pytest.raises(MLValidationError) as error:
        split_dataset(_available(delay=-1), _with_availability())
    assert error.value.code == "LABEL_AVAILABLE_BEFORE_PREDICTION"


def test_missing_declared_availability_column_is_rejected() -> None:
    with pytest.raises(MLValidationError) as error:
        split_dataset(_dated(), _with_availability())
    assert error.value.code == "LABEL_AVAILABILITY_NOT_FOUND"


def test_empty_partition_after_purge_blocks_training() -> None:
    with pytest.raises(MLValidationError) as error:
        split_dataset(_available(delay=300), _with_availability())
    assert error.value.code == "SPLIT_EMPTY"


def test_missing_class_after_purge_blocks_training() -> None:
    frame = _available(delay=0)
    frame.loc[(frame.index < 72) & (frame["target"] == 1), "resultado_disponible"] = "2027-01-01"
    with pytest.raises(MLValidationError) as error:
        split_dataset(frame, _with_availability())
    assert error.value.code == "SPLIT_CLASS_MISSING"


def test_single_instant_cannot_be_split_across_partitions() -> None:
    with pytest.raises(MLValidationError) as error:
        split_dataset(_dated().assign(momento="2026-01-01"), _temporal(_proposal()))
    assert error.value.code == "SPLIT_EMPTY"


@pytest.mark.parametrize(
    ("rows", "fractions"),
    [
        (2, {"train": 0.6, "validation": 0.2, "test": 0.2}),
        (120, {"train": 0.001, "validation": 0.799, "test": 0.2}),
        (120, {"train": 0.6, "validation": 0.4000000001, "test": 1e-12}),
    ],
)
def test_tiny_partitions_and_fraction_rounding_return_controlled_error(rows, fractions) -> None:
    value = _temporal(_proposal()).model_dump(mode="json")
    value["split"].update(fractions)
    with pytest.raises(MLValidationError) as error:
        split_dataset(_dated(rows), StrategyProposalV1.model_validate(value))
    assert error.value.code == "SPLIT_EMPTY"


def test_duplicate_dataframe_index_is_rejected_before_partitioning() -> None:
    frame = _available()
    frame.index = [0] * len(frame)
    with pytest.raises(MLValidationError) as error:
        split_dataset(frame, _with_availability())
    assert error.value.code == "DATASET_INDEX_NOT_UNIQUE"


@pytest.mark.parametrize("where", ["feature", "target", "random", "same_column"])
def test_availability_metadata_cannot_leak_into_features_or_wrong_configuration(where: str) -> None:
    value = _with_availability().model_dump(mode="json")
    if where == "feature":
        value["split"]["label_available_at_column"] = value["feature_columns"][0]
    elif where == "target":
        value["split"]["label_available_at_column"] = value["target_column"]
    elif where == "same_column":
        value["split"]["label_available_at_column"] = value["split"]["time_column"]
    else:
        value["split"].pop("time_column")
        value["split"]["method"] = "stratified_random"
    with pytest.raises(ValidationError):
        StrategyProposalV1.model_validate(value)


@pytest.mark.parametrize("task", ["multiclass_classification", "regression"])
def test_unimplemented_task_kernels_reject_availability_instead_of_ignoring_it(task: str) -> None:
    value = _with_availability().model_dump(mode="json")
    value.update(task_type=task, positive_class=None)
    if task == "regression":
        value.update(
            primary_metric="mae", candidates=["DummyRegressor", "Ridge", "RandomForestRegressor"]
        )
    else:
        value["primary_metric"] = "macro_f1"
    with pytest.raises(ValidationError, match="purga por disponibilidad"):
        StrategyProposalV1.model_validate(value)


def test_strategy_storage_accepts_availability_and_absent_field_preserves_legacy_json() -> None:
    current = _with_availability()
    stored = StrategyPayload.model_validate(current.model_dump(mode="json"))
    assert stored.split.label_available_at_column == "resultado_disponible"
    legacy = _proposal().model_dump(mode="json")
    assert "label_available_at_column" not in legacy["split"]
    assert canonical_json(StrategyProposalV1.model_validate(legacy)) == canonical_json(legacy)
    assert canonical_json(StrategyPayload.model_validate(legacy)) == canonical_json(legacy)
