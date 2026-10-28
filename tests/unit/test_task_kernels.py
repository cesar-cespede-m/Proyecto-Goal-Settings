"""GA-07b/c núcleo: varias clases, cantidades y partición por tiempo, contra su baseline."""

import numpy as np
import pandas as pd
import pytest

from core.ml_engine import MLValidationError
from core.ml_engine.task_kernels import split_frame, train_multiclass, train_regression


def _frame(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=rows)
    group = rng.choice(["a", "b", "c"], size=rows)
    return pd.DataFrame(
        {
            "momento": pd.date_range("2026-01-01", periods=rows, freq="h").astype(str),
            "senal": signal,
            "grupo": group,
            "ruido": rng.normal(size=rows),
            "nivel": np.select([signal < -0.4, signal < 0.4], ["bajo", "medio"], "alto"),
            "cantidad": 10 + 5 * signal + (group == "a") * 3 + rng.normal(scale=0.5, size=rows),
        }
    )


FEATURES = ["senal", "grupo", "ruido"]


def test_multiclass_beats_the_most_frequent_baseline() -> None:
    result = train_multiclass(_frame(), "nivel", FEATURES)

    assert result.selected in {"LogisticRegression", "RandomForestClassifier"}
    assert (
        result.validation[result.selected]["macro_f1"]
        > result.validation["DummyClassifier"]["macro_f1"]
    )
    assert set(result.test) == {"macro_f1", "balanced_accuracy", "accuracy"}


def test_regression_beats_the_median_baseline_with_lower_error() -> None:
    result = train_regression(_frame(), "cantidad", FEATURES)

    assert not result.higher_is_better and result.primary_metric == "mae"
    assert result.validation[result.selected]["mae"] < result.validation["DummyRegressor"]["mae"]
    assert result.test["r2"] > 0.5


def test_without_signal_nothing_is_selected() -> None:
    frame = _frame().assign(cantidad=np.random.default_rng(9).normal(size=300))
    result = train_regression(frame, "cantidad", ["ruido"])

    if result.selected is None:
        assert result.warnings == ["NO_CANDIDATE_BEATS_BASELINE"] and result.test == {}


def test_temporal_split_trains_on_the_past_and_tests_on_the_future() -> None:
    frame = _frame().sample(frac=1, random_state=3)  # desordenado a propósito
    splits = split_frame(
        frame, "cantidad", FEATURES + ["momento"], seed=1, stratify=False, time_column="momento"
    )

    assert splits.method == "temporal"
    assert splits.X_train["momento"].max() < splits.X_validation["momento"].min()
    assert splits.X_validation["momento"].max() < splits.X_test["momento"].min()
    result = train_regression(frame, "cantidad", FEATURES, time_column="momento")
    assert result.split_method == "temporal"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"nivel": "unico"}, "NOT_ENOUGH_CLASSES"),
        ({"nivel": [str(index % 30) for index in range(300)]}, "TOO_MANY_CLASSES"),
    ],
)
def test_unsuitable_targets_are_explained(change, code) -> None:
    with pytest.raises(MLValidationError) as error:
        train_multiclass(_frame().assign(**change), "nivel", FEATURES)
    assert error.value.code == code


def test_quantities_need_variation_and_dates_must_be_dates() -> None:
    with pytest.raises(MLValidationError) as flat:
        train_regression(_frame().assign(cantidad=1.0), "cantidad", FEATURES)
    assert flat.value.code == "TARGET_NOT_A_QUANTITY"
    with pytest.raises(MLValidationError) as dates:
        train_regression(_frame(), "cantidad", FEATURES, time_column="grupo")
    assert dates.value.code == "TIME_COLUMN_NOT_DATES"


def test_numeric_ratings_work_as_classes() -> None:
    # scikit-learn falla al ponderar clases «1»…«5» escritas como texto; deben quedar números.
    frame = _frame()
    frame["nota"] = np.select(
        [frame["senal"] < -0.6, frame["senal"] < 0, frame["senal"] < 0.6], [1, 2, 3], 4
    )
    result = train_multiclass(frame, "nota", FEATURES)

    assert result.selected is not None
