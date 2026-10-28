"""V2-22: tabla por caso con variables y resultados separados en el tiempo."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core.analysis.snapshot import (
    ObservationCoverageV1,
    SnapshotError,
    SnapshotSpecV1,
    build_snapshot,
)
from core.data_engine import validate_and_profile

ROOT = Path(__file__).parents[2]
JOURNEYS = ROOT / "data" / "synthetic" / "recorridos"
STAGE_ORDER = ["inicio", "intermedio", "final"]
COVERAGE = {
    "observed_from": "2026-01-01T00:00:00Z",
    "observed_until": "2026-12-31T23:59:59Z",
    "complete": True,
}


def _covered_snapshot(events, spec, entities=None):
    """Estos fixtures sintéticos declaran explícitamente captura completa de 2026."""

    return build_snapshot(events, spec, entities, coverage=COVERAGE)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "2026-01-01", "inicio", "web", 10.0),
            ("A", "2026-01-05", "intermedio", "app", 20.0),
            ("A", "2026-01-10", "final", "corte", 999.0),
            ("A", "2026-01-12", "final", "web", 30.0),
            ("B", "2026-01-03", "inicio", "web", 5.0),
            ("B", "2026-01-18", "intermedio", "app", 15.0),
            ("B", "2026-01-25", "final", "app", 25.0),
            ("C", "2026-01-09", "inicio", "app", 8.0),
        ],
        columns=["case", "at", "stage", "channel", "amount"],
    )


def _full_spec() -> SnapshotSpecV1:
    stage_features = [
        {"op": "stages_reached", "stage": "stage", "order": STAGE_ORDER},
        {"op": "current_stage", "stage": "stage", "order": STAGE_ORDER},
        {"op": "days_in_current_stage", "stage": "stage", "order": STAGE_ORDER},
    ]
    return SnapshotSpecV1.model_validate(
        {
            "entity": "case",
            "moment": "at",
            "cutoff": "2026-01-10T00:00:00",
            "windows_days": [7, 30],
            "features": [
                {"op": "count_events", "window_days": 7},
                {"op": "days_since_last_event"},
                {"op": "last_value", "column": "channel"},
                *stage_features,
                {"op": "sum", "column": "amount", "window_days": 30},
                {"op": "mean", "column": "amount", "window_days": 30},
                {"op": "max", "column": "amount", "window_days": 30},
                {
                    "op": "count_category",
                    "column": "channel",
                    "value": "app",
                    "window_days": 30,
                },
            ],
            "outcome": [
                {
                    "op": "not_reached_stage",
                    "stage": {"column": "stage", "value": "final"},
                    "within_days": 5,
                    "output_column": "not_final_5d",
                },
                {"op": "inactive", "within_days": 5, "output_column": "inactive_5d"},
                {
                    "op": "reached_stage",
                    "stage": {"column": "stage", "value": "final"},
                    "within_days": 5,
                    "output_column": "reached_final_5d",
                },
                {
                    "op": "last_stage_reached",
                    "stage": "stage",
                    "within_days": 10,
                    "output_column": "last_stage_10d",
                },
                {
                    "op": "days_until_stage",
                    "stage": {"column": "stage", "value": "final"},
                    "output_column": "days_until_final",
                },
                {
                    "op": "future_sum",
                    "column": "amount",
                    "within_days": 10,
                    "output_column": "future_amount_10d",
                },
            ],
        }
    )


def test_every_feature_and_outcome_recipe_on_hand_built_example() -> None:
    entities = pd.DataFrame(
        {"case": ["A", "B", "C"], "fixed_score": [3, 2, 1], "fixed_group": ["x", "y", "x"]}
    )
    result = _covered_snapshot(_events(), _full_spec(), entities).dataframe.set_index("case")

    assert list(result.index) == ["A", "B", "C"]
    assert result.loc["A", "count_events_7d"] == 1
    assert result.loc["A", "days_since_last_event"] == 5
    assert result.loc["A", "last_channel"] == "app"
    assert result.loc["A", "stages_reached_stage"] == 2
    assert result.loc["A", "current_stage_stage"] == "intermedio"
    assert result.loc["A", "days_in_current_stage_stage"] == 5
    assert result.loc["A", "sum_amount_30d"] == 30
    assert result.loc["A", "mean_amount_30d"] == 15
    assert result.loc["A", "max_amount_30d"] == 20
    assert result.loc["A", "count_channel_30d"] == 1
    assert result.loc["A", "fixed_score"] == 3

    assert bool(result.loc["A", "not_final_5d"]) is False
    assert bool(result.loc["B", "not_final_5d"]) is True
    assert result["inactive_5d"].tolist() == [False, True, True]
    assert result["reached_final_5d"].tolist() == [True, False, False]
    assert result.loc["A", "last_stage_10d"] == "final"
    assert result.loc["B", "last_stage_10d"] == "intermedio"
    assert pd.isna(result.loc["C", "last_stage_10d"])
    assert result.loc["A", "days_until_final"] == 2
    assert result.loc["B", "days_until_final"] == 15
    assert pd.isna(result.loc["C", "days_until_final"])
    assert result["future_amount_10d"].tolist() == [30.0, 15.0, 0.0]


def test_point_in_time_guarantee_in_both_directions() -> None:
    events = _events()
    spec = _full_spec()
    original = _covered_snapshot(events, spec).dataframe.sort_values("case").reset_index(drop=True)
    feature_names = _covered_snapshot(events, spec).summary.feature_names
    outcome_names = _covered_snapshot(events, spec).summary.outcome_names

    changed_future = events.copy()
    future = pd.to_datetime(changed_future["at"]).gt(pd.Timestamp("2026-01-10"))
    changed_future.loc[future, "stage"] = "inicio"
    changed_future.loc[future, "channel"] = "changed"
    changed_future.loc[future, "amount"] = -12345.0
    changed_future.loc[future, "at"] = "2026-01-11"
    after_future_change = _covered_snapshot(changed_future, spec).dataframe.sort_values("case")
    assert_frame_equal(
        original[feature_names],
        after_future_change.reset_index(drop=True)[feature_names],
    )

    changed_history = events.copy()
    history = pd.to_datetime(changed_history["at"]).lt(pd.Timestamp("2026-01-10"))
    changed_history.loc[history, "stage"] = "final"
    changed_history.loc[history, "channel"] = "changed"
    changed_history.loc[history, "amount"] = 777.0
    changed_history.loc[history, "at"] = "2026-01-09"
    after_history_change = _covered_snapshot(changed_history, spec).dataframe.sort_values("case")
    for name in outcome_names:
        assert_series_equal(
            original[name],
            after_history_change.reset_index(drop=True)[name],
            check_names=True,
        )


@pytest.mark.parametrize(
    ("events_name", "entities_name", "entity", "moment", "stage", "order", "numeric"),
    [
        (
            "altas_eventos.csv",
            "altas_entidades.csv",
            "codigo_incorporacion",
            "momento_actividad",
            "paso_incorporacion",
            ["registro", "verificacion", "primera_compra", "segunda_compra"],
            "valor_actividad",
        ),
        (
            "solicitudes_eventos.csv",
            "solicitudes_entidades.csv",
            "folio_gestion",
            "marca_temporal",
            "estado_tramite",
            ["recibida", "asignada", "en_revision", "resuelta", "cerrada_sin_resolver"],
            "minutos_esfuerzo",
        ),
    ],
)
def test_synthetic_journeys_become_one_valid_row_per_entity(
    events_name: str,
    entities_name: str,
    entity: str,
    moment: str,
    stage: str,
    order: list[str],
    numeric: str,
) -> None:
    events = pd.read_csv(JOURNEYS / events_name)
    entities = pd.read_csv(JOURNEYS / entities_name)
    spec = {
        "entity": entity,
        "moment": moment,
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 1},
        "windows_days": [7, 30, 90],
        "features": [
            {"op": "count_events", "window_days": 7},
            {"op": "days_since_last_event"},
            {"op": "current_stage", "stage": stage, "order": order},
            {"op": "mean", "column": numeric, "window_days": 30},
        ],
        "outcome": [{"op": "inactive", "within_days": 30}],
    }

    result = _covered_snapshot(events, spec, entities)

    assert len(result.dataframe) == events[entity].nunique() == len(entities)
    assert result.dataframe[entity].is_unique
    assert result.summary.rows == len(result.dataframe)
    content = result.dataframe.to_csv(index=False).encode("utf-8")
    profile = validate_and_profile("snapshot.csv", content)
    assert profile.row_count == len(result.dataframe)
    assert profile.column_count == len(result.dataframe.columns)


def test_simple_model_beats_the_majority_proportion_on_not_reached_stage() -> None:
    events = pd.read_csv(JOURNEYS / "altas_eventos.csv")
    entities = pd.read_csv(JOURNEYS / "altas_entidades.csv")
    spec = {
        "entity": "codigo_incorporacion",
        "moment": "momento_actividad",
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 1},
        "windows_days": [7, 30],
        "features": [
            {"op": "count_events", "window_days": 7},
            {"op": "last_value", "column": "valor_actividad"},
            {"op": "mean", "column": "valor_actividad", "window_days": 30},
        ],
        "outcome": [
            {
                "op": "not_reached_stage",
                "stage": {"column": "paso_incorporacion", "value": "primera_compra"},
                "within_days": 30,
                "output_column": "did_not_reach",
            }
        ],
        "entity_attributes": ["indice_interes_inicial"],
    }
    table = _covered_snapshot(events, spec, entities).training_frame()
    columns = [
        "count_events_7d",
        "last_valor_actividad",
        "mean_valor_actividad_30d",
        "indice_interes_inicial",
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        table[columns],
        table["did_not_reach"].astype(int),
        test_size=0.3,
        random_state=22,
        stratify=table["did_not_reach"],
    )
    model = LogisticRegression(max_iter=1_000).fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    majority_proportion = y_test.value_counts(normalize=True).max()

    assert accuracy > majority_proportion


@pytest.mark.parametrize(
    "change",
    [
        {"windows_days": [-1]},
        {"features": [{"op": "count_events", "window_days": -1}]},
    ],
)
def test_contract_rejects_negative_windows(change: dict[str, object]) -> None:
    raw = {
        "entity": "case",
        "moment": "at",
        "cutoff": "2026-01-10",
        "windows_days": [7],
        "features": [{"op": "count_events", "window_days": 7}],
        "outcome": [{"op": "inactive", "within_days": 7}],
    }
    raw.update(change)

    with pytest.raises(ValidationError, match="greater than 0|positiva"):
        SnapshotSpecV1.model_validate(raw)


def test_clear_errors_cover_missing_columns_wrong_types_and_unknown_stages() -> None:
    with pytest.raises(SnapshotError) as missing:
        build_snapshot(_events(), _full_spec().model_copy(update={"entity": "missing"}))
    assert missing.value.code == "SNAPSHOT_COLUMN_NOT_FOUND"
    assert "missing" in missing.value.message

    wrong_type = _events().assign(amount="not numeric")
    with pytest.raises(SnapshotError) as type_error:
        build_snapshot(wrong_type, _full_spec())
    assert type_error.value.code == "SNAPSHOT_TYPE_MISMATCH"
    assert "numérica" in type_error.value.message

    outside = _events().copy()
    outside.loc[0, "stage"] = "unknown"
    with pytest.raises(SnapshotError) as stage_error:
        build_snapshot(outside, _full_spec())
    assert stage_error.value.code == "SNAPSHOT_STAGE_OUT_OF_ORDER"
    assert "unknown" in stage_error.value.message


def _maturity_spec() -> dict[str, object]:
    return {
        "entity": "case",
        "moment": "at",
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
        "windows_days": [7],
        "features": [{"op": "count_events", "window_days": 7}],
        "outcome": [
            {"op": "inactive", "within_days": 30, "output_column": "inactive"},
            {
                "op": "not_reached_stage",
                "stage": {"column": "stage", "value": "final"},
                "within_days": 30,
                "output_column": "not_finished",
            },
            {"op": "future_sum", "column": "amount", "within_days": 30},
        ],
    }


@pytest.mark.parametrize(
    ("coverage", "reason"),
    [
        (None, "COVERAGE_NOT_DECLARED"),
        ({**COVERAGE, "complete": False}, "COVERAGE_INCOMPLETE"),
        (
            {**COVERAGE, "observed_from": "2026-01-04T00:00:00Z"},
            "COVERAGE_STARTS_AFTER_CUTOFF",
        ),
        (
            {**COVERAGE, "observed_until": "2026-01-03T00:00:00Z"},
            "OUTCOME_WINDOW_NOT_MATURE",
        ),
    ],
)
def test_missing_follow_up_is_never_abandonment_inactivity_or_zero_sum(coverage, reason) -> None:
    events = _events().query("case == 'A'")
    result = build_snapshot(events, _maturity_spec(), coverage=coverage)

    assert result.dataframe[result.summary.outcome_names].isna().all().all()
    assert result.training_frame().empty
    assert result.readiness.total_rows == result.readiness.pending_rows == 1
    # Tres outcomes pendientes no se cuentan como tres casos.
    assert result.readiness.reason_counts == {reason: 1}
    assert result.row_readiness["pending_reasons"].tolist() == [[reason]]
    assert result.summary.outcome_summaries["inactive"].proportion_true is None


def test_maturity_is_per_entity_and_waits_even_for_an_early_positive() -> None:
    events = pd.DataFrame(
        [
            ("old", "2026-01-01", "inicio", 1.0),
            ("recent", "2026-01-20", "inicio", 1.0),
            ("recent", "2026-01-23", "final", 10.0),
        ],
        columns=["case", "at", "stage", "amount"],
    )
    coverage = {**COVERAGE, "observed_until": "2026-02-02T00:00:00Z"}
    result = build_snapshot(events, _maturity_spec(), coverage=coverage)

    # El resultado del caso antiguo termina exactamente al final de la cobertura declarada.
    ready = result.training_frame()
    assert ready["case"].tolist() == ["old"]
    assert bool(ready.iloc[0]["not_finished"]) is True
    assert ready.iloc[0]["outcome_future_sum_amount_30d"] == 0
    assert result.dataframe.loc[1, result.summary.outcome_names].isna().all()
    assert result.readiness.eligible_rows == result.readiness.pending_rows == 1
    assert "eligible_for_training" not in ready.columns
    assert "pending_reasons" not in ready.columns


def test_no_coverage_is_inferred_from_a_late_event_and_features_remain_available() -> None:
    without = build_snapshot(_events(), _maturity_spec())
    covered = build_snapshot(_events(), _maturity_spec(), coverage=COVERAGE)

    assert without.readiness.eligible_rows == 0
    assert covered.readiness.eligible_rows == 3
    assert_frame_equal(
        without.dataframe[without.summary.feature_names],
        covered.dataframe[covered.summary.feature_names],
    )
    assert without.readiness.model_dump(mode="json")["coverage"] is None


def test_unbounded_time_to_event_remains_censored_until_event_is_observed() -> None:
    spec = _full_spec().model_dump(mode="json")
    spec["outcome"] = [
        {
            "op": "days_until_stage",
            "stage": {"column": "stage", "value": "final"},
            "output_column": "waiting_days",
        }
    ]
    # El archivo contiene también un evento posterior a la cobertura: no se usa aún.
    limited = build_snapshot(
        _events(), spec, coverage={**COVERAGE, "observed_until": "2026-01-20T00:00:00Z"}
    )
    assert limited.training_frame()["case"].tolist() == ["A"]
    assert limited.dataframe.set_index("case").loc[["B", "C"], "waiting_days"].isna().all()
    assert limited.readiness.reason_counts == {"OUTCOME_NOT_OBSERVED": 2}
    full = build_snapshot(_events(), spec, coverage=COVERAGE)
    assert full.training_frame()["case"].tolist() == ["A", "B"]


def test_coverage_rejects_reversed_dates_and_normalizes_timezones() -> None:
    with pytest.raises(SnapshotError) as error:
        build_snapshot(
            _events(),
            _maturity_spec(),
            coverage={**COVERAGE, "observed_from": "2027-01-01"},
        )
    assert error.value.code == "SNAPSHOT_COVERAGE_INVALID"
    coverage = ObservationCoverageV1.model_validate(
        {"observed_from": "2026-01-01T00:00:00-03:00", "observed_until": "2026-01-02"}
    )
    assert coverage.model_dump(mode="json")["observed_from"] == "2026-01-01T03:00:00Z"
    assert coverage.complete is False
