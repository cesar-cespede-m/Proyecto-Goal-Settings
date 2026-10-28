"""MV-14: promesa congelada por caso, sin falsos negativos por cobertura incompleta."""

import pandas as pd

from core.analysis.snapshot import build_snapshot


def test_frozen_per_case_deadline_and_pending_outcomes():
    events = pd.DataFrame(
        [
            ("ON_TIME", "2026-01-01", "order", "2026-01-10"),
            ("ON_TIME", "2026-01-08", "delivered", "2026-01-10"),
            ("LATE", "2026-01-01", "order", "2026-01-10"),
            ("LATE", "2026-01-12", "delivered", "2026-01-20"),
            ("OPEN", "2026-01-01", "order", "2026-01-10"),
            ("PENDING", "2026-01-01", "order", "2026-01-30"),
            ("CONFLICT", "2026-01-01", "order", "2026-01-10"),
            ("CONFLICT", "2026-01-02", "activity", "2026-01-12"),
        ],
        columns=["order_id", "moment", "stage", "promised_at"],
    )
    recipe = {
        "entity": "order_id",
        "moment": "moment",
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
        "windows_days": [2],
        "features": [{"op": "count_events", "window_days": 2, "output_column": "prior_events"}],
        "outcome": [
            {
                "op": "deadline_not_reached_stage",
                "stage": {"column": "stage", "value": "delivered"},
                "deadline_column": "promised_at",
                "output_column": "incumplimiento_promesa",
            }
        ],
        "label_available_at_output_column": "promised_at",
    }
    built = build_snapshot(
        events,
        recipe,
        coverage={
            "observed_from": "2026-01-01T00:00:00Z",
            "observed_until": "2026-01-20T00:00:00Z",
            "complete": True,
        },
    )
    rows = built.dataframe.set_index("order_id")
    assert not bool(rows.loc["ON_TIME", "incumplimiento_promesa"])
    assert bool(rows.loc["LATE", "incumplimiento_promesa"])
    assert bool(rows.loc["OPEN", "incumplimiento_promesa"])
    assert pd.isna(rows.loc["PENDING", "incumplimiento_promesa"])
    assert pd.isna(rows.loc["CONFLICT", "incumplimiento_promesa"])
    assert rows.loc["LATE", "promised_at"] == pd.Timestamp("2026-01-10", tz="UTC")
    assert set(built.training_frame()["order_id"]) == {"ON_TIME", "LATE", "OPEN"}
    assert built.readiness.reason_counts == {
        "DEADLINE_CHANGED_BEFORE_CUTOFF": 1,
        "OUTCOME_WINDOW_NOT_MATURE": 1,
    }
