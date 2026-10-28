"""V2-20: embudo y tiempo entre etapas sobre recorridos sintéticos."""

from pathlib import Path

import pandas as pd

from core.analysis.funnel import funnel

DATA = Path(__file__).parents[2] / "data" / "synthetic" / "recorridos"


def test_funnel_counts_where_cases_stay() -> None:
    frame = pd.read_csv(DATA / "altas_eventos.csv")
    entity, moment, stage = frame.columns[:3]
    order = ["registro", "verificacion", "primera_compra", "segunda_compra"]
    result = funnel(frame, entity=entity, moment=moment, stage=stage, order=order)

    reached = [step.reached for step in result.steps]
    assert reached == sorted(reached, reverse=True)  # nadie llega más lejos sin pasar antes
    assert reached[0] == 3000 and result.steps[0].share_of_start == 1
    assert result.biggest_drop in order[:-1]
    assert all(
        step.median_hours_to_next is None or step.median_hours_to_next >= 0 for step in result.steps
    )
    assert "Donde más se quedan" in result.summary


def test_tiny_handmade_funnel() -> None:
    frame = pd.DataFrame(
        {
            "caso": ["a", "a", "b", "c", "c", "c"],
            "cuando": [
                "2026-01-01 10:00",
                "2026-01-01 12:00",
                "2026-01-02 09:00",
                "2026-01-03 08:00",
                "2026-01-03 09:00",
                "2026-01-03 11:00",
            ],
            "paso": ["uno", "dos", "uno", "uno", "dos", "tres"],
        }
    )
    result = funnel(
        frame, entity="caso", moment="cuando", stage="paso", order=["uno", "dos", "tres"]
    )

    assert [s.reached for s in result.steps] == [3, 2, 1]
    assert result.steps[0].stayed_here == 1 and result.steps[1].stayed_here == 1
    assert result.steps[0].median_hours_to_next == 1.5


def test_cohorts_group_cases_by_their_first_month() -> None:
    from core.analysis.funnel import cohorts

    frame = pd.DataFrame(
        {
            "caso": ["a", "a", "b", "c", "c"],
            "cuando": ["2026-01-05", "2026-01-09", "2026-01-20", "2026-02-02", "2026-02-03"],
            "paso": ["uno", "dos", "uno", "uno", "dos"],
        }
    )
    result = cohorts(frame, entity="caso", moment="cuando", stage="paso", order=["uno", "dos"])

    assert [row.period for row in result.rows] == ["2026-01", "2026-02"]
    assert result.rows[0].entities == 2 and result.rows[0].reached_share["dos"] == 0.5
    assert result.rows[1].reached_share["dos"] == 1.0
    assert "2026-02" in result.summary


def test_suggestion_targets_the_stage_after_the_biggest_drop() -> None:
    from core.analysis.funnel import suggest_outcome

    frame = pd.read_csv(DATA / "altas_eventos.csv")
    entity, moment, stage = frame.columns[:3]
    order = ["registro", "verificacion", "primera_compra", "segunda_compra"]
    result = funnel(frame, entity=entity, moment=moment, stage=stage, order=order)
    suggestion = suggest_outcome(result)

    assert suggestion is not None
    assert order.index(suggestion.target_stage) == order.index(result.biggest_drop) + 1
    assert suggestion.within_days >= 7 and "se quedan" in suggestion.reason
