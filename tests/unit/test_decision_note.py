"""Recorrido 22/10: el simulador decía «84 de 100» y los resultados «6 de cada 10»."""

from app.ui.decision_view import precision_gap_note


def test_different_selection_is_not_a_claim_of_future_improvement() -> None:
    note = precision_gap_note(0.84, 0.605, 100)
    assert note is not None
    assert "6 de cada 10 en validación" in note and "atender 100 casos" in note
    assert "no garantiza" in note


def test_attending_more_cases_explains_the_lower_figure() -> None:
    note = precision_gap_note(0.40, 0.605, 400)
    assert note is not None and "conjunto distinto" in note
    assert "no garantiza" in note


def test_no_note_when_figures_are_close_or_unknown() -> None:
    assert precision_gap_note(0.62, 0.605, 150) is None
    assert precision_gap_note(0.84, None, 100) is None
