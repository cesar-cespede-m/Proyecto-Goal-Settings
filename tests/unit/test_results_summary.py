from app.ui.formatting import build_results_summary


def test_summary_uses_saved_metrics_and_confirmed_event() -> None:
    summary = build_results_summary(
        has_recommendation=True,
        recall=0.647,
        precision=0.612,
        baseline_recall=0.0,
        positive_class="abandono",
    )

    assert summary == (
        "En la prueba final, de cada 10 casos de «abandono», el modelo avisó de unos 6 "
        "(64,7 %). De cada 10 avisos, unos 6 fueron correctos (61,2 %). "
        "En los datos para comparar, la referencia simple no detectó ninguno "
        "de esos casos de «abandono»."
    )


def test_summary_falls_back_when_event_is_not_available() -> None:
    summary = build_results_summary(
        has_recommendation=True,
        recall=0.5,
        precision=0.8,
    )

    assert "5 (50,0 %)" in summary
    assert "8 fueron correctos (80,0 %)" in summary
    assert "casos del evento" in summary


def test_summary_explains_when_there_is_no_recommendation() -> None:
    assert build_results_summary(has_recommendation=False) == (
        "No hay un modelo recomendado. Ninguna alternativa le ganó a la referencia "
        "simple con las condiciones acordadas."
    )


def test_summary_names_the_target_column_when_known() -> None:
    summary = build_results_summary(
        has_recommendation=True,
        recall=0.647,
        precision=0.605,
        positive_class="sí",
        target_column="entregado_tarde",
    )
    assert "de cada 10 casos en que «entregado_tarde» fue «sí»" in summary
    assert "64,7 %" in summary and "60,5 %" in summary
