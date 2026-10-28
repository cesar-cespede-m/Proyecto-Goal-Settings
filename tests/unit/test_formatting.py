from datetime import UTC, datetime

from app.ui.formatting import (
    format_local_datetime,
    format_pct,
    format_pct_range,
    importance_table,
    round_per_ten,
    selection_reason,
)


def test_format_pct_uses_spanish_decimal_separator() -> None:
    assert format_pct(0.647) == "64,7 %"


def test_format_pct_handles_missing_value() -> None:
    assert format_pct(None) == "No disponible"


def test_format_pct_range_formats_both_limits() -> None:
    assert format_pct_range(0.573, 0.719) == "entre 57,3 % y 71,9 %"


def test_round_per_ten_uses_round() -> None:
    assert round_per_ten(0.64) == 6
    assert round_per_ten(0.66) == 7


def test_local_datetime_converts_stored_utc_to_local_time() -> None:
    expected = datetime(2026, 10, 22, 3, 25, tzinfo=UTC).astimezone()
    assert format_local_datetime("2026-10-22 03:25:34") == f"{expected:%d/%m/%Y, %H:%M}"
    assert "UTC" not in format_local_datetime("2026-10-22T03:25:34+00:00")
    assert format_local_datetime(None) == "fecha no disponible"
    assert format_local_datetime("ayer") == "ayer"


def test_selection_reason_names_the_metric_and_the_rivals() -> None:
    text = selection_reason(
        "alternativa de reglas combinadas",
        "de los eventos reales, cuántos detecta",
        0.692,
        [("alternativa lineal", 0.641)],
    )
    assert "porque es la que mejor cumple" in text
    assert "69,2 %" in text and "64,1 % de alternativa lineal" in text


def test_selection_reason_never_claims_a_lead_it_does_not_have() -> None:
    text = selection_reason("alternativa lineal", "detecta", 0.5, [("otra", 0.7)])
    assert text == "Se recomienda **alternativa lineal**, según las reglas que aprobaste."


def test_importance_table_is_sorted_relative_and_keeps_full_names() -> None:
    table = importance_table(
        [
            {"feature": "una_columna_con_nombre_largo", "importance_mean": 0.12},
            {"feature": "b", "importance_mean": 0.40},
            {"feature": "c", "importance_mean": -0.01},
        ]
    )
    assert list(table["Dato"]) == ["b", "una_columna_con_nombre_largo", "c"]
    assert list(table["Peso frente al que más pesa"]) == [100, 30, 0]
