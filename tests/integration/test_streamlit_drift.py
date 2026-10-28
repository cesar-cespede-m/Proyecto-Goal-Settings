"""MON-01: la sección «¿Los datos cambiaron?» se lee en lenguaje común."""

from streamlit.testing.v1 import AppTest


def _app(level: str) -> AppTest:
    def render(level: str) -> None:
        from app.ui.extension_views import _render_drift
        from core.ml_engine.drift import ColumnDriftV1, DriftReportV1

        column = ColumnDriftV1(
            column="temperatura_aire_c",
            kind="numeric",
            psi=8.2 if level == "REVIEW" else 0.01,
            out_of_range_rate=1.0 if level == "REVIEW" else 0.0,
            reference_missing_rate=0.0,
            current_missing_rate=0.0,
            reference_mean=25.0,
            current_mean=77.1 if level == "REVIEW" else 25.1,
            level="REVIEW" if level == "REVIEW" else "STABLE",
            reasons=["DISTRIBUTION_SHIFT", "OUT_OF_TRAINING_RANGE"] if level == "REVIEW" else [],
        )
        _render_drift(
            DriftReportV1(
                reference_rows=4000,
                current_rows=1000,
                level=level,
                max_psi=column.psi,
                columns=[column],
            )
        )

    return AppTest.from_function(render, args=(level,)).run(timeout=30)


def test_large_change_is_explained_with_the_affected_column() -> None:
    app = _app("REVIEW")

    assert not app.exception
    assert "cambiaron mucho" in app.error[0].value
    table = app.dataframe[0].value
    assert table.iloc[0]["Columna"] == "temperatura_aire_c"
    assert "fuera del rango visto al entrenar" in table.iloc[0]["Qué cambió"]
    assert table.iloc[0]["Nivel"] == "Cambió mucho"
    guided_text = next(
        item.value for item in app.caption if item.value.startswith("Se compara cada columna")
    )
    assert "cómo se reparten sus valores" in guided_text
    assert "índice de estabilidad" not in guided_text and "0,10" not in guided_text
    assert any(item.label == "Detalle técnico" for item in app.expander)
    assert any(
        "índice de estabilidad (PSI)" in item.value and "0,10" in item.value for item in app.caption
    )


def test_stable_data_shows_reassurance_without_a_table() -> None:
    app = _app("STABLE")

    assert not app.exception
    assert "se parecen" in app.success[0].value
    assert not app.dataframe
