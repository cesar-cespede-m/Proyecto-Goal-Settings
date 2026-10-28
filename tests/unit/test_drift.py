"""MON-01: cambios en los datos entre el entrenamiento y cada uso del modelo."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ui.extension_views import drift_column_text
from core.ml_engine.drift import MIN_ROWS_FOR_DRIFT, compute_drift
from tests.support.synthetic_model import VARIANTS, approved_maintenance_model

RNG = np.random.default_rng(7)


def _frame(rows: int, *, mean: float = 25.0, kind: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperatura": RNG.normal(mean, 2, rows),
            "tipo": RNG.choice(["A", "B", kind], rows),
        }
    )


def _drift(reference: pd.DataFrame, current: pd.DataFrame):
    return compute_drift(
        reference, current, numeric_columns=["temperatura"], categorical_columns=["tipo"]
    )


def test_same_distribution_is_stable() -> None:
    report = _drift(_frame(4000), _frame(1000))

    assert report.level == "STABLE"
    assert report.max_psi < 0.1
    assert all(column.reasons == [] for column in report.columns)


def test_known_shift_and_unit_change_need_review() -> None:
    reference = _frame(4000)
    shifted = _drift(reference, _frame(1000, mean=27.5))
    fahrenheit = reference.head(1000).assign(temperatura=lambda x: x.temperatura * 1.8 + 32)
    units = _drift(reference, fahrenheit)

    assert shifted.columns[0].level == "REVIEW"
    assert shifted.columns[0].psi > 0.25
    assert units.columns[0].out_of_range_rate == 1.0
    assert "OUT_OF_TRAINING_RANGE" in units.columns[0].reasons


def test_new_categories_and_missing_values_are_reported() -> None:
    reference = _frame(4000)
    current = _frame(1000, kind="Z")
    current.loc[current.index[:400], "temperatura"] = np.nan
    report = _drift(reference, current)
    numeric, categorical = report.columns

    assert categorical.new_category_rate > 0.25
    assert "NEW_CATEGORIES" in categorical.reasons
    assert numeric.current_missing_rate == 0.4
    assert "MISSING_RATE_CHANGE" in numeric.reasons
    assert report.level == "REVIEW"


def test_few_rows_are_not_enough_to_conclude() -> None:
    report = _drift(_frame(4000), _frame(MIN_ROWS_FOR_DRIFT - 1, mean=40))

    assert report.level == "INSUFFICIENT_DATA"


def test_report_never_contains_values_or_category_names() -> None:
    reference = _frame(4000)
    current = _frame(1000, kind="CATEGORIA-SECRETA")
    serialized = _drift(reference, current).model_dump_json()

    assert "CATEGORIA-SECRETA" not in serialized
    assert '"A"' not in serialized and '"B"' not in serialized


def test_plain_language_explains_each_reason() -> None:
    reference = _frame(4000)
    fahrenheit = reference.head(1000).assign(temperatura=lambda x: x.temperatura * 1.8 + 32)
    text = drift_column_text(_drift(reference, fahrenheit).columns[0])

    assert text.startswith("100,0 % de los valores queda fuera del rango")
    assert "el promedio pasó de 25," in text
    assert "PSI" not in text and "DISTRIBUTION_SHIFT" not in text


def test_scoring_records_drift_per_batch_without_rows(tmp_path: Path) -> None:
    repository, scoring, run_id = approved_maintenance_model(tmp_path)
    folder = VARIANTS / "mantenimiento_maquinas"
    expected = {
        "uso_01_estable.csv": "STABLE",
        "uso_02_cambio_leve.csv": "WATCH",
        "uso_03_cambio_fuerte.csv": "REVIEW",
        "uso_05_pocos_casos.csv": "INSUFFICIENT_DATA",
        "uso_06_categoria_nueva.csv": "REVIEW",
        "uso_07_muchos_vacios.csv": "REVIEW",
        "uso_08_unidades_cambiadas.csv": "REVIEW",
        "uso_09_columnas_reordenadas_y_extra.csv": "STABLE",
    }
    for name, level in expected.items():
        result = scoring.score(
            run_id, filename=name, content=(folder / name).read_bytes(), actor_label="Prueba"
        )
        assert result.drift.level == level, name

    events = repository.list_scorings(run_id)
    assert len(events) == len(expected)
    latest = json.loads(events[0]["detail_json"])
    assert latest["drift"]["level"] == "STABLE"
    stored = " ".join(event["detail_json"] for event in events)
    assert "MAQ-" not in stored
    assert '"D"' not in stored  # categoría nueva del lote 06, nunca su nombre
