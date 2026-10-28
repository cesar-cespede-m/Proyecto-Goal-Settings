"""MON-02: salud del modelo con reglas fijas a lo largo de sus usos."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.ui.extension_views import health_reason_text
from core.ml_engine.health import assess_model_health
from tests.support.synthetic_model import VARIANTS, approved_maintenance_model


def _use(drift: str | None = "STABLE", outcomes: int | None = None, **follow) -> dict:
    detail: dict = {"rows_scored": 500, "alerts": 50}
    if drift:
        detail["drift"] = {
            "level": drift,
            "max_psi": 0.5,
            "columns": [{"column": "temperatura", "level": drift}],
        }
    if outcomes:
        detail["follow_up"] = {"rows_with_outcome": outcomes, **follow}
    return detail


def _assess(*details: dict):
    return assess_model_health(
        [(f"2026-10-2{index}", detail) for index, detail in enumerate(details)],
        reference_precision=0.6,
        reference_recall=0.7,
    )


def test_without_uses_there_is_nothing_to_judge() -> None:
    health = _assess()

    assert health.status == "NO_USES"
    assert health.recommendation is None


def test_stable_data_without_outcomes_keeps_using_and_says_why() -> None:
    health = _assess(_use("STABLE"))

    assert health.status == "STABLE"
    assert health.recommendation == "KEEP_USING"
    assert [reason.code for reason in health.reasons] == ["NO_OUTCOMES_YET"]


def test_latest_data_change_decides_the_data_signal() -> None:
    assert _assess(_use("REVIEW"), _use("STABLE")).status == "STABLE"
    assert _assess(_use("STABLE"), _use("WATCH")).recommendation == "KEEP_WATCHING"
    review = _assess(_use("STABLE"), _use("REVIEW"))
    assert review.recommendation == "CHECK_DATA_SOURCE"
    assert review.reasons[0].columns == ["temperatura"]


def test_few_rows_are_not_used_to_claim_a_performance_drop() -> None:
    health = _assess(_use("STABLE", outcomes=29, precision=0.1, recall=0.1))

    assert health.status == "STABLE"
    assert health.reasons[0].code == "NOT_ENOUGH_OUTCOMES"


def test_performance_drop_thresholds_and_retraining_proposal() -> None:
    slipping = _assess(_use("STABLE", outcomes=200, precision=0.6, recall=0.6))
    dropped = _assess(_use("STABLE", outcomes=200, precision=0.6, recall=0.5))

    assert slipping.status == "WATCH"
    assert slipping.reasons[0].code == "PERFORMANCE_SLIPPING"
    assert dropped.status == "REVIEW"
    assert dropped.recommendation == "PROPOSE_RETRAINING"


def test_reason_text_is_plain_and_carries_the_numbers() -> None:
    reason = _assess(_use("STABLE", outcomes=200, precision=0.6, recall=0.43)).reasons[0]
    text = health_reason_text(reason)

    assert text.startswith("Con 200 casos de resultado conocido, el modelo detectó 43,0 %")
    assert "70,0 % en la prueba final" in text


def test_real_sequence_goes_from_stable_to_watch_to_review(tmp_path: Path) -> None:
    _, scoring, run_id = approved_maintenance_model(tmp_path)
    folder = VARIANTS / "mantenimiento_maquinas"
    statuses = []
    for name in ("uso_01_estable.csv", "uso_02_cambio_leve.csv", "uso_03_cambio_fuerte.csv"):
        scoring.score(
            run_id, filename=name, content=(folder / name).read_bytes(), actor_label="Prueba"
        )
        statuses.append(scoring.health(run_id))

    assert [health.status for health in statuses] == ["STABLE", "WATCH", "REVIEW"]
    assert statuses[-1].recommendation == "PROPOSE_RETRAINING"
    assert len(statuses[-1].points) == 3


def test_health_view_renders_status_reasons_and_history() -> None:
    def render() -> None:
        from app.ui.extension_views import render_model_health
        from core.ml_engine.health import assess_model_health

        render_model_health(
            assess_model_health(
                [
                    (
                        "2026-10-21",
                        {
                            "rows_scored": 1500,
                            "alerts": 819,
                            "drift": {
                                "level": "REVIEW",
                                "max_psi": 1.3,
                                "columns": [{"column": "desgaste", "level": "REVIEW"}],
                            },
                            "follow_up": {
                                "rows_with_outcome": 1500,
                                "precision": 0.24,
                                "recall": 0.43,
                            },
                        },
                    )
                ],
                reference_precision=0.35,
                reference_recall=0.66,
            )
        )

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert app.error[0].value.startswith("Revisar.")
    texts = " ".join(item.value for item in app.markdown)
    assert "«desgaste» cambió mucho" in texts
    assert "preparar un reentrenamiento" in texts
    assert app.dataframe[0].value.iloc[0]["Datos"] == "Cambió mucho"


def test_precision_reason_reads_as_a_full_sentence() -> None:
    """Recorrido 22/10: decía «el modelo sus avisos acertaron…»."""
    reasons = _assess(_use("STABLE", outcomes=200, precision=0.3, recall=0.43)).reasons
    precision = next(reason for reason in reasons if reason.metric == "precision")

    assert "los avisos del modelo acertaron 30,0 % de las veces" in health_reason_text(precision)
    assert "el modelo sus avisos" not in health_reason_text(precision)


def test_health_view_does_not_call_missing_evidence_stable() -> None:
    def render() -> None:
        from app.ui.extension_views import render_model_health
        from core.ml_engine.health import assess_model_health

        render_model_health(
            assess_model_health(
                [("2026-10-23", {"rows_scored": 20, "alerts": 3})],
                reference_precision=0.6,
                reference_recall=0.7,
            )
        )

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    assert not app.success
    assert "Sin alertas con la evidencia disponible" in app.info[0].value
    texts = " ".join(item.value for item in app.markdown)
    assert "no se puede saber" in texts
    assert "registra sus resultados" in texts
