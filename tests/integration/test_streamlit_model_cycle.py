"""QA-03: ciclo completo por la pantalla para los tres tipos de modelo."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
ACTOR = "Responsable QA-03"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


def _synthetic_csv(task_type: str) -> bytes:
    """Datos sin negocio real y con una señal clara frente a la referencia simple."""

    rng = np.random.default_rng(303)
    signal = np.tile(np.linspace(-3.0, 3.0, 60), 4)
    noise = rng.normal(scale=0.08, size=len(signal))
    if task_type == "binary_classification":
        target = np.where(signal + noise > 0, "si", "no")
    elif task_type == "multiclass_classification":
        target = np.select(
            [signal + noise < -0.75, signal + noise < 0.75],
            ["bajo", "medio"],
            default="alto",
        )
    else:
        target = np.round(20 + 7 * signal + noise, 4)
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(len(signal))],
            "signal": np.round(signal, 4),
            "group": np.resize(np.array(["a", "b", "c"]), len(signal)),
            "outcome": target,
        }
    )
    return frame.to_csv(index=False).encode()


def _project_with_dataset(runtime_dir: Path, task_type: str) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    project_id = repository.create_project(f"Ciclo {task_type}")
    DataIngestionService(repository, runtime_dir).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "qa03-synthetic.csv",
            "source_name": "Fixture sintético QA-03",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=_synthetic_csv(task_type),
    )
    connection.close()


def _by_label(widgets, label: str):
    return next(item for item in widgets if item.label == label)


def _by_key_suffix(widgets, suffix: str):
    return next(item for item in widgets if item.key and item.key.endswith(suffix))


def _run(app: AppTest, *, timeout: int = 30) -> AppTest:
    app.run(timeout=timeout)
    assert not app.exception, app.exception
    assert not app.error, [item.value for item in app.error]
    return app


def _draft_id(app: AppTest) -> str:
    key = next(item.key for item in app.button if (item.key or "").startswith("submit-"))
    return key.split("-", 1)[1].rsplit("-", 1)[0]


def _save_text(app: AppTest, field: str, value: str) -> None:
    next(item for item in app.text_area if f"-{field}-r" in (item.key or "")).input(value)
    _by_key_suffix(app.button, f"submit-{_draft_id(app)}-{field}").click()
    _run(app)


def _choose(app: AppTest, field: str, value: object, *, radio: bool = False) -> None:
    widgets = app.radio if radio else app.selectbox
    next(item for item in widgets if f"-{field}-r" in (item.key or "")).set_value(value)
    _run(app)
    # Elegir guarda al instante; sólo con advertencia aparece «Confirmar de todas formas».
    extra = [item for item in app.button if (item.key or "") == f"save-{_draft_id(app)}-{field}"]
    if extra:
        extra[0].click()
        _run(app)


def _complete_context(app: AppTest, task_type: str, metric: str) -> None:
    actor = next(item for item in app.text_input if (item.key or "").startswith("actor-input-"))
    actor.input(ACTOR)
    _by_key_suffix(app.button, "-actor").click()
    _run(app)

    _save_text(app, "objective", "Anticipar el resultado sintético de cada caso")
    _choose(app, "task_type", task_type, radio=True)
    _choose(app, "target_column", "outcome")
    if task_type == "binary_classification":
        _choose(app, "positive_class", "si", radio=True)
    else:
        assert not [item for item in app.radio if "-positive_class-r" in (item.key or "")]
    _save_text(app, "prediction_moment", "Antes de conocer el resultado sintético")
    _save_text(app, "false_positive_consequence", "Revisar un caso que no lo necesitaba")
    _save_text(app, "false_negative_consequence", "No revisar un caso que lo necesitaba")
    _choose(app, "primary_metric", metric, radio=True)
    _save_text(app, "metric_rationale", "Comparar de forma estable contra la referencia")
    _choose(app, "guardrail_mode", "no_quantitative_guardrail", radio=True)

    for column, answer in (("case_id", "known_skip"), ("signal", "use"), ("group", "use")):
        next(
            item for item in app.radio if f"feature-{_draft_id(app)}-{column}-r" in (item.key or "")
        ).set_value(answer)
        _run(app)

    _by_label(app.button, "Guardar mis decisiones").click()
    _run(app)


@pytest.mark.parametrize(
    ("task_type", "metric", "use_action"),
    [
        ("binary_classification", "f1", "Priorizar casos"),
        ("multiclass_classification", "macro_f1", "Estimar casos"),
        ("regression", "mae", "Estimar casos"),
    ],
)
def test_full_model_cycle_from_the_screen(
    runtime_dir: Path,
    task_type: str,
    metric: str,
    use_action: str,
) -> None:
    _project_with_dataset(runtime_dir, task_type)
    app = _run(AppTest.from_file(str(APP_PATH)))

    _complete_context(app, task_type, metric)
    _by_label(app.button, "Preparar el plan de entrenamiento").click()
    _run(app)

    _by_label(app.radio, "Decisión sobre el entrenamiento").set_value("APPROVED")
    _by_label(app.text_input, "Responsable de revisar la estrategia").input(ACTOR)
    for checkbox in app.checkbox:
        if checkbox.label == "Leí los puntos para revisar de este plan":
            checkbox.check()
    _by_label(app.button, "Registrar decisión").click()
    _run(app)

    _by_label(app.button, "Entrenar y comparar modelos").click()
    _run(app, timeout=180)
    assert any(
        "Se recomienda" in item.value or "modelo recomendado" in item.value
        for item in [*app.success, *app.markdown]
    )

    _by_label(app.button, "Generar reporte verificable").click()
    _run(app)
    _by_label(app.radio, "Decisión sobre el reporte").set_value("APPROVED")
    _by_label(app.text_input, "Responsable de revisar el reporte").input(ACTOR)
    _by_label(app.button, "Registrar decisión del reporte").click()
    _run(app)
    _by_label(app.button, "Publicar reporte aprobado").click()
    _run(app)
    _by_label(app.button, "Sí, publicar en la demo").click()
    _run(app)

    assert any(item.label == use_action for item in app.button)
    assert any(item.label == "Reentrenar con casos recientes" for item in app.expander)
    assert any(item.label == "Cómo va el ciclo de este proyecto" for item in app.expander)
    models_in_use = _by_label(app.metric, "Modelos en uso")
    assert str(models_in_use.value) == "1"
