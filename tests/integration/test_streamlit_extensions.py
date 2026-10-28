"""AppTest mínimo para las pantallas autorizadas EXT-A/B/C/D."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.ai_engine import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    register_provider,
)
from core.domain.context_service import BusinessContextService
from core.domain.strategy_service import StrategyService
from core.reporting import ReportService
from tests.support.ml_workflow import prepared_strategy

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield tmp_path


def _trained_report(runtime_dir: Path, *, approved: bool) -> tuple[str, str]:
    connection, repository, ml, strategy_id, _ = prepared_strategy(runtime_dir, storage_subdir=".")
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    run = ml.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    project_id = repository.get_run_project_id(run.run_id)
    if approved:
        repository.decide_publication(
            report_id=report_id,
            decision="APPROVED",
            actor_label="pytest-reviewer",
        )
    connection.close()
    return project_id, run.run_id


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def _text_area(app: AppTest, label: str):
    return next(item for item in app.text_area if item.label == label)


def test_new_cases_are_blocked_until_the_report_is_approved(runtime_dir: Path) -> None:
    _trained_report(runtime_dir, approved=False)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert any(item.value == "Usar el modelo con casos nuevos" for item in app.subheader)
    # Paso pendiente del flujo, no una falla: aviso informativo sin código de incidente.
    assert any("todavía no está aprobado" in item.value for item in app.info)
    assert not app.error
    assert not [item for item in app.button if item.label == "Priorizar casos"]


def test_new_cases_and_report_download_are_enabled_after_approval(runtime_dir: Path) -> None:
    _trained_report(runtime_dir, approved=True)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert any(item.label == "CSV con casos nuevos" for item in app.get("file_uploader"))
    assert any(item.label == "Priorizar casos" for item in app.button)
    downloads = app.get("download_button")
    assert any(item.label == "Descargar reporte en HTML" for item in downloads)
    assert any("Imprimir para guardarlo como PDF" in item.value for item in app.caption)


def test_comparison_is_visible_with_two_completed_runs(runtime_dir: Path) -> None:
    connection, repository, ml, strategy_id, storage = prepared_strategy(
        runtime_dir, storage_subdir="."
    )
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    ml.execute(strategy_id)
    strategy = repository.get_strategy(strategy_id)
    context_row = repository.get_context(strategy["context_version_id"])
    payload = json.loads(context_row["context_json"])["context"]
    payload["primary_metric"] = "precision"
    payload["metric_rationale"] = "Ahora importa que las alertas acierten"
    next_context = BusinessContextService(repository, storage).create_version(
        project_id=context_row["project_id"],
        dataset_id=context_row["dataset_id"],
        context=payload,
    )
    next_strategy = StrategyService(repository).create_template_version(
        project_id=context_row["project_id"],
        context_version_id=next_context.context_version_id,
    )
    repository.decide_training(
        strategy_version_id=next_strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    ml.execute(next_strategy.strategy_version_id)
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert any(item.value == "Comparar versiones" for item in app.subheader)
    assert any(item.label == "Versión anterior" for item in app.selectbox)
    assert any("Cambiaste" in item.value for item in app.markdown)


class _ResultsProvider:
    name = "pytest_ui_results"
    model = "results-model"

    def __init__(self, *, fail_results: bool = False) -> None:
        self.fail_results = fail_results

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.schema_name == "strategy_explanation":
            template = request.user.split("PLANTILLA (dato):\n", 1)[1]
            content = template
        elif self.fail_results:
            raise LLMProviderError("PROVIDER_TIMEOUT", retryable=True)
        else:
            content = json.dumps(
                {
                    "kind": "FACT",
                    "answer": "El reporte contiene una evaluación calculada.",
                    "evidence_refs": ["metrics.primary_metric"],
                    "suggestion": "Revisa también las limitaciones del reporte.",
                }
            )
        return LLMResponse(
            content=content,
            provider=self.name,
            model=self.model,
            latency_ms=1,
        )


def _enable_provider(monkeypatch: pytest.MonkeyPatch, *, fail_results: bool) -> None:
    provider_name = "pytest_ui_results_fallback" if fail_results else "pytest_ui_results_ok"
    register_provider(
        provider_name,
        lambda settings: _ResultsProvider(fail_results=fail_results),
    )
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", provider_name)
    monkeypatch.setenv("LLM_MODEL", "results-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")


@pytest.mark.parametrize("fail_results", [False, True])
def test_results_questions_show_verified_and_fallback_modes(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_results: bool,
) -> None:
    _trained_report(runtime_dir, approved=False)
    _enable_provider(monkeypatch, fail_results=fail_results)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    _text_area(app, "¿Qué quieres entender de estos resultados?").input(
        "¿Qué evidencia tiene el reporte?"
    )
    _button(app, "Preguntar").click().run(timeout=60)

    assert not app.exception
    if fail_results:
        assert any("Modo sin IA generativa" in item.value for item in app.warning)
        assert any("tardó demasiado" in item.value for item in app.warning)
    else:
        assert any("evaluación calculada" in item.value for item in app.markdown)
        assert any("Respuesta de IA verificada" in item.value for item in app.success)


def test_model_page_shows_health_and_retraining_entry(runtime_dir: Path) -> None:
    _trained_report(runtime_dir, approved=True)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert any(item.value == "#### Salud del modelo" for item in app.markdown)
    assert any("Todavía no hay usos registrados" in item.value for item in app.info)
    assert any(item.label == "Reentrenar con casos recientes" for item in app.expander)
    assert any(item.label == "Crear proyecto de reentrenamiento" for item in app.button)


def test_retraining_project_opens_with_copied_decisions(runtime_dir: Path) -> None:
    from core.db.connection import get_connection
    from core.db.repositories import WorkflowRepository
    from core.domain.retraining_service import RetrainingService
    from tests.support.ml_workflow import csv_bytes

    _, run_id = _trained_report(runtime_dir, approved=True)
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    RetrainingService(WorkflowRepository(connection), runtime_dir).start(
        source_run_id=run_id,
        filename="recientes.csv",
        content=csv_bytes(400),
        actor_label="pytest-reviewer",
        data_authorized=True,
    )
    connection.close()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert "reentrenamiento 1" in app.sidebar.selectbox[0].format_func(
        app.sidebar.selectbox[0].value
    )
    assert any("Decisiones guardadas: versión 1." in item.value for item in app.caption)


def test_deciding_with_the_model_by_capacity_and_by_cost(runtime_dir: Path) -> None:
    _trained_report(runtime_dir, approved=True)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert {"Atender", "Aciertos esperados", "Falsas alarmas", "Se escaparían"} <= set(metrics)
    assert int(metrics["Atender"]) <= 100
    next(
        item for item in app.number_input if item.label == "¿Cuántos puedes atender en ese periodo?"
    ).set_value(0)
    app.run(timeout=60)
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Atender") == "0"
    assert any("no atender casos" in item.value for item in app.info)
    next(item for item in app.radio if item.label == "¿Cómo quieres decidir?").set_value("cost")
    app.run(timeout=60)
    assert not app.exception
    assert any("no dinero" in item.value for item in app.caption)


def test_action_list_is_created_and_offers_measurement(runtime_dir: Path) -> None:
    import io

    import pandas as pd

    from core.db import get_connection
    from core.db.repositories import WorkflowRepository
    from core.ml_engine.scoring import ScoringService
    from tests.support.ml_workflow import csv_bytes

    _, run_id = _trained_report(runtime_dir, approved=True)
    cases = pd.read_csv(io.BytesIO(csv_bytes())).head(120).drop(columns=["target"])
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    result = ScoringService(WorkflowRepository(connection), runtime_dir).score(
        run_id, filename="nuevos.csv", content=cases.to_csv(index=False).encode(), actor_label="Ana"
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"scored-{run_id}"] = result
    app.run(timeout=60)
    next(i for i in app.text_input if i.label == "¿Quién decide atender estos casos?").input("Ana")
    next(item for item in app.button if item.label == "Crear lista de acción").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("Lista creada" in item.value for item in app.success)
    assert any(item.label == "Medir" for item in app.button)


def test_new_scoring_attempt_clears_old_result_and_keeps_actor(runtime_dir: Path) -> None:
    import io

    import pandas as pd

    from core.db import get_connection
    from core.db.repositories import WorkflowRepository
    from core.ml_engine.scoring import ScoringService
    from tests.support.ml_workflow import csv_bytes

    _, run_id = _trained_report(runtime_dir, approved=True)
    cases = pd.read_csv(io.BytesIO(csv_bytes())).head(120).drop(columns=["target"])
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    result = ScoringService(WorkflowRepository(connection), runtime_dir).score(
        run_id,
        filename="archivo-a.csv",
        content=cases.to_csv(index=False).encode(),
        actor_label="Ana",
    )
    connection.close()

    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"scored-{run_id}"] = result
    app.run(timeout=60)

    assert not app.exception
    metrics = {item.label: str(item.value) for item in app.metric}
    assert metrics["Punto de corte"] == f"{result.threshold:.3f}".replace(".", ",")
    scored_table = next(table.value for table in app.dataframe if "Puntaje" in table.value.columns)
    assert "," in scored_table.iloc[0]["Puntaje"]
    assert any(
        "Las alertas son todos los casos" in item.value
        and "sólo los que caben en tu capacidad" in item.value
        for item in app.caption
    )

    actor = next(item for item in app.text_input if item.label == "¿Quién usa el modelo?")
    actor.input("Equipo de operaciones")
    _button(app, "Priorizar casos").click().run(timeout=60)

    assert not app.exception
    assert any("Selecciona un CSV con casos nuevos." in item.value for item in app.error)
    assert not [item for item in app.metric if item.label == "Casos evaluados"]
    actor = next(item for item in app.text_input if item.label == "¿Quién usa el modelo?")
    assert actor.value == "Equipo de operaciones"
