"""MV-07/08: recorrido visible con proveedor simulado; ninguna llamada real."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.db.repositories.orchestration import OrchestrationRepository
from core.domain.capabilities import CapabilityService


def _seed(database: Path) -> str:
    connection = get_connection(database)
    try:
        migrate(connection)
        service = CapabilityService(WorkflowRepository(connection))
        product = service.create_product(
            name="Tienda sintética", actor_label="Persona", operation_id="product"
        )
        return service.create_capability(
            product.id,
            name="Primera compra",
            objective="Definir a quién ayudar a completar su compra",
            value_kind="revenue",
            decision_unit="Incorporación",
            actor_label="Persona",
            operation_id="capability",
        ).id
    finally:
        connection.close()


def _app(
    database: Path, capability_id: str, *, enabled: bool = False, finish: bool = False
) -> AppTest:
    def render(database_path: str, selected_capability: str, enabled: bool, finish: bool):
        import json

        from app.ui.agents_view import render_agents
        from core.ai_engine.provider import LLMResponse
        from core.db import get_connection, migrate
        from core.db.repositories import WorkflowRepository

        class FakeProvider:
            name = "test_provider"
            model = "scripted_test"

            def __init__(self):
                self.calls = 0

            def complete_json(self, request):
                self.calls += 1
                proposal = (
                    {
                        "action": "delegate",
                        "message": "Negocio revisará qué falta para definir la decisión.",
                        "delegate_role": "business",
                        "delegate_objective": "Identificar información pendiente de la decisión",
                    }
                    if self.calls == 1
                    else {
                        "action": "wait_human",
                        "message": "Necesito que definas quién revisará esta decisión.",
                        "questions": ["¿Quién será responsable de revisar la decisión?"],
                        "reason_code": "MISSING_OWNER",
                    }
                )
                if finish and self.calls > 1:
                    proposal = {
                        "action": "finish",
                        "message": "Revisión terminada. Completa la decisión antes de entrenar.",
                    }
                return LLMResponse(
                    content=json.dumps(proposal),
                    provider=self.name,
                    model=self.model,
                    latency_ms=1,
                    input_tokens=20,
                    output_tokens=15,
                )

        connection = get_connection(database_path)
        try:
            migrate(connection)
            render_agents(
                WorkflowRepository(connection),
                selected_capability,
                provider_factory=lambda: (FakeProvider() if enabled else None, False),
            )
        finally:
            connection.close()

    result = AppTest.from_function(
        render, args=(str(database), capability_id, enabled, finish)
    ).run(timeout=30)
    assert not result.exception, result.exception
    return result


def _input(app, label: str, value: str) -> None:
    next(item for item in app.text_input if item.label == label).input(value)


def _click(app, label: str) -> None:
    next(item for item in app.button if item.label == label).click().run(timeout=30)
    assert not app.exception, app.exception


def _tasks(database: Path):
    connection = get_connection(database)
    try:
        return OrchestrationRepository(WorkflowRepository(connection)).list_tasks()
    finally:
        connection.close()


def test_disabled_provider_prepares_durable_work_without_pretending_to_run(tmp_path):
    database = tmp_path / "agents.sqlite3"
    capability_id = _seed(database)
    app = _app(database, capability_id)
    _input(app, "Quién solicita el trabajo del equipo", "Persona")
    _click(app, "Preparar trabajo del equipo")
    tasks = _tasks(database)
    assert len(tasks) == 1
    assert tasks[0].state == "PENDING" and tasks[0].calls_used == 0
    assert next(item for item in app.button if item.label == "Ejecutar equipo").disabled
    assert any("desactivado" in item.value for item in app.info)
    # La revisión muestra exactamente el contexto del proveedor, con alias por turno.
    assert app.json and capability_id not in str(app.json[-1].value)
    reopened = _app(database, capability_id)
    assert len(reopened.selectbox) == 1


def test_consent_then_coordinator_delegates_and_question_is_visible(tmp_path):
    database = tmp_path / "agents.sqlite3"
    capability_id = _seed(database)
    app = _app(database, capability_id, enabled=True)
    _input(app, "Quién solicita el trabajo del equipo", "Persona")
    _click(app, "Preparar trabajo del equipo")
    assert next(item for item in app.button if item.label == "Ejecutar equipo").disabled
    _input(app, "Quién continúa el trabajo", "Persona")
    next(item for item in app.checkbox if "autorizo este recorrido" in item.label).check().run()
    _click(app, "Ejecutar equipo")
    tasks = _tasks(database)
    assert [task.role for task in tasks] == ["coordinator", "business"]
    assert tasks[0].calls_used == 2
    assert tasks[1].state == "WAITING_HUMAN"
    assert any("¿Quién será responsable" in item.value for item in app.markdown)
    assert not app.error

    _input(app, "Quién continúa el trabajo", "Persona")
    _click(app, "Revisé la información; reanudar")
    resumed = _tasks(database)
    assert all(task.state in {"PENDING", "RUNNING"} for task in resumed)
    assert resumed[0].calls_used == 2


def test_missing_actor_does_not_create_a_task(tmp_path):
    database = tmp_path / "agents.sqlite3"
    capability_id = _seed(database)
    app = _app(database, capability_id)
    _click(app, "Preparar trabajo del equipo")
    assert app.error
    assert _tasks(database) == []


def test_completed_team_shows_its_conclusion_and_next_step(tmp_path):
    database = tmp_path / "agents.sqlite3"
    capability_id = _seed(database)
    app = _app(database, capability_id, enabled=True, finish=True)
    _input(app, "Quién solicita el trabajo del equipo", "Persona")
    _click(app, "Preparar trabajo del equipo")
    _input(app, "Quién continúa el trabajo", "Persona")
    next(item for item in app.checkbox if "autorizo este recorrido" in item.label).check().run()
    _click(app, "Ejecutar equipo")
    assert all(task.state == "COMPLETED" for task in _tasks(database))
    assert any("El equipo terminó este encargo" in item.value for item in app.success)
    assert any("Conclusión del equipo (IA)" in item.value for item in app.markdown)
    assert any("Siguiente paso: revisa la conclusión" in item.value for item in app.caption)
    assert not any(item.label in {"Ejecutar equipo", "Continuar análisis"} for item in app.button)


def _readiness_app(summary: dict) -> AppTest:
    def render(value: dict):
        from app.ui.agents_view import _render_data_evidence

        _render_data_evidence(value)

    app = AppTest.from_function(render, args=(summary,)).run(timeout=30)
    assert not app.exception, app.exception
    return app


def test_ready_table_shows_pending_source_cases_as_excluded_without_granting_approval():
    app = _readiness_app(
        {
            "kind": "label_readiness",
            "status": "ready",
            "table_rows": 80,
            "verified_table_rows": 80,
            "pending_source_count": 12,
            "snapshot_readiness": {"eligible_rows": 80, "pending_rows": 12},
            "measurement_plan": ["Observar los casos pendientes antes del próximo lote."],
        }
    )
    assert app.success
    assert [item.value for item in app.metric] == ["80", "80"]
    assert any("pendientes en la fuente: 12" in item.value for item in app.caption)
    assert any("no aprueba un entrenamiento" in item.value for item in app.caption)


def test_unknown_observation_does_not_display_zero_pending_as_a_measured_fact():
    app = _readiness_app(
        {
            "kind": "label_readiness",
            "status": "needs_human",
            "table_rows": 80,
            "verified_table_rows": 0,
            "pending_source_count": 0,
            "snapshot_readiness": None,
            "measurement_plan": ["Declarar la cobertura de observación de la fuente."],
        }
    )
    assert not app.success
    assert app.info
    assert not any("pendientes en la fuente:" in item.value for item in app.caption)
    assert any("Declarar la cobertura" in item.value for item in app.markdown)


def test_model_viability_displays_partition_support_without_approval_or_metrics():
    app = _readiness_app(
        {
            "kind": "model_viability",
            "status": "ready",
            "split": {
                "train": {"row_count": 90, "positive_support": 40, "negative_support": 50},
                "validation": {"row_count": 30, "positive_support": 12, "negative_support": 18},
                "test": {"row_count": 35, "positive_support": 15, "negative_support": 20},
                "purged_train_rows": 4,
                "purged_validation_rows": 5,
            },
        }
    )
    assert not app.exception
    assert app.dataframe[0].value["Casos"].tolist() == [90, 30, 35]
    assert any("no aprueba" in item.value for item in app.caption)
    assert not app.button
