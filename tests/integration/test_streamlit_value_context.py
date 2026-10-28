"""MV-05: definición sin datos y escenarios económicos declarados en la interfaz."""

from decimal import Decimal
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService


def _seed(database: Path):
    connection = get_connection(database)
    try:
        migrate(connection)
        service = CapabilityService(WorkflowRepository(connection))
        product = service.create_product(
            name="Comercio sintético", actor_label="Ana", operation_id="product"
        )
        return [
            service.create_capability(
                product.id,
                name=name,
                objective="Mejorar una decisión con evidencia",
                value_kind=kind,
                decision_unit="Pedido",
                actor_label="Ana",
                operation_id=f"capability-{kind}",
            )
            for name, kind in (("Primera compra", "revenue"), ("Entregas", "savings"))
        ]
    finally:
        connection.close()


def _app(database: Path, capability_id: str | None = None) -> AppTest:
    def render(database_path: str, selected_capability: str | None):
        from app.ui.capability_view import render_capability_value
        from app.ui.product_view import render_product_workspace
        from core.db import get_connection, migrate
        from core.db.repositories import WorkflowRepository

        connection = get_connection(database_path)
        try:
            migrate(connection)
            repository = WorkflowRepository(connection)
            if selected_capability is None:
                render_product_workspace(repository, on_open=lambda project, message: None)
            else:
                render_capability_value(repository, selected_capability)
        finally:
            connection.close()

    app = AppTest.from_function(render, args=(str(database), capability_id)).run(timeout=30)
    assert not app.exception, app.exception
    return app


def _input(app, label: str, value: str):
    next(item for item in app.text_input if item.label == label).input(value)


def _click(app, label: str):
    next(item for item in app.button if item.label == label).click().run(timeout=30)
    assert not app.exception, app.exception


def _versions(database: Path, capability_id: str, kind: str):
    from core.domain.decision_value import DecisionValueService

    connection = get_connection(database)
    try:
        service = DecisionValueService(WorkflowRepository(connection))
        return getattr(service, f"list_{kind}_versions")(capability_id)
    finally:
        connection.close()


def _scenario(app, *, value: str = "20", cost: str = "1"):
    fields = {
        "Moneda (código de 3 letras)": "USD",
        "Decimales de la moneda": "2",
        "Período del escenario": "Cohorte sintética mensual",
        "Casos del período": "10",
        "Máximo de casos que puedes atender (opcional)": "4",
        "Fuente o fundamento de la tasa": "Supuesto sintético para explorar",
        "Fundamento de la reducción": "Hipótesis sin efecto causal medido",
        "Costo por intervención": cost,
        "Costo fijo al intervenir": "2",
        "Fundamento del valor económico": "Margen o pérdida sintética por resultado",
        "Quién registra el escenario": "Ana",
    }
    revenue = any(
        item.label == "Proporción con riesgo de perder ingresos" for item in app.text_input
    )
    fields.update(
        {
            "Proporción con riesgo de perder ingresos"
            if revenue
            else "Proporción con el problema sin la acción": "0.5",
            "Reducción relativa supuesta de esa pérdida"
            if revenue
            else "Reducción relativa supuesta del problema": "0.2",
            "Margen recuperable por caso (supuesto)"
            if revenue
            else "Costo evitado por problema resuelto (supuesto)": value,
        }
    )
    for label, field_value in fields.items():
        _input(app, label, field_value)
    next(
        item
        for item in app.checkbox
        if item.label == "Confirmo estos supuestos para explorar el escenario"
    ).check()
    _click(app, "Guardar escenario")


def _metric(app, label: str) -> Decimal:
    value = next(item for item in app.metric if item.label == label).value
    return Decimal(value.split(" ", maxsplit=1)[1])


def test_incomplete_decision_is_saved_without_data_or_invented_owner(tmp_path):
    database = tmp_path / "goal_settings.sqlite3"
    capability = _seed(database)[0]
    app = _app(database, capability.id)
    _input(app, "Qué decisión quieres mejorar", "A quién ofrecer ayuda para completar una compra")
    _input(app, "Quién registra la decisión", "Ana")
    _click(app, "Guardar decisión")
    versions = _versions(database, capability.id, "decision")
    assert len(versions) == 1
    assert versions[0].spec.question == "A quién ofrecer ayuda para completar una compra"
    assert versions[0].spec.owner_label is None
    assert "owner_label" in versions[0].missing_fields
    assert not versions[0].confirmed_human
    assert any("quién es responsable de la decisión" in item.value for item in app.info)
    assert any("Estimar ingresos o ahorro" in item.value for item in app.caption)
    connection = get_connection(database)
    try:
        for table in ("projects", "datasets", "context_versions", "runs"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        connection.close()
    reopened = _app(database, capability.id)
    assert (
        next(
            item for item in reopened.text_input if item.label == "Qué decisión quieres mejorar"
        ).value
        == versions[0].spec.question
    )
    _scenario(reopened)
    assert any(
        "Ya hay un escenario económico confirmado" in item.value for item in reopened.caption
    )
    assert not any("Para estimar valor, abre" in item.value for item in reopened.caption)


def test_invalid_or_unconfirmed_economic_input_is_not_presented_as_value(tmp_path):
    database = tmp_path / "goal_settings.sqlite3"
    capability = _seed(database)[0]
    app = _app(database, capability.id)
    _input(app, "Quién registra el escenario", "Ana")
    _input(app, "Casos del período", "no es un número")
    _click(app, "Guardar escenario")
    assert app.error
    assert _versions(database, capability.id, "economic") == []
    assert not app.metric

    _input(app, "Casos del período", "10")
    _click(app, "Guardar escenario")
    versions = _versions(database, capability.id, "economic")
    assert len(versions) == 1
    assert versions[0].scenario.external_risk_rate is None
    assert not versions[0].confirmed_human
    assert versions[0].missing_fields
    assert not app.metric


@pytest.mark.parametrize("capability_index", [0, 1], ids=["revenue", "savings"])
def test_confirmed_synthetic_scenario_shows_exposure_benefit_and_net_separately(
    tmp_path, capability_index: int
):
    database = tmp_path / "goal_settings.sqlite3"
    capability = _seed(database)[capability_index]
    app = _app(database, capability.id)
    _scenario(app)
    versions = _versions(database, capability.id, "economic")
    assert len(versions) == 1 and versions[0].confirmed_human
    assert _metric(app, "Exposición del grupo") == Decimal("40.00")
    assert _metric(app, "Beneficio bruto supuesto") == Decimal("8.00")
    assert _metric(app, "Valor neto supuesto") == Decimal("2.00")
    assert any("demostración sintética" in item.value for item in app.caption)
    assert any("no equivale a valor recuperable" in item.value for item in app.caption)


def test_cost_revision_preserves_prior_scenario_and_predictive_fingerprint(tmp_path):
    from core.domain.decision_value import DecisionValueService

    database = tmp_path / "goal_settings.sqlite3"
    capability = _seed(database)[0]
    connection = get_connection(database)
    try:
        predictive = DecisionValueService(WorkflowRepository(connection)).create_predictive_version(
            capability.id,
            spec={"population": "Clientes sintéticos elegibles"},
            actor_label="Ana",
            operation_id="predictive-draft",
        )
    finally:
        connection.close()
    app = _app(database, capability.id)
    _scenario(app)
    first = _versions(database, capability.id, "economic")[0]
    _scenario(app, cost="2")
    versions = _versions(database, capability.id, "economic")
    assert [item.version_no for item in versions] == [1, 2]
    assert versions[1].based_on_version_id == first.id
    assert versions[0] == first
    assert versions[1].sha256 != first.sha256
    assert _versions(database, capability.id, "predictive") == [predictive]
    assert _metric(app, "Valor neto supuesto") == Decimal("-2.00")


def test_switching_capability_does_not_carry_economic_inputs_or_versions(tmp_path):
    database = tmp_path / "goal_settings.sqlite3"
    revenue, savings = _seed(database)
    app = _app(database)
    next(item for item in app.selectbox if item.label == "Objetivo de negocio").set_value(
        revenue.id
    )
    app.run(timeout=30)
    _scenario(app)
    next(item for item in app.selectbox if item.label == "Objetivo de negocio").set_value(
        savings.id
    )
    app.run(timeout=30)
    assert not app.exception
    assert (
        next(
            item
            for item in app.text_input
            if item.label == "Costo evitado por problema resuelto (supuesto)"
        ).value
        == ""
    )
    assert not app.metric
    _scenario(app, value="50")
    assert _versions(database, revenue.id, "economic")[0].scenario.value_per_case == Decimal("20")
    assert _versions(database, savings.id, "economic")[0].scenario.value_per_case == Decimal("50")
    assert _metric(app, "Valor neto supuesto") == Decimal("14.00")


def test_predictive_definition_imports_context_and_binds_exact_cycle(tmp_path):
    from core.domain.decision_value import DecisionValueService

    database = tmp_path / "goal_settings.sqlite3"
    capability = _seed(database)[0]
    connection = get_connection(database)
    try:
        repository = WorkflowRepository(connection)
        cycle = CapabilityService(repository).create_cycle(
            capability.id,
            reason="Contexto sintético para probar la interfaz",
            actor_label="Ana",
            operation_id="predictive-cycle",
        )
        dataset_id = repository.add_dataset(
            project_id=cycle.project_id,
            sha256="a" * 64,
            original_filename_display="synthetic-ui.csv",
            source="synthetic UI fixture",
            license_name="test-only",
            anonymization_declared=True,
            admission_status="ACCEPTED",
            rejection_reason=None,
            n_rows=20,
            n_cols=2,
            byte_count=128,
            profile={"schema_version": "1.0", "columns": [], "fixture": True},
        )
        context_id = repository.add_context(
            project_id=cycle.project_id,
            dataset_id=dataset_id,
            version_no=1,
            payload={
                "schema_version": "1.0",
                "target_column": "no_purchase",
                "positive_class": 1,
                "prediction_moment": "Al segundo día de registro",
                "excluded_columns": [],
                "primary_metric": "recall",
                "context": {
                    "task_type": "binary_classification",
                    "feature_decisions": [
                        {
                            "column": "visits_before_cutoff",
                            "use": "include",
                            "available_at_prediction": True,
                            "reason": "Visitas anteriores al corte sintético",
                        }
                    ],
                },
            },
        )
    finally:
        connection.close()

    app = _app(database, capability.id)
    _input(app, "Población a la que se aplica", "Registros sintéticos sin compra al día dos")
    next(
        item for item in app.text_area if item.label == "Qué significa observar ese resultado"
    ).input("Sin primera compra al terminar la ventana declarada")
    _input(app, "Días de observación", "-1")
    _input(app, "Quién registra la definición predictiva", "Ana")
    next(
        item for item in app.checkbox if item.label == "Confirmo esta definición predictiva"
    ).check()
    _click(app, "Guardar definición predictiva")
    assert app.error
    assert _versions(database, capability.id, "predictive") == []

    _input(app, "Días de observación", "14")
    _click(app, "Guardar definición predictiva")
    versions = _versions(database, capability.id, "predictive")
    assert len(versions) == 1 and versions[0].confirmed_human
    assert versions[0].spec.source_context_version_id == context_id
    assert versions[0].spec.target_column == "no_purchase"
    assert versions[0].spec.outcome_window.days == 14
    assert versions[0].spec.feature_decisions[0].column == "visits_before_cutoff"
    _input(app, "Quién vincula la definición a la evaluación", "Ana")
    _click(app, "Vincular definición a la evaluación")
    connection = get_connection(database)
    try:
        binding = DecisionValueService(WorkflowRepository(connection)).get_cycle_binding(cycle.id)
        assert binding is not None
        assert binding.capability_id == capability.id
        assert binding.cycle_id == cycle.id
        assert binding.predictive_version_id == versions[0].id
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    finally:
        connection.close()
