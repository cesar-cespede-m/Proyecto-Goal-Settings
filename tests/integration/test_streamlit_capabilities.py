"""MV-04: objetivos sin datos y navegación por capacidades/ciclos aislados."""

from pathlib import Path
from uuid import uuid4

from streamlit.testing.v1 import AppTest

from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def _app(database: Path) -> AppTest:
    def render(database_path: str) -> None:
        import streamlit as st

        from app.ui.product_view import render_product_workspace
        from core.db import get_connection, migrate
        from core.db.repositories import WorkflowRepository

        def opened(project_id: str, message: str) -> None:
            st.session_state["opened-project"] = project_id
            st.session_state["opened-message"] = message
            st.rerun()

        connection = get_connection(database_path)
        try:
            migrate(connection)
            render_product_workspace(WorkflowRepository(connection), on_open=opened)
        finally:
            connection.close()

    return AppTest.from_function(render, args=(str(database),)).run(timeout=30)


def _input(app, label, value):
    next(item for item in app.text_input if item.label == label).input(value)


def _area(app, label, value):
    next(item for item in app.text_area if item.label == label).input(value)


def _select(app, label, value):
    next(item for item in app.selectbox if item.label == label).set_value(value)


def _click(app, label):
    next(item for item in app.button if item.label == label).click().run(timeout=30)
    assert not app.exception, app.exception


def _create_product(app):
    _input(app, "Nombre del producto", "Comercio sintético")
    _area(app, "Descripción del producto", "Producto de prueba sin datos productivos")
    _select(app, "Etapa", "operating")
    _input(app, "Quién registra este producto", "Ana")
    _click(app, "Crear producto")


def _create_capability(app, name, kind):
    _input(app, "Nombre del objetivo", name)
    _area(app, "Qué quieres mejorar", "Mejorar una decisión comercial con evidencia")
    _select(app, "Tipo de valor", kind)
    _input(app, "Sobre qué tomarás cada decisión", "Pedido")
    _input(app, "Quién registra este objetivo", "Ana")
    _click(app, "Crear objetivo")


def _seed_pair(database):
    from core.domain.capabilities import CapabilityService

    connection = get_connection(database)
    try:
        migrate(connection)
        service = CapabilityService(WorkflowRepository(connection))
        product = service.create_product(
            name="Producto sintético",
            description="Fixture de navegación",
            actor_label="Ana",
            operation_id=str(uuid4()),
        )
        capabilities = [
            service.create_capability(
                product.id,
                name=name,
                objective="Mejorar una decisión con evidencia",
                value_kind=kind,
                decision_unit="Pedido",
                actor_label="Ana",
                operation_id=str(uuid4()),
            )
            for name, kind in (("Activación", "revenue"), ("Entregas", "savings"))
        ]
        return product, capabilities
    finally:
        connection.close()


def _start_cycle(app, reason):
    _input(app, "Motivo de la evaluación", reason)
    _input(app, "Quién inicia esta evaluación", "Ana")
    _click(app, "Crear evaluación y continuar")


def test_product_and_two_value_capabilities_persist_without_data_or_training(tmp_path):
    from core.domain.capabilities import CapabilityService

    database = tmp_path / "goal_settings.sqlite3"
    app = _app(database)
    _create_product(app)
    _create_capability(app, "Activación", "revenue")
    _create_capability(app, "Entregas", "savings")
    app.run(timeout=30)
    assert not app.exception

    connection = get_connection(database)
    try:
        repository = WorkflowRepository(connection)
        service = CapabilityService(repository)
        products = service.list_products()
        assert len(products) == 1
        assert products[0].name == "Comercio sintético"
        assert products[0].stage == "operating"
        capabilities = service.list_capabilities(products[0].id)
        assert {(item.name, item.value_kind) for item in capabilities} == {
            ("Activación", "revenue"),
            ("Entregas", "savings"),
        }
        assert all(service.list_cycles(item.id) == [] for item in capabilities)
        assert repository.list_projects() == []
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
    finally:
        connection.close()

    reopened = _app(database)
    assert not reopened.exception
    assert set(
        next(item for item in reopened.selectbox if item.label == "Objetivo de negocio").options
    ) == {
        "Activación",
        "Entregas",
    }


def test_missing_actor_is_inline_error_and_retry_does_not_duplicate_product(tmp_path):
    from core.domain.capabilities import CapabilityService

    database = tmp_path / "goal_settings.sqlite3"
    app = _app(database)
    _input(app, "Nombre del producto", "Producto sin actor")
    _click(app, "Crear producto")
    assert app.error
    connection = get_connection(database)
    try:
        assert CapabilityService(WorkflowRepository(connection)).list_products() == []
    finally:
        connection.close()

    _input(app, "Quién registra este producto", "Ana")
    _click(app, "Crear producto")
    app.run(timeout=30)
    connection = get_connection(database)
    try:
        assert len(CapabilityService(WorkflowRepository(connection)).list_products()) == 1
    finally:
        connection.close()


def test_cycle_navigation_never_reuses_another_capability_project(tmp_path):
    from core.domain.capabilities import CapabilityService

    database = tmp_path / "goal_settings.sqlite3"
    _, (activation, delivery) = _seed_pair(database)
    app = _app(database)
    _select(app, "Objetivo de negocio", activation.id)
    app.run(timeout=30)
    _start_cycle(app, "Primera evaluación de activación")
    first_activation_project = app.session_state["opened-project"]

    _select(app, "Objetivo de negocio", delivery.id)
    app.run(timeout=30)
    _start_cycle(app, "Primera evaluación de entregas")
    delivery_project = app.session_state["opened-project"]
    assert delivery_project != first_activation_project

    _select(app, "Objetivo de negocio", activation.id)
    app.run(timeout=30)
    connection = get_connection(database)
    try:
        service = CapabilityService(WorkflowRepository(connection))
        first_activation = service.list_cycles(activation.id)[0]
        assert len(service.list_cycles(activation.id)) == 1
        assert len(service.list_cycles(delivery.id)) == 1
    finally:
        connection.close()
    _select(app, "Evaluación de origen", first_activation.id)
    _start_cycle(app, "Actualización de activación")
    updated_project = app.session_state["opened-project"]
    assert len({updated_project, delivery_project, first_activation_project}) == 3

    connection = get_connection(database)
    try:
        service = CapabilityService(WorkflowRepository(connection))
        cycles = service.list_cycles(activation.id)
        assert [item.cycle_no for item in cycles] == [1, 2]
        assert cycles[1].source_cycle_id == cycles[0].id
        assert cycles[0].project_id == first_activation_project
        assert cycles[1].project_id == updated_project
        assert service.list_cycles(delivery.id)[0].project_id == delivery_project
    finally:
        connection.close()

    _select(app, "Evaluación", first_activation.id)
    _click(app, "Continuar con datos y modelo")
    assert app.session_state["opened-project"] == first_activation_project
    _select(app, "Objetivo de negocio", delivery.id)
    app.run(timeout=30)
    assert len(next(item for item in app.selectbox if item.label == "Evaluación").options) == 1
    _click(app, "Continuar con datos y modelo")
    assert app.session_state["opened-project"] == delivery_project


def test_product_workspace_is_available_in_main_before_any_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception, app.exception
    assert any(
        item.value == "Tu producto: ingresos, ahorro y aprendizaje" for item in app.subheader
    )
    assert any(item.label == "Nombre del producto" for item in app.text_input)
    assert any(item.label == "Crear producto" for item in app.button)
    assert not any(item.label == "Nombre del proyecto" for item in app.text_input)


def test_main_opens_the_selected_cycle_and_returns_to_its_objective(tmp_path, monkeypatch):
    from core.domain.capabilities import CapabilityService

    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    database = tmp_path / "goal_settings.sqlite3"
    _, (activation, delivery) = _seed_pair(database)
    connection = get_connection(database)
    try:
        service = CapabilityService(WorkflowRepository(connection))
        selected_cycle = service.create_cycle(
            delivery.id, reason="Datos de entregas", actor_label="Ana", operation_id=str(uuid4())
        )
    finally:
        connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    _select(app, "Objetivo de negocio", delivery.id)
    app.run(timeout=30)
    _click(app, "Continuar con datos y modelo")
    assert app.sidebar.radio(key="workspace-section").value == "Datos y modelos"
    assert app.sidebar.selectbox[0].value == selected_cycle.project_id
    assert any(item.value == "Sube tus datos" for item in app.subheader)
    assert not any(item.label == "Crear producto" for item in app.button)
    app.sidebar.radio(key="workspace-section").set_value("Producto y decisiones").run(timeout=30)
    assert not app.exception
    assert (
        next(item for item in app.selectbox if item.label == "Objetivo de negocio").value
        == delivery.id
    )


def test_archived_cycle_remains_visible_but_cannot_open_an_unrelated_project(tmp_path):
    from core.domain.capabilities import CapabilityService

    database = tmp_path / "goal_settings.sqlite3"
    _, (activation, delivery) = _seed_pair(database)
    connection = get_connection(database)
    try:
        repository = WorkflowRepository(connection)
        service = CapabilityService(repository)
        archived = service.create_cycle(
            activation.id,
            reason="Evaluación histórica",
            actor_label="Ana",
            operation_id=str(uuid4()),
        )
        active = service.create_cycle(
            delivery.id,
            reason="Evaluación vigente",
            actor_label="Ana",
            operation_id=str(uuid4()),
        )
        repository.archive_project(archived.project_id, actor_label="Ana")
    finally:
        connection.close()

    app = _app(database)
    app.session_state["opened-project"] = active.project_id
    _select(app, "Objetivo de negocio", activation.id)
    app.run(timeout=30)
    assert not app.exception, app.exception
    assert next(
        item for item in app.button if item.label == "Continuar con datos y modelo"
    ).disabled
    assert any("archivado" in item.value for item in app.info)
    assert app.session_state["opened-project"] == active.project_id

    _select(app, "Objetivo de negocio", delivery.id)
    app.run(timeout=30)
    assert not next(
        item for item in app.button if item.label == "Continuar con datos y modelo"
    ).disabled
    _click(app, "Continuar con datos y modelo")
    assert app.session_state["opened-project"] == active.project_id
