"""Smoke test de la UI mínima sin levantar un servidor de red."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def test_minimal_app_renders_and_declares_its_limits(tmp_path: Path) -> None:
    previous = os.environ.get("GOAL_SETTINGS_RUNTIME_DIR")
    os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = str(tmp_path / "runtime")
    try:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.sidebar.radio(key="workspace-section").set_value("Datos y modelos").run(timeout=10)
        next(item for item in app.text_input if item.label == "Nombre del proyecto").input(
            "Proyecto UI fixture"
        )
        next(item for item in app.button if item.label == "Crear proyecto").click().run(timeout=10)
    finally:
        if previous is None:
            os.environ.pop("GOAL_SETTINGS_RUNTIME_DIR", None)
        else:
            os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = previous

    assert not app.exception
    assert app.title[0].value == "Goal Settings Analytics Studio"
    assert "nada se entrena ni se usa sin tu aprobación" in " ".join(c.value for c in app.caption)
    assert "Demo privada" in app.warning[0].value
    assert any(item.value == "Sube tus datos" for item in app.subheader)
    assert any("25 MB, 100.000 filas y 100 columnas" in item.value for item in app.markdown)
    assert any(
        "«Proyecto UI fixture» creado. Ahora sube su archivo" in item.value for item in app.success
    )
    # UX-11: el proyecto nuevo queda activo y crear otro pasa a ser secundario.
    assert any("Estás trabajando en **Proyecto UI fixture**" in item.value for item in app.markdown)
    # UX-07 / DU-1: sin plantilla de fichas ni numeración imposible.
    page = " ".join(item.value for item in app.markdown)
    assert "Qué es" not in page and "Para qué sirve" not in page
    assert "Paso 1 de 5" in page and "de 5</div>" in page
