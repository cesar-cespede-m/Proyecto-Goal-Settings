"""Pruebas mínimas de la estructura creada por TEC-01."""

import importlib


def test_core_imports_without_streamlit_side_effects() -> None:
    core = importlib.import_module("core")

    assert core.__version__ == "0.9.0"


def test_future_data_engine_boundary_is_importable() -> None:
    module = importlib.import_module("core.data_engine")

    assert module.__doc__ is not None
