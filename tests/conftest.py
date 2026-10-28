"""Configuración compartida de pytest para Goal Settings."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ninguna prueba llama a un proveedor real, aunque exista un `.env` local con clave."""

    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "no-env-file"))
