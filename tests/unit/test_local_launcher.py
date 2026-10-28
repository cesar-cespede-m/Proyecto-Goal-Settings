"""Entrega local: aislamiento, errores accionables y conservación de la demo."""

import socket
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import local_launcher as launcher


def test_default_launch_disables_ai_even_when_parent_has_it_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", "other-runtime")
    env = launcher.local_environment(tmp_path)
    assert env["LLM_ENABLED"] == "false"
    assert env["GOAL_SETTINGS_RUNTIME_DIR"] == str(tmp_path.resolve())
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert launcher.local_environment(tmp_path, enable_ai=True)["LLM_ENABLED"] == "true"


def test_demo_reopening_preserves_database_and_does_not_retrain(tmp_path, monkeypatch):
    database = tmp_path / "execution/private-storage/goal_settings.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"existing reviewer work")
    runner = Mock()
    monkeypatch.setattr(launcher, "run", runner)
    assert launcher.prepare_demo(Path("python"), tmp_path) == database.parent
    assert database.read_bytes() == b"existing reviewer work"
    runner.assert_not_called()


def test_partial_demo_requires_new_location_and_preserves_evidence(tmp_path):
    output = tmp_path / "execution"
    output.mkdir()
    evidence = output / "trace.json"
    evidence.write_text("unfinished", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preparación incompleta"):
        launcher.prepare_demo(Path("python"), tmp_path)
    assert evidence.read_text(encoding="utf-8") == "unfinished"


def test_busy_port_has_actionable_error():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(RuntimeError, match="ocupado.*--port"):
            launcher.ensure_free_port(sock.getsockname()[1])


def test_check_never_installs_or_starts_server(tmp_path, monkeypatch, capsys):
    installer = Mock()
    monkeypatch.setattr(launcher, "install", installer)
    assert launcher.main(["--check", "--venv", str(tmp_path / "missing")]) == 1
    installer.assert_not_called()
    assert "--install-only" in capsys.readouterr().err


def test_prepared_demo_cannot_enable_provider_implicitly():
    with pytest.raises(SystemExit) as error:
        launcher.main(["--demo", "--enable-ai"])
    assert error.value.code == 2
