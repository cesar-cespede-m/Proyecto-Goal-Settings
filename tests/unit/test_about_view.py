"""V2-06: lo que hace Goal Settings y lo que no, dentro del producto."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.ui.about_view import CAN, CANNOT, LATER

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


def test_every_limit_comes_with_an_alternative() -> None:
    assert CAN and LATER
    assert all(item and alternative for item, alternative in CANNOT)


def test_first_screen_explains_the_product(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert any(
        item.label == "Qué hace Goal Settings, cómo funciona y qué no puede hacer"
        for item in app.expander
    )
    texts = " ".join(item.value for item in app.markdown)
    assert "Todavía no puedes" in texts and "25 MB" in texts


def test_without_ai_the_app_says_so_and_does_not_promise_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert any("Modo sin IA generativa" in item.value for item in app.info)
    assert not any("La IA redacta textos" in item.value for item in app.warning)
