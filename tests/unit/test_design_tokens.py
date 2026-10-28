"""Paleta «Grafito y pizarra»: una sola fuente de verdad y contraste AA verificado."""

import re
from pathlib import Path

from app.ui.accessibility import validate_contrast_pairs
from app.ui.theme import build_theme_css
from app.ui.tokens import CONTRAST_PAIRS, SEMANTIC_COLORS, color, streamlit_theme

ROOT = Path(__file__).parents[2]


def test_every_contrast_pair_meets_its_minimum() -> None:
    results = validate_contrast_pairs(CONTRAST_PAIRS)

    assert results["acento sobre superficie"] >= 4.5
    assert results["borde de elemento bloqueado"] >= 3.0


def test_streamlit_config_is_generated_from_the_tokens() -> None:
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    for key, value in streamlit_theme().items():
        assert f'{key} = "{value}"' in config, f"{key}: ejecuta scripts/sync_streamlit_theme.py"


def test_state_colors_are_reserved_and_distinct_from_actions_and_accent() -> None:
    states = {color(name) for name in ("success", "warning", "error")}

    assert color("primary") not in states and color("accent") not in states
    assert color("accent") != color("primary")
    assert SEMANTIC_COLORS["danger"] == SEMANTIC_COLORS["error"]


def test_theme_uses_tokens_instead_of_literal_colors() -> None:
    source = (ROOT / "app" / "ui" / "theme.py").read_text(encoding="utf-8")

    assert not re.findall(r"#[0-9A-Fa-f]{6}\b", source)
    assert "var(--gsa-color-accent)" in build_theme_css()


def test_component_spacing_and_touch_targets_come_from_tokens() -> None:
    source = (ROOT / "app" / "ui" / "theme.py").read_text(encoding="utf-8")
    assert not re.findall(
        r"(?:padding|margin|gap|min-height|border-radius):[^;\n]*\d+(?:\.\d+)?(?:rem|px)",
        source,
    )
    assert "min-height: var(--gsa-touch-target)" in build_theme_css()
    assert "var(--gsa-color-primary-hover)" in build_theme_css()
