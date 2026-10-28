"""Contrato del sistema visual: tokens, contraste y navegación escalable."""

import tomllib
from pathlib import Path

import pytest

from app.ui.accessibility import contrast_ratio, validate_contrast_pairs
from app.ui.navigation import StepStatus, WorkflowProgress, build_navigation
from app.ui.theme import build_theme_css
from app.ui.tokens import (
    CONTRAST_PAIRS,
    PRIMITIVE_COLORS,
    SEMANTIC_COLORS,
    SEMANTIC_SPACING,
    SEMANTIC_TYPE,
    SPACING,
    TYPE_SCALE,
    color,
    streamlit_theme,
)

ROOT = Path(__file__).parents[2]


def test_every_semantic_color_resolves_to_a_primitive() -> None:
    assert SEMANTIC_COLORS
    assert all(reference in PRIMITIVE_COLORS for reference in SEMANTIC_COLORS.values())


def test_all_required_color_pairs_pass_wcag_aa() -> None:
    results = validate_contrast_pairs(CONTRAST_PAIRS)
    text_pairs = [label for _, _, minimum, label in CONTRAST_PAIRS if minimum >= 4.5]

    assert results
    # Texto: 4,5:1 (AA). Bordes de componentes declaran 3:1 (WCAG 1.4.11).
    assert min(results[label] for label in text_pairs) >= 4.5
    assert min(results.values()) >= 3.0


def test_primary_action_has_enhanced_contrast() -> None:
    assert contrast_ratio(color("on-primary"), color("primary")) >= 7.0


def test_invalid_color_and_low_contrast_are_rejected() -> None:
    with pytest.raises(ValueError, match="Color hexadecimal inválido"):
        contrast_ratio("#fff", "#000000")
    with pytest.raises(ValueError, match="Contraste insuficiente"):
        validate_contrast_pairs((("#FFFFFF", "#FFFFFF", 4.5, "invisible"),))


def test_spacing_scale_is_monotonic_and_uses_quarter_rem_steps() -> None:
    values = [float(value.removesuffix("rem")) for value in SPACING.values() if value != "0"]

    assert values == sorted(values)
    assert all((value * 4).is_integer() for value in values)


def test_semantic_spacing_and_type_resolve_to_primitives() -> None:
    assert all(reference in SPACING for reference in SEMANTIC_SPACING.values())
    assert all(reference in TYPE_SCALE for reference in SEMANTIC_TYPE.values())


def test_theme_is_generated_from_semantic_tokens_after_validation() -> None:
    css = build_theme_css()

    assert f"--gsa-color-primary: {color('primary')}" in css
    assert f"--gsa-color-accent: {color('accent')}" in css
    assert "--gsa-touch-target: 44px" in css
    assert "min-height: var(--gsa-touch-target)" in css
    assert "focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_phone_layout_keeps_content_and_sidebar_inside_the_viewport() -> None:
    css = build_theme_css()

    assert "@media (max-width: 640px)" in css
    assert "--gsa-space-component: 1rem" in css
    assert "--gsa-space-page-bottom-compact: 3rem" in css
    assert (
        "padding: var(--gsa-space-component) var(--gsa-space-component) "
        "var(--gsa-space-page-bottom-compact)"
    ) in " ".join(css.split())
    assert "width: min(85vw, 20rem)" in css
    assert "overflow-x: hidden" in css


def test_sidebar_uses_streamlit_automatic_initial_state() -> None:
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert 'initial_sidebar_state="auto"' in source


def test_streamlit_native_theme_uses_the_validated_palette() -> None:
    with (ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
        theme = tomllib.load(config_file)["theme"]

    for key, value in streamlit_theme().items():
        assert theme[key] == value
    assert contrast_ratio(theme["textColor"], theme["backgroundColor"]) >= 4.5
    assert contrast_ratio("#FFFFFF", theme["primaryColor"]) >= 4.5


def test_navigation_unlocks_only_the_next_real_stage() -> None:
    initial = build_navigation(WorkflowProgress())
    approved = build_navigation(
        WorkflowProgress(
            has_project=True,
            has_dataset=True,
            has_context=True,
            has_training_approval=True,
        )
    )
    finished = build_navigation(
        WorkflowProgress(
            has_project=True,
            has_dataset=True,
            has_context=True,
            has_training_approval=True,
            has_reportable_run=True,
            report_published=True,
            model_usable=True,
            has_scoring=True,
        )
    )

    assert [item.status for item in initial] == [
        StepStatus.CURRENT,
        StepStatus.LOCKED,
        StepStatus.LOCKED,
        StepStatus.LOCKED,
        StepStatus.LOCKED,
    ]
    assert [item.status for item in approved] == [
        StepStatus.COMPLETE,
        StepStatus.COMPLETE,
        StepStatus.CURRENT,
        StepStatus.LOCKED,
        StepStatus.LOCKED,
    ]
    assert all(item.status == StepStatus.COMPLETE for item in finished)
