"""Recorrido 22/10: el panel mostraba 6 tarjetas mientras el encabezado decía «de 5»."""

import pytest

from app.ui import components
from app.ui.navigation import NavigationItem, StepStatus


def _item(number: int) -> NavigationItem:
    return NavigationItem(number, f"Paso {number}", f"a{number}", StepStatus.LOCKED, "Falta algo")


def test_steps_after_approval_are_separated_from_the_five_preparation_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(components.st, "markdown", lambda body, **_: shown.append(body))

    components.render_workflow_navigation(tuple(_item(number) for number in range(1, 7)))

    html = shown[0]
    divider = html.index("Cuando el modelo ya está aprobado")
    assert html.index("Paso 5") < divider < html.index("Paso 6")
    assert html.count("gsa-nav-divider") == 1


def test_view_moves_only_when_the_current_step_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, str] = {}
    scripts: list[str] = []
    monkeypatch.setattr(components.st, "session_state", state)
    monkeypatch.setattr(components.components, "html", lambda body, **_: scripts.append(body))

    def items(current: int) -> tuple[NavigationItem, ...]:
        return tuple(
            NavigationItem(
                number,
                f"Paso {number}",
                f"a{number}",
                StepStatus.CURRENT if number == current else StepStatus.LOCKED,
                "",
            )
            for number in range(1, 4)
        )

    assert components.scroll_to_new_step("p", items(1)) is False  # primera vez: no se mueve
    assert components.scroll_to_new_step("p", items(1)) is False  # mismo paso
    assert components.scroll_to_new_step("p", items(2)) is True
    assert len(scripts) == 1 and '"a2"' in scripts[0]


def test_profile_counts_use_spanish_thousands_separator() -> None:
    assert components._count(3500) == "3.500"
    assert components._count(12) == "12"
    assert components._count("—") == "—"
