"""El responsable es un nombre o rol corto; nunca se copia texto a una aprobación."""

import pytest

from app.ui.assistant_view import looks_like_name


@pytest.mark.parametrize("text", ["Ana", "Ana, jefa de logística", "Responsable de prueba"])
def test_names_and_roles_are_accepted(text: str) -> None:
    assert looks_like_name(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Necesito comprender como los usuarios antiguos abandonan y tratar de entender que "
        "hacer para que no abandonen.",
        "¿Qué clientes se van a ir?",
        "Quiero saber qué envíos llegarán tarde para avisar",
    ],
)
def test_descriptions_are_rejected(text: str) -> None:
    assert not looks_like_name(text)
