"""UX-03: el evento a detectar se lee sin código y advierte la elección mayoritaria."""

from app.ui.assistant_view import event_option_label, majority_event_warning


def test_option_label_has_no_code_quotes_and_uses_thousands_separator() -> None:
    label = event_option_label("entregado_tarde", "sí", 2721)
    assert label == "Que «entregado_tarde» sea «sí» (2.721 casos)"
    assert "'" not in label


def test_boolean_values_are_readable() -> None:
    assert "«verdadero»" in event_option_label("hubo_falla", True, 10)


def test_majority_choice_is_warned() -> None:
    counts = {"no": 2721, "sí": 779}
    message = majority_event_warning("no", counts)
    assert message is not None
    assert "2.721 de 3.500" in message


def test_minority_or_balanced_choice_is_not_warned() -> None:
    assert majority_event_warning("sí", {"no": 2721, "sí": 779}) is None
    assert majority_event_warning(1, {0: 100, 1: 100}) is None
