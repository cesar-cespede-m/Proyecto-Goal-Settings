"""Comprobaciones WCAG para el sistema visual de Goal Settings."""

from collections.abc import Iterable


def _rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"Color hexadecimal inválido: {hex_color}")
    try:
        channels = tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))
    except ValueError as error:
        raise ValueError(f"Color hexadecimal inválido: {hex_color}") from error
    return channels  # type: ignore[return-value]


def relative_luminance(hex_color: str) -> float:
    """Calcula luminancia relativa según WCAG 2.x."""

    linear = tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in _rgb(hex_color)
    )
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Retorna el contraste entre dos colores, entre 1:1 y 21:1."""

    first = relative_luminance(foreground)
    second = relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def validate_contrast_pairs(
    pairs: Iterable[tuple[str, str, float, str]],
) -> dict[str, float]:
    """Valida pares foreground/background y devuelve evidencia numérica."""

    results: dict[str, float] = {}
    failures: list[str] = []
    for foreground, background, minimum, label in pairs:
        ratio = contrast_ratio(foreground, background)
        results[label] = ratio
        if ratio < minimum:
            failures.append(f"{label}: {ratio:.2f}:1 < {minimum:.2f}:1")
    if failures:
        raise ValueError("Contraste insuficiente: " + "; ".join(failures))
    return results
