"""Verificador de afirmaciones: cada cifra de un insight debe salir de su evidencia.

Sirve igual para el reporte por plantilla y para el texto que proponga un LLM (AI-01): una
cifra que no coincida, tras redondeo, con un valor de sus `evidence_refs` invalida el reporte.
"""

import re
from typing import Any

from core.domain.models import ReportPayload

# Números sueltos; no toma dígitos pegados a letras ("v2", "B12") ni partes de otro número.
_NUMBER = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")
_ABSOLUTE_TOLERANCE = 1e-9


def resolve_evidence(metrics: dict[str, Any], ref: str) -> Any:
    """Resuelve `metrics.a.b.0.c` dentro del bloque de métricas del reporte."""

    parts = ref.split(".")
    if not parts or parts[0] != "metrics":
        raise KeyError(ref)
    value: Any = metrics
    for part in parts[1:]:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(ref)
    return value


def _leaves(value: Any) -> tuple[list[float], list[str]]:
    numbers: list[float] = []
    strings: list[str] = []
    if isinstance(value, bool) or value is None:
        return numbers, strings
    if isinstance(value, int | float):
        numbers.append(float(value))
    elif isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            nested_numbers, nested_strings = _leaves(item)
            numbers.extend(nested_numbers)
            strings.extend(nested_strings)
    elif isinstance(value, list):
        for item in value:
            nested_numbers, nested_strings = _leaves(item)
            numbers.extend(nested_numbers)
            strings.extend(nested_strings)
    return numbers, strings


def _supported(token: str, percent: bool, evidence: list[float]) -> bool:
    normalized = token.replace(",", ".")
    decimals = len(normalized.split(".")[1]) if "." in normalized else 0
    number = float(normalized)
    tolerance = 0.5 * 10**-decimals + _ABSOLUTE_TOLERANCE
    for value in evidence:
        if abs(number - value) <= tolerance:
            return True
        if percent and abs(number - value * 100) <= tolerance:
            return True
    return False


def verify_report(payload: ReportPayload) -> list[str]:
    """Devuelve errores; lista vacía significa que todas las cifras tienen evidencia."""

    errors: list[str] = []
    for index, insight in enumerate(payload.insights):
        evidence: list[float] = []
        names: list[str] = []
        for ref in insight.evidence_refs:
            try:
                numbers, strings = _leaves(resolve_evidence(payload.metrics, ref))
            except KeyError:
                errors.append(f"UNRESOLVED_EVIDENCE:{index}:{ref}")
                continue
            evidence.extend(numbers)
            names.extend(strings)
        text = insight.claim
        # Los nombres citados (p. ej. una feature "sensor_2") no son cifras afirmadas.
        for name in sorted(names, key=len, reverse=True):
            text = text.replace(name, " ")
        for match in _NUMBER.finditer(text):
            if not _supported(match.group(1), bool(match.group(2)), evidence):
                errors.append(f"UNSUPPORTED_NUMBER:{index}:{match.group(0).strip()}")
    return errors
