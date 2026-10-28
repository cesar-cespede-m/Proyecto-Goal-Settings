"""Formateo determinista para presentar resultados sin alterar las métricas guardadas."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def format_pct(value: float | None) -> str:
    """Presenta una proporción como porcentaje con decimal y separador español."""

    if value is None:
        return "No disponible"
    return f"{value * 100:.1f}".replace(".", ",") + " %"


def format_number(value: float, decimals: int = 1) -> str:
    """Número con coma decimal y punto de miles, como se lee en español."""

    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def format_pct_range(lower: float, upper: float) -> str:
    """Presenta los extremos de un intervalo sin modificar sus valores."""

    return f"entre {format_pct(lower)} y {format_pct(upper)}"


def round_per_ten(value: float) -> int:
    """Convierte una proporción a una aproximación honesta de casos por cada diez."""

    return round(value * 10)


def format_local_datetime(value: str | None) -> str:
    """Fecha guardada en UTC, mostrada en la hora local del equipo como «22/10/2026, 00:25»."""

    if not value:
        return "fecha no disponible"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return f"{stamp.astimezone():%d/%m/%Y, %H:%M}"


def event_cases_label(positive_class: Any | None, target_column: str | None = None) -> str:
    """Devuelve una referencia breve al evento confirmado, si está disponible."""

    if positive_class is None or not str(positive_class).strip():
        return "casos del evento"
    if isinstance(positive_class, bool):
        label = "Sí" if positive_class else "No"
    else:
        label = str(positive_class).strip()
    if target_column:
        return f"casos en que «{target_column}» fue «{label}»"
    return f"casos de «{label}»"


def build_results_summary(
    *,
    has_recommendation: bool,
    recall: float | None = None,
    precision: float | None = None,
    baseline_recall: float | None = None,
    positive_class: Any | None = None,
    target_column: str | None = None,
) -> str:
    """Resume cifras ya calculadas con frases comunes y sin usar IA."""

    if not has_recommendation:
        return (
            "No hay un modelo recomendado. Ninguna alternativa le ganó a la referencia "
            "simple con las condiciones acordadas."
        )

    if recall is None or precision is None:
        return "Hay un modelo recomendado, pero faltan cifras para explicar su resultado."

    event_cases = event_cases_label(positive_class, target_column)
    sentences = [
        (
            f"En la prueba final, de cada 10 {event_cases}, el modelo avisó de unos "
            f"{round_per_ten(recall)} ({format_pct(recall)})."
        ),
        (
            f"De cada 10 avisos, unos {round_per_ten(precision)} fueron correctos "
            f"({format_pct(precision)})."
        ),
    ]
    if baseline_recall is not None:
        if baseline_recall == 0:
            sentences.append(
                "En los datos para comparar, la referencia simple no detectó ninguno "
                f"de esos {event_cases}."
            )
        else:
            sentences.append(
                f"En los datos para comparar, la referencia simple detectó unos "
                f"{round_per_ten(baseline_recall)} de cada 10 {event_cases} "
                f"({format_pct(baseline_recall)})."
            )
    return " ".join(sentences)


def selection_reason(
    selected_label: str,
    metric_label: str,
    chosen_value: float | None,
    others: list[tuple[str, float | None]],
) -> str:
    """Explica por qué se recomienda un modelo, sólo con las cifras de comparación."""

    known = [(label, value) for label, value in others if value is not None]
    if chosen_value is None or any(value > chosen_value for _, value in known):
        return f"Se recomienda **{selected_label}**, según las reglas que aprobaste."
    reason = (
        f"Se recomienda **{selected_label}** porque es la que mejor cumple lo que elegiste "
        f"como más importante («{metric_label}»: {format_pct(chosen_value)}"
    )
    if known:
        comparisons = ", ".join(f"{format_pct(value)} de {label}" for label, value in known)
        reason += f", frente a {comparisons}"
    return reason + "). Las demás cifras sirven para entender sus costos."


def importance_table(importance: list[dict[str, Any]]) -> pd.DataFrame:
    """Influencia de cada columna ordenada y relativa a la mayor, con el nombre completo."""

    values = {str(item["feature"]): max(0.0, float(item["importance_mean"])) for item in importance}
    top = max(values.values(), default=0.0)
    rows = sorted(values.items(), key=lambda pair: pair[1], reverse=True)
    return pd.DataFrame(
        {
            "Dato": [name for name, _ in rows],
            "Peso frente al que más pesa": [
                round(value / top * 100) if top > 0 else 0 for _, value in rows
            ],
        }
    )
