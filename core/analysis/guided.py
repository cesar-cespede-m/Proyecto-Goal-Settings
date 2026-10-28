"""Planes guiados sin IA: constructor, preguntas sugeridas y explicación en palabras (GA-04a).

Todo se decide por tipo de columna, nunca por nombre: sirve igual para cualquier negocio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.analysis.contracts import AnalysisPlanV1

COUNT = "__count__"
MEASURE_LABELS = {
    "sum": "Suma",
    "mean": "Promedio",
    "median": "Mediana",
    "min": "Mínimo",
    "max": "Máximo",
}
DATE_PARTS = {
    "hour_of_day": "hora del día",
    "day_of_week": "día de la semana (0 = lunes)",
    "day_part": "franja del día",
    "month": "mes",
    "date": "fecha",
    "week": "semana",
}
GROUPABLE = {"category", "datetime", "number"}
MAX_GROUPS = 50
SOURCE = "archivo"
MAX_SUGGESTIONS = 6


def build_guided_plan(
    schema: Mapping[str, str],
    *,
    group_by: str,
    date_part: str | None = None,
    measure: str = COUNT,
    function: str = "count",
    split_by: str | None = None,
    source: str = SOURCE,
) -> AnalysisPlanV1:
    """Contar o medir por grupos; con fechas, por la parte elegida."""

    steps: list[dict[str, Any]] = []
    key = group_by
    if schema[group_by] == "datetime":
        part = date_part or "date"
        key = f"{group_by} ({DATE_PARTS[part]})"
        steps.append(
            {"op": "derive_datetime", "column": group_by, "output_column": key, "part": part}
        )
    keys = [key] + ([split_by] if split_by else [])
    if measure == COUNT:
        value = "casos"
        aggregation: dict[str, Any] = {"function": "count", "output_column": value}
    else:
        value = f"{MEASURE_LABELS[function].lower()} de {measure}"
        aggregation = {"function": function, "column": measure, "output_column": value}
    steps.append({"op": "group_by", "by": keys, "aggregations": [aggregation]})
    ordered = schema[group_by] in {"datetime", "number"}
    steps.append({"op": "sort", "by": [key] if ordered else [value], "descending": not ordered})
    steps.append({"op": "limit", "rows": MAX_GROUPS})
    return AnalysisPlanV1.model_validate(
        {
            "sources": [source],
            "steps": steps,
            "output": {
                "chart": "line" if ordered else "bar",
                "x": key,
                "y": value,
                "color": split_by,
            },
        }
    )


@dataclass(frozen=True)
class SuggestedQuestion:
    text: str
    plan: AnalysisPlanV1


def suggested_questions(
    schemas: Mapping[str, Mapping[str, str]], *, limit: int = MAX_SUGGESTIONS
) -> list[SuggestedQuestion]:
    """Preguntas frecuentes de cualquier negocio, derivadas de los tipos de columna."""

    suggestions: list[SuggestedQuestion] = []
    for source, schema in schemas.items():
        dates = [name for name, kind in schema.items() if kind == "datetime"]
        categories = [name for name, kind in schema.items() if kind == "category"]
        numbers = [name for name, kind in schema.items() if kind == "number"]
        for date in dates[:1]:
            suggestions.append(
                SuggestedQuestion(
                    f"¿A qué hora del día hay más casos según «{date}»?",
                    build_guided_plan(
                        schema, group_by=date, date_part="hour_of_day", source=source
                    ),
                )
            )
            suggestions.append(
                SuggestedQuestion(
                    f"¿Qué día de la semana hay más casos según «{date}»?",
                    build_guided_plan(
                        schema, group_by=date, date_part="day_of_week", source=source
                    ),
                )
            )
        for category in categories[:2]:
            suggestions.append(
                SuggestedQuestion(
                    f"¿Cuántos casos hay en cada valor de «{category}»?",
                    build_guided_plan(schema, group_by=category, source=source),
                )
            )
        if categories and numbers:
            suggestions.append(
                SuggestedQuestion(
                    f"¿Cómo cambia el promedio de «{numbers[0]}» según «{categories[0]}»?",
                    build_guided_plan(
                        schema,
                        group_by=categories[0],
                        measure=numbers[0],
                        function="mean",
                        source=source,
                    ),
                )
            )
    return suggestions[:limit]


_FILTER_WORDS = {
    "eq": "es igual a",
    "ne": "es distinto de",
    "gt": "es mayor que",
    "gte": "es mayor o igual que",
    "lt": "es menor que",
    "lte": "es menor o igual que",
    "in": "está entre",
    "not_in": "no está entre",
    "is_null": "está vacío",
    "not_null": "no está vacío",
}
_FUNCTION_WORDS = {
    "count": "cantidad de casos",
    "nunique": "cantidad de valores distintos de",
    "sum": "suma de",
    "mean": "promedio de",
    "median": "mediana de",
    "min": "mínimo de",
    "max": "máximo de",
    "share": "proporción del total de",
}


def _quote(names: list[str]) -> str:
    return ", ".join(f"«{name}»" for name in names)


def describe_plan(plan: AnalysisPlanV1) -> list[str]:
    """Qué se calculó, paso a paso y en lenguaje común, sin IA."""

    lines = [f"Usé {_quote(list(plan.sources))}."]
    for step in plan.steps:
        data = step.model_dump()
        op = data["op"]
        if op == "select":
            lines.append(f"Me quedé con las columnas {_quote(data['columns'])}.")
        elif op == "rename":
            pairs = ", ".join(f"«{a}» como «{b}»" for a, b in data["mapping"].items())
            lines.append(f"Renombré {pairs}.")
        elif op == "filter":
            value = "" if data["value"] is None else f" {data['value']}"
            lines.append(
                f"Dejé sólo los casos donde «{data['column']}» "
                f"{_FILTER_WORDS[data['operator']]}{value}."
            )
        elif op == "derive_datetime":
            part = DATE_PARTS.get(data.get("part") or "", "fecha completa")
            lines.append(f"Saqué «{part}» de «{data['column']}».")
        elif op == "derive_bucket":
            lines.append(f"Agrupé «{data['column']}» en tramos.")
        elif op == "derive_ratio":
            lines.append(f"Dividí «{data['numerator']}» por «{data['denominator']}».")
        elif op == "union":
            lines.append(f"Agregué debajo las filas de «{data['source']}».")
        elif op == "join":
            lines.append(
                f"Crucé con «{data['source']}» uniendo «{data['left_on']}» con "
                f"«{data['right_on']}»."
            )
        elif op == "group_by":
            measures = ", ".join(
                _FUNCTION_WORDS[item["function"]]
                + (f" «{item['column']}»" if item.get("column") else "")
                for item in data["aggregations"]
            )
            target = f" por {_quote(data['by'])}" if data["by"] else " en total"
            lines.append(f"Calculé {measures}{target}.")
        elif op == "sort":
            order = "de mayor a menor" if data["descending"] else "de menor a mayor"
            lines.append(f"Ordené por {_quote(data['by'])} {order}.")
        elif op == "limit":
            lines.append(f"Mostré como máximo {data['rows']} filas.")
        elif op == "pivot":
            lines.append(
                f"Armé una tabla con filas {_quote(data['index'])} y columnas «{data['columns']}»."
            )
        elif op == "threshold_label":
            cut = (
                f"el percentil {round(data['percentile'] * 100)}"
                if data.get("percentile") is not None
                else f"el valor {data['threshold']}"
            )
            lines.append(f"Marqué «{data['column']}» como «{data['labels'][1]}» desde {cut}.")
    return lines
