"""GA-03: tipos de análisis automáticos, registrados y ampliables.

Cada tipo decide si aplica según los tipos de columna (nunca por nombres) y arma planes
`AnalysisPlanV1` que ejecuta el mismo motor. Agregar otro tipo (cohortes, estacionalidad,
anomalías) es escribir una clase con `applies` y `plans` y sumarla a `ANALYSIS_KINDS`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.guided import SOURCE, build_guided_plan

MAX_PLANS_PER_KIND = 4
MAX_BINS = 5


@dataclass(frozen=True)
class KindPlan:
    title: str
    plan: AnalysisPlanV1


class AnalysisKind(Protocol):
    key: str
    title: str
    description: str

    def applies(self, schema: Mapping[str, str], *, outcome: str | None = None) -> bool: ...

    def plans(
        self,
        schema: Mapping[str, str],
        *,
        source: str = SOURCE,
        outcome: str | None = None,
        missing: Mapping[str, int] | None = None,
    ) -> list[KindPlan]: ...


def _of(schema: Mapping[str, str], kind: str, *, skip: str | None = None) -> list[str]:
    return [name for name, value in schema.items() if value == kind and name != skip]


def _plan(source: str, steps: list[dict[str, Any]], output: dict[str, Any]) -> AnalysisPlanV1:
    return AnalysisPlanV1.model_validate({"sources": [source], "steps": steps, "output": output})


class Descriptive:
    key = "descriptive"
    title = "Resumen"
    description = "Cuántos casos hay en cada categoría y cómo se reparten los números."

    def applies(self, schema, *, outcome=None):
        return bool(_of(schema, "category") or _of(schema, "number"))

    def plans(self, schema, *, source=SOURCE, outcome=None, missing=None):
        found = [
            KindPlan(
                f"Casos por «{column}»", build_guided_plan(schema, group_by=column, source=source)
            )
            for column in _of(schema, "category")[:2]
        ]
        for column in _of(schema, "number")[:2]:
            tramo = f"{column} (tramo)"
            found.append(
                KindPlan(
                    f"Cómo se reparte «{column}»",
                    _plan(
                        source,
                        [
                            {
                                "op": "derive_bucket",
                                "column": column,
                                "output_column": tramo,
                                "method": "quantile",
                                "bins": MAX_BINS,
                            },
                            {
                                "op": "group_by",
                                "by": [tramo],
                                "aggregations": [{"function": "count", "output_column": "casos"}],
                            },
                            {"op": "sort", "by": [tramo]},
                        ],
                        {"chart": "bar", "x": tramo, "y": "casos"},
                    ),
                )
            )
        return found[:MAX_PLANS_PER_KIND]


class Temporal:
    key = "temporal"
    title = "En el tiempo"
    description = "Cómo cambian los casos por hora, día de la semana, semana y mes."

    def applies(self, schema, *, outcome=None):
        return bool(_of(schema, "datetime"))

    def plans(self, schema, *, source=SOURCE, outcome=None, missing=None):
        date = _of(schema, "datetime")[0]
        labels = {
            "hour_of_day": "hora del día",
            "day_of_week": "día de la semana",
            "week": "semana",
            "month": "mes",
        }
        return [
            KindPlan(
                f"Casos por {label} según «{date}»",
                build_guided_plan(schema, group_by=date, date_part=part, source=source),
            )
            for part, label in labels.items()
        ]


class Segmentation:
    key = "segmentation"
    title = "Comparar grupos"
    description = "Cómo cambia el promedio de cada número entre los grupos."

    def applies(self, schema, *, outcome=None):
        return bool(_of(schema, "category") and _of(schema, "number", skip=outcome))

    def plans(self, schema, *, source=SOURCE, outcome=None, missing=None):
        group = next(name for name in _of(schema, "category") if name != outcome)
        return [
            KindPlan(
                f"Promedio de «{number}» por «{group}»",
                build_guided_plan(
                    schema, group_by=group, measure=number, function="mean", source=source
                ),
            )
            for number in _of(schema, "number", skip=outcome)[:MAX_PLANS_PER_KIND]
        ]


class OutcomeRelation:
    key = "outcome"
    title = "Relación con el resultado"
    description = "Cómo se reparte lo que quieres anticipar dentro de cada grupo."

    def applies(self, schema, *, outcome=None):
        return (
            outcome is not None
            and schema.get(outcome) == "category"
            and bool(_of(schema, "category", skip=outcome))
        )

    def plans(self, schema, *, source=SOURCE, outcome=None, missing=None):
        assert outcome is not None
        return [
            KindPlan(
                f"«{outcome}» según «{group}»",
                build_guided_plan(schema, group_by=group, split_by=outcome, source=source),
            )
            for group in _of(schema, "category", skip=outcome)[:MAX_PLANS_PER_KIND]
        ]


class DataQuality:
    key = "quality"
    title = "Calidad de los datos"
    description = "Cuántos casos traen vacía cada columna que tiene huecos."

    def applies(self, schema, *, outcome=None):
        return True

    def plans(self, schema, *, source=SOURCE, outcome=None, missing=None):
        holes = sorted(
            ((count, name) for name, count in (missing or {}).items() if count and name in schema),
            reverse=True,
        )
        return [
            KindPlan(
                f"Casos con «{name}» vacío",
                _plan(
                    source,
                    [
                        {"op": "filter", "column": name, "operator": "is_null"},
                        {
                            "op": "group_by",
                            "by": [],
                            "aggregations": [{"function": "count", "output_column": "casos"}],
                        },
                    ],
                    {"chart": "table"},
                ),
            )
            for _, name in holes[:MAX_PLANS_PER_KIND]
        ]


ANALYSIS_KINDS: tuple[AnalysisKind, ...] = (
    Descriptive(),
    Temporal(),
    Segmentation(),
    OutcomeRelation(),
    DataQuality(),
)


def applicable_kinds(
    schema: Mapping[str, str], *, outcome: str | None = None
) -> list[AnalysisKind]:
    return [kind for kind in ANALYSIS_KINDS if kind.applies(schema, outcome=outcome)]


def missing_counts(profile: Mapping[str, Any]) -> dict[str, int]:
    """Vacíos por columna desde el perfil guardado (0 si el perfil es antiguo)."""

    return {
        str(column["name"]): int(column.get("missing_count") or 0)
        for column in profile.get("columns", [])
    }
