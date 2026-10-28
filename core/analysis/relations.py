"""GA-05: cómo se relacionan los archivos de un proyecto, de forma determinista y sin IA.

Encuentra columnas que sirven para cruzar dos archivos (mismo tipo, valores en común, cuántas
filas encajan y si la relación es uno a uno o muchos a uno), archivos que se pueden apilar
(mismas columnas) y desajustes de formato. Sólo mira tipos y valores, nunca nombres de negocio;
la persona confirma y la IA, si se usa, sólo lo explica.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from core.analysis.engine import LoadedSource

KEY_TYPES = {"identifier", "category", "number"}
MIN_MATCH_RATE = 0.3  # bajo esto no se propone como clave
LOW_MATCH_RATE = 0.8  # bajo esto se advierte que muchas filas no encajan
MAX_DISTINCT_SAMPLE = 50_000


@dataclass(frozen=True, slots=True)
class KeyCandidate:
    left: str
    left_column: str
    right: str
    right_column: str
    match_rate: float  # filas de `left` cuyo valor existe en `right`
    cardinality: str  # one_to_one, many_to_one, one_to_many o many_to_many
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StackCandidate:
    first: str
    second: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileRelations:
    keys: tuple[KeyCandidate, ...]
    stacks: tuple[StackCandidate, ...]


def _normalized(series: pd.Series) -> pd.Series:
    """Texto sin espacios ni ceros decimales sobrantes: «7», «7.0» y « 7 » son el mismo valor."""

    text = series.dropna().astype(str).str.strip()
    return text.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _cardinality(left: pd.Series, right: pd.Series) -> str:
    left_unique = left.is_unique
    right_unique = right.is_unique
    if left_unique and right_unique:
        return "one_to_one"
    if right_unique:
        return "many_to_one"
    if left_unique:
        return "one_to_many"
    return "many_to_many"


def _candidate(
    left_name: str,
    left_column: str,
    left_values: pd.Series,
    right_name: str,
    right_column: str,
    right_values: pd.Series,
) -> KeyCandidate | None:
    if left_values.empty or right_values.empty:
        return None
    right_set = set(right_values.head(MAX_DISTINCT_SAMPLE))
    match_rate = float(left_values.isin(right_set).mean())
    if match_rate < MIN_MATCH_RATE:
        return None
    # Una clave útil distingue casos: descartar columnas con muy pocos valores distintos.
    if left_values.nunique() < 3 or right_values.nunique() < 3:
        return None
    cardinality = _cardinality(left_values, right_values)
    warnings: list[str] = []
    if match_rate < LOW_MATCH_RATE:
        warnings.append(f"LOW_MATCH_RATE:{match_rate:.2f}")
    if cardinality in {"one_to_many", "many_to_many"}:
        warnings.append("ROWS_WOULD_MULTIPLY")
    return KeyCandidate(
        left=left_name,
        left_column=left_column,
        right=right_name,
        right_column=right_column,
        match_rate=round(match_rate, 4),
        cardinality=cardinality,
        warnings=tuple(warnings),
    )


def find_relations(sources: Mapping[str, LoadedSource], *, limit: int = 20) -> FileRelations:
    """Claves para cruzar y archivos apilables entre todos los pares de archivos."""

    names = list(sources)
    keys: list[KeyCandidate] = []
    stacks: list[StackCandidate] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = sources[left_name], sources[right_name]
            if set(left.schema) == set(right.schema) and all(
                left.schema[column] == right.schema[column] for column in left.schema
            ):
                stacks.append(StackCandidate(left_name, right_name, tuple(left.schema)))
                continue
            for left_column, left_type in left.schema.items():
                if getattr(left_type, "value", str(left_type)) not in KEY_TYPES:
                    continue
                left_values = _normalized(left.frame[left_column])
                for right_column, right_type in right.schema.items():
                    right_kind = getattr(right_type, "value", str(right_type))
                    if right_kind not in KEY_TYPES:
                        continue
                    right_values = _normalized(right.frame[right_column])
                    for a, a_col, a_values, b, b_col, b_values in (
                        (
                            left_name,
                            left_column,
                            left_values,
                            right_name,
                            right_column,
                            right_values,
                        ),
                        (
                            right_name,
                            right_column,
                            right_values,
                            left_name,
                            left_column,
                            left_values,
                        ),
                    ):
                        found = _candidate(a, a_col, a_values, b, b_col, b_values)
                        if found is not None:
                            keys.append(found)
    keys.sort(
        key=lambda item: (
            item.cardinality not in {"many_to_one", "one_to_one"},
            -item.match_rate,
            item.left,
            item.left_column,
        )
    )
    return FileRelations(keys=tuple(keys[:limit]), stacks=tuple(stacks))


def describe_relations(relations: FileRelations) -> list[str]:
    """Las relaciones en palabras, para la persona y como base para la explicación de la IA."""

    lines: list[str] = []
    for key in relations.keys:
        share = round(key.match_rate * 100)
        kind = {
            "one_to_one": "cada fila encaja con una sola fila",
            "many_to_one": "varias filas pueden apuntar a la misma fila del otro archivo",
            "one_to_many": "cada fila encaja con varias filas del otro: al cruzar se multiplican",
            "many_to_many": "hay repetidos en ambos lados: al cruzar se multiplican filas",
        }[key.cardinality]
        lines.append(
            f"«{key.left}.{key.left_column}» encaja con «{key.right}.{key.right_column}» en el "
            f"{share} % de las filas; {kind}."
        )
    for stack in relations.stacks:
        lines.append(f"«{stack.first}» y «{stack.second}» tienen las mismas columnas: se apilan.")
    return lines
