"""GA-03b: hallazgos deterministas de calidad a partir de tipos y valores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field

from core.analysis.contracts import StrictAnalysisModel
from core.analysis.engine import LoadedSource
from core.analysis.relations import find_relations

type QualityKind = Literal[
    "EXTREME_OUTLIER",
    "DATE_INVALID_OR_OUT_OF_RANGE",
    "POSSIBLE_SIGN_ERROR",
    "NEAR_DUPLICATE_CATEGORY",
    "KEY_MISMATCH",
]

MIN_REASONABLE_DATE = pd.Timestamp("1900-01-01", tz="UTC")
MAX_DETAIL_VARIANTS = 10


class QualityFindingV1(StrictAnalysisModel):
    """Hallazgo acotado y serializable, sin campos implícitos ni coerciones."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    source: str = Field(min_length=1)
    column: str = Field(min_length=1)
    kind: QualityKind
    count: int = Field(gt=0)
    share: float = Field(ge=0, le=1)
    detail: str = Field(min_length=1)


def _kind(value: object) -> str:
    return str(getattr(value, "value", value))


def _finding(
    source: str,
    column: str,
    kind: QualityKind,
    count: int,
    total: int,
    detail: str,
) -> QualityFindingV1:
    return QualityFindingV1(
        source=source,
        column=column,
        kind=kind,
        count=count,
        share=round(count / total, 4),
        detail=detail,
    )


def _numeric_findings(source: str, column: str, values: pd.Series) -> list[QualityFindingV1]:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    populated = numeric.dropna()
    if populated.empty:
        return []

    findings: list[QualityFindingV1] = []
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty:
        first_quartile = float(finite.quantile(0.25))
        third_quartile = float(finite.quantile(0.75))
        spread = third_quartile - first_quartile
        lower = first_quartile - 3 * spread
        upper = third_quartile + 3 * spread
        extreme = numeric.notna() & ((numeric < lower) | (numeric > upper))
        count = int(extreme.sum())
        if count:
            findings.append(
                _finding(
                    source,
                    column,
                    "EXTREME_OUTLIER",
                    count,
                    len(populated),
                    f"{count} valores quedan fuera del intervalo esperado [{lower:g}, {upper:g}].",
                )
            )

    negative = numeric.notna() & numeric.lt(0)
    non_negative = numeric.notna() & numeric.ge(0)
    negative_count = int(negative.sum())
    if negative_count and bool(non_negative.any()):
        findings.append(
            _finding(
                source,
                column,
                "POSSIBLE_SIGN_ERROR",
                negative_count,
                len(populated),
                f"{negative_count} valores son negativos y los restantes no lo son.",
            )
        )
    return findings


def _datetime_findings(source: str, column: str, values: pd.Series) -> list[QualityFindingV1]:
    present = values.notna()
    total = int(present.sum())
    if not total:
        return []

    parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    invalid = present & parsed.isna()
    too_old = parsed.notna() & parsed.lt(MIN_REASONABLE_DATE)
    tomorrow = pd.Timestamp(date.today(), tz="UTC") + pd.Timedelta(days=1)
    future = parsed.notna() & parsed.ge(tomorrow)
    count = int((invalid | too_old | future).sum())
    if not count:
        return []
    return [
        _finding(
            source,
            column,
            "DATE_INVALID_OR_OUT_OF_RANGE",
            count,
            total,
            (
                f"{int(invalid.sum())} no se pueden interpretar, {int(too_old.sum())} son "
                f"anteriores a 1900 y {int(future.sum())} son futuras."
            ),
        )
    ]


def _normalized_category(value: object) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _category_findings(source: str, column: str, values: pd.Series) -> list[QualityFindingV1]:
    present = values.dropna()
    if present.empty:
        return []

    variants: dict[str, set[str]] = {}
    normalized: list[str] = []
    for value in present:
        original = str(value)
        key = _normalized_category(original)
        normalized.append(key)
        if key:
            variants.setdefault(key, set()).add(original)
    duplicate_keys = {key for key, originals in variants.items() if len(originals) > 1}
    count = sum(key in duplicate_keys for key in normalized)
    if not count:
        return []

    examples = [
        " / ".join(
            repr(item) for item in sorted(variants[key], key=lambda item: (item.casefold(), item))
        )
        for key in sorted(duplicate_keys)
    ]
    shown = examples[:MAX_DETAIL_VARIANTS]
    suffix = "" if len(examples) == len(shown) else f" y {len(examples) - len(shown)} grupos más"
    return [
        _finding(
            source,
            column,
            "NEAR_DUPLICATE_CATEGORY",
            count,
            len(present),
            f"Variantes que sólo cambian en mayúsculas o espacios: {'; '.join(shown)}{suffix}.",
        )
    ]


def _normalized_key(values: pd.Series) -> pd.Series:
    text = values.dropna().astype(str).str.strip()
    return text.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _relation_findings(sources: Mapping[str, LoadedSource]) -> list[QualityFindingV1]:
    findings: list[QualityFindingV1] = []
    for key in find_relations(sources).keys:
        left = _normalized_key(sources[key.left].frame[key.left_column])
        right = set(_normalized_key(sources[key.right].frame[key.right_column]))
        count = int((~left.isin(right)).sum())
        if count:
            findings.append(
                _finding(
                    key.left,
                    key.left_column,
                    "KEY_MISMATCH",
                    count,
                    len(left),
                    f"{count} valores no existen en «{key.right}.{key.right_column}».",
                )
            )
    return findings


def quality_findings(sources: Mapping[str, LoadedSource]) -> list[QualityFindingV1]:
    """Detecta problemas de calidad usando solamente tipos declarados y valores."""

    ordered_sources = {name: sources[name] for name in sorted(sources)}
    findings: list[QualityFindingV1] = []
    for source_name, source in ordered_sources.items():
        for column in sorted(source.schema):
            column_kind = _kind(source.schema[column])
            values = source.frame[column]
            if column_kind == "number":
                findings.extend(_numeric_findings(source_name, column, values))
            elif column_kind == "datetime":
                findings.extend(_datetime_findings(source_name, column, values))
            elif column_kind == "category":
                findings.extend(_category_findings(source_name, column, values))
    findings.extend(_relation_findings(ordered_sources))
    return findings


def describe_findings(findings: Sequence[QualityFindingV1]) -> list[str]:
    """Explica los hallazgos con frases breves para personas no técnicas."""

    descriptions = {
        "EXTREME_OUTLIER": "valores numéricos muy alejados del resto",
        "DATE_INVALID_OR_OUT_OF_RANGE": "fechas que no se pueden usar o están fuera de rango",
        "POSSIBLE_SIGN_ERROR": "valores negativos que podrían tener el signo equivocado",
        "NEAR_DUPLICATE_CATEGORY": (
            "valores de categoría repetidos con diferencias de mayúsculas o espacios"
        ),
        "KEY_MISMATCH": "valores que no aparecen en la columna relacionada",
    }
    return [
        (
            f"«{finding.source}.{finding.column}»: {finding.count} "
            f"{descriptions[finding.kind]} ({finding.share:.1%} de los valores revisados)."
        )
        for finding in findings
    ]
