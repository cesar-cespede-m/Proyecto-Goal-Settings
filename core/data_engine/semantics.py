"""Tipo de cada columna para el análisis (GA-01), sin conocer ningún negocio.

Se deduce sólo de los valores y de patrones genéricos de datos personales; nunca de nombres
de un dominio. Es un aviso: la persona decide qué usar y qué enviar a la IA (ADR-014).
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_numeric_dtype

SemanticType = Literal["number", "category", "datetime", "text", "identifier", "personal"]

# Palabras genéricas de datos personales, en español e inglés (no son de ningún negocio).
_PERSONAL_WORDS = {
    "email",
    "correo",
    "mail",
    "telefono",
    "phone",
    "celular",
    "movil",
    "rut",
    "dni",
    "pasaporte",
    "passport",
    "nombre",
    "name",
    "apellido",
    "surname",
    "direccion",
    "address",
    "domicilio",
    "ip",
    "nacimiento",
    "birth",
}
_CONTACT_KINDS = {"EMAIL", "URL", "RUT", "PHONE", "SECRET"}
DATETIME_MIN_SHARE = 0.95
CATEGORY_MAX_UNIQUE = 1000
CATEGORY_MAX_RATIO = 0.5
PERSONAL_VALUE_SHARE = 0.5
_SAMPLE = 500


def _words(name: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return {word for word in re.split(r"[^a-z0-9]+", plain) if word}


def looks_personal(name: str, series: pd.Series) -> bool:
    """Nombre con una palabra de dato personal, o mayoría de valores con forma de contacto."""

    if _words(name) & _PERSONAL_WORDS:
        return True
    if is_numeric_dtype(series):
        return False
    # Import diferido: el detector vive en ai_engine, que depende de data_engine.
    from core.ai_engine.outgoing_review import redact

    sample = series.dropna().astype(str).head(_SAMPLE)
    if sample.empty:
        return False
    # Sólo formas de contacto o secretos; un número largo suele ser un código, no una persona.
    flagged = sum(bool(set(redact(value)[1]) & _CONTACT_KINDS) for value in sample)
    return flagged / len(sample) >= PERSONAL_VALUE_SHARE


def looks_datetime(series: pd.Series) -> bool:
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).head(_SAMPLE)
    if sample.empty or not sample.str.contains(r"\d", regex=True).all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= DATETIME_MIN_SHARE


def semantic_type(
    name: str,
    series: pd.Series,
    *,
    possible_identifier: bool,
    inferred_type: str,
) -> SemanticType:
    """Prioridad: fecha > personal > identificador > número > categoría > texto."""

    if looks_datetime(series) and not _words(name) & _PERSONAL_WORDS:
        return "datetime"
    if looks_personal(name, series):
        return "personal"
    if possible_identifier and not is_float_dtype(series):
        return "identifier"
    if inferred_type == "numeric":
        return "number"
    if inferred_type in {"boolean", "categorical"}:
        return "category"
    present = series.dropna()
    unique = present.nunique()
    if (
        present.size
        and unique <= CATEGORY_MAX_UNIQUE
        and unique / present.size <= CATEGORY_MAX_RATIO
    ):
        return "category"
    return "text"


_FALLBACK: dict[str, SemanticType] = {
    "numeric": "number",
    "boolean": "category",
    "categorical": "category",
    "string": "text",
}


def analysis_schema(profile: dict) -> dict[str, SemanticType]:
    """Esquema columna → tipo para validar planes de análisis (contrato con GA-00).

    Los perfiles guardados antes de GA-01 no tienen `semantic_type`: se deduce del tipo
    técnico y de la marca de identificador.
    """

    schema: dict[str, SemanticType] = {}
    for column in profile["columns"]:
        kind = column.get("semantic_type")
        if kind is None:
            kind = (
                "identifier"
                if column.get("possible_identifier") and column["inferred_type"] != "numeric"
                else _FALLBACK[column["inferred_type"]]
            )
        schema[str(column["name"])] = kind
    return schema
