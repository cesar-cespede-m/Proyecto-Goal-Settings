"""GA-14: el código del producto no conoce ningún negocio de prueba.

Reúne los nombres de columna y los valores de texto repetidos de todos los archivos de
`data/synthetic` (incluidas sus subcarpetas) y falla si alguno aparece como palabra completa en
`core/` o `app/`. Las pruebas, la documentación, los datos y los generadores pueden nombrarlos.
"""

import re
from pathlib import Path

import pandas as pd

from core.analysis.roles import infer_roles
from core.data_engine import validate_and_profile

ROOT = Path(__file__).parents[2]
PRODUCT_DIRS = ("core", "app")
MIN_LENGTH = 4
MAX_VALUES_PER_COLUMN = 50

# Palabras de uso común en español que también aparecen en algún archivo de prueba. Cada una
# está justificada: no es un concepto de negocio sino vocabulario del producto o del idioma.
COMMON_WORDS = {
    "plan",  # plan de análisis (AnalysisPlanV1) y plan de entrenamiento
    "revisar",  # «puntos para revisar»
    "pendiente",  # estados de revisión
    "tarjeta",  # tarjetas de la interfaz
    "franja",  # franja visual de cada etapa
    "noche",  # tramo horario genérico exigido por derive_datetime(day_part)
    "turno",  # «propuesta por turno» de la conversación con la IA
    "estándar",  # «entrada estándar» del transporte de evidencia
    "baja",  # nivel genérico de importancia, riesgo o intensidad
    "formulario",  # control genérico de entrada en la interfaz
    "normal",  # nivel genérico y nombre de un token visual
    "recibida",  # estado genérico de recepción de eventos externos
    "registro",  # acción transversal de guardar o auditar información
    "moneda",  # unidad económica transversal, no nombre propio de un negocio sintético
}


def _test_business_terms() -> set[str]:
    terms: set[str] = set()
    for path in (ROOT / "data" / "synthetic").rglob("*.csv"):
        try:
            frame = pd.read_csv(path, nrows=2000)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue  # archivos mal formados a propósito
        terms.update(str(column) for column in frame.columns)
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            values = frame[column].dropna().astype(str)
            if values.nunique() <= MAX_VALUES_PER_COLUMN:
                terms.update(values.unique())
    return {
        term
        for term in terms
        if len(term) >= MIN_LENGTH
        and not re.fullmatch(r"[\d\W_]+", term)
        and term.casefold() not in COMMON_WORDS
    }


def test_product_code_does_not_name_any_test_business() -> None:
    sources = {
        path: path.read_text(encoding="utf-8")
        for folder in PRODUCT_DIRS
        for path in (ROOT / folder).rglob("*.py")
    }
    found = {}
    for term in sorted(_test_business_terms()):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        files = [
            str(path.relative_to(ROOT)) for path, text in sources.items() if pattern.search(text)
        ]
        if files:
            found[term] = files

    assert not found, f"Términos de negocios de prueba en el producto: {found}"


def test_the_audit_would_catch_a_leak() -> None:
    terms = {term.casefold() for term in _test_business_terms()}

    assert len(terms) > 100
    assert any("_" in term for term in terms)  # incluye nombres de columna compuestos


def test_column_roles_do_not_depend_on_column_names() -> None:
    frame = pd.DataFrame(
        {
            "columna_a": ["x1", "x1", "x2", "x2", "x2", "x3", "x3"],
            "columna_b": [
                "2026-01-01 09:00:00",
                "2026-01-02 09:00:00",
                "2026-02-01 09:00:00",
                "2026-02-02 09:00:00",
                "2026-02-03 09:00:00",
                "2026-03-01 09:00:00",
                "2026-03-02 09:00:00",
            ],
            "columna_c": ["inicio", "fin", "inicio", "medio", "fin", "inicio", "medio"],
            "columna_d": [1.2, 2.1, 3.4, 2.8, 4.1, 1.0, 1.8],
        }
    )
    renamed = frame.rename(
        columns={
            "columna_a": "campo_w",
            "columna_b": "campo_x",
            "columna_c": "campo_y",
            "columna_d": "campo_z",
        }
    )

    def roles_by_position(candidate: pd.DataFrame) -> list[str]:
        content = candidate.to_csv(index=False).encode()
        profile = validate_and_profile("datos.csv", content)
        return [item.role for item in infer_roles(candidate, profile).items]

    assert roles_by_position(frame) == roles_by_position(renamed)
