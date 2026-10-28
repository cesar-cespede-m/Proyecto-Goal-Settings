"""UX-05: advertencias técnicas traducidas a lenguaje común junto a cada decisión.

Los códigos los produce `core/`; aquí sólo se explican. Se separan los problemas que la
persona debe revisar de las notas que describen cómo trabaja el método.
"""

from typing import Literal, NamedTuple

import streamlit as st

Kind = Literal["review", "note"]


class WarningText(NamedTuple):
    kind: Kind
    text: str


def _columns(detail: str) -> str:
    names = [name for name in detail.split(",") if name]
    return ", ".join(f"«{name}»" for name in names)


def describe_warning(code: str) -> WarningText:
    """Traduce un código de advertencia; los desconocidos se muestran sin ocultarlos."""

    code = code.removeprefix("DATA_PROFILE:")
    name, _, detail = code.partition(":")
    if name == "DUPLICATE_ROWS":
        return WarningText(
            "review",
            f"Hay {detail} filas repetidas en el archivo. Si no son casos reales distintos, "
            "pueden hacer que los resultados parezcan mejores de lo que son.",
        )
    if name == "CONSTANT_COLUMNS":
        return WarningText(
            "review",
            f"{_columns(detail)} tiene el mismo valor en todas las filas, así que no ayuda a "
            "distinguir casos. Conviene excluirla.",
        )
    if name == "HIGH_CARDINALITY_COLUMNS":
        return WarningText(
            "review",
            f"{_columns(detail)} tiene casi un valor distinto por fila. Si es un código o un "
            "texto libre, el modelo puede memorizarlo en lugar de aprender. Revisa si "
            "conviene excluirla.",
        )
    if name == "POSSIBLE_IDENTIFIER_COLUMNS":
        return WarningText(
            "review",
            f"{_columns(detail)} parece un identificador (como un número de folio o de "
            "registro). Identifica cada caso pero no explica el resultado: conviene excluirla.",
        )
    if name == "POSSIBLE_PERSONAL_DATA_COLUMNS":
        return WarningText(
            "review",
            f"{_columns(detail)} parece contener datos personales. Tú decides si se usa y si "
            "se envía a la IA; si no es necesaria para tu objetivo, conviene no usarla.",
        )
    if name == "JOIN_KEYS_MUST_BE_EXCLUDED":
        return WarningText(
            "review",
            f"{_columns(detail)} se usó para unir los archivos. Identifica cada caso pero no "
            "explica el resultado: exclúyela del aprendizaje.",
        )
    if name == "LOW_MATCH_RATE":
        return WarningText(
            "review",
            "Una parte importante de las filas no encuentra su valor en el otro archivo. Revisa "
            "si las claves están escritas igual antes de cruzar.",
        )
    if name == "ROWS_WOULD_MULTIPLY":
        return WarningText(
            "review",
            "El valor se repite en el otro archivo: al cruzar, un caso se convertiría en varias "
            "filas. Resume ese archivo por la clave antes de cruzar.",
        )
    if name == "LOW_JOIN_MATCH":
        return WarningText(
            "review",
            f"Menos de la mitad de los casos encontró su fila en el archivo «{detail}». "
            "Revisa si las claves están escritas igual en ambos archivos; los casos sin "
            "coincidencia quedan con esas columnas vacías.",
        )
    if name == "AUXILIARY_REPEATS_PRIMARY_COLUMNS":
        return WarningText(
            "review",
            f"{_columns(detail)} viene de un archivo auxiliar con el mismo nombre que una "
            "columna del archivo principal, por ejemplo el resultado. Podría ser una copia o un "
            "dato conocido después del evento: revísala y exclúyela si es así.",
        )
    if name in {"LOW_POSITIVE_SUPPORT_VALIDATION", "LOW_POSITIVE_SUPPORT_HOLDOUT"}:
        return WarningText(
            "review",
            "Hay pocos casos del evento en la parte usada para medir (menos de 100). Cada caso "
            "mueve mucho las cifras: léelas junto con su rango de incertidumbre.",
        )
    if name == "STALE_RUN_RECOVERED":
        return WarningText(
            "review",
            "Una ejecución anterior quedó interrumpida y se cerró automáticamente.",
        )
    if name == "AGGREGATE_SUPPORT_TOO_SMALL":
        return WarningText(
            "review",
            "Hay muy pocos casos para mostrar este agregado sin revelar información individual.",
        )
    if name == "POLICY_SELECTED_ON_VALIDATION_NOT_INDEPENDENT_EVIDENCE":
        return WarningText(
            "review",
            "La regla para elegir a quién atender se ajustó con los datos de comparación. "
            "Evalúala con casos nuevos antes de concluir que funcionará igual.",
        )
    if name == "PROBABILITY_DIAGNOSTICS_DO_NOT_AUTHORIZE_MONETARY_USE":
        return WarningText(
            "review",
            "La evaluación de los puntajes ayuda a revisar el modelo, pero no demuestra "
            "ingresos ni ahorro. Para estimarlos hacen falta supuestos de negocio; para "
            "comprobarlos, resultados de una intervención evaluada.",
        )
    if name.startswith("RUN_") and name != "RUN_STARTED_PENDING_RESULTS":
        return WarningText(
            "review", "La ejecución no terminó normalmente; sus cifras no son válidas."
        )
    notes = {
        "DETERMINISTIC_TEMPLATE_NO_LLM": (
            "La estrategia sigue una plantilla fija del producto; la IA no la decide."
        ),
        "HOLDOUT_EVALUATED_ONCE_AFTER_SELECTION": (
            "La prueba final se usó una sola vez, después de elegir el modelo."
        ),
        "HOLDOUT_NOT_EVALUATED_ML01": "La prueba final todavía no se usó.",
        "PERMUTATION_IMPORTANCE_IS_NOT_CAUSAL": (
            "La influencia de cada variable muestra con qué se asocia la predicción, "
            "no qué causa el resultado."
        ),
        "THRESHOLD_TUNED_ON_VALIDATION_HOLDOUT_UNTOUCHED": (
            "El punto desde el que se emite una alerta se eligió con los datos de comparación, "
            "sin mirar la prueba final."
        ),
        "RUN_STARTED_PENDING_RESULTS": "La ejecución empezó y todavía no tiene resultados.",
    }
    if name in notes:
        return WarningText("note", notes[name])
    return WarningText("review", f"Advertencia técnica sin explicación disponible ({code}).")


_COLUMN_WARNINGS = (
    "CONSTANT_COLUMNS",
    "HIGH_CARDINALITY_COLUMNS",
    "POSSIBLE_IDENTIFIER_COLUMNS",
    "POSSIBLE_PERSONAL_DATA_COLUMNS",
)


def drop_settled_columns(codes: list[str], excluded: set[str]) -> list[str]:
    """Quita de los avisos las columnas que ya quedaron fuera y no repite una misma columna.

    Recorrido 22/10: el plan avisaba de columnas ya excluidas y nombraba dos veces el mismo
    identificador (como «casi un valor por fila» y como «identificador»).
    """

    identifiers: set[str] = set()
    for code in codes:
        name, _, detail = code.removeprefix("DATA_PROFILE:").partition(":")
        if name == "POSSIBLE_IDENTIFIER_COLUMNS":
            identifiers.update(item for item in detail.split(",") if item)
    kept: list[str] = []
    for code in codes:
        prefix = "DATA_PROFILE:" if code.startswith("DATA_PROFILE:") else ""
        name, _, detail = code.removeprefix(prefix).partition(":")
        if name not in _COLUMN_WARNINGS:
            kept.append(code)
            continue
        skip = excluded | (identifiers if name == "HIGH_CARDINALITY_COLUMNS" else set())
        columns = [item for item in detail.split(",") if item and item not in skip]
        if columns:
            kept.append(f"{prefix}{name}:{','.join(columns)}")
    return kept


def split_warnings(
    codes: list[str], *, excluded: set[str] | None = None
) -> tuple[list[str], list[str]]:
    review: list[str] = []
    notes: list[str] = []
    for code in drop_settled_columns(codes, excluded or set()):
        described = describe_warning(code)
        target = review if described.kind == "review" else notes
        if described.text not in target:
            target.append(described.text)
    return review, notes


def render_warnings(
    codes: list[str], *, empty_text: str | None = None, excluded: set[str] | None = None
) -> list[str]:
    """Muestra la lista legible y devuelve los puntos que requieren revisión."""

    review, notes = split_warnings(codes, excluded=excluded)
    if review:
        heading = (
            "Hay 1 punto para revisar antes de seguir:"
            if len(review) == 1
            else f"Hay {len(review)} puntos para revisar antes de seguir:"
        )
        st.warning(heading + "\n\n" + "\n".join(f"- {text}" for text in review))
    elif empty_text:
        st.write(empty_text)
    if notes:
        st.caption("Sobre el método: " + " ".join(notes))
    return review
