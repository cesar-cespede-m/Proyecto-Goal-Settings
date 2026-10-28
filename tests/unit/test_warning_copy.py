"""UX-05: toda advertencia producida por core/ tiene texto en español."""

import re
from pathlib import Path

import pytest

from app.ui.warnings import describe_warning, split_warnings

ROOT = Path(__file__).parents[2]

# Códigos producidos hoy por core/ (con un detalle de ejemplo cuando lo llevan).
KNOWN_CODES = [
    "LOW_MATCH_RATE:0.60",
    "ROWS_WOULD_MULTIPLY",
    "DUPLICATE_ROWS:12",
    "CONSTANT_COLUMNS:pais_operacion",
    "HIGH_CARDINALITY_COLUMNS:comentario",
    "POSSIBLE_IDENTIFIER_COLUMNS:id_envio",
    "DATA_PROFILE:POSSIBLE_IDENTIFIER_COLUMNS:id_envio",
    "DETERMINISTIC_TEMPLATE_NO_LLM",
    "HOLDOUT_EVALUATED_ONCE_AFTER_SELECTION",
    "HOLDOUT_NOT_EVALUATED_ML01",
    "PERMUTATION_IMPORTANCE_IS_NOT_CAUSAL",
    "THRESHOLD_TUNED_ON_VALIDATION_HOLDOUT_UNTOUCHED",
    "LOW_POSITIVE_SUPPORT_VALIDATION",
    "LOW_POSITIVE_SUPPORT_HOLDOUT",
    "RUN_STARTED_PENDING_RESULTS",
    "RUN_FAILED",
    "STALE_RUN_RECOVERED",
    "JOIN_KEYS_MUST_BE_EXCLUDED:case_id",
    "LOW_JOIN_MATCH:mant",
    "AUXILIARY_REPEATS_PRIMARY_COLUMNS:mant__falla_maquina",
    "POSSIBLE_PERSONAL_DATA_COLUMNS:correo",
    "AGGREGATE_SUPPORT_TOO_SMALL",
    "POLICY_SELECTED_ON_VALIDATION_NOT_INDEPENDENT_EVIDENCE",
    "PROBABILITY_DIAGNOSTICS_DO_NOT_AUTHORIZE_MONETARY_USE",
]


@pytest.mark.parametrize("code", KNOWN_CODES)
def test_every_known_code_has_plain_spanish_text(code: str) -> None:
    described = describe_warning(code)
    assert "sin explicación disponible" not in described.text
    assert not re.search(r"\b[A-Z]{3,}_[A-Z_]+\b", described.text)


def test_known_codes_cover_the_literals_emitted_by_core() -> None:
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "core").rglob("*.py"))
    emitted = set(re.findall(r'warnings(?:\.add|\.append)?\(\s*f?"([A-Z][A-Z0-9_]+)', sources))
    emitted |= set(re.findall(r'"([A-Z][A-Z0-9_]+)",\s*\n?\s*"[A-Z]', sources)) & {
        "HOLDOUT_EVALUATED_ONCE_AFTER_SELECTION",
        "PERMUTATION_IMPORTANCE_IS_NOT_CAUSAL",
    }
    known = {code.removeprefix("DATA_PROFILE:").split(":")[0] for code in KNOWN_CODES}
    assert emitted <= known, emitted - known


def test_identifier_column_is_named_and_counted_as_review_point() -> None:
    review, notes = split_warnings(
        ["DETERMINISTIC_TEMPLATE_NO_LLM", "DATA_PROFILE:POSSIBLE_IDENTIFIER_COLUMNS:id_envio"]
    )
    assert len(review) == 1
    assert "«id_envio»" in review[0]
    assert len(notes) == 1


def test_unknown_code_is_still_shown() -> None:
    described = describe_warning("SOMETHING_NEW")
    assert described.kind == "review"
    assert "SOMETHING_NEW" in described.text


def test_settled_columns_are_not_warned_again_nor_twice() -> None:
    """Recorrido 22/10: el plan avisaba de columnas ya excluidas y repetía el identificador."""
    codes = [
        "DATA_PROFILE:CONSTANT_COLUMNS:region",
        "DATA_PROFILE:HIGH_CARDINALITY_COLUMNS:folio,notas",
        "DATA_PROFILE:POSSIBLE_IDENTIFIER_COLUMNS:folio",
        "DETERMINISTIC_TEMPLATE_NO_LLM",
    ]

    review, notes = split_warnings(codes, excluded={"region"})

    assert not any("«region»" in text for text in review)
    assert sum("«folio»" in text for text in review) == 1
    assert any("«notas»" in text for text in review)
    assert notes

    review, _ = split_warnings(codes, excluded={"region", "folio", "notas"})
    assert review == []
