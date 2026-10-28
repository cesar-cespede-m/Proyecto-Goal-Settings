"""UX-18: una sola pregunta por columna y confirmación en bloque acotada."""

from types import SimpleNamespace

from app.ui.assistant_view import (
    FEATURE_ANSWERS,
    bulk_candidates,
    feature_answer,
    unsafe_columns,
)
from core.domain.assistant import (
    Availability,
    ConfirmationStatus,
    DraftUse,
    FeatureDraft,
    FieldSource,
)


def test_every_answer_round_trips() -> None:
    for key, (use, available, _) in FEATURE_ANSWERS.items():
        assert feature_answer(use, available) == key


def test_any_unknown_reads_as_no_se() -> None:
    assert feature_answer(DraftUse.INCLUDE, Availability.UNKNOWN) == "unknown"
    assert feature_answer(DraftUse.UNKNOWN, Availability.YES) == "unknown"


def _stored(*features: FeatureDraft) -> SimpleNamespace:
    return SimpleNamespace(draft=SimpleNamespace(candidate_features=lambda: list(features)))


def test_bulk_only_offers_ai_reviewed_columns_without_alerts() -> None:
    stored = _stored(
        FeatureDraft(
            column="distancia_km",
            use=DraftUse.INCLUDE,
            note="Disponible antes del despacho.",
            note_source=FieldSource.LLM,
            status=ConfirmationStatus.PROPOSED,
        ),
        FeatureDraft(
            column="minutos_retraso_real",
            use=DraftUse.INCLUDE,
            note="Se conoce después.",
            note_source=FieldSource.LLM,
            possible_post_event=True,
            status=ConfirmationStatus.PROPOSED,
        ),
        FeatureDraft(
            column="id_envio",
            use=DraftUse.EXCLUDE,
            note="Parece un identificador.",
            note_source=FieldSource.DATASET,
            status=ConfirmationStatus.PROPOSED,
        ),
        FeatureDraft(column="sin_revisar"),
    )
    assert bulk_candidates(stored) == ["distancia_km"]


def test_bulk_never_offers_identifier_or_constant_columns() -> None:
    stored = _stored(
        FeatureDraft(
            column="pais_operacion",
            use=DraftUse.INCLUDE,
            note="La IA la propone.",
            note_source=FieldSource.LLM,
            status=ConfirmationStatus.PROPOSED,
        )
    )
    assert bulk_candidates(stored, {"pais_operacion"}) == []


def test_unsafe_columns_explain_why_in_plain_words() -> None:
    profile = SimpleNamespace(
        columns=[
            SimpleNamespace(name="pais", constant=True, possible_identifier=False),
            SimpleNamespace(name="id_envio", constant=False, possible_identifier=True),
            SimpleNamespace(name="peso_kg", constant=False, possible_identifier=False),
        ]
    )
    reasons = unsafe_columns(profile)
    assert reasons == {
        "pais": "tiene el mismo valor en todas las filas",
        "id_envio": "parece un identificador (distinto en cada fila)",
    }
