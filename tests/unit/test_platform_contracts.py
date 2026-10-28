"""GA-11: contratos del ciclo 2, genéricos y sin conectores habilitados."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from core.contracts.platform import (
    ChangeProposalV1,
    DataSourceV1,
    DecisionRecordV1,
    ObservationEventV1,
    RecipeV1,
)
from core.domain.canonical import canonical_hash

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
SHA = "a" * 64


def test_only_csv_upload_is_enabled_for_now() -> None:
    assert DataSourceV1(kind="csv_upload", name="Archivo").enabled
    assert not DataSourceV1(kind="api_push", name="Sistema externo").enabled
    with pytest.raises(ValidationError):
        DataSourceV1(kind="ftp", name="No soportado")


def test_observation_events_are_generic_and_time_consistent() -> None:
    event = ObservationEventV1(
        source="sistema_a",
        entity_type="caso",
        entity_id="123",
        occurred_at=NOW,
        attributes={"atributo_numerico": 3.5, "atributo_texto": "x", "vacio": None},
    )
    later = event.model_copy(
        update={"outcome": "sí", "outcome_observed_at": NOW + timedelta(days=1)}
    )

    assert ObservationEventV1.model_validate(later.model_dump()) == later
    with pytest.raises(ValidationError):
        ObservationEventV1(**{**event.model_dump(), "occurred_at": NOW.replace(tzinfo=None)})
    with pytest.raises(ValidationError):
        ObservationEventV1(**{**event.model_dump(), "outcome": "sí"})
    with pytest.raises(ValidationError):
        ObservationEventV1(
            **{**event.model_dump(), "outcome": 1, "outcome_observed_at": NOW - timedelta(1)}
        )


def test_recipe_hash_is_stable_and_changes_with_any_version() -> None:
    recipe = RecipeV1(
        context={"id": "c1", "sha256": SHA},
        strategy={"id": "s1", "sha256": SHA},
        task_kind="binary_classification",
        sources=[{"id": "d1", "sha256": SHA}],
    )
    changed = recipe.model_copy(update={"task_kind": "regression"})

    assert canonical_hash(recipe) == canonical_hash(RecipeV1.model_validate(recipe.model_dump()))
    assert canonical_hash(recipe) != canonical_hash(changed)


def test_decisions_record_their_later_result() -> None:
    decision = DecisionRecordV1(
        decision="Reforzar la atención en el grupo con más casos",
        based_on="analysis",
        reference_id="analisis-1",
        actor_label="Responsable",
        decided_at=NOW,
    )
    assert decision.observed_result is None
    with pytest.raises(ValidationError):
        DecisionRecordV1(
            **{
                **decision.model_dump(),
                "observed_result": "Bajaron los casos",
                "observed_at": NOW - timedelta(days=1),
            }
        )


def test_change_proposals_only_travel_as_reviewed_pull_requests() -> None:
    proposal = ChangeProposalV1(
        kind="analysis_kind",
        title="Nuevo tipo de análisis",
        rationale="Las preguntas frecuentes del negocio lo requieren.",
    )

    assert proposal.delivery == "pull_request"
    assert proposal.requires_human_review is True
    with pytest.raises(ValidationError):
        ChangeProposalV1(**{**proposal.model_dump(), "delivery": "execute"})
    with pytest.raises(ValidationError):
        ChangeProposalV1(**{**proposal.model_dump(), "requires_human_review": False})
