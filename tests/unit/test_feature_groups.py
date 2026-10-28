"""V2-10: columnas agrupadas y recomendaciones reversibles."""

from core.data_engine import DatasetProfileV1, validate_and_profile
from core.domain.assistant import ConfirmationStatus, ContextDraftV1, FieldSource
from core.domain.feature_groups import (
    FeatureGroup,
    accept_recommendations,
    group_features,
    undo_recommendations,
)
from tests.support.domains import DOMAINS

SHA = "a" * 64
ACTOR = "pytest-responsable"


def _draft(path, target: str):
    profile = validate_and_profile(path.name, path.read_bytes())
    draft = ContextDraftV1.start(
        project_id="project-test",
        dataset_id="dataset-test",
        dataset_sha256=SHA,
        profile=profile,
    )
    return draft, profile, group_features(draft, profile, target_column=target)


def test_subscription_columns_are_grouped_with_plain_reasons() -> None:
    path = next(domain.files[0] for domain in DOMAINS if domain.key == "suscripciones")
    _, _, groups = _draft(path, "abandona")
    by_column = {item.column: item for item in groups.items}

    assert by_column["id_suscriptor"].group is FeatureGroup.EXCLUDE
    assert by_column["moneda"].group is FeatureGroup.EXCLUDE
    assert by_column["encuesta_salida_completada"].group in {
        FeatureGroup.EXCLUDE,
        FeatureGroup.ASK,
    }
    assert all(item.reason and "_" not in item.reason for item in groups.items)
    assert all(
        item.group is FeatureGroup.USE
        for name, item in by_column.items()
        if name not in {"id_suscriptor", "moneda", "encuesta_salida_completada"}
    )


def test_accept_never_answers_ask_and_undo_restores_pending_state() -> None:
    path = next(domain.files[0] for domain in DOMAINS if domain.key == "suscripciones")
    draft, _, groups = _draft(path, "abandona")
    ask = {item.column for item in groups.items if item.group is FeatureGroup.ASK}
    accepted_columns = [item.column for item in groups.items if item.group is not FeatureGroup.ASK]

    accepted = accept_recommendations(draft, groups, actor_label=ACTOR)

    assert accepted == len(accepted_columns)
    assert all(
        item.status is not ConfirmationStatus.CONFIRMED
        for item in draft.features
        if item.column in ask
    )
    assert all(
        item.note_source is FieldSource.GOAL_SETTINGS_RECOMMENDATION and item.confirmed_by == ACTOR
        for item in draft.features
        if item.column in accepted_columns
    )

    undo_recommendations(draft, accepted_columns, actor_label=ACTOR)

    assert all(
        item.status is ConfirmationStatus.PROPOSED and item.confirmed_by is None
        for item in draft.features
        if item.column in accepted_columns
    )


def test_grouping_runs_for_every_versioned_business() -> None:
    for domain in DOMAINS:
        for path in domain.files:
            profile = validate_and_profile(path.name, path.read_bytes())
            draft = ContextDraftV1.start(
                project_id=f"project-{domain.key}",
                dataset_id=path.stem,
                dataset_sha256=SHA,
                profile=profile,
            )
            target = profile.columns[-1].name
            groups = group_features(draft, profile, target_column=target)

            assert len(groups.items) == profile.column_count - 1


def test_numeric_result_with_near_perfect_correlation_is_excluded() -> None:
    rows = ["input_measure,late_measure,result_measure"]
    rows.extend(f"{index % 5},{index / 10},{index / 20}" for index in range(1, 101))
    profile = validate_and_profile("numeric.csv", ("\n".join(rows) + "\n").encode())
    draft = ContextDraftV1.start(
        project_id="project-numeric",
        dataset_id="dataset-numeric",
        dataset_sha256=SHA,
        profile=profile,
    )

    groups = group_features(draft, profile, target_column="result_measure")
    late = next(item for item in groups.items if item.column == "late_measure")

    assert late.group is FeatureGroup.EXCLUDE
    assert "después del resultado" in late.reason


def test_profile_saved_before_association_signals_still_opens() -> None:
    path = next(domain.files[0] for domain in DOMAINS if domain.key == "suscripciones")
    current = validate_and_profile(path.name, path.read_bytes()).model_dump(mode="json")
    current.pop("outcome_associations")

    restored = DatasetProfileV1.model_validate(current)

    assert restored.outcome_associations == []
