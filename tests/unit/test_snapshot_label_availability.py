"""MV-09: disponibilidad de etiqueta derivada de una receta explícita, nunca predictor."""

from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from core.analysis.snapshot import SnapshotSpecV1, build_snapshot
from core.domain.canonical import canonical_hash
from tests.unit.test_data_readiness import COVERAGE, RECIPE


def _events():
    return pd.DataFrame(
        [
            ("a", "2026-01-01", "signup"),
            ("a", "2026-01-04", "purchase_private"),
            ("b", "2026-01-29", "signup"),
        ],
        columns=["entity", "moment", "stage"],
    )


def test_legacy_recipe_keeps_hash_and_csv_without_availability_opt_in():
    recipe = SnapshotSpecV1.model_validate(RECIPE)
    old_payload = recipe.model_dump(mode="json")
    assert "label_available_at_output_column" not in old_payload
    explicit_none = SnapshotSpecV1.model_validate(
        {**RECIPE, "label_available_at_output_column": None}
    )
    assert canonical_hash(recipe) == canonical_hash(explicit_none) == canonical_hash(old_payload)
    old = build_snapshot(_events(), RECIPE, coverage=COVERAGE)
    same = build_snapshot(_events(), explicit_none, coverage=COVERAGE)
    assert old.dataframe.to_csv(index=False) == same.dataframe.to_csv(index=False)


def test_available_at_is_full_horizon_in_utc_even_for_early_positive_and_pending_rows():
    recipe = {**RECIPE, "label_available_at_output_column": "label_available_at"}
    result = build_snapshot(_events(), recipe, coverage=COVERAGE)
    frame = result.dataframe
    expected = pd.to_datetime(frame["snapshot_cutoff"], utc=True) + pd.Timedelta(days=7)
    assert frame["label_available_at"].equals(expected)
    assert "label_available_at" not in result.summary.feature_names
    assert "label_available_at" not in result.summary.outcome_names
    assert result.readiness.eligible_rows == 1 and result.readiness.pending_rows == 1
    assert len(result.training_frame()) == 1
    assert str(frame["label_available_at"].iloc[0]) == "2026-01-10 00:00:00+00:00"


@pytest.mark.parametrize(
    "column",
    [
        "entity",
        "moment",
        "snapshot_cutoff",
        "SNAPSHOT_CUTOFF",
        "count_events_3d",
        "adverse",
        "attribute",
    ],
)
def test_metadata_column_cannot_shadow_identity_inputs_features_outputs_or_attributes(column):
    with pytest.raises(ValidationError, match="metadata exclusiva"):
        SnapshotSpecV1.model_validate(
            {
                **RECIPE,
                "entity_attributes": ["attribute"],
                "label_available_at_output_column": column,
            }
        )


@pytest.mark.parametrize(
    "outcomes",
    [
        [*RECIPE["outcome"], {**RECIPE["outcome"][0], "output_column": "other"}],
        [{"op": "days_until_stage", "stage": {"column": "stage", "value": "purchase"}}],
        [{"op": "future_sum", "column": "value", "within_days": 7}],
        [{"op": "last_stage_reached", "stage": "stage", "within_days": 7}],
    ],
)
def test_opt_in_rejects_multiple_uncapped_or_nonbinary_outcomes(outcomes):
    with pytest.raises(ValidationError, match="único resultado binario"):
        SnapshotSpecV1.model_validate(
            {
                **deepcopy(RECIPE),
                "outcome": outcomes,
                "label_available_at_output_column": "label_available_at",
            }
        )


def test_temporal_metadata_never_substitutes_a_coverage_declaration():
    result = build_snapshot(
        _events(), {**RECIPE, "label_available_at_output_column": "label_available_at"}
    )
    assert result.dataframe["label_available_at"].notna().all()
    assert result.training_frame().empty
    assert result.readiness.reason_counts == {"COVERAGE_NOT_DECLARED": 2}
