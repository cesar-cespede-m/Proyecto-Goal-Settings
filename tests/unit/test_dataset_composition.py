"""DATA-02.1: composición trazable de varios CSV permitidos."""

import hashlib
import io

import pandas as pd
import pytest
from pydantic import ValidationError

from core.data_engine import (
    CompositionLimits,
    CSVSourceInput,
    DatasetCompositionError,
    DatasetCompositionSpecV1,
    IngestionLimits,
    compose_csv_datasets,
)


def _source(content: bytes, filename: str) -> tuple[dict[str, str], CSVSourceInput]:
    digest = hashlib.sha256(content).hexdigest()
    return (
        {
            "dataset_id": filename.removesuffix(".csv"),
            "dataset_sha256": digest,
            "classification": "synthetic",
        },
        CSVSourceInput(filename=filename, content=content),
    )


def _composition(
    *auxiliary_contents: tuple[bytes, str, str, str, str],
    primary_content: bytes = b"case_id,target,value\n1,0,10\n2,1,20\n3,0,30\n",
) -> tuple[dict[str, object], dict[str, CSVSourceInput]]:
    primary_ref, primary_input = _source(primary_content, "primary.csv")
    joins: list[dict[str, object]] = []
    sources = {primary_ref["dataset_id"]: primary_input}
    for content, filename, alias, primary_key, auxiliary_key in auxiliary_contents:
        source_ref, source_input = _source(content, filename)
        joins.append(
            {
                "source": source_ref,
                "alias": alias,
                "primary_key": primary_key,
                "auxiliary_key": auxiliary_key,
                "cardinality": "many_to_one",
            }
        )
        sources[source_ref["dataset_id"]] = source_input
    return {"primary": primary_ref, "joins": joins}, sources


def test_composes_five_sources_without_multiplying_primary_rows() -> None:
    auxiliaries = tuple(
        (
            f"key,feature\n1,{index}1\n2,{index}2\n3,{index}3\n".encode(),
            f"aux-{index}.csv",
            f"aux{index}",
            "case_id",
            "key",
        )
        for index in range(1, 5)
    )
    spec, sources = _composition(*auxiliaries)

    result = compose_csv_datasets(spec, sources)
    frame = pd.read_csv(io.BytesIO(result.content))

    assert result.summary.source_count == 5
    assert result.summary.output_sha256 == hashlib.sha256(result.content).hexdigest()
    assert result.summary.protected_join_columns == ["case_id"]
    assert len(frame) == 3
    assert list(frame.columns) == [
        "case_id",
        "target",
        "value",
        "aux1__feature",
        "aux2__feature",
        "aux3__feature",
        "aux4__feature",
    ]
    assert all(item.matched_primary_rows == 3 for item in result.summary.joins)
    assert "JOIN_KEYS_MUST_BE_EXCLUDED:case_id" in result.profile.warnings


def test_left_join_reports_unmatched_rows_without_exposing_values() -> None:
    spec, sources = _composition(
        (
            b"key,segment\n1,SENSITIVE_ROW_VALUE\n3,C\n4,D\n",
            "segments.csv",
            "segment",
            "case_id",
            "key",
        )
    )

    result = compose_csv_datasets(spec, sources)
    stats = result.summary.joins[0]
    frame = pd.read_csv(io.BytesIO(result.content))

    assert stats.matched_primary_rows == 2
    assert stats.unmatched_primary_rows == 1
    assert stats.unmatched_auxiliary_rows == 1
    assert pd.isna(frame.loc[1, "segment__segment"])
    assert "SENSITIVE_ROW_VALUE" not in result.summary.model_dump_json()


def test_composition_is_deterministic_for_same_sources_and_spec() -> None:
    spec, sources = _composition(
        (b"key,segment\n1,A\n2,B\n3,C\n", "segments.csv", "segment", "case_id", "key")
    )

    first = compose_csv_datasets(spec, sources)
    second = compose_csv_datasets(spec, sources)

    assert first.content == second.content
    assert first.summary == second.summary


@pytest.mark.parametrize(
    ("auxiliary", "expected_code"),
    [
        (b"key,segment\n1,A\n1,B\n", "AUXILIARY_KEY_NOT_UNIQUE"),
        (b"key,segment\n1,A\n,B\n", "JOIN_KEY_HAS_MISSING"),
        (b"other,segment\n1,A\n", "JOIN_KEY_NOT_FOUND"),
        (b"key,segment\none,A\ntwo,B\n", "JOIN_KEY_TYPE_MISMATCH"),
    ],
)
def test_rejects_unsafe_auxiliary_keys(auxiliary: bytes, expected_code: str) -> None:
    spec, sources = _composition((auxiliary, "segments.csv", "segment", "case_id", "key"))

    with pytest.raises(DatasetCompositionError) as captured:
        compose_csv_datasets(spec, sources)

    assert captured.value.code == expected_code


def test_one_to_one_rejects_duplicates_in_primary_key() -> None:
    spec, sources = _composition(
        (b"key,segment\n1,A\n2,B\n", "segments.csv", "segment", "case_id", "key"),
        primary_content=b"case_id,target\n1,0\n1,1\n2,0\n",
    )
    spec["joins"][0]["cardinality"] = "one_to_one"  # type: ignore[index]

    with pytest.raises(DatasetCompositionError) as captured:
        compose_csv_datasets(spec, sources)

    assert captured.value.code == "PRIMARY_KEY_NOT_UNIQUE"


def test_rejects_changed_source_content_and_inexact_source_set() -> None:
    spec, sources = _composition(
        (b"key,segment\n1,A\n2,B\n3,C\n", "segments.csv", "segment", "case_id", "key")
    )
    changed = dict(sources)
    changed["segments"] = CSVSourceInput(
        filename="segments.csv", content=b"key,segment\n1,X\n2,B\n3,C\n"
    )
    with pytest.raises(DatasetCompositionError) as hash_error:
        compose_csv_datasets(spec, changed)
    assert hash_error.value.code == "SOURCE_HASH_MISMATCH"

    with pytest.raises(DatasetCompositionError) as set_error:
        compose_csv_datasets(spec, {"primary": sources["primary"]})
    assert set_error.value.code == "SOURCE_SET_MISMATCH"


def test_enforces_aggregate_bytes_and_output_columns() -> None:
    spec, sources = _composition(
        (b"key,segment\n1,A\n2,B\n3,C\n", "segments.csv", "segment", "case_id", "key")
    )
    total = sum(len(item.content) for item in sources.values())
    with pytest.raises(DatasetCompositionError) as size_error:
        compose_csv_datasets(
            spec,
            sources,
            composition_limits=CompositionLimits(max_total_input_bytes=total - 1),
        )
    assert size_error.value.code == "COMPOSITION_INPUT_TOO_LARGE"

    with pytest.raises(DatasetCompositionError) as columns_error:
        compose_csv_datasets(
            spec,
            sources,
            ingestion_limits=IngestionLimits(max_columns=3),
        )
    assert columns_error.value.code == "COMPOSED_TOO_MANY_COLUMNS"


def test_contract_caps_composition_at_five_sources() -> None:
    primary_ref, _ = _source(b"id,target\n1,0\n2,1\n", "primary.csv")
    joins = []
    for index in range(5):
        source_ref, _ = _source(b"id,value\n1,A\n2,B\n", f"aux-{index}.csv")
        joins.append(
            {
                "source": source_ref,
                "alias": f"aux{index}",
                "primary_key": "id",
                "auxiliary_key": "id",
            }
        )

    with pytest.raises(ValidationError):
        DatasetCompositionSpecV1.model_validate({"primary": primary_ref, "joins": joins})


def test_rejects_duplicate_source_or_alias_in_contract() -> None:
    primary_ref, _ = _source(b"id,target\n1,0\n2,1\n", "primary.csv")
    auxiliary_ref, _ = _source(b"id,value\n1,A\n2,B\n", "aux.csv")
    duplicate_join = {
        "source": auxiliary_ref,
        "alias": "lookup",
        "primary_key": "id",
        "auxiliary_key": "id",
    }
    with pytest.raises(ValidationError):
        DatasetCompositionSpecV1.model_validate(
            {"primary": primary_ref, "joins": [duplicate_join, duplicate_join]}
        )


def test_contract_rejects_source_pending_review() -> None:
    primary_ref, _ = _source(b"id,target\n1,0\n2,1\n", "primary.csv")
    auxiliary_ref, _ = _source(b"id,value\n1,A\n2,B\n", "aux.csv")
    auxiliary_ref["classification"] = "pending_review"

    with pytest.raises(ValidationError):
        DatasetCompositionSpecV1.model_validate(
            {
                "primary": primary_ref,
                "joins": [
                    {
                        "source": auxiliary_ref,
                        "alias": "lookup",
                        "primary_key": "id",
                        "auxiliary_key": "id",
                    }
                ],
            }
        )
