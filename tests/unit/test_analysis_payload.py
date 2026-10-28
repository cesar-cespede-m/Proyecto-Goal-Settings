"""GA-02d: columnas autorizadas para IA, linaje y round-trip JSON."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.analysis import (
    ANALYSIS_ENGINE_VERSION,
    AnalysisEngine,
    AnalysisLineageV1,
    AnalysisOutputV1,
    AnalysisPlanError,
    AnalysisPlanV1,
    AnalysisResultV1,
    ColumnType,
    LlmPayloadV1,
    LoadedSource,
    analysis_lineage,
    to_llm_payload,
)
from core.data_engine import IngestionLimits, analysis_schema, validate_and_profile
from core.data_engine.loading import read_validated_frame
from core.data_engine.validator import validate_csv_bytes
from tests.support.domains import DOMAINS
from tests.support.domains import TestDomain as DomainCase


def _load_domain(domain: DomainCase) -> LoadedSource:
    path: Path = domain.files[0]
    content = path.read_bytes()
    profile = validate_and_profile(path.name, content)
    validated = validate_csv_bytes(path.name, content, IngestionLimits())
    return LoadedSource(
        frame=read_validated_frame(validated),
        sha256=hashlib.sha256(content).hexdigest(),
        schema={
            name: ColumnType(kind)
            for name, kind in analysis_schema(profile.model_dump(mode="json")).items()
        },
    )


def _result(domain: DomainCase) -> AnalysisResultV1:
    source = _load_domain(domain)
    plan = AnalysisPlanV1(
        sources=["source"],
        steps=[],
        output=AnalysisOutputV1(),
    )
    return AnalysisEngine().run(plan, {"source": source})


@pytest.mark.parametrize("domain", DOMAINS[:2], ids=lambda domain: domain.key)
def test_payload_uses_only_authorized_columns_in_result_order(domain: DomainCase) -> None:
    result = _result(domain)
    first = result.columns[0].name
    last = result.columns[-1].name
    excluded = next(column.name for column in result.columns if column.name not in {first, last})

    payload = to_llm_payload(result, [last, first])
    serialized = payload.model_dump_json()

    assert [column.name for column in payload.columns] == [first, last]
    assert all(len(row) == 2 for row in payload.rows)
    assert excluded not in serialized


def test_payload_reports_truncation_and_rejects_unknown_columns() -> None:
    result = _result(DOMAINS[0])
    selected = result.columns[0].name

    payload = to_llm_payload(result, [selected], max_rows=2)

    assert payload.sent_row_count == 2
    assert payload.excluded_row_count == result.total_rows - 2
    assert payload.truncated is True
    with pytest.raises(AnalysisPlanError, match="no existe en el resultado") as captured:
        to_llm_payload(result, ["missing_column"])
    assert captured.value.code == "COLUMN_NOT_FOUND"


@pytest.mark.parametrize("domain", DOMAINS[:2], ids=lambda domain: domain.key)
def test_result_payload_and_lineage_round_trip_through_json(domain: DomainCase) -> None:
    result = _result(domain)
    allowed = [column.name for column in result.columns[:2]]
    payload = to_llm_payload(result, allowed, max_rows=3)

    restored_result = AnalysisResultV1.model_validate_json(result.model_dump_json())
    restored_payload = LlmPayloadV1.model_validate_json(payload.model_dump_json())
    lineage = analysis_lineage(restored_result)
    restored_lineage = AnalysisLineageV1.model_validate_json(lineage.model_dump_json())

    assert restored_result == result
    assert restored_payload == payload
    assert all(isinstance(column.type, ColumnType) for column in restored_result.columns)
    assert all(isinstance(column.type, ColumnType) for column in restored_payload.columns)
    assert restored_lineage == lineage
    assert restored_lineage.engine_version == ANALYSIS_ENGINE_VERSION
    assert restored_lineage.created_at.utcoffset() is not None


def test_lineage_rejects_a_created_at_without_timezone() -> None:
    result = _result(DOMAINS[0])
    raw = analysis_lineage(result).model_dump()
    raw["created_at"] = result.created_at.replace(tzinfo=None)

    with pytest.raises(ValidationError, match="zona horaria"):
        AnalysisLineageV1.model_validate(raw)


def test_payload_reports_the_columns_left_out() -> None:
    from core.analysis.contracts import ColumnType
    from core.analysis.engine import AnalysisResultV1

    result = AnalysisResultV1(
        columns=[
            {"name": "a", "type": ColumnType.CATEGORY},
            {"name": "b", "type": ColumnType.NUMBER},
            {"name": "c", "type": ColumnType.TEXT},
        ],
        rows=[["x", 1, "t"]],
        total_rows=1,
        plan_hash="a" * 64,
        source_hashes={},
        warnings=[],
    )
    from core.analysis.payload import omitted_columns

    payload = to_llm_payload(result, ["b"])
    sent = payload.model_dump_json()

    assert [column.name for column in payload.columns] == ["b"]
    assert omitted_columns(result, ["b"]) == ["a", "c"]
    assert '"a"' not in sent and '"c"' not in sent and '"x"' not in sent
