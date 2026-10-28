"""GA-03b: detecciones deterministas de calidad sin conocimiento del negocio."""

from datetime import date, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from core.analysis.contracts import ColumnType
from core.analysis.engine import LoadedSource
from core.analysis.quality import QualityFindingV1, describe_findings, quality_findings
from tests.support.domains import DOMAINS
from tests.unit.test_relations import _loaded


def _source(frame: pd.DataFrame, schema: dict[str, str]) -> LoadedSource:
    return LoadedSource(
        frame=frame,
        sha256="0" * 64,
        schema={column: ColumnType(kind) for column, kind in schema.items()},
    )


def _of_kind(findings: list[QualityFindingV1], kind: str) -> list[QualityFindingV1]:
    return [finding for finding in findings if finding.kind == kind]


def test_extreme_numeric_outlier_is_counted() -> None:
    source = _source(
        pd.DataFrame({"medida": [10, 10, 10, 10, 10, 10, 10, 10, 1000]}),
        {"medida": "number"},
    )

    finding = _of_kind(quality_findings({"archivo": source}), "EXTREME_OUTLIER")[0]

    assert finding.count == 1
    assert finding.share == pytest.approx(1 / 9, abs=0.0001)


def test_impossible_old_and_future_dates_are_counted_together() -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    source = _source(
        pd.DataFrame({"momento": ["2020-01-01", "no-es-fecha", "1899-12-31", tomorrow]}),
        {"momento": "datetime"},
    )

    finding = _of_kind(quality_findings({"archivo": source}), "DATE_INVALID_OR_OUT_OF_RANGE")[0]

    assert finding.count == 3
    assert finding.share == 0.75
    assert "1 no se pueden interpretar" in finding.detail


def test_negative_among_non_negative_values_is_flagged() -> None:
    source = _source(
        pd.DataFrame({"cantidad": [0, 4, -7, 12]}),
        {"cantidad": "number"},
    )

    finding = _of_kind(quality_findings({"archivo": source}), "POSSIBLE_SIGN_ERROR")[0]

    assert finding.count == 1
    assert finding.share == 0.25


def test_categories_that_only_differ_in_case_or_spaces_are_grouped() -> None:
    source = _source(
        pd.DataFrame({"clase": ["A", "a ", "A", "B"]}),
        {"clase": "category"},
    )

    finding = _of_kind(quality_findings({"archivo": source}), "NEAR_DUPLICATE_CATEGORY")[0]

    assert finding.count == 3
    assert finding.share == 0.75
    assert "'A'" in finding.detail and "'a '" in finding.detail


def test_missing_values_are_counted_for_each_relation_direction() -> None:
    left = _source(
        pd.DataFrame({"clave_a": [1, 2, 3, 9], "nota": ["x", "x", "x", "x"]}),
        {"clave_a": "identifier", "nota": "text"},
    )
    right = _source(
        pd.DataFrame({"clave_b": [1, 2, 3, 8], "detalle": ["y", "y", "y", "y"]}),
        {"clave_b": "identifier", "detalle": "text"},
    )

    findings = _of_kind(quality_findings({"primero": left, "segundo": right}), "KEY_MISMATCH")

    assert [(finding.source, finding.column, finding.count) for finding in findings] == [
        ("primero", "clave_a", 1),
        ("segundo", "clave_b", 1),
    ]
    assert all(finding.share == 0.25 for finding in findings)


def test_contract_is_strict_and_descriptions_are_clear() -> None:
    with pytest.raises(ValidationError):
        QualityFindingV1(
            source="archivo",
            column="columna",
            kind="EXTREME_OUTLIER",
            count="1",
            share=0.5,
            detail="detalle",
        )

    finding = QualityFindingV1(
        source="archivo",
        column="columna",
        kind="EXTREME_OUTLIER",
        count=1,
        share=0.5,
        detail="detalle",
    )
    assert describe_findings([finding]) == [
        "«archivo.columna»: 1 valores numéricos muy alejados del resto "
        "(50.0% de los valores revisados)."
    ]


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_quality_runs_in_every_business(domain) -> None:
    sources = {path.stem: _loaded(path) for path in domain.files}

    findings = quality_findings(sources)

    assert all(isinstance(finding, QualityFindingV1) for finding in findings)
    assert len(describe_findings(findings)) == len(findings)
