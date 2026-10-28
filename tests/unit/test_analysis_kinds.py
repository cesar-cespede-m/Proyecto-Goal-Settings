"""GA-03: tipos de análisis automáticos válidos y ejecutables en cada negocio de prueba."""

import pytest

from core.analysis.engine import AnalysisEngine
from core.analysis.kinds import ANALYSIS_KINDS, applicable_kinds, missing_counts
from core.analysis.validation import validate_plan
from tests.support.domains import DOMAINS
from tests.unit.test_relations import _loaded

KEYS = {"descriptive", "temporal", "segmentation", "outcome", "quality"}


def test_registry_has_the_mvp_kinds() -> None:
    assert {kind.key for kind in ANALYSIS_KINDS} == KEYS


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_applicable_kind_runs_in_every_business(domain) -> None:
    from core.data_engine import validate_and_profile

    for path in domain.files:
        source = _loaded(path)
        schema = {name: str(kind.value) for name, kind in source.schema.items()}
        categories = [name for name, kind in schema.items() if kind == "category"]
        outcome = categories[-1] if len(categories) > 1 else None
        profile = validate_and_profile(path.name, path.read_bytes()).model_dump()
        kinds = applicable_kinds(schema, outcome=outcome)

        assert ("temporal" in {kind.key for kind in kinds}) == ("datetime" in schema.values())
        for kind in kinds:
            for item in kind.plans(schema, outcome=outcome, missing=missing_counts(profile)):
                validate_plan(item.plan, {"archivo": schema})
                result = AnalysisEngine().run(item.plan, {"archivo": source})
                assert result.total_rows >= 1, item.title


def test_quality_lists_only_columns_with_holes() -> None:
    schema = {"a": "number", "b": "category"}
    kind = next(item for item in ANALYSIS_KINDS if item.key == "quality")

    plans = kind.plans(schema, missing={"a": 0, "b": 4})
    assert [item.title for item in plans] == ["Casos con «b» vacío"]
