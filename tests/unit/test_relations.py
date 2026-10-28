"""GA-05: relaciones entre archivos, deterministas y genéricas."""

import pandas as pd

from core.analysis.contracts import ColumnType
from core.analysis.engine import LoadedSource
from core.analysis.relations import describe_relations, find_relations
from core.data_engine import analysis_schema, validate_and_profile
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from tests.support.domains import DOMAINS


def _source(frame: pd.DataFrame, schema: dict[str, str]) -> LoadedSource:
    return LoadedSource(
        frame=frame, sha256="0" * 64, schema={k: ColumnType(v) for k, v in schema.items()}
    )


CASES = pd.DataFrame(
    {"codigo": [f"c{i}" for i in range(40)], "grupo_id": [i % 5 for i in range(40)]}
)
GROUPS = pd.DataFrame({"id": [0, 1, 2, 3, 4], "nivel": ["a", "b", "a", "c", "b"]})


def test_many_to_one_key_is_found_and_described() -> None:
    relations = find_relations(
        {
            "casos": _source(CASES, {"codigo": "identifier", "grupo_id": "number"}),
            "grupos": _source(GROUPS, {"id": "identifier", "nivel": "category"}),
        }
    )

    best = relations.keys[0]
    assert (best.left, best.left_column, best.right, best.right_column) == (
        "casos",
        "grupo_id",
        "grupos",
        "id",
    )
    assert best.cardinality == "many_to_one" and best.match_rate == 1.0
    assert "100 %" in describe_relations(relations)[0]


def test_format_differences_still_match_and_low_overlap_is_flagged() -> None:
    texts = GROUPS.assign(id=[" 0", "1.0", "2", "9", "8"])
    relations = find_relations(
        {
            "casos": _source(CASES, {"codigo": "identifier", "grupo_id": "number"}),
            "grupos": _source(texts, {"id": "category", "nivel": "category"}),
        }
    )

    key = next(item for item in relations.keys if item.left == "casos")
    assert key.match_rate == 0.6 and key.warnings == ("LOW_MATCH_RATE:0.60",)


def test_same_columns_are_offered_as_stack() -> None:
    schema = {"codigo": "identifier", "grupo_id": "number"}
    relations = find_relations(
        {"enero": _source(CASES, schema), "febrero": _source(CASES.head(10), schema)}
    )

    assert relations.stacks[0].first == "enero" and not relations.keys


def _loaded(path) -> LoadedSource:
    content = path.read_bytes()
    profile = validate_and_profile(path.name, content).model_dump()
    frame = read_validated_frame(validate_csv_bytes(path.name, content, IngestionLimits()))
    return _source(frame, analysis_schema(profile))


def test_multi_file_business_has_keys_to_cross() -> None:
    domain = next(item for item in DOMAINS if item.multi_file)
    relations = find_relations({path.stem: _loaded(path) for path in domain.files})

    assert relations.keys
    assert all(0.3 <= key.match_rate <= 1 for key in relations.keys)
