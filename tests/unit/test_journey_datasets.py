"""V2-19: recorridos sintéticos reproducibles, válidos y con etapas coherentes."""

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from core.data_engine import IngestionLimits, validate_and_profile

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "synthetic" / "recorridos"
SPEC = importlib.util.spec_from_file_location(
    "generate_journey_datasets", ROOT / "scripts" / "generate_journey_datasets.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)

JOURNEYS = (
    {
        "events": "altas_eventos.csv",
        "entities": "altas_entidades.csv",
        "entity": "codigo_incorporacion",
        "moment": "momento_actividad",
        "stage": "paso_incorporacion",
        "sequences": {
            ("registro",),
            ("registro", "verificacion"),
            ("registro", "verificacion", "primera_compra"),
            ("registro", "verificacion", "primera_compra", "segunda_compra"),
        },
        "entity_count": 3_000,
    },
    {
        "events": "solicitudes_eventos.csv",
        "entities": "solicitudes_entidades.csv",
        "entity": "folio_gestion",
        "moment": "marca_temporal",
        "stage": "estado_tramite",
        "sequences": {
            ("recibida",),
            ("recibida", "asignada"),
            ("recibida", "asignada", "en_revision"),
            ("recibida", "asignada", "en_revision", "resuelta"),
            ("recibida", "asignada", "en_revision", "cerrada_sin_resolver"),
        },
        "entity_count": 2_000,
    },
)


def _hashes(folder: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.glob("*.csv"))
    }


def test_generator_is_deterministic_and_matches_versioned_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator.write_datasets(first, seed=generator.SEED)
    generator.write_datasets(second, seed=generator.SEED)

    assert _hashes(first) == _hashes(second) == _hashes(DATA)


@pytest.mark.parametrize("name", sorted(generator.build()))
def test_every_file_passes_ingestion_limits(name: str) -> None:
    path = DATA / name
    profile = validate_and_profile(path.name, path.read_bytes())

    assert path.stat().st_size < IngestionLimits().max_upload_bytes
    assert profile.column_count < IngestionLimits().max_columns
    assert 0 < profile.row_count <= IngestionLimits().max_rows


@pytest.mark.parametrize("journey", JOURNEYS, ids=lambda item: item["events"])
def test_entities_have_ordered_valid_stage_sequences(journey: dict[str, object]) -> None:
    events = pd.read_csv(DATA / str(journey["events"]))
    entities = pd.read_csv(DATA / str(journey["entities"]))
    entity_column = str(journey["entity"])
    moment_column = str(journey["moment"])
    stage_column = str(journey["stage"])
    valid_sequences = journey["sequences"]

    assert len(entities) == journey["entity_count"]
    assert entities[entity_column].is_unique
    assert set(events[entity_column]) == set(entities[entity_column])
    for _, history in events.groupby(entity_column, sort=False):
        moments = pd.to_datetime(history[moment_column])
        assert moments.is_monotonic_increasing
        assert moments.is_unique
        assert tuple(history[stage_column]) in valid_sequences


@pytest.mark.parametrize("journey", JOURNEYS, ids=lambda item: item["events"])
def test_some_entities_stop_at_every_possible_stage(journey: dict[str, object]) -> None:
    events = pd.read_csv(DATA / str(journey["events"]))
    last_stage = events.groupby(str(journey["entity"]), sort=False).tail(1)
    stop_counts = last_stage[str(journey["stage"])].value_counts()
    expected_stages = {sequence[-1] for sequence in journey["sequences"]}

    assert set(stop_counts.index) == expected_stages
    assert (stop_counts > 0).all()
