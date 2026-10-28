"""GA-10: los negocios de prueba existen, se aceptan y tienen las propiedades declaradas."""

import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from core.data_engine import validate_and_profile
from tests.support.domains import DOMAINS, ROOT

_SPEC = importlib.util.spec_from_file_location(
    "generate_transport_datasets", ROOT / "scripts" / "generate_transport_datasets.py"
)
transport = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = transport
_SPEC.loader.exec_module(transport)


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_domain_file_is_accepted_by_the_current_ingestion(domain) -> None:
    assert len(DOMAINS) >= 6
    for path in domain.files:
        profile = validate_and_profile(path.name, path.read_bytes())
        assert profile.row_count > 0, path.name


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_declared_shape_matches_the_files(domain) -> None:
    frames = [pd.read_csv(path) for path in domain.files]
    assert (len(frames) > 1) == domain.multi_file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas avisa al probar columnas que no son fechas
        has_datetime = any(
            pd.to_datetime(frame[column], errors="coerce").notna().mean() > 0.95
            for frame in frames
            for column in frame.select_dtypes(include=["object", "string"]).columns
        )
    assert has_datetime == domain.has_datetimes


def test_multi_file_business_is_reproducible(tmp_path: Path) -> None:
    for name, frame in transport.build().items():
        regenerated = tmp_path / name
        frame.to_csv(regenerated, index=False, encoding="utf-8", lineterminator="\n")
        versioned = ROOT / "data" / "synthetic" / "transporte" / name
        assert regenerated.read_bytes() == versioned.read_bytes(), name


def test_multi_file_business_has_the_deliberate_patterns() -> None:
    folder = ROOT / "data" / "synthetic" / "transporte"
    trips = pd.read_csv(folder / "transporte_transacciones.csv", parse_dates=["fecha_hora"])
    supply = pd.read_csv(folder / "transporte_buses.csv")
    surveys = pd.read_csv(folder / "transporte_encuestas.csv")

    workday = trips[trips["fecha_hora"].dt.weekday < 5]
    by_hour = workday.groupby(workday["fecha_hora"].dt.hour).size()
    assert by_hour.loc[[7, 8, 18, 19]].min() > 1.2 * by_hour.loc[[11, 12, 13, 14]].mean()
    per_station = trips.groupby("Estacion").size()
    assert per_station.max() > 5 * per_station.min()
    assert set(supply["Estacion"]) - set(trips["Estacion"])  # clave sólo en un archivo
    assert surveys["espera_declarada_min"].corr(surveys["satisfaccion"]) < -0.3


def test_owner_case_has_no_temporal_variation_to_learn() -> None:
    folder = ROOT / "data" / "synthetic" / "caso_responsable"
    trips = pd.read_csv(folder / "datos_bip_simulados.csv", parse_dates=["Horario"])
    flows = pd.read_csv(folder / "datos_flujos_simulados.csv")

    gaps = trips["Horario"].sort_values().diff().dropna().dt.total_seconds().round()
    assert gaps.nunique() == 1  # una transacción cada intervalo fijo: nada que aprender
    assert set(trips["Estacion"]) ^ set(flows["Estacion"])  # claves que no calzan
