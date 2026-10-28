"""Los datasets sintéticos versionados son reproducibles y contienen sus trampas."""

import importlib.util
from pathlib import Path

import pytest

from core.data_engine import validate_and_profile
from core.domain.assistant import is_binary_candidate

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_synthetic_datasets", ROOT / "scripts" / "generate_synthetic_datasets.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)

TRAPS = {
    "mantenimiento_maquinas.csv": ("falla_maquina", "id_maquina", "planta", "velocidad_rpm"),
    "abandono_suscripcion.csv": ("abandona", "id_suscriptor", "moneda", "minutos_promedio_sesion"),
    "retraso_entregas.csv": ("entregado_tarde", "id_envio", "pais_operacion", "clima_previsto"),
    "reservas_restaurante.csv": ("no_asistio", "id_reserva", "cadena", "pronostico_lluvia"),
}


@pytest.mark.parametrize("name", sorted(generator.DATASETS))
def test_versioned_file_matches_the_seeded_generator(name: str, tmp_path: Path) -> None:
    frame = generator.DATASETS[name]()
    regenerated = tmp_path / name
    frame.to_csv(regenerated, index=False, encoding="utf-8", lineterminator="\n")
    assert regenerated.read_bytes() == (ROOT / "data" / "synthetic" / name).read_bytes()


@pytest.mark.parametrize("name", sorted(TRAPS))
def test_profile_exposes_the_deliberate_traps(name: str) -> None:
    target, identifier, constant, with_missing = TRAPS[name]
    content = (ROOT / "data" / "synthetic" / name).read_bytes()
    profile = validate_and_profile(name, content)
    by_name = {column.name: column for column in profile.columns}

    assert by_name[identifier].possible_identifier
    assert constant in profile.constant_columns
    assert by_name[with_missing].missing_count > 0
    binary = [column.name for column in profile.columns if is_binary_candidate(column)]
    assert target in binary and len(binary) >= 2
