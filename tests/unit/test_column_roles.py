"""V2-18: papeles de columna inferidos sólo por comportamiento."""

import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from core.analysis.roles import ColumnRolesV1, ColumnRoleV1, infer_roles
from core.data_engine import validate_and_profile
from tests.support.domains import DOMAINS

ROOT = Path(__file__).parents[2]
SYNTHETIC = ROOT / "data" / "synthetic"
JOURNEYS = (
    (
        "altas_eventos.csv",
        "codigo_incorporacion",
        "momento_actividad",
        "paso_incorporacion",
        ["registro", "verificacion", "primera_compra", "segunda_compra"],
    ),
    (
        "solicitudes_eventos.csv",
        "folio_gestion",
        "marca_temporal",
        "estado_tramite",
        ["recibida", "asignada", "en_revision", "cerrada_sin_resolver", "resuelta"],
    ),
)


def _infer(path: Path) -> ColumnRolesV1:
    frame = pd.read_csv(path)
    profile = validate_and_profile(path.name, path.read_bytes())
    return infer_roles(frame, profile)


def _by_column(roles: ColumnRolesV1) -> dict[str, ColumnRoleV1]:
    return {item.column: item for item in roles.items}


@pytest.mark.parametrize(
    ("filename", "entity", "moment", "stage", "expected_order"),
    JOURNEYS,
)
def test_journeys_detect_entity_moment_and_ordered_stage(
    filename: str,
    entity: str,
    moment: str,
    stage: str,
    expected_order: list[str],
) -> None:
    roles = _by_column(_infer(SYNTHETIC / "recorridos" / filename))

    assert roles[entity].role == "entity"
    assert roles[moment].role == "moment"
    assert roles[stage].role == "stage"
    assert roles[stage].stage_order == expected_order
    assert min(roles[name].confidence for name in (entity, moment, stage)) >= 0.8


def test_one_row_per_case_does_not_invent_entity_or_stage() -> None:
    roles = _infer(SYNTHETIC / "abandono_suscripcion.csv")

    assert not [item for item in roles.items if item.role == "entity"]
    assert not [item for item in roles.items if item.role == "stage"]
    assert {item.role for item in roles.items} <= {
        "measure",
        "attribute",
        "outcome_candidate",
        "other",
    }


@pytest.mark.parametrize(
    "path",
    [path for domain in DOMAINS for path in domain.files],
    ids=lambda path: str(path.relative_to(SYNTHETIC)),
)
def test_role_inference_runs_for_every_test_business(path: Path) -> None:
    roles = _infer(path)

    assert [item.column for item in roles.items] == list(pd.read_csv(path, nrows=0).columns)


def test_reasons_do_not_expose_uppercase_codes() -> None:
    code = re.compile(r"\b[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9_]{2,}\b")

    for filename, *_ in JOURNEYS:
        roles = _infer(SYNTHETIC / "recorridos" / filename)
        assert all(not code.search(item.reason) for item in roles.items)


def test_confirmation_is_explicit_immutable_and_strict() -> None:
    proposed = _infer(SYNTHETIC / "recorridos" / "altas_eventos.csv")
    column = proposed.items[0].column

    confirmed = proposed.with_confirmation(column, "event", "Analista de procesos")

    assert proposed.items[0].confirmation is None
    assert confirmed.items[0].confirmation is not None
    assert confirmed.items[0].confirmation.role == "event"
    assert confirmed.items[0].confirmation.actor_label == "Analista de procesos"
    with pytest.raises(ValidationError):
        ColumnRolesV1.model_validate(
            {
                "items": [item.model_dump() for item in confirmed.items],
                "campo_inesperado": True,
            }
        )
