"""Genera dos negocios 100 % sintéticos con muchas actividades por entidad.

Los nombres, códigos y valores no provienen de personas, empresas ni sistemas reales. La
semilla fija permite regenerar exactamente los mismos cuatro CSV usados por V2-19.

Uso:
    python scripts/generate_journey_datasets.py
    python scripts/generate_journey_datasets.py --output otra/carpeta --seed 20261021
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261021
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic" / "recorridos"
ALTA_ENTITIES = 3_000
SOLICITUD_ENTITIES = 2_000


def _advances(rng: np.random.Generator, logit: float) -> bool:
    probability = 1.0 / (1.0 + np.exp(-logit))
    return bool(rng.random() < probability)


def _positive_delay(
    rng: np.random.Generator,
    *,
    typical_hours: float,
    sigma: float = 0.55,
) -> pd.Timedelta:
    hours = max(1 / 60, float(rng.lognormal(np.log(typical_hours), sigma)))
    return pd.Timedelta(seconds=max(60, round(hours * 3_600)))


def _solicitud_event(
    request_id: str,
    moment: pd.Timestamp,
    stage: str,
    area: str,
    effort: float,
) -> dict[str, object]:
    return {
        "folio_gestion": request_id,
        "marca_temporal": moment.strftime("%Y-%m-%d %H:%M:%S"),
        "estado_tramite": stage,
        "grupo_en_turno": area,
        "minutos_esfuerzo": round(max(1.0, effort), 1),
    }


def _build_altas(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    channels = np.array(["portal", "aplicacion", "alianza", "punto_asistido"])
    channel_effect = {
        "portal": 0.05,
        "aplicacion": 0.40,
        "alianza": -0.35,
        "punto_asistido": 0.22,
    }
    channel_delay = {
        "portal": 1.0,
        "aplicacion": 0.72,
        "alianza": 1.35,
        "punto_asistido": 0.90,
    }
    stages = ("registro", "verificacion", "primera_compra", "segunda_compra")
    transition_logits = (1.05, 0.45, 0.05)
    stage_delays = (8.0, 48.0, 240.0)
    entity_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-05 08:00:00")

    for index in range(1, ALTA_ENTITIES + 1):
        entity_id = f"INC-{index:05d}"
        channel = str(rng.choice(channels, p=[0.34, 0.31, 0.18, 0.17]))
        plan = str(rng.choice(["esencial", "flexible", "ampliado"], p=[0.47, 0.36, 0.17]))
        zone = str(rng.choice(["zona_norte", "zona_centro", "zona_sur", "zona_costera"]))
        interest = float(
            np.clip(
                rng.normal(56, 16) + 8 * (channel == "aplicacion") + 5 * (plan == "ampliado"),
                5,
                98,
            )
        )
        entity_rows.append(
            {
                "codigo_incorporacion": entity_id,
                "modalidad_elegida": plan,
                "zona_servicio": zone,
                "indice_interes_inicial": round(interest, 1),
            }
        )

        moment = start + pd.Timedelta(
            days=int(rng.integers(0, 120)), seconds=int(rng.integers(0, 43_200))
        )
        for stage_index, stage in enumerate(stages):
            activity_value = float(
                np.clip(
                    interest + stage_index * 4 + rng.normal(0, 10),
                    1,
                    100,
                )
            )
            event_channel = channel
            if stage_index and rng.random() < 0.12:
                event_channel = str(rng.choice(channels))
            event_rows.append(
                {
                    "codigo_incorporacion": entity_id,
                    "momento_actividad": moment.strftime("%Y-%m-%d %H:%M:%S"),
                    "paso_incorporacion": stage,
                    "canal_interaccion": event_channel,
                    "valor_actividad": round(activity_value, 2),
                }
            )
            if stage_index == len(transition_logits):
                break
            logit = (
                transition_logits[stage_index]
                + channel_effect[event_channel]
                + (activity_value - 55) * 0.025
            )
            if not _advances(rng, logit):
                break
            typical = stage_delays[stage_index] * channel_delay[event_channel]
            typical *= max(0.55, 1.25 - activity_value / 180)
            moment += _positive_delay(rng, typical_hours=typical)

    entities = pd.DataFrame(entity_rows)
    events = pd.DataFrame(event_rows)
    return entities, events


def _build_solicitudes(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    priorities = np.array(["baja", "normal", "preferente", "urgente"])
    areas = np.array(["operacion", "cobros", "asistencia", "coordinacion"])
    sources = np.array(["formulario", "linea_voz", "mensajeria", "derivacion"])
    priority_speed = {"baja": 1.65, "normal": 1.0, "preferente": 0.62, "urgente": 0.30}
    area_speed = {"operacion": 1.0, "cobros": 1.20, "asistencia": 0.82, "coordinacion": 1.42}
    area_resolution = {
        "operacion": 0.18,
        "cobros": -0.12,
        "asistencia": 0.30,
        "coordinacion": -0.28,
    }
    entity_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-03-02 07:00:00")

    for index in range(1, SOLICITUD_ENTITIES + 1):
        request_id = f"GST-{index:05d}"
        priority = str(rng.choice(priorities, p=[0.20, 0.48, 0.24, 0.08]))
        area = str(rng.choice(areas, p=[0.31, 0.24, 0.30, 0.15]))
        source = str(rng.choice(sources, p=[0.40, 0.22, 0.27, 0.11]))
        complexity = int(
            np.clip(
                round(
                    rng.normal(48, 18) + 10 * (area == "coordinacion") + 6 * (priority == "urgente")
                ),
                5,
                95,
            )
        )
        entity_rows.append(
            {
                "folio_gestion": request_id,
                "nivel_atencion": priority,
                "sector_resolutor": area,
                "via_recepcion": source,
                "complejidad_estimada": complexity,
            }
        )

        moment = start + pd.Timedelta(
            days=int(rng.integers(0, 90)), seconds=int(rng.integers(0, 57_600))
        )

        event_rows.append(
            _solicitud_event(request_id, moment, "recibida", area, rng.normal(8, 2.2))
        )
        assignment_logit = 2.40 - complexity * 0.010 + 0.22 * (priority == "urgente")
        if not _advances(rng, assignment_logit):
            continue
        speed = priority_speed[priority] * area_speed[area]
        moment += _positive_delay(rng, typical_hours=5.0 * speed)
        event_rows.append(
            _solicitud_event(
                request_id,
                moment,
                "asignada",
                area,
                rng.normal(18 + complexity * 0.12, 5),
            )
        )

        review_logit = 1.85 - complexity * 0.012 + 0.18 * (priority in {"preferente", "urgente"})
        if not _advances(rng, review_logit):
            continue
        moment += _positive_delay(rng, typical_hours=18.0 * speed, sigma=0.70)
        event_rows.append(
            _solicitud_event(
                request_id,
                moment,
                "en_revision",
                area,
                rng.normal(35 + complexity * 0.45, 10),
            )
        )

        closure_logit = 1.35 - complexity * 0.015 + 0.30 * (priority == "urgente")
        if not _advances(rng, closure_logit):
            continue
        moment += _positive_delay(rng, typical_hours=38.0 * speed, sigma=0.78)
        resolution_logit = 1.20 + area_resolution[area] - complexity * 0.022
        final_stage = "resuelta" if _advances(rng, resolution_logit) else "cerrada_sin_resolver"
        event_rows.append(
            _solicitud_event(
                request_id,
                moment,
                final_stage,
                area,
                rng.normal(52 + complexity * 0.70, 15),
            )
        )

    entities = pd.DataFrame(entity_rows)
    events = pd.DataFrame(event_rows)
    return entities, events


def build(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Construye los cuatro archivos en memoria usando una sola secuencia aleatoria."""

    rng = np.random.default_rng(seed)
    alta_entities, alta_events = _build_altas(rng)
    request_entities, request_events = _build_solicitudes(rng)
    return {
        "altas_entidades.csv": alta_entities,
        "altas_eventos.csv": alta_events,
        "solicitudes_entidades.csv": request_entities,
        "solicitudes_eventos.csv": request_events,
    }


def write_datasets(output: Path, seed: int = SEED) -> dict[str, Path]:
    """Escribe CSV UTF-8 con finales de línea estables y devuelve sus rutas."""

    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in build(seed).items():
        path = output / name
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        written[name] = path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for name, path in write_datasets(args.output, args.seed).items():
        frame = pd.read_csv(path)
        print(f"{name}: {len(frame)} filas, {frame.shape[1]} columnas, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
