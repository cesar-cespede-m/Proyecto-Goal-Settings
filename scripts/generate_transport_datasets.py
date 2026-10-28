"""Negocio de prueba con varios archivos y fechas (suite de dominios, GA-10).

Es sólo uno de los negocios con que se valida que Goal Settings es genérica: nada del producto
puede depender de estos nombres. 100 % sintético, con semilla fija y patrones deliberados.
Cuatro archivos relacionados por `Estacion`:

- `transporte_estaciones.csv`: una fila por estación (tipo, línea, comuna, torniquetes).
- `transporte_transacciones.csv`: viajes individuales de tres semanas con punta de mañana y de
  tarde en días hábiles, fines de semana más bajos y estaciones con demanda muy distinta.
- `transporte_buses.csv`: buses por hora que alimentan cada estación por franja; algunas
  estaciones de barrio quedan con poca oferta para su demanda (hallazgo deliberado).
- `transporte_encuestas.csv`: satisfacción declarada, que baja con la espera y en la punta.

Trampas deliberadas para que el análisis las detecte: `Estación_31` sólo existe en buses y
`Estación_30` no tiene encuestas; la tarifa depende de la franja horaria.

Uso:
    python scripts/generate_transport_datasets.py
    python scripts/generate_transport_datasets.py --output otra/carpeta
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261021
ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2026-09-07")  # lunes
DAYS = 21
STATIONS = 30
UNDERSERVED = {"Estación_7", "Estación_12", "Estación_19", "Estación_23", "Estación_28"}

# Perfil horario relativo (5:00 a 23:00) por tipo de estación en día hábil.
_HOURS = np.arange(24)


def _profile(morning: float, midday: float, evening: float) -> np.ndarray:
    shape = (
        morning * np.exp(-0.5 * ((_HOURS - 7.8) / 1.0) ** 2)
        + midday * np.exp(-0.5 * ((_HOURS - 13.0) / 2.2) ** 2)
        + evening * np.exp(-0.5 * ((_HOURS - 18.6) / 1.2) ** 2)
        + 0.08
    )
    shape[(_HOURS < 5) | (_HOURS > 23)] = 0
    return shape / shape.sum()


PROFILES = {
    "barrio": _profile(1.0, 0.25, 0.55),
    "centro": _profile(0.45, 0.6, 1.0),
    "intermodal": _profile(0.9, 0.45, 0.95),
}
WEEKEND = _profile(0.15, 0.9, 0.35)
BASE_DAILY = {"intermodal": 230, "centro": 140, "barrio": 60}


def _time_band(hour: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    workday = weekday < 5
    band = np.full(hour.shape, "valle", dtype=object)
    band[workday & (hour >= 7) & (hour < 9)] = "punta_manana"
    band[workday & (hour >= 18) & (hour < 20)] = "punta_tarde"
    band[(hour < 6) | (hour >= 21)] = "noche"
    band[~workday] = "fin_de_semana"
    return band


def estaciones(rng: np.random.Generator) -> pd.DataFrame:
    kinds = np.array(["intermodal"] * 6 + ["centro"] * 10 + ["barrio"] * 14)
    rng.shuffle(kinds)
    names = [f"Estación_{index}" for index in range(1, STATIONS + 1)]
    comunas = rng.choice(
        ["Comuna Norte", "Comuna Centro", "Comuna Sur", "Comuna Oriente"], STATIONS
    )
    turnstiles = np.where(kinds == "intermodal", 12, np.where(kinds == "centro", 8, 4))
    turnstiles = turnstiles + rng.integers(-1, 2, STATIONS)
    return pd.DataFrame(
        {
            "Estacion": names,
            "tipo_estacion": kinds,
            "linea": rng.choice(["L1", "L2", "L3"], STATIONS, p=[0.4, 0.35, 0.25]),
            "comuna": comunas,
            "torniquetes": turnstiles,
        }
    )


def transacciones(rng: np.random.Generator, stations: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for row in stations.itertuples():
        daily = BASE_DAILY[row.tipo_estacion] * rng.lognormal(0, 0.35)
        if row.Estacion in UNDERSERVED:
            daily *= 1.6  # barrios que crecieron más rápido que su oferta de buses
        for day in range(DAYS):
            date = START + pd.Timedelta(days=day)
            weekend = date.weekday() >= 5
            profile = WEEKEND if weekend else PROFILES[row.tipo_estacion]
            total = rng.poisson(daily * (0.55 if weekend else 1.0))
            hours = rng.choice(24, total, p=profile)
            seconds = rng.integers(0, 3600, total)
            pieces.append(
                pd.DataFrame(
                    {
                        "fecha_hora": date + pd.to_timedelta(hours * 3600 + seconds, unit="s"),
                        "Estacion": row.Estacion,
                    }
                )
            )
    frame = pd.concat(pieces, ignore_index=True).sort_values("fecha_hora", kind="stable")
    hour = frame["fecha_hora"].dt.hour.to_numpy()
    weekday = frame["fecha_hora"].dt.weekday.to_numpy()
    band = _time_band(hour, weekday)
    fare = np.where(
        np.isin(band, ["punta_manana", "punta_tarde"]), 800, np.where(band == "noche", 710, 760)
    )
    users = rng.choice(["adulto", "estudiante", "adulto_mayor"], len(frame), p=[0.72, 0.18, 0.10])
    fare = np.where(users == "estudiante", 260, np.where(users == "adulto_mayor", 390, fare))
    frame = frame.reset_index(drop=True)
    return pd.DataFrame(
        {
            "id_transaccion": [f"TX-{index:06d}" for index in range(len(frame))],
            "fecha_hora": frame["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Estacion": frame["Estacion"],
            "tipo_usuario": users,
            "medio_pago": rng.choice(["tarjeta", "app"], len(frame), p=[0.8, 0.2]),
            "tarifa": fare,
        }
    )


def buses(rng: np.random.Generator, trips: pd.DataFrame) -> pd.DataFrame:
    stamp = pd.to_datetime(trips["fecha_hora"])
    band = _time_band(stamp.dt.hour.to_numpy(), stamp.dt.weekday.to_numpy())
    demand = (
        pd.DataFrame({"Estacion": trips["Estacion"], "franja": band})
        .groupby(["Estacion", "franja"])
        .size()
        .rename("viajes")
        .reset_index()
    )
    hours_per_band = {
        "punta_manana": 2,
        "punta_tarde": 2,
        "valle": 9,
        "noche": 4,
        "fin_de_semana": 18,
    }
    days = demand["franja"].map(lambda franja: 6 if franja == "fin_de_semana" else 15)
    per_hour = demand["viajes"] / (demand["franja"].map(hours_per_band) * days)
    # Frecuencia mínima de servicio: 4 buses por hora; las estaciones con poca oferta reciben
    # menos de la mitad de lo que su demanda justificaría.
    supply = np.maximum(per_hour / 6, 4) * rng.lognormal(0, 0.15, len(demand))
    supply = np.where(demand["Estacion"].isin(UNDERSERVED), supply * 0.4, supply)
    table = demand[["Estacion", "franja"]].copy()
    table["buses_por_hora"] = np.clip(np.round(supply, 1), 0.5, None)
    extra = pd.DataFrame(
        {
            "Estacion": "Estación_31",
            "franja": list(hours_per_band),
            "buses_por_hora": [6.0, 6.0, 3.0, 1.5, 2.5],
        }
    )
    return pd.concat([table, extra], ignore_index=True)


def encuestas(rng: np.random.Generator, trips: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    rows = 1500
    sample = trips[trips["Estacion"] != "Estación_30"].sample(rows, random_state=SEED)
    stamp = pd.to_datetime(sample["fecha_hora"])
    band = _time_band(stamp.dt.hour.to_numpy(), stamp.dt.weekday.to_numpy())
    joined = pd.DataFrame({"Estacion": sample["Estacion"].to_numpy(), "franja": band}).merge(
        supply, on=["Estacion", "franja"], how="left"
    )
    wait = np.clip(60 / joined["buses_por_hora"] / 2 + rng.normal(0, 2.5, rows), 1, 45)
    peak = np.isin(band, ["punta_manana", "punta_tarde"])
    score = 4.7 - 0.11 * wait - 0.45 * peak + rng.normal(0, 0.6, rows)
    return pd.DataFrame(
        {
            "id_respuesta": [f"ENC-{index:05d}" for index in range(rows)],
            "Estacion": joined["Estacion"],
            "franja_viaje": band,
            "edad": np.clip(rng.normal(38, 14, rows).round(), 15, 85).astype(int),
            "frecuencia_uso": rng.choice(
                ["diaria", "semanal", "ocasional"], rows, p=[0.6, 0.3, 0.1]
            ),
            "espera_declarada_min": wait.round().astype(int),
            "satisfaccion": np.clip(np.round(score), 1, 5).astype(int),
        }
    )


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    stations = estaciones(rng)
    trips = transacciones(rng, stations)
    supply = buses(rng, trips)
    return {
        "transporte_estaciones.csv": stations,
        "transporte_transacciones.csv": trips,
        "transporte_buses.csv": supply,
        "transporte_encuestas.csv": encuestas(rng, trips, supply),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic" / "transporte")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in build().items():
        frame.to_csv(args.output / name, index=False, encoding="utf-8", lineterminator="\n")
        print(f"{name}: {len(frame)} filas, {frame.shape[1]} columnas")


if __name__ == "__main__":
    main()
