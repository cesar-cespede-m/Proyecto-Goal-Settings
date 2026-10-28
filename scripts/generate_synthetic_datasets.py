"""Genera datasets 100 % sintéticos para demos y pruebas de Goal Settings.

Ningún valor proviene de personas, empresas ni sistemas reales: todo sale de un generador
pseudoaleatorio con semilla fija, así que cada ejecución produce exactamente los mismos
archivos. Cada dataset incluye trampas deliberadas para comprobar que el asistente y la
persona las detectan:

- un identificador único por fila (no sirve para predecir);
- una columna constante;
- valores vacíos en alguna variable;
- más de una columna con dos valores (el resultado a anticipar no es obvio);
- al menos un dato que sólo se conoce DESPUÉS del evento (fuga de información).

Uso:
    python scripts/generate_synthetic_datasets.py               # escribe en data/synthetic/
    python scripts/generate_synthetic_datasets.py --output otra/carpeta
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261020
ROOT = Path(__file__).resolve().parents[1]


def _yes_no(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, "sí", "no")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def mantenimiento_maquinas(rows: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Fallas de máquinas en una planta ficticia. Evento: `falla_maquina` = sí (~12 %)."""

    rng = np.random.default_rng(seed)
    tipo = rng.choice(["A", "B", "C"], rows, p=[0.5, 0.3, 0.2])
    temperatura_aire = rng.normal(25, 2, rows)
    temperatura_proceso = temperatura_aire + rng.normal(10, 1, rows)
    velocidad = rng.normal(1500, 180, rows)
    torque = rng.normal(40, 9, rows)
    desgaste = rng.integers(0, 250, rows)
    turno_noche = rng.random(rows) < 0.35
    riesgo = (
        -4.2
        + 0.045 * (torque - 40)
        + 0.012 * (desgaste - 125)
        + 0.6 * (temperatura_proceso - temperatura_aire < 8.6)
        + 0.5 * (tipo == "C")
        + 0.25 * turno_noche
        - 0.002 * (velocidad - 1500)
    )
    falla = rng.random(rows) < _sigmoid(riesgo * 1.6 + 3.6)
    # Trampa: horas detenida sólo existe cuando la falla ya ocurrió.
    horas_detenida = np.where(falla, rng.gamma(2.0, 3.0, rows).round(1), 0.0)
    velocidad_registrada = velocidad.round(0)
    velocidad_registrada[rng.random(rows) < 0.06] = np.nan  # sensor intermitente
    return pd.DataFrame(
        {
            "id_maquina": [f"MAQ-{index:05d}" for index in range(rows)],
            "tipo_maquina": tipo,
            "temperatura_aire_c": temperatura_aire.round(1),
            "temperatura_proceso_c": temperatura_proceso.round(1),
            "velocidad_rpm": velocidad_registrada,
            "torque_nm": torque.round(1),
            "desgaste_herramienta_min": desgaste,
            "turno_noche": _yes_no(turno_noche),
            "planta": "Planta Demo",
            "horas_detenida_post_falla": horas_detenida,
            "falla_maquina": _yes_no(falla),
        }
    )


def abandono_suscripcion(rows: int = 3000, seed: int = SEED + 1) -> pd.DataFrame:
    """Bajas en un servicio de suscripción ficticio. Evento: `abandona` = sí (~18 %)."""

    rng = np.random.default_rng(seed)
    meses = rng.integers(1, 61, rows)
    plan = rng.choice(["básico", "estándar", "premium"], rows, p=[0.45, 0.4, 0.15])
    sesiones = rng.poisson(9, rows)
    minutos = rng.gamma(4, 9, rows)
    tickets = rng.poisson(0.8, rows)
    pago_automatico = rng.random(rows) < 0.6
    region = rng.choice(["Norte", "Centro", "Sur"], rows, p=[0.25, 0.5, 0.25])
    riesgo = (
        -1.0
        + 1.1 * (meses <= 6)
        + 0.9 * (sesiones <= 4)
        + 0.5 * (tickets >= 2)
        + 0.5 * (plan == "básico")
        + 0.7 * ~pago_automatico
        - 0.012 * (minutos - 36)
        + rng.normal(0, 0.6, rows)
    )
    abandona = rng.random(rows) < _sigmoid(riesgo * 1.4 - 1.5)
    # Trampa: la encuesta de salida se contesta al darse de baja.
    encuesta_salida = np.where(abandona, rng.random(rows) < 0.85, rng.random(rows) < 0.02)
    minutos_registrados = minutos.round(1)
    minutos_registrados[rng.random(rows) < 0.08] = np.nan
    return pd.DataFrame(
        {
            "id_suscriptor": [f"SUS-{index:05d}" for index in range(rows)],
            "meses_antiguedad": meses,
            "plan": plan,
            "sesiones_ultimo_mes": sesiones,
            "minutos_promedio_sesion": minutos_registrados,
            "tickets_soporte_90d": tickets,
            "pago_automatico": _yes_no(pago_automatico),
            "region": region,
            "moneda": "CLP-demo",
            "encuesta_salida_completada": _yes_no(encuesta_salida),
            "abandona": _yes_no(abandona),
        }
    )


def retraso_entregas(rows: int = 3500, seed: int = SEED + 2) -> pd.DataFrame:
    """Envíos de una logística ficticia. Evento: `entregado_tarde` = sí (~22 %): más de 40 min."""

    rng = np.random.default_rng(seed)
    distancia = rng.gamma(3, 12, rows)
    peso = rng.gamma(2, 4, rows)
    servicio = rng.choice(["estándar", "express"], rows, p=[0.7, 0.3])
    dia = rng.choice(["lun", "mar", "mié", "jue", "vie", "sáb"], rows)
    clima = rng.choice(["despejado", "lluvia", "tormenta"], rows, p=[0.7, 0.24, 0.06])
    pedidos_centro = rng.poisson(420, rows)
    conductor_nuevo = rng.random(rows) < 0.2
    minutos_retraso = (
        rng.normal(-15, 20, rows)
        + 0.6 * distancia
        + 25 * (clima == "lluvia")
        + 60 * (clima == "tormenta")
        + 0.08 * (pedidos_centro - 420)
        + 18 * conductor_nuevo
        - 10 * (servicio == "express")
        + 8 * (dia == "sáb")
    )
    tarde = minutos_retraso > 40
    # Trampas: el retraso real y el reclamo se conocen después de entregar.
    reclamo = np.where(tarde, rng.random(rows) < 0.45, rng.random(rows) < 0.03)
    clima_registrado = clima.astype(object)
    clima_registrado[rng.random(rows) < 0.05] = None
    return pd.DataFrame(
        {
            "id_envio": [f"ENV-{index:06d}" for index in range(rows)],
            "distancia_km": distancia.round(1),
            "peso_kg": peso.round(2),
            "tipo_servicio": servicio,
            "dia_semana": dia,
            "clima_previsto": clima_registrado,
            "pedidos_dia_centro": pedidos_centro,
            "conductor_nuevo": _yes_no(conductor_nuevo),
            "pais_operacion": "País Demo",
            "minutos_retraso_real": minutos_retraso.round(0),
            "reclamo_cliente": _yes_no(reclamo),
            "entregado_tarde": _yes_no(tarde),
        }
    )


def reservas_restaurante(rows: int = 2500, seed: int = SEED + 3) -> pd.DataFrame:
    """Reservas de una cadena de restaurantes ficticia. Evento: `no_asistio` = sí (~15 %).

    Dataset adicional para comprobar reutilización en un dominio nuevo.
    """

    rng = np.random.default_rng(seed)
    anticipacion = rng.integers(0, 31, rows)
    personas = rng.choice([1, 2, 3, 4, 5, 6, 8], rows, p=[0.08, 0.38, 0.14, 0.22, 0.08, 0.07, 0.03])
    dia = rng.choice(["lun", "mar", "mié", "jue", "vie", "sáb", "dom"], rows)
    turno = rng.choice(["almuerzo", "cena"], rows, p=[0.4, 0.6])
    canal = rng.choice(["app", "teléfono", "web"], rows, p=[0.45, 0.25, 0.3])
    frecuente = rng.random(rows) < 0.3
    canceladas_previas = rng.poisson(0.4, rows)
    lluvia = rng.random(rows) < 0.2
    confirmo_recordatorio = rng.random(rows) < 0.55
    riesgo = (
        -2.2
        + 0.05 * anticipacion
        + 0.6 * canceladas_previas
        - 0.9 * frecuente
        + 0.5 * lluvia
        - 1.1 * confirmo_recordatorio
        + 0.4 * (canal == "web")
        + 0.25 * (personas >= 6)
        + rng.normal(0, 0.5, rows)
    )
    no_asistio = rng.random(rows) < _sigmoid(riesgo * 1.5 + 0.2)
    # Trampas: la propina y la espera sólo existen si la persona llegó.
    propina = np.where(no_asistio, 0.0, rng.gamma(2, 2500, rows).round(0))
    espera = np.where(no_asistio, 0, rng.integers(0, 40, rows))
    lluvia_registrada = _yes_no(lluvia).astype(object)
    lluvia_registrada[rng.random(rows) < 0.04] = None
    return pd.DataFrame(
        {
            "id_reserva": [f"RES-{index:05d}" for index in range(rows)],
            "dias_anticipacion": anticipacion,
            "personas_mesa": personas,
            "dia_semana": dia,
            "turno": turno,
            "canal_reserva": canal,
            "cliente_frecuente": _yes_no(frecuente),
            "cancelaciones_previas": canceladas_previas,
            "pronostico_lluvia": lluvia_registrada,
            "confirmo_recordatorio": _yes_no(confirmo_recordatorio),
            "cadena": "Restaurantes Demo",
            "propina_registrada": propina,
            "minutos_espera_mesa": espera,
            "no_asistio": _yes_no(no_asistio),
        }
    )


DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "mantenimiento_maquinas.csv": mantenimiento_maquinas,
    "abandono_suscripcion.csv": abandono_suscripcion,
    "retraso_entregas.csv": retraso_entregas,
    "reservas_restaurante.csv": reservas_restaurante,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, build in DATASETS.items():
        frame = build()
        path = args.output / name
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        event = frame.columns[-1]
        positives = int((frame[event] == "sí").sum())
        print(
            f"{name}: {len(frame)} filas, {frame.shape[1]} columnas, "
            f"evento «{event} = sí» en {positives} filas ({positives / len(frame):.1%})"
        )


if __name__ == "__main__":
    main()
