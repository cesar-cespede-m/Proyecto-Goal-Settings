"""Genera variantes 100 % sintéticas de los datasets demo para monitoreo y pruebas de fallo.

Parte de los generadores de `generate_synthetic_datasets.py` (misma semilla base) y crea, por
dominio, en `data/synthetic/variantes/<dominio>/`:

- tamaños de entrenamiento distintos (grande, pequeño, mínimo);
- lotes de uso del modelo: estable, cambios leves y fuertes, sin resultado, pocos casos,
  categoría nueva, muchos vacíos, unidades cambiadas, columnas reordenadas y filas repetidas;
- lotes que deben fallar o advertir: columna faltante o renombrada, resultado desconocido,
  separador y decimales distintos, otra codificación, texto en una columna numérica, sólo
  encabezado;
- un lote reciente para reentrenar;
- archivos auxiliares para cruzar varios CSV con claves sanas, duplicadas, faltantes, con otro
  formato y con columnas repetidas.

Los cambios se crean de dos formas declaradas:
- **cambio de población:** se re-muestrea un conjunto grande dando más peso a valores altos (o
  bajos) de una columna clave; la relación entre datos y resultado no cambia;
- **cambio de comportamiento:** en el lote fuerte y el de reentrenamiento, una parte de los casos
  que el modelo consideraba de bajo riesgo pasa a tener el evento; las columnas posteriores al
  evento se recalculan para seguir siendo coherentes.

Uso:
    python scripts/generate_synthetic_variants.py
    python scripts/generate_synthetic_variants.py --output otra/carpeta
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "generate_synthetic_datasets", ROOT / "scripts" / "generate_synthetic_datasets.py"
)
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

SEED = base.SEED + 100
YES, NO = "sí", "no"


@dataclass(frozen=True)
class DomainSpec:
    folder: str
    base_file: str
    build: Callable[..., pd.DataFrame]
    target: str
    id_column: str
    id_prefix: str
    shift_column: str  # columna cuyo cambio de población se simula
    shift_direction: float  # +1 favorece valores altos; -1, bajos
    category_column: str
    new_category: str
    missing_column: str
    unit_column: str
    unit_factor: float
    unit_note: str
    renamed_column: str
    renamed_to: str
    fix_post_event: PostEventFix
    auxiliary_name: str
    auxiliary_columns: Callable[[pd.DataFrame, np.random.Generator], dict[str, object]]


PostEventFix = Callable[[pd.DataFrame, np.ndarray, bool, np.random.Generator], None]


def _fix_maquinas(
    frame: pd.DataFrame, rows: np.ndarray, event: bool, rng: np.random.Generator
) -> None:
    hours = rng.gamma(2.0, 3.0, rows.sum()).round(1) if event else 0.0
    frame.loc[rows, "horas_detenida_post_falla"] = hours


def _fix_suscripcion(
    frame: pd.DataFrame, rows: np.ndarray, event: bool, rng: np.random.Generator
) -> None:
    share = 0.85 if event else 0.02
    frame.loc[rows, "encuesta_salida_completada"] = np.where(
        rng.random(rows.sum()) < share, YES, NO
    )


def _fix_entregas(
    frame: pd.DataFrame, rows: np.ndarray, event: bool, rng: np.random.Generator
) -> None:
    low, high = (41, 95) if event else (-20, 40)
    frame.loc[rows, "minutos_retraso_real"] = rng.integers(low, high, rows.sum()).astype(float)
    share = 0.45 if event else 0.03
    frame.loc[rows, "reclamo_cliente"] = np.where(rng.random(rows.sum()) < share, YES, NO)


def _fix_reservas(
    frame: pd.DataFrame, rows: np.ndarray, event: bool, rng: np.random.Generator
) -> None:
    if event:
        frame.loc[rows, "propina_registrada"] = 0.0
        frame.loc[rows, "minutos_espera_mesa"] = 0
    else:
        frame.loc[rows, "propina_registrada"] = rng.gamma(2, 2500, rows.sum()).round(0)
        frame.loc[rows, "minutos_espera_mesa"] = rng.integers(0, 40, rows.sum())


def _aux_maquinas(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    wear = frame["desgaste_herramienta_min"].to_numpy(dtype=float)
    return {
        "dias_desde_ultima_mantencion": np.clip(
            wear * 0.25 + rng.normal(10, 8, len(frame)), 0, None
        )
        .round(0)
        .astype(int),
        "proveedor_mantencion": rng.choice(["Servicios Demo", "Técnicos Ficticios"], len(frame)),
    }


def _aux_suscripcion(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    sessions = frame["sesiones_ultimo_mes"].to_numpy(dtype=float)
    return {
        "correos_abiertos_30d": np.clip(sessions * 0.6 + rng.normal(1, 2, len(frame)), 0, None)
        .round(0)
        .astype(int),
        "canal_alta": rng.choice(["web", "app", "tienda"], len(frame), p=[0.5, 0.35, 0.15]),
    }


def _aux_entregas(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    distance = frame["distancia_km"].to_numpy(dtype=float)
    return {
        "paradas_en_ruta": np.clip(distance / 6 + rng.normal(2, 2, len(frame)), 1, None)
        .round(0)
        .astype(int),
        "zona_destino": rng.choice(
            ["urbana", "periurbana", "rural"], len(frame), p=[0.6, 0.3, 0.1]
        ),
    }


def _aux_reservas(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    return {
        "visitas_ultimo_anio": rng.poisson(2, len(frame)),
        "tiene_alergias_registradas": np.where(rng.random(len(frame)) < 0.15, YES, NO),
    }


DOMAINS = (
    DomainSpec(
        folder="mantenimiento_maquinas",
        base_file="mantenimiento_maquinas.csv",
        build=base.mantenimiento_maquinas,
        target="falla_maquina",
        id_column="id_maquina",
        id_prefix="MAQ",
        shift_column="desgaste_herramienta_min",
        shift_direction=1.0,
        category_column="tipo_maquina",
        new_category="D",
        missing_column="torque_nm",
        unit_column="temperatura_aire_c",
        unit_factor=1.8,  # se suma 32 aparte: °C → °F
        unit_note="temperatura_aire_c llega en °F aunque el nombre dice °C",
        renamed_column="torque_nm",
        renamed_to="torque",
        fix_post_event=_fix_maquinas,
        auxiliary_name="mantenciones",
        auxiliary_columns=_aux_maquinas,
    ),
    DomainSpec(
        folder="abandono_suscripcion",
        base_file="abandono_suscripcion.csv",
        build=base.abandono_suscripcion,
        target="abandona",
        id_column="id_suscriptor",
        id_prefix="SUS",
        shift_column="sesiones_ultimo_mes",
        shift_direction=-1.0,
        category_column="plan",
        new_category="familiar",
        missing_column="tickets_soporte_90d",
        unit_column="minutos_promedio_sesion",
        unit_factor=1 / 60,
        unit_note="minutos_promedio_sesion llega en horas aunque el nombre dice minutos",
        renamed_column="sesiones_ultimo_mes",
        renamed_to="sesiones_mes",
        fix_post_event=_fix_suscripcion,
        auxiliary_name="campanas_correo",
        auxiliary_columns=_aux_suscripcion,
    ),
    DomainSpec(
        folder="retraso_entregas",
        base_file="retraso_entregas.csv",
        build=base.retraso_entregas,
        target="entregado_tarde",
        id_column="id_envio",
        id_prefix="ENV",
        shift_column="distancia_km",
        shift_direction=1.0,
        category_column="clima_previsto",
        new_category="nieve",
        missing_column="pedidos_dia_centro",
        unit_column="distancia_km",
        unit_factor=0.621371,
        unit_note="distancia_km llega en millas aunque el nombre dice km",
        renamed_column="distancia_km",
        renamed_to="distancia",
        fix_post_event=_fix_entregas,
        auxiliary_name="rutas",
        auxiliary_columns=_aux_entregas,
    ),
    DomainSpec(
        folder="reservas_restaurante",
        base_file="reservas_restaurante.csv",
        build=base.reservas_restaurante,
        target="no_asistio",
        id_column="id_reserva",
        id_prefix="RES",
        shift_column="dias_anticipacion",
        shift_direction=1.0,
        category_column="canal_reserva",
        new_category="whatsapp",
        missing_column="cancelaciones_previas",
        unit_column="dias_anticipacion",
        unit_factor=24.0,
        unit_note="dias_anticipacion llega en horas aunque el nombre dice días",
        renamed_column="dias_anticipacion",
        renamed_to="anticipacion",
        fix_post_event=_fix_reservas,
        auxiliary_name="historial_clientes",
        auxiliary_columns=_aux_reservas,
    ),
)


def _with_ids(frame: pd.DataFrame, spec: DomainSpec, code: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame[spec.id_column] = [f"{spec.id_prefix}-{code}-{index:05d}" for index in range(len(frame))]
    return frame


def _population_shift(
    spec: DomainSpec, rows: int, seed: int, strength: float, code: str
) -> pd.DataFrame:
    """Re-muestrea un conjunto grande favoreciendo un extremo de la columna clave."""

    rng = np.random.default_rng(seed)
    pool = spec.build(rows=rows * 4, seed=seed)
    values = pool[spec.shift_column].astype(float)
    zscore = ((values - values.mean()) / values.std()).fillna(0).to_numpy()
    weights = np.exp(strength * spec.shift_direction * zscore)
    chosen = rng.choice(len(pool), rows, replace=False, p=weights / weights.sum())
    return _with_ids(pool.iloc[np.sort(chosen)].copy(), spec, code)


def _behavior_shift(frame: pd.DataFrame, spec: DomainSpec, seed: int, share: float) -> pd.DataFrame:
    """Cambia la relación entre datos y resultado en ambos sentidos.

    Casos de bajo riesgo según la columna clave pasan a tener el evento y casos de alto riesgo
    dejan de tenerlo (como si una nueva práctica corrigiera la causa conocida).
    """

    rng = np.random.default_rng(seed)
    values = frame[spec.shift_column].astype(float) * spec.shift_direction
    low = (values < values.median()).to_numpy()
    event = (frame[spec.target] == YES).to_numpy()
    to_event = low & ~event & (rng.random(len(frame)) < share)
    to_no_event = ~low & event & (rng.random(len(frame)) < share * 2)
    frame = frame.copy()
    frame.loc[to_event, spec.target] = YES
    frame.loc[to_no_event, spec.target] = NO
    spec.fix_post_event(frame, to_event, True, rng)
    spec.fix_post_event(frame, to_no_event, False, rng)
    return frame


def _stable(spec: DomainSpec, rows: int, seed: int, code: str) -> pd.DataFrame:
    return _with_ids(spec.build(rows=rows, seed=seed), spec, code)


def domain_variants(spec: DomainSpec, index: int) -> dict[str, pd.DataFrame | bytes]:
    seed = SEED + index * 1000
    rng = np.random.default_rng(seed)
    training = pd.read_csv(ROOT / "data" / "synthetic" / spec.base_file)
    files: dict[str, pd.DataFrame | bytes] = {}

    # Tamaños de entrenamiento.
    files["entrenamiento_grande.csv"] = _stable(spec, 15000, seed + 1, "G")
    files["entrenamiento_pequeno.csv"] = _stable(spec, 300, seed + 2, "P")
    files["entrenamiento_minimo.csv"] = _stable(spec, 60, seed + 3, "M")

    # Lotes de uso del modelo.
    files["uso_01_estable.csv"] = _stable(spec, 1000, seed + 10, "U01")
    files["uso_02_cambio_leve.csv"] = _population_shift(spec, 1000, seed + 11, 0.4, "U02")
    files["uso_03_cambio_fuerte.csv"] = _behavior_shift(
        _population_shift(spec, 1500, seed + 12, 1.6, "U03"), spec, seed + 13, 0.4
    )
    files["uso_04_sin_resultado.csv"] = _stable(spec, 800, seed + 14, "U04").drop(
        columns=[spec.target]
    )
    files["uso_05_pocos_casos.csv"] = _stable(spec, 20, seed + 15, "U05")

    new_category = _stable(spec, 1000, seed + 16, "U06")
    mask = rng.random(len(new_category)) < 0.3
    new_category[spec.category_column] = new_category[spec.category_column].astype(object)
    new_category.loc[mask, spec.category_column] = spec.new_category
    files["uso_06_categoria_nueva.csv"] = new_category

    many_missing = _stable(spec, 1000, seed + 17, "U07")
    many_missing[spec.missing_column] = many_missing[spec.missing_column].astype(float)
    many_missing.loc[rng.random(len(many_missing)) < 0.45, spec.missing_column] = np.nan
    files["uso_07_muchos_vacios.csv"] = many_missing

    units = _stable(spec, 1000, seed + 18, "U08")
    converted = units[spec.unit_column].astype(float) * spec.unit_factor
    if spec.unit_column == "temperatura_aire_c":
        converted = converted + 32
    units[spec.unit_column] = converted.round(1)
    files["uso_08_unidades_cambiadas.csv"] = units

    reordered = _stable(spec, 1000, seed + 19, "U09")
    reordered["observacion_operador"] = rng.choice(["sin novedad", "revisar", ""], len(reordered))
    files["uso_09_columnas_reordenadas_y_extra.csv"] = reordered[
        list(reversed(reordered.columns.tolist()))
    ]

    repeated = pd.concat(
        [
            training.sample(200, random_state=seed + 20),
            _stable(spec, 200, seed + 21, "U10"),
        ],
        ignore_index=True,
    )
    files["uso_10_filas_ya_vistas.csv"] = repeated

    # Lotes que deben fallar o advertir con un mensaje claro.
    files["falla_01_columna_faltante.csv"] = _stable(spec, 300, seed + 30, "F01").drop(
        columns=[spec.missing_column]
    )
    files["falla_02_columna_renombrada.csv"] = _stable(spec, 300, seed + 31, "F02").rename(
        columns={spec.renamed_column: spec.renamed_to}
    )
    unknown = _stable(spec, 300, seed + 32, "F03")
    unknown[spec.target] = unknown[spec.target].astype(object)
    unknown.loc[unknown.index[:40], spec.target] = "pendiente"
    unknown.loc[unknown.index[40:60], spec.target] = "Sí"
    files["falla_03_resultado_desconocido.csv"] = unknown
    files["falla_04_punto_y_coma_decimal_coma.csv"] = (
        _stable(spec, 300, seed + 33, "F04")
        .to_csv(index=False, sep=";", decimal=",", lineterminator="\n")
        .encode("utf-8")
    )
    files["falla_05_codificacion_latin1.csv"] = (
        _stable(spec, 300, seed + 34, "F05")
        .to_csv(index=False, lineterminator="\n")
        .encode("latin-1", errors="replace")
    )
    text_in_number = _stable(spec, 300, seed + 35, "F06")
    text_in_number[spec.shift_column] = text_in_number[spec.shift_column].astype(object)
    text_in_number.loc[text_in_number.index[::25], spec.shift_column] = "n/d"
    files["falla_06_texto_en_columna_numerica.csv"] = text_in_number
    files["falla_07_solo_encabezado.csv"] = training.head(0)

    # Lote reciente para reentrenar: población y comportamiento nuevos, con resultado.
    files["reentrenamiento_reciente.csv"] = _behavior_shift(
        _population_shift(spec, 6000, seed + 40, 1.6, "R"), spec, seed + 41, 0.4
    )

    # Auxiliares para cruzar varios CSV con la tabla base del dominio.
    ids = training[spec.id_column]
    auxiliary = pd.DataFrame({spec.id_column: ids, **spec.auxiliary_columns(training, rng)})
    name = spec.auxiliary_name
    files[f"auxiliar_{name}_ok.csv"] = auxiliary
    files[f"auxiliar_{name}_claves_duplicadas.csv"] = pd.concat(
        [auxiliary, auxiliary.sample(150, random_state=seed + 50)], ignore_index=True
    )
    files[f"auxiliar_{name}_claves_faltantes.csv"] = auxiliary.sample(
        frac=0.7, random_state=seed + 51
    ).sort_index()
    other_format = auxiliary.copy()
    other_format[spec.id_column] = other_format[spec.id_column].str.lower().str.replace("-", "_")
    files[f"auxiliar_{name}_claves_otro_formato.csv"] = other_format
    clash = auxiliary.copy()
    clash[spec.target] = rng.choice([YES, NO], len(clash))
    clash[spec.missing_column] = rng.normal(0, 1, len(clash)).round(2)
    files[f"auxiliar_{name}_columnas_repetidas.csv"] = clash

    # Auxiliar que comparte claves con entrenamiento_grande.csv (generador propio para no
    # alterar los archivos anteriores).
    large = files["entrenamiento_grande.csv"]
    large_rng = np.random.default_rng(seed + 60)
    files[f"auxiliar_{name}_para_entrenamiento_grande.csv"] = pd.DataFrame(
        {spec.id_column: large[spec.id_column], **spec.auxiliary_columns(large, large_rng)}
    )
    return files


def write_variants(output: Path) -> list[Path]:
    written: list[Path] = []
    for index, spec in enumerate(DOMAINS):
        folder = output / spec.folder
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in domain_variants(spec, index).items():
            path = folder / name
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                content.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic" / "variantes")
    args = parser.parse_args()
    written = write_variants(args.output)
    for path in written:
        print(path.relative_to(args.output).as_posix())
    print(f"{len(written)} archivos")


if __name__ == "__main__":
    main()
