"""Suite de negocios intercambiables (GA-10): la misma plataforma, sin cambiar código.

Cada negocio declara sus archivos y lo que la plataforma debe poder demostrar con ellos. Las
pruebas de cada capacidad (catálogo, análisis, preguntas, modelos) recorren esta lista; nada
del código de producción puede depender de estos nombres (auditoría GA-14).
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parents[2]
SYNTHETIC = ROOT / "data" / "synthetic"


@dataclass(frozen=True)
class TestDomain:
    key: str
    files: tuple[Path, ...]
    # Qué debe poder hacer la plataforma con este negocio.
    has_outcome: bool  # existe una columna con el resultado a anticipar
    task_kinds: tuple[str, ...]  # tipos de modelo aplicables
    has_datetimes: bool  # alguna columna es fecha y hora
    multi_file: bool  # requiere relacionar archivos
    expected_findings: tuple[str, ...] = field(default_factory=tuple)


DOMAINS: tuple[TestDomain, ...] = (
    TestDomain(
        key="mantenimiento",
        files=(SYNTHETIC / "mantenimiento_maquinas.csv",),
        has_outcome=True,
        task_kinds=("binary_classification", "regression"),
        has_datetimes=False,
        multi_file=False,
        expected_findings=("POST_EVENT_LEAKAGE", "IDENTIFIER", "CONSTANT"),
    ),
    TestDomain(
        key="suscripciones",
        files=(SYNTHETIC / "abandono_suscripcion.csv",),
        has_outcome=True,
        task_kinds=("binary_classification", "multiclass_classification"),
        has_datetimes=False,
        multi_file=False,
        expected_findings=("POST_EVENT_LEAKAGE", "IDENTIFIER", "CONSTANT"),
    ),
    TestDomain(
        key="entregas",
        files=(SYNTHETIC / "retraso_entregas.csv",),
        has_outcome=True,
        task_kinds=("binary_classification", "regression"),
        has_datetimes=False,
        multi_file=False,
        expected_findings=("POST_EVENT_LEAKAGE", "IDENTIFIER", "CONSTANT"),
    ),
    TestDomain(
        key="reservas",
        files=(SYNTHETIC / "reservas_restaurante.csv",),
        has_outcome=True,
        task_kinds=("binary_classification",),
        has_datetimes=False,
        multi_file=False,
        expected_findings=("POST_EVENT_LEAKAGE", "IDENTIFIER", "CONSTANT"),
    ),
    TestDomain(
        key="varios_archivos_con_fechas",
        files=tuple(
            SYNTHETIC / "transporte" / name
            for name in (
                "transporte_estaciones.csv",
                "transporte_transacciones.csv",
                "transporte_buses.csv",
                "transporte_encuestas.csv",
            )
        ),
        has_outcome=True,
        task_kinds=("binary_classification", "multiclass_classification", "regression"),
        has_datetimes=True,
        multi_file=True,
        expected_findings=("TIME_PEAKS", "KEY_MISMATCH", "GROUP_DIFFERENCES"),
    ),
    TestDomain(
        key="caso_responsable",
        files=(
            SYNTHETIC / "caso_responsable" / "datos_bip_simulados.csv",
            SYNTHETIC / "caso_responsable" / "datos_flujos_simulados.csv",
        ),
        has_outcome=False,
        task_kinds=(),
        has_datetimes=True,
        multi_file=True,
        # La plataforma debe decir que no hay resultado ni patrón temporal, en vez de inventar.
        expected_findings=("NO_OUTCOME", "KEY_MISMATCH", "NO_TEMPORAL_VARIATION"),
    ),
    TestDomain(
        key="recorrido_altas",
        files=(
            SYNTHETIC / "recorridos" / "altas_eventos.csv",
            SYNTHETIC / "recorridos" / "altas_entidades.csv",
        ),
        has_outcome=False,
        task_kinds=(),
        has_datetimes=True,
        multi_file=True,
        expected_findings=("NO_OUTCOME",),
    ),
    TestDomain(
        key="recorrido_solicitudes",
        files=(
            SYNTHETIC / "recorridos" / "solicitudes_eventos.csv",
            SYNTHETIC / "recorridos" / "solicitudes_entidades.csv",
        ),
        has_outcome=False,
        task_kinds=(),
        has_datetimes=True,
        multi_file=True,
        expected_findings=("NO_OUTCOME",),
    ),
)
