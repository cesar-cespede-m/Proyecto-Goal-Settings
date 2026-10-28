"""MON-04: variantes sintéticas reproducibles y su comportamiento esperado al cargar y usar."""

import importlib.util
import sys
from pathlib import Path

import pytest

from core.data_engine import validate_and_profile
from core.data_engine.errors import DataValidationError
from core.ml_engine import RunBlockedError
from tests.support.synthetic_model import VARIANTS, approved_maintenance_model

ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "generate_synthetic_variants", ROOT / "scripts" / "generate_synthetic_variants.py"
)
generator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = generator
_SPEC.loader.exec_module(generator)

REJECTED_AT_UPLOAD = {
    "falla_04_punto_y_coma_decimal_coma.csv": "AMBIGUOUS_DELIMITER",
    "falla_05_codificacion_latin1.csv": "INVALID_ENCODING",
    "falla_07_solo_encabezado.csv": "NO_DATA_ROWS",
}
BLOCKED_AT_SCORING = {
    "falla_01_columna_faltante.csv": "MISSING_FEATURE_COLUMNS",
    "falla_02_columna_renombrada.csv": "MISSING_FEATURE_COLUMNS",
    "falla_03_resultado_desconocido.csv": "UNKNOWN_OUTCOME_VALUES",
    "falla_06_texto_en_columna_numerica.csv": "NON_NUMERIC_FEATURE_VALUES",
}


def test_versioned_variants_match_the_seeded_generator(tmp_path: Path) -> None:
    written = generator.write_variants(tmp_path)

    versioned = sorted(path.relative_to(VARIANTS) for path in VARIANTS.rglob("*.csv"))
    assert sorted(path.relative_to(tmp_path) for path in written) == versioned
    for relative in versioned:
        assert (tmp_path / relative).read_bytes() == (VARIANTS / relative).read_bytes(), relative


@pytest.mark.parametrize("domain", [spec.folder for spec in generator.DOMAINS])
def test_upload_rejects_only_the_malformed_files(domain: str) -> None:
    for path in sorted((VARIANTS / domain).glob("*.csv")):
        if path.name in REJECTED_AT_UPLOAD:
            with pytest.raises(DataValidationError) as captured:
                validate_and_profile(path.name, path.read_bytes())
            assert captured.value.code == REJECTED_AT_UPLOAD[path.name]
        else:
            assert validate_and_profile(path.name, path.read_bytes()).row_count > 0, path.name


def test_maintenance_batches_behave_as_documented(tmp_path: Path) -> None:
    _, scoring, run_id = approved_maintenance_model(tmp_path)
    folder = VARIANTS / "mantenimiento_maquinas"

    def score(name: str):
        content = (folder / name).read_bytes()
        return scoring.score(run_id, filename=name, content=content, actor_label="Prueba")

    for name, code in BLOCKED_AT_SCORING.items():
        with pytest.raises(RunBlockedError) as captured:
            score(name)
        assert captured.value.code == code, name

    stable = score("uso_01_estable.csv").follow_up
    strong = score("uso_03_cambio_fuerte.csv").follow_up
    assert strong.recall < stable.recall - 0.1
    assert strong.precision < stable.precision

    assert score("uso_04_sin_resultado.csv").follow_up is None
    assert score("uso_05_pocos_casos.csv").rows_scored == 20
    assert score("uso_06_categoria_nueva.csv").rows_scored == 1000
    assert score("uso_07_muchos_vacios.csv").rows_scored == 1000
    # Sin monitoreo, un cambio de unidades pasa inadvertido: todo queda en alerta.
    units = score("uso_08_unidades_cambiadas.csv")
    assert units.alerts == units.rows_scored
    reordered = score("uso_09_columnas_reordenadas_y_extra.csv")
    assert "observacion_operador" in reordered.ignored_columns
    repeated = score("uso_10_filas_ya_vistas.csv")
    assert repeated.seen_in_training == 200
    assert repeated.follow_up.rows_with_outcome == 200


@pytest.mark.parametrize("spec", generator.DOMAINS, ids=lambda spec: spec.folder)
def test_auxiliary_files_reveal_join_problems(spec) -> None:
    import hashlib

    from core.data_engine.composition import CSVSourceInput, compose_csv_datasets
    from core.data_engine.errors import DatasetCompositionError

    base = (ROOT / "data" / "synthetic" / spec.base_file).read_bytes()

    def compose(kind: str):
        content = (
            VARIANTS / spec.folder / f"auxiliar_{spec.auxiliary_name}_{kind}.csv"
        ).read_bytes()
        return compose_csv_datasets(
            {
                "primary": {
                    "dataset_id": "p",
                    "dataset_sha256": hashlib.sha256(base).hexdigest(),
                    "classification": "synthetic",
                },
                "joins": [
                    {
                        "source": {
                            "dataset_id": "a",
                            "dataset_sha256": hashlib.sha256(content).hexdigest(),
                            "classification": "synthetic",
                        },
                        "alias": "aux",
                        "primary_key": spec.id_column,
                        "auxiliary_key": spec.id_column,
                    }
                ],
            },
            {"p": CSVSourceInput("p.csv", base), "a": CSVSourceInput("a.csv", content)},
        )

    assert compose("ok").summary.joins[0].unmatched_primary_rows == 0
    assert compose("claves_faltantes").summary.joins[0].matched_primary_rows > 0
    for kind, code in (
        ("claves_duplicadas", "AUXILIARY_KEY_NOT_UNIQUE"),
        ("claves_otro_formato", "JOIN_KEYS_FORMAT_MISMATCH"),
    ):
        with pytest.raises(DatasetCompositionError) as captured:
            compose(kind)
        assert captured.value.code == code
    warnings = compose("columnas_repetidas").profile.warnings
    repeated = next(w for w in warnings if w.startswith("AUXILIARY_REPEATS_PRIMARY_COLUMNS:"))
    assert f"aux__{spec.target}" in repeated.split(":", 1)[1].split(",")


def test_unrelated_files_are_told_apart_from_badly_written_keys() -> None:
    import hashlib

    from core.data_engine.composition import CSVSourceInput, compose_csv_datasets
    from core.data_engine.errors import DatasetCompositionError

    primary = (VARIANTS / "mantenimiento_maquinas" / "entrenamiento_grande.csv").read_bytes()
    auxiliary = (ROOT / "data" / "synthetic" / "mantenimiento_maquinas.csv").read_bytes()

    def ref(name: str, content: bytes) -> dict:
        return {
            "dataset_id": name,
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "classification": "synthetic",
        }

    with pytest.raises(DatasetCompositionError) as captured:
        compose_csv_datasets(
            {
                "primary": ref("p", primary),
                "joins": [
                    {
                        "source": ref("a", auxiliary),
                        "alias": "aux1",
                        "primary_key": "id_maquina",
                        "auxiliary_key": "id_maquina",
                    }
                ],
            },
            {"p": CSVSourceInput("p.csv", primary), "a": CSVSourceInput("a.csv", auxiliary)},
        )
    assert captured.value.code == "JOIN_HAS_NO_MATCHES"


@pytest.mark.parametrize("spec", generator.DOMAINS, ids=lambda spec: spec.folder)
def test_large_training_file_has_its_own_matching_auxiliary(spec) -> None:
    import hashlib

    from core.data_engine.composition import CSVSourceInput, compose_csv_datasets

    folder = VARIANTS / spec.folder
    primary = (folder / "entrenamiento_grande.csv").read_bytes()
    auxiliary = (
        folder / f"auxiliar_{spec.auxiliary_name}_para_entrenamiento_grande.csv"
    ).read_bytes()

    def ref(name: str, content: bytes) -> dict:
        return {
            "dataset_id": name,
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "classification": "synthetic",
        }

    result = compose_csv_datasets(
        {
            "primary": ref("p", primary),
            "joins": [
                {
                    "source": ref("a", auxiliary),
                    "alias": "aux1",
                    "primary_key": spec.id_column,
                    "auxiliary_key": spec.id_column,
                }
            ],
        },
        {"p": CSVSourceInput("p.csv", primary), "a": CSVSourceInput("a.csv", auxiliary)},
    )
    assert result.summary.joins[0].matched_primary_rows == 15000
