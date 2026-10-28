"""C2-09: reentrenar por receta desde la terminal; nunca entrena ni publica solo."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from core.ml_engine import RunBlockedError
from tests.unit.test_recipe_retraining import _files, trained  # noqa: F401

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_recipe.py"
spec = importlib.util.spec_from_file_location("run_recipe", SCRIPT)
run_recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_recipe)


def _args(tmp_path: Path, run_id: str, *, authorized: bool = True) -> argparse.Namespace:
    pairs = []
    for name, (filename, content) in _files(9).items():
        path = tmp_path / filename
        path.write_bytes(content)
        pairs.append(f"{name}={path}")
    return argparse.Namespace(run=run_id, file=pairs, actor="sistema-x", authorized=authorized)


def test_cli_rebuilds_the_table_and_waits_for_approval(trained, tmp_path: Path) -> None:  # noqa: F811
    repository, storage, run_id = trained
    output = run_recipe.run(repository, storage, _args(tmp_path, run_id))

    assert repository.get_analysis_table(output["dataset_id"]) is not None
    runs = repository.connection.execute(
        "SELECT COUNT(*) FROM runs r JOIN strategy_versions s ON s.id = r.strategy_version_id "
        "JOIN context_versions c ON c.id = s.context_version_id WHERE c.project_id = ?",
        (output["project_id"],),
    ).fetchone()[0]
    assert runs == 0  # no entrena por su cuenta


def test_cli_requires_explicit_data_authorization(trained, tmp_path: Path) -> None:  # noqa: F811
    repository, storage, run_id = trained
    with pytest.raises(RunBlockedError) as error:
        run_recipe.run(repository, storage, _args(tmp_path, run_id, authorized=False))
    assert error.value.code == "DATA_AUTHORIZATION_REQUIRED"
