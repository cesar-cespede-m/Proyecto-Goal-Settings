"""V2-25: ciclo de aprendizaje en un paso; mide y avisa, nunca entrena ni aprueba."""

import importlib.util
from pathlib import Path

from tests.unit.test_recipe_retraining import trained  # noqa: F401

SCRIPT = Path(__file__).parents[2] / "scripts" / "learning_cycle.py"
spec = importlib.util.spec_from_file_location("learning_cycle", SCRIPT)
cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cycle)


def test_cycle_reports_every_project_without_training(trained) -> None:  # noqa: F811
    repository, storage, _ = trained
    runs_before = repository.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    results = cycle.run_cycle(repository, storage, actor="ciclo", classification="synthetic")

    assert [item["modelos_en_uso"] for item in results] == [1]
    assert results[0]["necesita_una_persona"] == []
    assert repository.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == runs_before
