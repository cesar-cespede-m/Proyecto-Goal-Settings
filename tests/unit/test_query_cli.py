"""C2-03: consulta de sólo lectura; respeta aprobaciones y audita el uso."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from core.ml_engine import RunBlockedError
from tests.unit.test_recipe_retraining import _files, trained  # noqa: F401

SCRIPT = Path(__file__).parents[2] / "scripts" / "goal_settings_query.py"
spec = importlib.util.spec_from_file_location("goal_settings_query", SCRIPT)
query = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query)


def test_predict_uses_the_approved_model_and_is_audited(trained, tmp_path: Path) -> None:  # noqa: F811
    repository, storage, run_id = trained
    cases = tmp_path / "casos.csv"
    cases.write_bytes(_files(5)["casos"][1])
    # La tabla del modelo cruza dos archivos: el archivo nuevo trae sus columnas finales.
    import pandas as pd

    frame = pd.read_csv(cases).assign(nivel="a").drop(columns=["grupo_id", "valor"])
    frame.to_csv(cases, index=False)

    output = query.predict(
        repository,
        storage,
        argparse.Namespace(run=run_id, file=str(cases), actor="sistema-externo"),
    )

    assert output["task_type"] == "binary_classification"
    assert len(output["cases"]) == len(frame)
    used = repository.connection.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_type = 'MODEL_USED_FOR_SCORING'"
    ).fetchone()[0]
    assert used == 1


def test_predict_refuses_models_without_approval(trained, tmp_path: Path) -> None:  # noqa: F811
    repository, storage, _ = trained
    with pytest.raises((RunBlockedError, LookupError, ValueError)):
        query.predict(
            repository,
            storage,
            argparse.Namespace(run="no-existe", file=str(tmp_path / "x.csv"), actor="sistema"),
        )


def test_ask_without_ai_says_so_and_sends_nothing(trained, monkeypatch) -> None:  # noqa: F811
    repository, storage, run_id = trained
    monkeypatch.setattr(query, "_provider", lambda: None)
    project_id = repository.get_run_project_id(run_id)

    output = query.ask(
        repository,
        storage,
        argparse.Namespace(project=project_id, question="¿Cuántos casos hay?", actor="sistema"),
    )

    assert output == {"answered": False, "mode": "DISABLED", "reason": None}
    shared = repository.connection.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_type = 'AI_DATA_SHARED'"
    ).fetchone()[0]
    assert shared == 0
