"""V2-16: razones por caso coherentes con el modelo y en palabras."""

import numpy as np
import pandas as pd

from core.ml_engine.explain import case_reasons, reasons_text, typical_values


def _score(frame: pd.DataFrame) -> np.ndarray:
    # Modelo de juguete: sólo «a» importa, y «grupo» = x suma un poco.
    return np.clip(0.1 + 0.08 * frame["a"].astype(float) + (frame["grupo"] == "x") * 0.1, 0, 1)


def test_reasons_point_to_the_columns_that_move_the_score() -> None:
    reference = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0] * 5, "grupo": list("yyyyx")})
    cases = pd.DataFrame({"a": [9, 3], "b": [7, 0], "grupo": ["x", "y"]})
    reasons = case_reasons(
        _score, cases, ["a", "b", "grupo"], typical_values(reference, ["a", "b", "grupo"])
    )

    first = reasons[0]
    assert first[0].column == "a" and first[0].direction == "sube"
    assert "b" not in [item.column for item in first]  # no mueve el puntaje
    assert reasons[1] == []  # caso típico
    assert "lo sube" in reasons_text(first)
    assert "cerca del típico" in reasons_text(reasons[1])


def test_prioritized_cases_carry_their_reasons(tmp_path) -> None:
    import io

    from core.ml_engine.scoring import ScoringService
    from core.reporting.service import ReportService
    from tests.support.ml_workflow import prepared_strategy
    from tests.unit.test_scoring import _approve, _new_cases  # noqa: F401

    connection, repository, service, strategy_id, storage = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="pytest"
    )
    run = service.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="pytest")
    result = ScoringService(repository, storage).score(
        run.run_id, filename="nuevos.csv", content=_new_cases(), actor_label="pytest"
    )
    connection.close()

    assert result.cases[0].reasons
    exported = pd.read_csv(io.BytesIO(result.to_csv()))
    assert "por_que" in exported.columns
