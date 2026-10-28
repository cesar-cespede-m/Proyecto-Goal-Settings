"""EXT-A: usar un modelo aprobado con casos nuevos, sólo con aprobación vigente."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from core.domain.strategy_service import StrategyService
from core.ml_engine.execution import RunBlockedError
from core.ml_engine.scoring import ScoringService
from core.reporting import ReportService
from tests.support.ml_workflow import csv_bytes, prepared_strategy

ACTOR = "pytest-usuario"


@pytest.fixture
def trained(tmp_path: Path):
    connection, repository, ml, strategy_id, storage = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label=ACTOR
    )
    result = ml.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(result.run_id)
    yield connection, repository, ScoringService(repository, storage), result, report_id
    connection.close()


def _new_cases(rows: int = 60, *, drop: tuple[str, ...] = ("target",)) -> bytes:
    frame = pd.read_csv(io.BytesIO(csv_bytes(rows)))
    frame["fixture_id"] = [f"nuevo-{index}" for index in range(rows)]
    return frame.drop(columns=list(drop)).to_csv(index=False).encode("utf-8")


def _approve(repository, report_id: str) -> None:
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label=ACTOR)


def test_model_cannot_be_used_before_its_report_is_approved(trained) -> None:
    _, _, scoring, result, _ = trained
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(result.run_id, filename="nuevos.csv", content=_new_cases(), actor_label=ACTOR)
    assert captured.value.code == "MODEL_NOT_APPROVED_FOR_USE"


def test_approved_model_ranks_new_cases_and_audits_without_rows(trained) -> None:
    connection, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    scored = scoring.score(
        result.run_id, filename="nuevos.csv", content=_new_cases(), actor_label=ACTOR
    )

    assert scored.rows_scored == 60 and scored.id_column == "fixture_id"
    scores = [case.score for case in scored.cases]
    assert scores == sorted(scores, reverse=True)
    assert [case.rank for case in scored.cases] == list(range(1, 61))
    assert scored.alerts == sum(score >= scored.threshold for score in scores)
    assert not scored.target_present
    exported = pd.read_csv(io.BytesIO(scored.to_csv()))
    assert list(exported.columns) == [
        "prioridad",
        "fila_original",
        "fixture_id",
        "puntaje",
        "alerta",
        "por_que",
    ]

    events = repository.list_scorings(result.run_id)
    assert len(events) == 1
    detail = json.loads(events[0]["detail_json"])
    assert detail["rows_scored"] == 60 and detail["actor_label"] == ACTOR
    assert "nuevo-0" not in events[0]["detail_json"]


def test_missing_training_columns_are_rejected(trained) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    frame = pd.read_csv(io.BytesIO(_new_cases()))
    frame = frame.rename(columns={"signal": "signals"})
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(
            result.run_id,
            filename="nuevos.csv",
            content=frame.to_csv(index=False).encode("utf-8"),
            actor_label=ACTOR,
        )
    assert captured.value.code == "MISSING_FEATURE_COLUMNS"
    assert captured.value.missing_columns == ("signal",)
    assert "signals" in captured.value.available_columns


def test_tampered_model_file_is_rejected(trained, tmp_path: Path) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    artifact = scoring._artifact(result.artifact_path)
    artifact.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(result.run_id, filename="n.csv", content=_new_cases(), actor_label=ACTOR)
    assert captured.value.code == "ARTIFACT_INTEGRITY_FAILED"


def test_superseded_model_cannot_be_used(trained) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    strategy = repository.get_strategy(repository.get_run(result.run_id)["strategy_version_id"])
    edited = json.loads(strategy["strategy_json"])
    edited["seed"] += 1
    project_id = repository.get_context(strategy["context_version_id"])["project_id"]
    StrategyService(repository).create_version(project_id=project_id, proposal=edited)
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(result.run_id, filename="n.csv", content=_new_cases(), actor_label=ACTOR)
    assert captured.value.code == "MODEL_NOT_CURRENT"


def test_new_cases_with_known_outcome_report_follow_up_metrics(trained) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    scored = scoring.score(
        result.run_id, filename="con-resultado.csv", content=_new_cases(drop=()), actor_label=ACTOR
    )
    assert scored.target_present
    follow_up = scored.follow_up
    assert follow_up is not None and follow_up.rows_with_outcome == 60
    assert 0 <= follow_up.precision <= 1 and 0 <= follow_up.recall <= 1
    assert follow_up.training_average_precision is not None


def test_unknown_outcome_values_are_rejected_not_counted_as_negative(trained) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    frame = pd.read_csv(io.BytesIO(_new_cases(drop=())))
    frame["target"] = frame["target"].astype(object)
    frame.loc[0, "target"] = "desconocido"
    content = frame.to_csv(index=False).encode("utf-8")
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(result.run_id, filename="n.csv", content=content, actor_label=ACTOR)
    assert captured.value.code == "UNKNOWN_OUTCOME_VALUES"
    assert repository.list_scorings(result.run_id) == []


@pytest.mark.parametrize("actor", ["", "   "])
def test_scoring_requires_a_declared_actor(trained, actor: str) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    with pytest.raises(RunBlockedError) as captured:
        scoring.score(result.run_id, filename="n.csv", content=_new_cases(), actor_label=actor)
    assert captured.value.code == "ACTOR_REQUIRED"


def test_rows_from_the_training_file_are_flagged_and_left_out_of_follow_up(trained) -> None:
    """UX-04: reutilizar filas del entrenamiento no puede mostrar un 100 % engañoso."""
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    training_rows = pd.read_csv(io.BytesIO(csv_bytes())).head(30)
    content = training_rows.to_csv(index=False).encode("utf-8")

    scored = scoring.score(result.run_id, filename="viejos.csv", content=content, actor_label=ACTOR)

    assert scored.seen_in_training == 30
    assert all(case.seen_in_training for case in scored.cases)
    assert scored.follow_up is None
    detail = json.loads(repository.list_scorings(result.run_id)[0]["detail_json"])
    assert detail["seen_in_training"] == 30


def test_mixed_file_measures_follow_up_only_on_unseen_rows(trained) -> None:
    _, repository, scoring, result, report_id = trained
    _approve(repository, report_id)
    old = pd.read_csv(io.BytesIO(csv_bytes())).head(10)
    new = pd.read_csv(io.BytesIO(_new_cases(drop=())))
    content = pd.concat([old, new]).to_csv(index=False).encode("utf-8")

    scored = scoring.score(result.run_id, filename="mixto.csv", content=content, actor_label=ACTOR)

    assert scored.seen_in_training == 10
    assert scored.follow_up is not None and scored.follow_up.rows_with_outcome == 60
