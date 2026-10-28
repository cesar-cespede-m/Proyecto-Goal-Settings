"""MON-03: reentrenamiento con aprobación humana y comparación con el modelo anterior."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.domain.retraining_service import RetrainingService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService, RunBlockedError
from core.reporting import ReportService
from tests.support.synthetic_model import VARIANTS, approved_maintenance_model

FOLDER = VARIANTS / "mantenimiento_maquinas"


def _start(repository, tmp_path: Path, run_id: str, **overrides):
    arguments = {
        "source_run_id": run_id,
        "filename": "reentrenamiento_reciente.csv",
        "content": (FOLDER / "reentrenamiento_reciente.csv").read_bytes(),
        "actor_label": "Responsable de prueba",
        "data_authorized": True,
        **overrides,
    }
    return RetrainingService(repository, tmp_path / "runtime").start(**arguments)


def _train_and_approve(repository, tmp_path: Path, started) -> str:
    strategies = StrategyService(repository)
    template = strategies.build_template(
        project_id=started.project_id, context_version_id=started.context_version_id
    ).model_dump(mode="json")
    template["model_params"]["random_forest"]["n_estimators"] = 30
    strategy = strategies.create_version(
        project_id=started.project_id, proposal=StrategyProposalV1.model_validate(template)
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="Responsable de prueba",
    )
    run = MLRunService(repository, tmp_path / "runtime").execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(run.run_id)
    repository.decide_publication(
        report_id=report_id, decision="APPROVED", actor_label="Responsable de prueba"
    )
    return run.run_id


def test_retraining_needs_a_person_and_authorized_data(tmp_path: Path) -> None:
    repository, _, run_id = approved_maintenance_model(tmp_path)

    with pytest.raises(RunBlockedError) as missing_actor:
        _start(repository, tmp_path, run_id, actor_label="pendiente")
    with pytest.raises(RunBlockedError) as not_authorized:
        _start(repository, tmp_path, run_id, data_authorized=False)

    assert missing_actor.value.code == "ACTOR_REQUIRED"
    assert not_authorized.value.code == "DATA_AUTHORIZATION_REQUIRED"
    assert len(repository.list_projects()) == 1


def test_retraining_creates_a_linked_project_and_keeps_the_current_model(
    tmp_path: Path,
) -> None:
    repository, scoring, run_id = approved_maintenance_model(tmp_path)
    started = _start(repository, tmp_path, run_id)

    names = [row["name"] for row in repository.list_projects()]
    assert "Mantenimiento sintético · reentrenamiento 1" in names
    context = json.loads(repository.get_context(started.context_version_id)["context_json"])
    assert context["context"]["decision_actor_label"] == "Responsable de prueba"
    assert context["target_column"] == "falla_maquina"
    source = RetrainingService(repository, tmp_path / "runtime").source_of(started.project_id)
    assert source["source_run_id"] == run_id
    # El modelo anterior sigue vigente: nada se entrenó ni se aprobó por la persona.
    usable = scoring.evaluate(
        run_id, filename="x.csv", content=(FOLDER / "uso_01_estable.csv").read_bytes()
    )
    assert usable.rows_scored == 1000
    assert repository.get_latest_strategy(started.context_version_id) is None


def test_comparison_uses_the_same_cases_and_is_not_a_model_use(tmp_path: Path) -> None:
    repository, scoring, run_id = approved_maintenance_model(tmp_path)
    started = _start(repository, tmp_path, run_id)
    candidate_run_id = _train_and_approve(repository, tmp_path, started)

    current, candidate = scoring.compare(
        current_run_id=run_id,
        candidate_run_id=candidate_run_id,
        filename="uso_03_cambio_fuerte.csv",
        content=(FOLDER / "uso_03_cambio_fuerte.csv").read_bytes(),
        actor_label="Responsable de prueba",
    )

    assert current.input_sha256 == candidate.input_sha256
    assert candidate.follow_up.recall > current.follow_up.recall + 0.2
    assert candidate.follow_up.precision > current.follow_up.precision
    assert repository.list_scorings(run_id) == []
    assert repository.list_scorings(candidate_run_id) == []
    event = repository.connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'MODEL_COMPARISON'"
    ).fetchone()
    detail = json.loads(event["detail_json"])
    assert detail["current_run_id"] == run_id
    assert "MAQ-" not in event["detail_json"]

    with pytest.raises(RunBlockedError) as no_outcome:
        scoring.compare(
            current_run_id=run_id,
            candidate_run_id=candidate_run_id,
            filename="uso_04_sin_resultado.csv",
            content=(FOLDER / "uso_04_sin_resultado.csv").read_bytes(),
            actor_label="Responsable de prueba",
        )
    assert no_outcome.value.code == "COMPARISON_NEEDS_OUTCOMES"


def test_comparison_view_states_the_verdict_without_deciding() -> None:
    def render() -> None:
        from app.ui.extension_views import render_comparison_result
        from core.ml_engine.scoring import FollowUpMetrics, ScoringResultV1

        def result(precision: float, recall: float) -> ScoringResultV1:
            return ScoringResultV1(
                run_id="r",
                report_id="rep",
                selected_model="RandomForestClassifier",
                positive_class="sí",
                threshold=0.5,
                id_column=None,
                rows_scored=1500,
                alerts=800,
                input_sha256="0" * 64,
                cases=[],
                ignored_columns=[],
                target_present=True,
                follow_up=FollowUpMetrics(
                    rows_with_outcome=1500, prevalence=0.3, precision=precision, recall=recall
                ),
            )

        render_comparison_result(result(0.24, 0.43), result(0.52, 0.89))

    app = AppTest.from_function(render).run(timeout=30)

    assert not app.exception
    table = app.dataframe[0].value
    assert table.iloc[0]["Modelo anterior"] == "43,0 %"
    assert table.iloc[0]["Modelo nuevo"] == "89,0 %"
    assert "el modelo nuevo detecta más" in app.info[0].value
    assert "apruébalo y publícalo" in app.info[0].value
