"""MV-12: candidata de otra cohorte sintética, con gate nuevo de entrenamiento."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from core.domain.capability_learning import CapabilityLearningError, CapabilityLearningService
from core.domain.errors import ApprovalRequiredError
from core.ml_engine.scoring import ScoringService
from tests.unit.test_modeling_agent import COVERAGE, _propose
from tests.unit.test_prediction_batches import _registered

pytest_plugins = ["tests.unit.test_modeling_agent"]


def _fresh_events(*, overlap: str | None = None, reverse: bool = False) -> bytes:
    rows = []
    for number in range(120):
        first = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=number * 2)
        entity = overlap if number == 0 and overlap else f"UPDATE-NEW-{number:03d}"
        rows.append((entity, first.isoformat(), "start"))
        for _ in range(1 if number % 2 else 3):
            rows.append((entity, (first + timedelta(days=1)).isoformat(), "activity"))
        if (number % 2 == 0) != reverse:
            rows.append((entity, (first + timedelta(days=5)).isoformat(), "purchase"))
    return pd.DataFrame(rows, columns=["entity", "moment", "stage"]).to_csv(index=False).encode()


def test_update_candidate_is_isolated_and_needs_its_own_training_approval(state):
    item, _batch_service, batch, scored, _event = _registered(state)
    learning = CapabilityLearningService(state.workflow, state.storage)
    content = _fresh_events()
    started = learning.start_from_snapshot(
        item.cap.id,
        source_run_id=scored.run_id,
        filename="update-events.csv",
        content=content,
        coverage=COVERAGE,
        reason="Cohorte sintética fresca para evaluar actualización",
        reference_unintervened=True,
        reference_basis="Fixture sin acciones",
        actor_label="pytest",
        data_authorized=True,
        operation_id="candidate",
    )
    assert started.source_cycle_id == item.cycle.id
    assert started.cycle_id != item.cycle.id
    assert started.project_id != item.cycle.project_id
    assert (
        started.predictive_version_id
        == state.values.get_cycle_binding(item.cycle.id).predictive_version_id
    )
    assert (
        learning.start_from_snapshot(
            item.cap.id,
            source_run_id=scored.run_id,
            filename="update-events.csv",
            content=content,
            coverage=COVERAGE,
            reason="Cohorte sintética fresca para evaluar actualización",
            reference_unintervened=True,
            reference_basis="Fixture sin acciones",
            actor_label="pytest",
            data_authorized=True,
            operation_id="candidate",
        )
        == started
    )
    ScoringService(state.workflow, state.storage).approved_run(scored.run_id)
    candidate = SimpleNamespace(
        cap=item.cap,
        cycle=state.caps.get_cycle(started.cycle_id),
        dataset_id=started.dataset_id,
    )
    _supervisor, _task, strategy_id = _propose(state, candidate, name="candidate-task")
    with pytest.raises(ApprovalRequiredError):
        state.service.execute(strategy_id)
    assert (
        state.connection.execute(
            "SELECT COUNT(*) FROM runs WHERE strategy_version_id=?", (strategy_id,)
        ).fetchone()[0]
        == 0
    )
    assert batch.policy_version_id


def test_update_rejects_overlap_with_scored_intervention_cohort(state):
    item, _batch_service, batch, scored, _event = _registered(state)
    learning = CapabilityLearningService(state.workflow, state.storage)
    before = len(state.caps.list_cycles(item.cap.id))
    with pytest.raises(CapabilityLearningError, match="referencia sin intervención"):
        learning.start_from_snapshot(
            item.cap.id,
            source_run_id=scored.run_id,
            filename="unverified.csv",
            content=_fresh_events(),
            coverage=COVERAGE,
            reason="Sin procedencia",
            reference_unintervened=False,
            reference_basis="No demostrada",
            actor_label="pytest",
            data_authorized=True,
            operation_id="unverified",
        )
    with pytest.raises(CapabilityLearningError, match="cohorte separada"):
        learning.start_from_snapshot(
            item.cap.id,
            source_run_id=scored.run_id,
            filename="overlap.csv",
            content=_fresh_events(overlap=batch.cases[0].case_id),
            coverage=COVERAGE,
            reason="Otra cohorte",
            reference_unintervened=True,
            reference_basis="Fixture sin acciones",
            actor_label="pytest",
            data_authorized=True,
            operation_id="overlap",
        )
    assert len(state.caps.list_cycles(item.cap.id)) == before
