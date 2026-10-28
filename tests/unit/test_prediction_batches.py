"""MV-11: lote y hechos con procedencia sintética, nunca impacto comercial real."""

from datetime import UTC, datetime

import pytest

from core.domain.canonical import canonical_hash
from core.domain.prediction_batches import PredictionBatchError, PredictionBatchService
from core.domain.value_policy import ValuePolicyService
from core.ml_engine.scoring import ScoringService
from core.reporting import ReportService
from tests.unit.test_value_policy import _economy, _trained

pytest_plugins = ["tests.unit.test_modeling_agent"]


def _registered(state):
    item, _strategy_id, run_id = _trained(state)
    economic = _economy(state, item)
    policies = ValuePolicyService(state.workflow)
    policy = policies.propose(
        item.cap.id,
        cycle_id=item.cycle.id,
        run_id=run_id,
        economic_version_id=economic.id,
        actor_label="pytest",
        operation_id="policy",
    )
    policies.review(
        policy.id,
        approved_hash=policy.sha256,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        operation_id="policy-review",
    )
    report_id = ReportService(state.workflow).create_template_report(run_id)
    state.workflow.decide_publication(
        report_id=report_id, decision="APPROVED", actor_label="pytest-reviewer"
    )
    frame = item.frame.head(40).copy()
    frame["activity_count"] = frame["activity_count"] + 10
    frame["entity"] = [f"NEW-SYNTH-{index}" for index in range(len(frame))]
    frame = frame.drop(columns=[item.context_payload["target_column"]])
    scored = ScoringService(state.workflow, state.storage).score(
        run_id,
        filename="sandbox-new.csv",
        content=frame.to_csv(index=False).encode(),
        actor_label="pytest",
    )
    event_id = str(state.workflow.list_scorings(run_id)[0]["id"])
    service = PredictionBatchService(state.workflow, state.storage)
    batch = service.register(
        item.cap.id,
        policy_version_id=policy.id,
        scoring=scored,
        scoring_event_id=event_id,
        period="2026-10",
        data_classification="synthetic",
        provenance_basis="Fixture de prueba",
        actor_label="pytest",
        operation_id="batch",
    )
    return item, service, batch, scored, event_id


def test_batch_tracks_assigned_acted_pending_and_observed_separately(state):
    item, service, batch, scored, event_id = _registered(state)
    assert batch.run_id == scored.run_id
    assert batch.scoring_event_id == event_id
    assert len(batch.cases) == scored.rows_scored
    assigned = next(case for case in batch.cases if case.assigned)
    unassigned = next(case for case in batch.cases if not case.assigned)
    before = service.summary(batch.id)
    assert before.assigned > 0 and before.acted == 0 and before.pending == before.cases
    now = datetime.now(UTC)
    with pytest.raises(PredictionBatchError, match="asignación"):
        service.record(
            batch.id,
            case_id=unassigned.case_id,
            event_type="ACTED",
            occurred_at=now,
            actor_label="pytest",
            operation_id="unassigned-action",
        )
    action = service.record(
        batch.id,
        case_id=assigned.case_id,
        event_type="ACTED",
        occurred_at=now,
        actor_label="pytest",
        operation_id="action",
    )
    assert (
        service.record(
            batch.id,
            case_id=assigned.case_id,
            event_type="ACTED",
            occurred_at=now,
            actor_label="pytest",
            operation_id="action",
        )
        == action
    )
    service.record(
        batch.id,
        case_id=assigned.case_id,
        event_type="OUTCOME_PENDING",
        occurred_at=now,
        actor_label="pytest",
        operation_id="pending",
    )
    assert service.summary(batch.id).pending == before.cases
    service.record(
        batch.id,
        case_id=assigned.case_id,
        event_type="OUTCOME_OBSERVED",
        outcome_positive=True,
        occurred_at=now,
        actor_label="pytest",
        operation_id="observed",
    )
    after = service.summary(batch.id)
    assert after.acted == 1 and after.assigned_not_acted == after.assigned - 1
    assert after.observed == 1 and after.pending == before.cases - 1
    assert after.positive_observed == 1
    with pytest.raises(PredictionBatchError, match="final"):
        service.record(
            batch.id,
            case_id=assigned.case_id,
            event_type="OUTCOME_PENDING",
            occurred_at=now,
            actor_label="pytest",
            operation_id="stale-pending",
        )
    audit = state.connection.execute(
        "SELECT detail_json FROM audit_events WHERE entity_id=?", (batch.id,)
    ).fetchall()
    assert audit and all(assigned.case_id not in row["detail_json"] for row in audit)
    assert len(service.list_batches(item.cap.id)) == 1


def test_registration_rejects_mutated_scores_and_operation_conflicts(state):
    item, service, batch, scored, event_id = _registered(state)
    with pytest.raises(PredictionBatchError, match="productivos reales"):
        service.register(
            item.cap.id,
            policy_version_id=batch.policy_version_id,
            scoring=scored,
            scoring_event_id=event_id,
            period="2026-10",
            data_classification="production",
            provenance_basis="No permitido",
            actor_label="pytest",
            operation_id="production-blocked",
        )
    assert (
        service.register(
            item.cap.id,
            policy_version_id=batch.policy_version_id,
            scoring=scored,
            scoring_event_id=event_id,
            period="2026-10",
            data_classification="synthetic",
            provenance_basis="Fixture de prueba",
            actor_label="pytest",
            operation_id="batch",
        )
        == batch
    )
    changed = scored.model_copy(deep=True)
    changed.cases[0].score = 1 - changed.cases[0].score
    assert canonical_hash(changed) != canonical_hash(scored)
    with pytest.raises(PredictionBatchError, match="otro efecto"):
        service.register(
            item.cap.id,
            policy_version_id=batch.policy_version_id,
            scoring=changed,
            scoring_event_id=event_id,
            period="2026-10",
            data_classification="synthetic",
            provenance_basis="Fixture de prueba",
            actor_label="pytest",
            operation_id="batch",
        )
    with pytest.raises(PredictionBatchError, match="no coincide"):
        service.register(
            item.cap.id,
            policy_version_id=batch.policy_version_id,
            scoring=changed,
            scoring_event_id=event_id,
            period="2026-10",
            data_classification="synthetic",
            provenance_basis="Fixture de prueba",
            actor_label="pytest",
            operation_id="tampered",
        )
    current_economy = state.values.list_economic_versions(item.cap.id)[-1]
    _economy(
        state,
        item,
        operation="new-cost",
        base=current_economy.id,
        action_cost_per_case="300",
    )
    with pytest.raises(PredictionBatchError, match="política vigente"):
        service.register(
            item.cap.id,
            policy_version_id=batch.policy_version_id,
            scoring=scored,
            scoring_event_id=event_id,
            period="2026-11",
            data_classification="synthetic",
            provenance_basis="Fixture de prueba",
            actor_label="pytest",
            operation_id="stale-policy",
        )
