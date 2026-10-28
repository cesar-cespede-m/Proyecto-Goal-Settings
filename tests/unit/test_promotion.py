"""MV-13: la candidata se evalúa sin publicar y sólo cambia el vigente con revisión."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from core.domain.canonical import canonical_hash
from core.domain.capability_learning import CapabilityLearningService
from core.domain.promotion import PromotionError, PromotionService
from core.domain.value_policy import ValuePolicyService
from core.ml_engine.execution import RunBlockedError
from core.ml_engine.scoring import ScoringService
from core.reporting import ReportService
from tests.unit.test_capability_learning import _fresh_events
from tests.unit.test_modeling_agent import COVERAGE, _approve, _propose
from tests.unit.test_prediction_batches import _registered

pytest_plugins = ["tests.unit.test_modeling_agent"]


def _comparison_state(state):
    item, _batch_service, batch, scored, _event = _registered(state)
    started = CapabilityLearningService(state.workflow, state.storage).start_from_snapshot(
        item.cap.id,
        source_run_id=scored.run_id,
        filename="reversed-events.csv",
        content=_fresh_events(reverse=True),
        coverage=COVERAGE,
        reason="Probar candidata sobre datos sintéticos nuevos",
        reference_unintervened=True,
        reference_basis="Fixture sin actuaciones",
        actor_label="pytest",
        data_authorized=True,
        operation_id="candidate",
    )
    candidate = SimpleNamespace(
        cap=item.cap,
        cycle=state.caps.get_cycle(started.cycle_id),
        dataset_id=started.dataset_id,
    )
    _supervisor, _task, strategy_id = _propose(state, candidate, name="candidate-task")
    _approve(state, strategy_id)
    run = state.service.execute(strategy_id)
    assert run.status == "COMPLETED"
    report_id = ReportService(state.workflow).create_template_report(run.run_id)
    policies = ValuePolicyService(state.workflow)
    economic_id = policies.get(batch.policy_version_id).content.economic_version_id
    candidate_policy = policies.propose(
        item.cap.id,
        cycle_id=candidate.cycle.id,
        run_id=run.run_id,
        economic_version_id=economic_id,
        based_on_version_id=batch.policy_version_id,
        actor_label="pytest",
        operation_id="candidate-policy",
    )
    policies.review(
        candidate_policy.id,
        approved_hash=candidate_policy.sha256,
        decision="APPROVED",
        actor_label="pytest-reviewer",
        operation_id="candidate-policy-review",
    )
    assert policies.is_review_current(batch.policy_version_id)
    assert policies.is_review_current(candidate_policy.id)
    frame = pd.DataFrame(
        {
            "entity": [f"FUTURE-SEALED-{index:03d}" for index in range(60)],
            "activity_count": [3 if index % 2 == 0 else 1 for index in range(60)],
            "no_first_purchase": [index % 2 == 0 for index in range(60)],
            "label_available_at": ["2026-09-01T00:00:00Z"] * 60,
        }
    )
    content = frame.to_csv(index=False).encode()
    service = PromotionService(state.workflow, state.storage)
    cohort = service.seal(
        item.cap.id,
        current_policy_id=batch.policy_version_id,
        candidate_policy_id=candidate_policy.id,
        filename="future-synthetic.csv",
        content=content,
        data_classification="synthetic",
        provenance_basis="Cohorte sintética reservada",
        coverage_complete=True,
        coverage_basis="Fixture con ventanas maduras",
        observed_until=datetime(2026, 9, 30, tzinfo=UTC),
        min_rows=30,
        min_f1_improvement=0.01,
        actor_label="pytest",
    )
    return item, batch, run.run_id, report_id, candidate_policy, service, cohort, content


def test_candidate_evaluation_does_not_publish_and_promotion_uses_exact_review(state):
    item, batch, candidate_run, report_id, candidate_policy, service, cohort, content = (
        _comparison_state(state)
    )
    with pytest.raises(RunBlockedError):
        ScoringService(state.workflow, state.storage).score(
            candidate_run, filename="future-synthetic.csv", content=content, actor_label="pytest"
        )
    with pytest.raises(PromotionError, match="archivo sellado exacto"):
        service.evaluate(
            cohort.id, filename="other.csv", content=content + b"\n", actor_label="pytest"
        )
    comparison = service.evaluate(
        cohort.id, filename="future-synthetic.csv", content=content, actor_label="pytest"
    )
    assert comparison.rows == 60
    assert comparison.recommendation == "CANDIDATE_BETTER"
    assert comparison.candidate_f1 > comparison.current_f1
    assert state.workflow.list_scorings(candidate_run) == []
    assert comparison.current_scenario.status == "scenario_estimate"
    assert comparison.candidate_scenario.status == "scenario_estimate"
    assert (
        service.evaluate(
            cohort.id, filename="future-synthetic.csv", content=content, actor_label="pytest"
        )
        == comparison
    )
    assert service.active_policy_id(item.cap.id) is None
    with pytest.raises(PromotionError, match="revisión humana"):
        service.promote(
            comparison.id, actor_label="pytest", approved_hash=canonical_hash(comparison)
        )
    with pytest.raises(PromotionError, match="comparación exacta"):
        service.review(
            comparison.id,
            approved_hash="0" * 64,
            decision="APPROVED",
            actor_label="pytest-reviewer",
        )
    service.review(
        comparison.id,
        approved_hash=canonical_hash(comparison),
        decision="APPROVED",
        actor_label="pytest-reviewer",
    )
    with pytest.raises(RunBlockedError):
        service.promote(
            comparison.id, actor_label="pytest", approved_hash=canonical_hash(comparison)
        )
    state.workflow.decide_publication(
        report_id=report_id, decision="APPROVED", actor_label="pytest-reviewer"
    )
    with pytest.raises(RunBlockedError, match="vigente"):
        ScoringService(state.workflow, state.storage).score(
            candidate_run, filename="future-synthetic.csv", content=content, actor_label="pytest"
        )
    assignment = service.promote(
        comparison.id, actor_label="pytest-reviewer", approved_hash=canonical_hash(comparison)
    )
    assert assignment
    assert service.active_policy_id(item.cap.id) == candidate_policy.id
    with pytest.raises(RunBlockedError, match="vigente"):
        ScoringService(state.workflow, state.storage).score(
            batch.run_id, filename="future-synthetic.csv", content=content, actor_label="pytest"
        )
    ScoringService(state.workflow, state.storage).approved_run(candidate_run)
    assert (
        service.promote(
            comparison.id, actor_label="pytest-reviewer", approved_hash=canonical_hash(comparison)
        )
        == assignment
    )
    assert batch.policy_version_id != candidate_policy.id


def test_leakage_and_insufficient_evidence_preserve_current(state):
    item, batch, _candidate_run, _report_id, candidate_policy, service, cohort, content = (
        _comparison_state(state)
    )
    with pytest.raises(PromotionError, match="criterio sellado"):
        service.seal(
            item.cap.id,
            current_policy_id=batch.policy_version_id,
            candidate_policy_id=candidate_policy.id,
            filename="same.csv",
            content=content,
            data_classification="synthetic",
            provenance_basis="Mismo archivo",
            coverage_complete=True,
            coverage_basis="Fixture con ventanas maduras",
            observed_until=datetime(2026, 9, 30, tzinfo=UTC),
            min_rows=1,
            min_f1_improvement=0,
            actor_label="pytest",
        )
    with pytest.raises(PromotionError, match="productivos reales"):
        service.seal(
            item.cap.id,
            current_policy_id=batch.policy_version_id,
            candidate_policy_id=candidate_policy.id,
            filename="production.csv",
            content=content,
            data_classification="production",
            provenance_basis="No autorizado",
            coverage_complete=True,
            coverage_basis="Fixture con ventanas maduras",
            observed_until=datetime(2026, 9, 30, tzinfo=UTC),
            min_rows=30,
            min_f1_improvement=0.01,
            actor_label="pytest",
        )
    frame = pd.read_csv(pd.io.common.BytesIO(content))
    frame.loc[0, "entity"] = batch.cases[0].case_id
    with pytest.raises(PromotionError, match="casos de acción"):
        service.seal(
            item.cap.id,
            current_policy_id=batch.policy_version_id,
            candidate_policy_id=candidate_policy.id,
            filename="overlap.csv",
            content=frame.to_csv(index=False).encode(),
            data_classification="synthetic",
            provenance_basis="Fixture con solapamiento",
            coverage_complete=True,
            coverage_basis="Fixture con ventanas maduras",
            observed_until=datetime(2026, 9, 30, tzinfo=UTC),
            min_rows=30,
            min_f1_improvement=0.01,
            actor_label="pytest",
        )
    with pytest.raises(PromotionError, match="archivo sellado exacto"):
        service.evaluate(
            cohort.id,
            filename="changed.csv",
            content=frame.to_csv(index=False).encode(),
            actor_label="pytest",
        )
    fresh = pd.read_csv(pd.io.common.BytesIO(content))
    fresh["entity"] = fresh["entity"] + "-SECOND"
    fresh_content = fresh.to_csv(index=False).encode()
    insufficient = service.seal(
        item.cap.id,
        current_policy_id=batch.policy_version_id,
        candidate_policy_id=candidate_policy.id,
        filename="future-synthetic.csv",
        content=fresh_content,
        data_classification="synthetic",
        provenance_basis="Cohorte sintética reservada",
        coverage_complete=True,
        coverage_basis="Fixture con ventanas maduras",
        observed_until=datetime(2026, 9, 30, tzinfo=UTC),
        min_rows=100,
        min_f1_improvement=0.01,
        actor_label="pytest",
    )
    comparison = service.evaluate(
        insufficient.id,
        filename="future-synthetic.csv",
        content=fresh_content,
        actor_label="pytest",
    )
    assert comparison.recommendation == "INSUFFICIENT"
    with pytest.raises(PromotionError, match="no favorece"):
        service.promote(
            comparison.id, actor_label="pytest", approved_hash=canonical_hash(comparison)
        )
    assert service.active_policy_id(item.cap.id) is None
