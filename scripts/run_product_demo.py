"""Ejecuta el circuito de dominio con datos y revisiones simuladas, sin proveedor ni red.

Uso: python -m scripts.run_product_demo --data data/synthetic/value_demo \
     --output .tmp/goal_settings-value-demo --simulate-reviews
Las aprobaciones de esta CLI son exclusivamente fixtures sintéticos; no acreditan HITL real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.analysis.snapshot import build_snapshot
from core.analysis.snapshot_recipe import record_snapshot_recipe
from core.contracts.orchestration import ToolCallV1
from core.data_engine import DataIngestionService
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.capability_learning import CapabilityLearningService
from core.domain.context_service import BusinessContextService
from core.domain.decision_value import DecisionValueService
from core.domain.prediction_batches import PredictionBatchService
from core.domain.promotion import PromotionService
from core.domain.value_policy import ValuePolicyService
from core.ml_engine import MLRunService
from core.ml_engine.scoring import ScoringService
from core.orchestration.runtime import Supervisor
from core.reporting import ReportService
from scripts.generate_value_demo_datasets import generate

ACTOR = "synthetic-demo-reviewer-not-a-real-human"


def _file(folder: Path, name: str, digest: str) -> bytes:
    content = (folder / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"El archivo sintético {name} no coincide con el manifiesto")
    return content


def _ingest(repository, storage: Path, project_id: str, name: str, content: bytes) -> str:
    result = DataIngestionService(repository, storage).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": name,
            "source_name": "Generador sintético Goal Settings MV-14",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=content,
    )
    if result.metadata.admission_status != "ACCEPTED":
        raise ValueError(f"El dataset {name} no fue aceptado")
    return result.metadata.dataset_id


def _prepare_initial(repository, storage, cap, cycle, spec, folder):
    content = _file(folder, "initial_events.csv", spec["files_sha256"]["initial_events.csv"])
    events = pd.read_csv(folder / "initial_events.csv")
    built = build_snapshot(events, spec["recipe"], coverage=spec["coverage"]["initial"])
    if built.readiness.eligible_rows < 100:
        raise ValueError("La cohorte inicial sintética no tiene soporte maduro suficiente")
    source_id = _ingest(repository, storage, cycle.project_id, "initial_events.csv", content)
    table_id = _ingest(
        repository,
        storage,
        cycle.project_id,
        "initial_snapshot.csv",
        built.training_frame().to_csv(index=False).encode(),
    )
    record_snapshot_recipe(
        repository,
        project_id=cycle.project_id,
        table_dataset_id=table_id,
        source_dataset_id=source_id,
        spec=spec["recipe"],
        actor_label=ACTOR,
        coverage=spec["coverage"]["initial"],
        readiness=built.readiness,
    )
    repository.set_modeling_table(cycle.project_id, table_id, actor_label=ACTOR)
    frame = built.training_frame()
    target = spec["target"]
    feature_columns = set(built.summary.feature_names)
    context_payload = {
        "objective": spec["question"],
        "target_column": target,
        "positive_class": True,
        "prediction_moment": "Dos días después del primer evento sintético",
        "false_positive_consequence": "Intervenir un caso sin necesidad",
        "false_negative_consequence": "No intervenir un caso con riesgo",
        "primary_metric": "f1",
        "metric_rationale": "Compara señal con baseline bajo capacidad limitada",
        "decision_actor_label": ACTOR,
        "feature_decisions": [
            {
                "column": column,
                "use": "include" if column in feature_columns else "exclude",
                "available_at_prediction": column in feature_columns,
                "reason": "Disponible antes de actuar"
                if column in feature_columns
                else "Identidad, corte o disponibilidad excluidos",
            }
            for column in frame.columns
            if column != target
        ],
        "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
    }
    context = BusinessContextService(repository, storage).create_version(
        project_id=cycle.project_id, dataset_id=table_id, context=context_payload
    )
    values = DecisionValueService(repository)
    draft = values.draft_predictive_from_context(
        cap.id, context_version_id=context.context_version_id
    )
    predictive = draft.model_dump(mode="json")
    predictive.update(
        population="Casos elegibles del comercio sintético",
        target_definition="Resultado binario definido por la receta temporal sintética",
        outcome_window=spec["outcome_window"],
    )
    version = values.create_predictive_version(
        cap.id,
        spec=predictive,
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id=f"predictive-{cap.id}",
    )
    values.bind_predictive_to_cycle(
        cap.id,
        cycle_id=cycle.id,
        predictive_version_id=version.id,
        actor_label=ACTOR,
        operation_id=f"bind-{cycle.id}",
    )
    economic = values.create_economic_version(
        cap.id,
        scenario=spec["economic_scenario"],
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id=f"economy-{cap.id}",
    )
    return table_id, economic


def _train(repository, storage: Path, cap, cycle, dataset_id: str, label: str):
    supervisor = Supervisor(repository, storage_root=storage)
    task = supervisor.create_task(
        capability_id=cap.id,
        cycle_id=cycle.id,
        role="modeling",
        objective="Proponer un ML binario para una cohorte sintética",
        actor_label=ACTOR,
        operation_id=f"task-{label}",
    )
    result = supervisor.execute_tool(
        task.id,
        ToolCallV1(tool_name="propose_training_strategy", arguments={"dataset_id": dataset_id}),
        operation_id=f"strategy-{label}",
        expected_revision=task.revision,
        actor_label=ACTOR,
    )
    if result.status != "COMPLETED":
        raise ValueError(f"La propuesta de modelado no terminó: {result.status}")
    strategy_id = next(ref.id for ref in result.output_refs if ref.kind == "strategy")
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label=ACTOR
    )
    trained = MLRunService(repository, storage).execute(strategy_id)
    return task.id, strategy_id, trained


def _policy(repository, cap, cycle, run_id: str, economic_id: str, *, based_on=None, label: str):
    service = ValuePolicyService(repository)
    policy = service.propose(
        cap.id,
        cycle_id=cycle.id,
        run_id=run_id,
        economic_version_id=economic_id,
        based_on_version_id=based_on,
        actor_label=ACTOR,
        operation_id=f"policy-{label}",
    )
    service.review(
        policy.id,
        approved_hash=policy.sha256,
        decision="APPROVED",
        actor_label=ACTOR,
        operation_id=f"policy-review-{label}",
    )
    return policy


def _actions(repository, storage, cap, policy, run_id: str, spec: dict, folder: Path):
    content = _file(folder, "new_cases.csv", spec["files_sha256"]["new_cases.csv"])
    scored = ScoringService(repository, storage).score(
        run_id, filename="new_cases.csv", content=content, actor_label=ACTOR
    )
    event_id = str(repository.list_scorings(run_id)[0]["id"])
    batch_service = PredictionBatchService(repository, storage)
    batch = batch_service.register(
        cap.id,
        policy_version_id=policy.id,
        scoring=scored,
        scoring_event_id=event_id,
        period="2026-08",
        data_classification="synthetic",
        provenance_basis="Manifiesto Goal Settings MV-14, cohorte score",
        actor_label=ACTOR,
        operation_id=f"batch-{cap.id}",
    )
    observed = pd.read_csv(folder / "action_outcomes.csv").set_index("entity")
    acted = 0
    for case in batch.cases[:10]:
        occurred = datetime.fromisoformat("2026-09-30T00:00:00+00:00")
        if case.assigned and acted < 3:
            batch_service.record(
                batch.id,
                case_id=case.case_id,
                event_type="ACTED",
                occurred_at=occurred,
                actor_label=ACTOR,
                operation_id=f"acted-{batch.id}-{case.row_number}",
            )
            acted += 1
        batch_service.record(
            batch.id,
            case_id=case.case_id,
            event_type="OUTCOME_OBSERVED" if case.row_number % 3 else "OUTCOME_PENDING",
            outcome_positive=bool(observed.loc[case.case_id, spec["target"]])
            if case.row_number % 3
            else None,
            occurred_at=occurred,
            actor_label=ACTOR,
            operation_id=f"outcome-{batch.id}-{case.row_number}",
        )
    return batch_service.summary(batch.id)


def _one(repository, storage: Path, product, kind: str, spec: dict, folder: Path) -> dict:
    caps = CapabilityService(repository)
    cap = caps.create_capability(
        product.id,
        name="Primera compra" if kind == "revenue" else "Promesa de entrega",
        objective=spec["question"],
        value_kind=kind,
        decision_unit="caso sintético",
        actor_label=ACTOR,
        operation_id=f"capability-{kind}",
    )
    cycle = caps.create_cycle(
        cap.id,
        reason="Histórico sintético inicial",
        actor_label=ACTOR,
        operation_id=f"cycle-{kind}",
    )
    dataset_id, economic = _prepare_initial(repository, storage, cap, cycle, spec, folder)
    initial_task, strategy_id, initial = _train(
        repository, storage, cap, cycle, dataset_id, f"{kind}-initial"
    )
    trace = {
        "capability_id": cap.id,
        "value_kind": kind,
        "initial_cycle_id": cycle.id,
        "initial_task_id": initial_task,
        "initial_strategy_id": strategy_id,
        "initial_run_id": initial.run_id,
        "initial_status": initial.status,
    }
    if initial.status != "COMPLETED":
        trace["next_step"] = "Revisar métricas reales; no se fuerza un modelo ni valor."
        return trace
    report_id = ReportService(repository).create_template_report(initial.run_id)
    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label=ACTOR)
    policy = _policy(repository, cap, cycle, initial.run_id, economic.id, label=f"{kind}-initial")
    trace.update(
        initial_report_id=report_id,
        initial_policy_id=policy.id,
        initial_policy_hash=policy.sha256,
        initial_scenario_net=str(policy.content.economic_evaluation.net_value),
        initial_metrics=json.loads(repository.get_run(initial.run_id)["candidate_metrics_json"]),
    )
    trace["action_summary"] = _actions(
        repository, storage, cap, policy, initial.run_id, spec, folder
    ).model_dump(mode="json")
    update_content = _file(folder, "update_events.csv", spec["files_sha256"]["update_events.csv"])
    candidate = CapabilityLearningService(repository, storage).start_from_snapshot(
        cap.id,
        source_run_id=initial.run_id,
        filename="update_events.csv",
        content=update_content,
        coverage=spec["coverage"]["update"],
        reason="Actualizar con cohorte sintética de referencia no intervenida",
        reference_unintervened=True,
        reference_basis="IDs UPDATE del generador; disjuntos de score",
        actor_label=ACTOR,
        data_authorized=True,
        operation_id=f"learning-{kind}",
    )
    candidate_cycle = caps.get_cycle(candidate.cycle_id)
    task_id, candidate_strategy, candidate_run = _train(
        repository, storage, cap, candidate_cycle, candidate.dataset_id, f"{kind}-candidate"
    )
    trace.update(
        candidate_cycle_id=candidate.cycle_id,
        candidate_task_id=task_id,
        candidate_strategy_id=candidate_strategy,
        candidate_run_id=candidate_run.run_id,
        candidate_status=candidate_run.status,
    )
    if candidate_run.status != "COMPLETED":
        trace["next_step"] = "Candidata sin modelo recomendado; conservar vigente."
        return trace
    candidate_report_id = ReportService(repository).create_template_report(candidate_run.run_id)
    candidate_policy = _policy(
        repository,
        cap,
        candidate_cycle,
        candidate_run.run_id,
        economic.id,
        based_on=policy.id,
        label=f"{kind}-candidate",
    )
    promotion = PromotionService(repository, storage)
    comparison_content = _file(
        folder, "comparison_cohort.csv", spec["files_sha256"]["comparison_cohort.csv"]
    )
    cohort = promotion.seal(
        cap.id,
        current_policy_id=policy.id,
        candidate_policy_id=candidate_policy.id,
        filename="comparison_cohort.csv",
        content=comparison_content,
        data_classification="synthetic",
        provenance_basis="Manifiesto Goal Settings MV-14, cohorte compare",
        coverage_complete=True,
        coverage_basis="Generador temporal y cobertura declarada en el manifiesto",
        observed_until=datetime.fromisoformat(
            spec["coverage"]["compare"]["observed_until"].replace("Z", "+00:00")
        ),
        min_rows=30,
        min_f1_improvement=0.01,
        actor_label=ACTOR,
    )
    evaluation = promotion.evaluate(
        cohort.id,
        filename="comparison_cohort.csv",
        content=comparison_content,
        actor_label=ACTOR,
    )
    trace.update(
        candidate_report_id=candidate_report_id,
        candidate_policy_id=candidate_policy.id,
        comparison_cohort_id=cohort.id,
        comparison_id=evaluation.id,
        comparison_hash=canonical_hash(evaluation),
        comparison=evaluation.model_dump(mode="json"),
        candidate_metrics=json.loads(
            repository.get_run(candidate_run.run_id)["candidate_metrics_json"]
        ),
    )
    if evaluation.recommendation == "CANDIDATE_BETTER":
        promotion.review(
            evaluation.id,
            approved_hash=canonical_hash(evaluation),
            decision="APPROVED",
            actor_label=ACTOR,
        )
        repository.decide_publication(
            report_id=candidate_report_id, decision="APPROVED", actor_label=ACTOR
        )
        trace["assignment_id"] = promotion.promote(
            evaluation.id, actor_label=ACTOR, approved_hash=canonical_hash(evaluation)
        )
        trace["active_policy_id"] = promotion.active_policy_id(cap.id)
    else:
        trace["active_policy_id"] = policy.id
        trace["next_step"] = (
            "Conservar vigente; obtener nueva evidencia independiente si se revisa."
        )
    return trace


def run(data: Path, output: Path, *, simulate_reviews: bool) -> dict:
    if not simulate_reviews:
        raise ValueError("Esta CLI requiere --simulate-reviews y sólo datos sintéticos")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True:
        raise ValueError("El manifiesto no acredita datos sintéticos")
    with TemporaryDirectory(prefix="goal_settings-synthetic-check-") as directory:
        expected = generate(Path(directory), seed=manifest["seed"])
    if manifest != expected:
        raise ValueError("El manifiesto no coincide con el generador sintético")
    output.mkdir(parents=True, exist_ok=True)
    connection = get_connection(output / "goal_settings_demo.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage = output / "private-storage"
    product = CapabilityService(repository).create_product(
        name="Comercio digital sintético",
        stage="building",
        actor_label=ACTOR,
        operation_id="demo-product",
    )
    trace = {
        "schema_version": "goal_settings-demo-run.v1",
        "synthetic": True,
        "reviews_simulated": True,
        "llm_provider_used": False,
        "product_id": product.id,
        "capabilities": {},
        "limits": [
            "Aprobaciones simuladas exclusivamente para el fixture; no equivalen a HITL real.",
            "Los escenarios económicos son supuestos; no hay ingresos ni ahorros reales.",
            "La CLI ejercita herramientas del especialista de modelado sin llamadas LLM.",
        ],
    }
    for kind, spec in manifest["capabilities"].items():
        trace["capabilities"][kind] = _one(repository, storage, product, kind, spec, data / kind)
    (output / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    connection.close()
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/value_demo"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/goal_settings-value-demo"))
    parser.add_argument("--simulate-reviews", action="store_true")
    args = parser.parse_args()
    result = run(args.data, args.output, simulate_reviews=args.simulate_reviews)
    for kind, value in result["capabilities"].items():
        actions = value.get("action_summary", {})
        comparison = value.get("comparison", {})
        print(
            f"{kind}: inicial={value['initial_status']}, "
            f"candidata={value.get('candidate_status')}, "
            f"escenario inicial={value.get('initial_scenario_net')} CLP"
        )
        print(
            f"  lote: {actions.get('cases')} casos, {actions.get('assigned')} asignados, "
            f"{actions.get('acted')} actuados; comparación={comparison.get('recommendation')}"
        )
    print(f"Traza sintética: {args.output / 'trace.json'}")


if __name__ == "__main__":
    main()
