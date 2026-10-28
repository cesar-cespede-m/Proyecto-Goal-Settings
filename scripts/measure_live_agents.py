"""Prueba acotada del equipo con un proveedor LLM real y datos sintéticos.

Ejemplo: python -m scripts.measure_live_agents --output .tmp/live-agents-01
La clave se lee de .env; la evidencia no guarda prompts ni respuestas del proveedor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from core.ai_engine.agents.runner import AgentRunner
from core.ai_engine.config import LLMSettings
from core.ai_engine.openai_compatible import OpenAICompatibleProvider
from core.contracts.orchestration import AgentBudgetV1
from core.db import get_connection
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from scripts.generate_value_demo_datasets import generate
from scripts.run_product_demo import run as run_domain_demo

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = {
    "revenue": (
        "Revisa sólo la calidad y maduración de la evidencia sintética de primera compra. "
        "Delega esta comprobación en data. Cuando termine, cierra tu análisis con finish "
        "y señala que la decisión de actuar corresponde a una persona. Si falta información "
        "indispensable, usa wait_human o wait_data. No entrenes, apruebes ni publiques nada."
    ),
    "savings": (
        "Revisa sólo la política y los límites del escenario sintético de entrega tardía. "
        "Delega esta revisión en value_evidence. Cuando termine, cierra tu análisis con finish "
        "y aclara que el valor real necesita medición posterior. Si falta información "
        "indispensable, usa wait_human o wait_data. No entrenes, apruebes ni publiques nada."
    ),
}


def _check_preview(preview: dict, *, secret: str | None, output: Path, capability_id: str) -> None:
    encoded = json.dumps(preview, ensure_ascii=False)
    forbidden = [value for value in (secret, str(output.resolve()), capability_id) if value]
    forbidden += ["REV-initial-", "SAV-initial-", "OPENAI_API_KEY"]
    if any(value in encoded for value in forbidden):
        raise ValueError("La vista previa contiene datos que no deben salir al proveedor")


def measure(output: Path, *, env_file: Path, max_turns: int = 12) -> dict:
    if output.exists():
        raise ValueError("Elige una carpeta nueva para conservar una sola ejecución por evidencia")
    settings = LLMSettings.from_env(env_file=env_file)
    if not settings.active or settings.api_key is None:
        raise ValueError("Se requiere proveedor remoto activo y clave local configurada")
    if settings.provider not in {"openai", "openai_compatible"}:
        raise ValueError("La medición usa el adaptador OpenAI compatible del repositorio")

    data = output / "synthetic"
    domain = output / "domain"
    generate(data)
    trace = run_domain_demo(data, domain, simulate_reviews=True)
    connection = get_connection(domain / "goal_settings_demo.sqlite3")
    evidence = {
        "schema_version": "goal_settings-live-llm-measurement.v1",
        "synthetic": True,
        "domain_reviews_simulated": True,
        "llm_provider_used": True,
        "provider": settings.provider,
        "configured_model": settings.model,
        "domain_trace_sha256": canonical_hash(trace),
        "capabilities": {},
        "limits": [
            "El proveedor es real; las aprobaciones del fixture de dominio son simuladas.",
            "Los agentes proponen; el supervisor valida y no concede aprobaciones humanas.",
            "Las cifras monetarias son supuestos sintéticos, no impacto observado.",
        ],
    }
    try:
        workflow = WorkflowRepository(connection)
        provider = OpenAICompatibleProvider(settings)
        runner = AgentRunner(workflow, provider, storage_root=domain / "private-storage")
        for kind, objective in OBJECTIVES.items():
            capability = trace["capabilities"][kind]
            root = runner.start(
                capability_id=capability["capability_id"],
                cycle_id=capability["initial_cycle_id"],
                objective=objective,
                actor_label="Revisión sintética autorizada",
                operation_id=f"real-llm-{kind}-{uuid4().hex}",
                budget=AgentBudgetV1(max_steps=48, max_calls=20, max_seconds=600),
            )
            preview = runner.preview(root.id)
            _check_preview(
                preview,
                secret=settings.api_key.get_secret_value(),
                output=output,
                capability_id=capability["capability_id"],
            )
            started = perf_counter()
            results = runner.run(
                root.id,
                expected_revision=root.revision,
                actor_label="Revisión sintética autorizada",
                operation_id=f"run-{kind}-{uuid4().hex}",
                max_turns=max_turns,
            )
            elapsed_ms = round((perf_counter() - started) * 1000)
            children = runner.supervisor.repository.list_children(root.id)
            tasks = [runner.supervisor.repository.get_task(root.id), *children]
            turns = [turn for task in tasks for turn in runner.turns.list_turns(task.id)]
            root_state = tasks[0].state
            waiting = any(task.state in {"WAITING_HUMAN", "WAITING_DATA"} for task in tasks)
            evidence["capabilities"][kind] = {
                "wall_time_ms": elapsed_ms,
                "preview_sha256": canonical_hash(preview),
                "root_state": root_state,
                "resolution": (
                    "COMPLETED"
                    if root_state == "COMPLETED"
                    else "WAITING_INPUT"
                    if waiting
                    else "INCOMPLETE"
                ),
                "tasks": [{"role": task.role, "state": task.state} for task in tasks],
                "task_states": {task.role: task.state for task in tasks},
                "turns": [
                    {
                        "role": next(task.role for task in tasks if task.id == turn.task_id),
                        "proposal_action": turn.proposal.action if turn.proposal else None,
                        "delegate_role": turn.proposal.delegate_role if turn.proposal else None,
                        "tool_name": (
                            turn.proposal.tool_call.tool_name
                            if turn.proposal and turn.proposal.tool_call
                            else None
                        ),
                        "dispatch_error": (
                            turn.dispatch_result.error_code if turn.dispatch_result else None
                        ),
                        "tool_status": (
                            turn.dispatch_result.tool_result.status
                            if turn.dispatch_result and turn.dispatch_result.tool_result
                            else None
                        ),
                        "error_code": turn.error_code,
                        "latency_ms": turn.latency_ms,
                        "input_tokens": turn.input_tokens,
                        "output_tokens": turn.output_tokens,
                        "response_sha256": turn.response_sha256,
                    }
                    for turn in turns
                ],
                "valid_proposals": sum(turn.proposal is not None for turn in turns),
                "result_errors": [result.error_code for result in results if result.error_code],
                "total_input_tokens": sum(turn.input_tokens or 0 for turn in turns),
                "total_output_tokens": sum(turn.output_tokens or 0 for turn in turns),
                "total_provider_latency_ms": sum(turn.latency_ms or 0 for turn in turns),
            }
    finally:
        connection.close()
    path = output / "live_llm_evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-turns", type=int, default=12, choices=range(1, 21))
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()
    evidence = measure(args.output, env_file=args.env_file, max_turns=args.max_turns)
    for kind, result in evidence["capabilities"].items():
        print(
            f"{kind}: {len(result['turns'])} llamadas, {result['valid_proposals']} propuestas "
            f"válidas, {result['total_input_tokens']} tokens entrada, "
            f"{result['total_output_tokens']} salida, {result['wall_time_ms']} ms"
        )
        print(
            f"  resolución: {result['resolution']}; roles/estados: "
            f"{result['task_states']}; errores: {result['result_errors']}"
        )
    print(f"Evidencia: {args.output / 'live_llm_evidence.json'}")
    if args.require_completed and any(
        result["resolution"] != "COMPLETED" for result in evidence["capabilities"].values()
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
