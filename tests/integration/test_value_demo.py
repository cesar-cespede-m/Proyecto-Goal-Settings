"""El fixture completo de valor recorre los gates sin datos ni revisiones reales."""

import json

import pytest

from core.ai_engine.agents.context import build_context
from core.ai_engine.agents.runner import AgentRunner
from core.db import get_connection
from core.db.repositories import WorkflowRepository
from scripts.generate_value_demo_datasets import generate
from scripts.run_product_demo import run


def test_two_synthetic_capabilities_complete_with_trace(tmp_path):
    data = tmp_path / "synthetic"
    manifest = generate(data)
    assert set(manifest["capabilities"]) == {"revenue", "savings"}
    assert manifest["capabilities"]["savings"]["outcome_window"]["kind"] == "deadline_column"

    trace = run(data, tmp_path / "demo", simulate_reviews=True)
    assert trace["synthetic"] is True
    assert trace["reviews_simulated"] is True
    assert trace["llm_provider_used"] is False
    assert set(trace["capabilities"]) == {"revenue", "savings"}
    for capability in trace["capabilities"].values():
        assert capability["initial_status"] == "COMPLETED"
        assert capability["candidate_status"] == "COMPLETED"
        assert capability["initial_scenario_net"] is not None
        assert capability["action_summary"]["cases"] == 40
        assert capability["action_summary"]["assigned"] > capability["action_summary"]["acted"]
        assert capability["action_summary"]["pending"] > 0
        assert capability["comparison"]["rows"] == 60
        assert capability["comparison"]["recommendation"] in {
            "KEEP_CURRENT",
            "CANDIDATE_BETTER",
        }
        assert capability["active_policy_id"]
    persisted = json.loads((tmp_path / "demo" / "trace.json").read_text(encoding="utf-8"))
    assert persisted == trace


def test_demo_requires_explicit_simulation_and_generator_manifest(tmp_path):
    data = tmp_path / "synthetic"
    manifest = generate(data)
    with pytest.raises(ValueError, match="simulate-reviews"):
        run(data, tmp_path / "no-review", simulate_reviews=False)

    manifest["capabilities"]["revenue"]["economic_scenario"]["value_per_case"] = "999999"
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generador sintético"):
        run(data, tmp_path / "tampered", simulate_reviews=True)


def test_agent_context_offers_only_the_policy_of_its_cycle(tmp_path):
    data = tmp_path / "synthetic"
    generate(data)
    trace = run(data, tmp_path / "demo", simulate_reviews=True)
    connection = get_connection(tmp_path / "demo" / "goal_settings_demo.sqlite3")
    try:
        workflow = WorkflowRepository(connection)
        runner = AgentRunner(workflow, None)
        capability = trace["capabilities"]["revenue"]
        for cycle_kind in ("initial", "candidate"):
            task = runner.start(
                capability_id=capability["capability_id"],
                cycle_id=capability[f"{cycle_kind}_cycle_id"],
                objective="Revisar evidencia sintética de este ciclo",
                actor_label="Prueba sintética",
                operation_id=f"context-{cycle_kind}",
            )
            context = build_context(workflow, task.id)
            assert context["policy"]["id"] == capability[f"{cycle_kind}_policy_id"]
            policy_refs = [
                ref for ref in context["available_refs"] if ref["kind"] == "value_policy"
            ]
            assert [ref["id"] for ref in policy_refs] == [capability[f"{cycle_kind}_policy_id"]]
    finally:
        connection.close()
