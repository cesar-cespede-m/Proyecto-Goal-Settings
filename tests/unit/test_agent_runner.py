"""MV-07: recorridos con proveedores falsos; ninguna prueba usa red ni claves reales."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.ai_engine.agents import AgentRunner
from core.ai_engine.agents.prompts import build_request, proposal_schema
from core.ai_engine.provider import LLMProviderError, LLMResponse
from core.contracts.orchestration import AgentBudgetV1
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from core.db.repositories.agent_turns import AgentTurnRepository
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.orchestration.runtime import OrchestrationError
from tests.unit.test_decision_value import _economic

ACTOR = "Persona de prueba sintética"


class FakeAgentProvider:
    """Proveedor determinista de pruebas, sin transporte HTTP ni credenciales."""

    name = "pytest_fake_agent"
    model = "fake-agent-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
        assert self.responses, "El runner excedió las llamadas esperadas al proveedor falso"
        response = self.responses.pop(0)
        if callable(response):
            response = response(request)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(
            content=json.dumps(response, ensure_ascii=False)
            if isinstance(response, dict)
            else response,
            provider=self.name,
            model=self.model,
            response_id=f"fake-response-{len(self.requests)}",
            latency_ms=3,
            input_tokens=40,
            output_tokens=20,
        )


@pytest.fixture
def state(tmp_path):
    connection = get_connection(tmp_path / "agent-runner.sqlite3")
    migrate(connection)
    workflow = WorkflowRepository(connection)
    capabilities = CapabilityService(workflow)
    product = capabilities.create_product(
        name="Producto sintético", actor_label=ACTOR, operation_id="product"
    )
    caps = [
        capabilities.create_capability(
            product.id,
            name=f"Capacidad {kind}",
            objective="Mejorar decisiones sobre primera compra",
            value_kind=kind,
            decision_unit="persona elegible",
            actor_label=ACTOR,
            operation_id=kind,
        )
        for kind in ("revenue", "savings")
    ]
    yield SimpleNamespace(
        connection=connection,
        workflow=workflow,
        product=product,
        capabilities=capabilities,
        caps=caps,
        values=DecisionValueService(workflow),
        turns=AgentTurnRepository(workflow),
    )
    connection.close()


def _start(state, provider, *, budget=None, objective="Aclarar la decisión de producto"):
    runner = AgentRunner(state.workflow, provider)
    task = runner.start(
        capability_id=state.caps[0].id,
        objective=objective,
        actor_label=ACTOR,
        operation_id="start",
        budget=budget or AgentBudgetV1(max_steps=30, max_calls=12, max_seconds=120),
    )
    return runner, task


def _business(state, provider, *, budget=None):
    runner = AgentRunner(state.workflow, provider)
    task = runner.supervisor.create_task(
        capability_id=state.caps[0].id,
        role="business",
        objective="Aclarar la decisión y pedir supuestos pendientes",
        actor_label=ACTOR,
        operation_id="business-task",
        budget=budget or AgentBudgetV1(max_steps=30, max_calls=12, max_seconds=120),
    )
    return runner, task


def _step(runner, task, operation="turn"):
    current = runner.supervisor.repository.get_task(task.id)
    return runner.step(
        task.id,
        expected_revision=current.revision,
        actor_label=ACTOR,
        operation_id=operation,
    )


def _wait(action="wait_human"):
    return {
        "action": action,
        "message": "Falta información para continuar con una propuesta revisable.",
        "questions": ["¿Qué acción puede realizar el equipo?"],
        "reason_code": "MISSING_BUSINESS_INFORMATION",
    }


def _tool(name="read_capability_context", arguments=None):
    return {
        "action": "tool",
        "message": "Consultaré la evidencia disponible para esta capacidad.",
        "tool_call": {"tool_name": name, "arguments": arguments or {}},
    }


def _finish():
    return {"action": "finish", "message": "La revisión solicitada está terminada."}


def _wire_context(request):
    return json.loads(request.user.split("\n", 1)[1])["context"]


def test_live_schema_exposes_value_specialist_and_repairs_unsupported_claim():
    role_field = proposal_schema("coordinator")["properties"]["delegate_role"]
    assert "value_evidence" in role_field["anyOf"][0]["enum"]
    request = build_request(
        {"role": "coordinator"},
        timeout_seconds=30,
        repair_error="AGENT_UNSUPPORTED_CLAIM",
    )
    repair = json.loads(request.user.split("\n", 1)[1])["repair"]
    assert "delegate_objective" in repair["instruction"]
    assert "dígitos" in repair["instruction"]


def _no_training_or_approval(state):
    assert state.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert state.connection.execute("SELECT COUNT(*) FROM approval_events").fetchone()[0] == 0


def test_start_is_business_first_and_never_calls_provider(state):
    provider = FakeAgentProvider()
    runner, task = _start(state, provider)
    assert task.role == "coordinator"
    assert task.cycle_id is None
    assert task.state == "PENDING"
    assert provider.requests == []
    assert runner.supervisor.repository.list_children(task.id) == []
    assert state.connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
    _no_training_or_approval(state)


@pytest.mark.parametrize(
    "action,expected", [("wait_human", "WAITING_HUMAN"), ("wait_data", "WAITING_DATA")]
)
def test_missing_information_creates_visible_wait_without_inventing_values(state, action, expected):
    provider = FakeAgentProvider(_wait(action))
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is None
    assert result.task.state == expected
    assert result.proposal.questions == ["¿Qué acción puede realizar el equipo?"]
    assert result.task.calls_used == 1
    assert state.values.list_decision_versions(state.caps[0].id) == []
    assert state.values.list_economic_versions(state.caps[0].id) == []
    _no_training_or_approval(state)


def test_role_loop_can_read_then_save_draft_then_request_human_information(state):
    provider = FakeAgentProvider(
        _tool(),
        _tool("save_decision_draft", {"spec": {"question": "¿A quién ofrecer ayuda?"}}),
        _wait(),
    )
    runner, task = _business(state, provider)
    first = _step(runner, task, "read")
    assert first.tool_result.status == "COMPLETED"
    assert first.task.state == "RUNNING"
    second = _step(runner, task, "draft")
    assert second.tool_result.status == "COMPLETED"
    final = _step(runner, task, "wait")
    assert final.task.state == "WAITING_HUMAN"
    versions = state.values.list_decision_versions(state.caps[0].id)
    assert len(versions) == 1
    assert versions[0].spec.owner_label is None
    assert versions[0].confirmed_human is False
    assert final.task.calls_used == 3
    assert len(state.turns.list_turns(task.id)) == 3
    assert all(request.max_retries == 0 for request in provider.requests)
    assert all(0 < request.max_timeout_seconds <= 60 for request in provider.requests)
    _no_training_or_approval(state)


def test_coordinator_delegates_to_real_business_task_with_separate_prompt(state):
    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "Encargo la definición de la decisión al especialista de negocio.",
            "delegate_role": "business",
            "delegate_objective": "Identificar la decisión y solicitar los supuestos faltantes",
        },
        _tool(),
        _finish(),
        _wait(),
    )
    runner, root = _start(state, provider)
    delegated = _step(runner, root, "delegate")
    assert delegated.child_task_id is not None
    child = runner.supervisor.repository.get_task(delegated.child_task_id)
    assert child.role == "business"
    assert child.parent_task_id == root.id
    assert child.root_task_id == root.id
    assert child.capability_id == root.capability_id
    assert child.id != root.id
    _step(runner, child, "business-read")
    done = _step(runner, child, "business-finish")
    assert done.task.state == "COMPLETED"
    continued = _step(runner, root, "coordinator-continue")
    assert continued.task.state == "WAITING_HUMAN"
    assert continued.task.calls_used == 4
    assert provider.requests[0].system != provider.requests[1].system
    assert "business" in provider.requests[1].user
    assert child.id not in provider.requests[-1].user
    assert _wire_context(provider.requests[-1])["children"][0]["id"].startswith("R")


def test_coordinator_can_finish_after_a_bounded_run_and_continue(state):
    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "Encargo una revisión acotada al especialista de negocio.",
            "delegate_role": "business",
            "delegate_objective": "Revisar la definición existente y devolver lo pendiente",
        },
        _finish(),
        _finish(),
    )
    runner, root = _start(
        state,
        provider,
        budget=AgentBudgetV1(max_steps=8, max_calls=4, max_seconds=120),
    )
    first = runner.run(
        root.id,
        expected_revision=root.revision,
        actor_label=ACTOR,
        operation_id="first-bounded-run",
        max_turns=2,
    )
    assert len(first) == 2
    current = runner.supervisor.repository.get_task(root.id)
    assert current.state == "RUNNING"
    assert current.calls_used == 2
    second = runner.run(
        root.id,
        expected_revision=current.revision,
        actor_label=ACTOR,
        operation_id="continue-bounded-run",
        max_turns=1,
    )
    assert len(second) == 1
    assert second[0].task.state == "COMPLETED"
    assert runner.supervisor.repository.get_task(root.id).calls_used == 3


def test_invalid_json_has_one_bounded_repair_and_records_both_calls(state):
    provider = FakeAgentProvider("this is not JSON", _wait())
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state == "WAITING_HUMAN"
    assert result.error_code is None
    assert len(provider.requests) == 2
    assert result.task.calls_used == 2
    turns = state.turns.list_turns(task.id)
    assert len(turns) == 2
    assert turns[0].proposal is None
    assert turns[1].repair_of_operation_id == turns[0].operation_id
    assert turns[1].proposal is not None
    assert all(turn.response_sha256 for turn in turns)
    assert "this is not JSON" not in provider.requests[1].user


def test_second_invalid_json_stops_without_third_call_or_raw_response_storage(state):
    secret = "TEST_SECRET_TOKEN_NEVER_PERSIST_1234567890"
    provider = FakeAgentProvider(f"broken {secret}", "still invalid")
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is not None
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}
    assert len(provider.requests) == 2
    assert result.task.calls_used == 2
    turns = state.turns.list_turns(task.id)
    assert all(turn.proposal is None for turn in turns)
    assert secret not in json.dumps([turn.model_dump(mode="json") for turn in turns])
    assert secret not in "".join(request.user for request in provider.requests)
    _no_training_or_approval(state)


def test_repair_cannot_exceed_root_call_budget(state):
    provider = FakeAgentProvider("invalid JSON")
    runner, task = _business(state, provider, budget=AgentBudgetV1(max_calls=1))
    result = _step(runner, task)
    assert len(provider.requests) == 1
    assert result.task.calls_used == 1
    assert result.error_code is not None
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}


@pytest.mark.parametrize(
    "code", ["PROVIDER_TIMEOUT", "PROVIDER_REFUSAL", "PROVIDER_QUOTA_EXHAUSTED"]
)
def test_provider_failure_is_visible_and_not_retried_or_applied(state, code):
    provider = FakeAgentProvider(LLMProviderError(code, retryable=True))
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code == code
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}
    assert result.task.calls_used == 1
    assert len(provider.requests) == 1
    assert state.values.list_decision_versions(state.caps[0].id) == []
    _no_training_or_approval(state)


def test_exact_step_replay_does_not_repeat_provider_or_domain_effect(state):
    provider = FakeAgentProvider(
        _tool("save_decision_draft", {"spec": {"question": "¿A quién ayudar?"}})
    )
    runner, task = _business(state, provider)
    result = _step(runner, task, "same-turn")
    replay = _step(runner, task, "same-turn")
    assert result.tool_result.status == "COMPLETED"
    assert replay.replayed is True
    assert replay.tool_result == result.tool_result
    assert len(provider.requests) == 1
    assert len(state.values.list_decision_versions(state.caps[0].id)) == 1


def test_changed_context_during_provider_call_invalidates_response_before_write(state):
    def mutate_context(_request):
        state.values.create_decision_version(
            state.caps[0].id,
            spec={"question": "Pregunta nueva declarada por una persona"},
            actor_label=ACTOR,
            operation_id="human-edit-during-call",
        )
        return _tool("save_decision_draft", {"spec": {"question": "Propuesta ya obsoleta"}})

    provider = FakeAgentProvider(mutate_context)
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code in {"STALE_CONTEXT", "AGENT_STALE_CONTEXT"}
    assert result.task.state == "WAITING_HUMAN"
    assert result.task.calls_used == 1
    versions = state.values.list_decision_versions(state.caps[0].id)
    assert len(versions) == 1
    assert versions[0].spec.question == "Pregunta nueva declarada por una persona"
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "call",
    [
        _tool("execute_python", {"code": "raise SystemExit()"}),
        _tool("approve_training"),
        _tool("train_model"),
        _tool("save_decision_draft", {"spec": {}, "confirmed_human": True}),
        _tool("save_decision_draft", {"spec": {"owner_label": "Persona inventada"}}),
        _tool("save_decision_draft", {"spec": {}, "capability_id": "foreign"}),
    ],
)
def test_forbidden_tool_requests_never_create_effects(state, call):
    provider = FakeAgentProvider(call, _wait())
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}
    assert result.tool_result is None or result.tool_result.status == "FAILED"
    assert state.values.list_decision_versions(state.caps[0].id) == []
    assert state.values.list_economic_versions(state.caps[0].id) == []
    _no_training_or_approval(state)


def test_foreign_capability_evidence_and_economics_are_not_accepted(state):
    foreign = state.values.create_economic_version(
        state.caps[1].id,
        scenario=_economic(),
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="foreign-economics",
    )
    proposal = _tool("evaluate_economic_scenario", {"economic_version_id": foreign.id})
    proposal["evidence_refs"] = [
        {"kind": "economic_scenario", "id": foreign.id, "sha256": foreign.sha256}
    ]
    provider = FakeAgentProvider(proposal, _wait())
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}
    assert result.tool_result is None or result.tool_result.status == "FAILED"
    assert foreign.id not in provider.requests[0].user
    _no_training_or_approval(state)


def test_provider_cannot_present_invented_economic_amounts_as_verified(state):
    forged = {"action": "finish", "message": "El ahorro comprobado es de 900000 USD."}
    provider = FakeAgentProvider(forged, _wait())
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state != "COMPLETED"
    assert all(
        turn.proposal is None or "900000" not in turn.proposal.message
        for turn in state.turns.list_turns(task.id)
    )
    assert state.values.list_economic_versions(state.caps[0].id) == []


def test_outgoing_context_redacts_private_text_and_never_sends_rows(state):
    email = "person@example.test"
    secret = "TEST_SECRET_TOKEN_OUTGOING_123456789012"
    objective = f"Revisar casos, contacto {email}; token {secret}"
    provider = FakeAgentProvider(_wait())
    runner, task = _start(state, provider, objective=objective)
    preview = runner.preview(task.id)
    _step(runner, task)
    outgoing = json.dumps(preview, ensure_ascii=False) + provider.requests[0].user
    assert email not in outgoing
    assert secret not in outgoing
    assert "redactado" in outgoing
    assert "storage_path" not in outgoing
    assert state.connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


def test_disabled_provider_allows_preparation_but_not_execution(state):
    runner, task = _start(state, None)
    assert runner.preview(task.id)["role"] == "coordinator"
    with pytest.raises(OrchestrationError):
        _step(runner, task)
    assert state.turns.list_turns(task.id) == []
    assert runner.supervisor.repository.get_task(task.id).calls_used == 0


def test_business_specialist_cannot_delegate_to_another_specialist(state):
    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "Delegaré una nueva revisión.",
            "delegate_role": "business",
            "delegate_objective": "Revisar la capacidad",
        },
        _wait(),
    )
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state != "COMPLETED"
    assert runner.supervisor.repository.list_children(task.id) == []


def test_declared_outcome_horizon_can_be_reused_without_becoming_an_economic_claim(state):
    state.values.create_decision_version(
        state.caps[0].id,
        spec={"question": "¿Cómo mejorar la primera compra a 14 días?"},
        actor_label=ACTOR,
        operation_id="human-horizon",
    )
    provider = FakeAgentProvider(
        {
            **_wait(),
            "message": "Falta decidir la intervención para la primera compra a 14 días.",
        }
    )
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is None
    assert "14" in result.proposal.message
    assert len(provider.requests) == 1


def test_economic_amounts_are_computed_by_tool_and_preserve_scenario_evidence_type(state):
    scenario = state.values.create_economic_version(
        state.caps[0].id,
        scenario=_economic(),
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="human-economic-assumptions",
    )

    def request_calculation(request):
        return _tool(
            "evaluate_economic_scenario",
            {"economic_version_id": _wire_context(request)["economics"]["id"]},
        )

    provider = FakeAgentProvider(request_calculation, _wait())
    runner, task = _business(state, provider)
    calculated = _step(runner, task, "economic-calculation")
    assert calculated.tool_result.status == "COMPLETED"
    assert calculated.tool_result.summary["net_value"] == "875"
    assert calculated.tool_result.summary["evidence_type"] == "synthetic_scenario"
    assert calculated.tool_result.summary["status"] == "scenario_estimate"
    _step(runner, task, "interpretation")
    assert '"net_value": "875"' in provider.requests[-1].user
    assert "synthetic_scenario" in provider.requests[-1].user
    assert len(state.values.list_economic_versions(state.caps[0].id)) == 1
    assert calculated.proposal.tool_call.arguments["economic_version_id"] == scenario.id
    _no_training_or_approval(state)


def test_child_cannot_escape_call_budget_already_consumed_by_coordinator(state):
    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "El especialista revisará la decisión.",
            "delegate_role": "business",
            "delegate_objective": "Identificar información pendiente",
        }
    )
    runner, root = _start(state, provider, budget=AgentBudgetV1(max_calls=1))
    result = _step(runner, root, "delegate")
    child = runner.supervisor.repository.get_task(result.child_task_id)
    with pytest.raises(OrchestrationError) as error:
        _step(runner, child, "child-turn")
    assert error.value.code == "BUDGET_EXHAUSTED"
    assert len(provider.requests) == 1
    assert runner.supervisor.repository.get_task(root.id).calls_used == 1


def test_bounded_team_run_stops_for_child_question_and_human_resume_keeps_budget(state):
    provider = FakeAgentProvider(
        {
            "action": "delegate",
            "message": "El especialista revisará la decisión.",
            "delegate_role": "business",
            "delegate_objective": "Identificar información pendiente",
        },
        _wait(),
    )
    runner, root = _start(state, provider)
    runner.run(
        root.id,
        expected_revision=root.revision,
        actor_label=ACTOR,
        operation_id="team-run",
        max_turns=8,
    )
    current = runner.supervisor.repository.get_task(root.id)
    assert current.state == "RUNNING"
    children = runner.supervisor.repository.list_children(root.id)
    assert len(children) == 1
    assert children[0].state == "WAITING_HUMAN"
    assert len(provider.requests) == 2
    counters = (current.calls_used, current.steps_used, current.elapsed_ms)
    runner.resume(root.id, expected_revision=current.revision, actor_label=ACTOR)
    resumed = runner.supervisor.repository.get_task(root.id)
    assert resumed.state == "RUNNING"
    assert (resumed.calls_used, resumed.steps_used, resumed.elapsed_ms) == counters
    assert runner.supervisor.repository.get_task(children[0].id).state == "PENDING"
    _no_training_or_approval(state)


def test_currency_question_without_amount_is_allowed(state):
    proposal = {**_wait(), "questions": ["¿En qué moneda, por ejemplo USD, expresas el costo?"]}
    provider = FakeAgentProvider(proposal)
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is None
    assert result.proposal.questions == proposal["questions"]
    assert len(provider.requests) == 1


def test_known_horizon_number_does_not_authorize_an_invented_monetary_amount(state):
    state.values.create_decision_version(
        state.caps[0].id,
        spec={"question": "¿Cómo mejorar la primera compra a 14 días?"},
        actor_label=ACTOR,
        operation_id="human-horizon",
    )
    provider = FakeAgentProvider(
        {"action": "finish", "message": "El ahorro comprobado es de 14 USD."}, _wait()
    )
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.task.state != "COMPLETED"
    assert all(
        turn.proposal is None or "14 USD" not in turn.proposal.message
        for turn in state.turns.list_turns(task.id)
    )


def test_unexpected_provider_exception_never_persists_exception_payload(state):
    secret = "TEST_SECRET_TOKEN_DIAGNOSTIC_1234567890"
    provider = FakeAgentProvider(RuntimeError(f"network diagnostic {secret}"))
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is not None
    assert result.task.state in {"WAITING_HUMAN", "FAILED"}
    assert len(provider.requests) == 1
    assert secret not in json.dumps(result.model_dump(mode="json"))
    assert secret not in json.dumps(
        [turn.model_dump(mode="json") for turn in state.turns.list_turns(task.id)]
    )


def test_conversational_context_change_during_provider_call_invalidates_proposal(state):
    holder = {}

    def add_child(_request):
        runner = holder["runner"]
        root = holder["root"]
        runner.supervisor.create_task(
            capability_id=state.caps[0].id,
            role="business",
            parent_task_id=root.id,
            objective="Encargo concurrente declarado fuera del proveedor",
            actor_label=ACTOR,
            operation_id="concurrent-child",
        )
        return _wait()

    provider = FakeAgentProvider(add_child)
    runner, root = _start(state, provider)
    holder.update(runner=runner, root=root)
    result = _step(runner, root)
    assert result.error_code == "STALE_CONTEXT"
    assert result.proposal is None
    assert result.task.state == "WAITING_HUMAN"
    assert len(provider.requests) == 1
    assert len(runner.supervisor.repository.list_children(root.id)) == 1


def test_provider_response_after_lease_expiry_is_not_applied(state):
    clock = SimpleNamespace(now=datetime(2030, 1, 1, tzinfo=UTC))

    def late_response(_request):
        clock.now += timedelta(seconds=61)
        return _tool("save_decision_draft", {"spec": {"question": "Borrador tardío"}})

    provider = FakeAgentProvider(late_response)
    runner = AgentRunner(state.workflow, provider, clock=lambda: clock.now)
    task = runner.supervisor.create_task(
        capability_id=state.caps[0].id,
        role="business",
        objective="Revisar la decisión",
        actor_label=ACTOR,
        operation_id="business",
        budget=AgentBudgetV1(max_seconds=120),
    )
    result = _step(runner, task)
    assert result.error_code == "OPERATION_LEASE_EXPIRED"
    assert result.task.state == "WAITING_HUMAN"
    assert result.task.elapsed_ms == 61000
    assert result.task.calls_used == 1
    assert result.proposal is None
    assert state.values.list_decision_versions(state.caps[0].id) == []


@pytest.mark.parametrize("effect", ["draft", "delegate"])
def test_restart_after_persisted_effect_recovers_without_another_provider_call(
    state, monkeypatch, effect
):
    proposal = (
        _tool("save_decision_draft", {"spec": {"question": "¿A quién ayudar?"}})
        if effect == "draft"
        else {
            "action": "delegate",
            "message": "Encargo la revisión al especialista de negocio.",
            "delegate_role": "business",
            "delegate_objective": "Aclarar información pendiente",
        }
    )
    provider = FakeAgentProvider(proposal)
    runner, task = _business(state, provider) if effect == "draft" else _start(state, provider)

    def crash_after_effect(*_args, **_kwargs):
        raise KeyboardInterrupt("Caída sintética después del efecto, antes de registrar despacho")

    monkeypatch.setattr(runner.turns, "mark_dispatched", crash_after_effect)
    with pytest.raises(KeyboardInterrupt):
        _step(runner, task, "interrupted-turn")
    assert len(provider.requests) == 1
    assert state.turns.get("interrupted-turn").dispatch_result is None

    # Una instancia nueva representa el reinicio; recupera la evidencia persistida.
    restarted = AgentRunner(state.workflow, provider)
    recovered = _step(restarted, task, "interrupted-turn")
    assert recovered.error_code is None
    assert recovered.replayed is True
    assert len(provider.requests) == 1
    assert state.turns.get("interrupted-turn").dispatch_result is not None
    if effect == "draft":
        assert recovered.tool_result.status == "COMPLETED"
        assert len(state.values.list_decision_versions(state.caps[0].id)) == 1
    else:
        children = restarted.supervisor.repository.list_children(task.id)
        assert [child.id for child in children] == [recovered.child_task_id]
    _no_training_or_approval(state)


def test_only_aliases_leave_process_and_evidence_is_resolved_back_to_local_artifact(state):
    scenario = state.values.create_economic_version(
        state.caps[0].id,
        scenario=_economic(),
        confirmed_human=True,
        actor_label=ACTOR,
        operation_id="human-economic",
    )
    decision = state.values.create_decision_version(
        state.caps[0].id,
        spec={"question": "¿A quién ayudar?", "economic_version_id": scenario.id},
        actor_label=ACTOR,
        operation_id="human-decision",
    )
    captured = {}

    def cite_alias(request):
        context = _wire_context(request)
        ref = next(
            item for item in context["available_refs"] if item["kind"] == "economic_scenario"
        )
        captured["ref"] = ref
        return {
            **_tool("evaluate_economic_scenario", {"economic_version_id": ref["id"]}),
            "evidence_refs": [ref],
        }

    provider = FakeAgentProvider(cite_alias)
    runner, task = _business(state, provider)
    preview = runner.preview(task.id)
    result = _step(runner, task)
    outgoing = json.dumps(preview) + provider.requests[0].user
    for internal in (
        state.product.id,
        state.caps[0].id,
        task.id,
        scenario.id,
        scenario.sha256,
        decision.id,
        decision.sha256,
        task.context_sha256,
    ):
        assert internal not in outgoing
    assert captured["ref"]["id"].startswith("R")
    assert captured["ref"].get("sha256") is None
    assert result.error_code is None
    assert result.proposal.evidence_refs[0].id == scenario.id
    assert result.proposal.tool_call.arguments["economic_version_id"] == scenario.id
    assert result.tool_result.output_refs[0].id == scenario.id
    assert state.turns.list_turns(task.id)[0].proposal.evidence_refs[0].id == scenario.id


def test_malformed_provider_error_code_is_sanitized_without_abandoning_claim(state):
    provider = FakeAgentProvider(LLMProviderError("invalid provider code with private detail"))
    runner, task = _business(state, provider)
    result = _step(runner, task)
    assert result.error_code is not None
    assert result.task.state == "WAITING_HUMAN"
    assert result.task.lease_owner is None
    assert len(provider.requests) == 1
    assert "private detail" not in json.dumps(result.model_dump(mode="json"))
