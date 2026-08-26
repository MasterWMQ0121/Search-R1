from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from apps.enterprise_agent_workbench.graph import (
    PLANNER_FAILURE_ANSWER,
    WorkbenchGraph,
    deterministic_idempotency_key,
)
from apps.enterprise_agent_workbench.model_client import VLLMHTTPModelClient
from apps.enterprise_agent_workbench.state import AgentState, initial_agent_state
from apps.enterprise_agent_workbench.tools.campaign_api import (
    UpdateCampaignBudgetInput,
)


_LIVE_DESCRIPTIVE_ACTION = (
    "Run the structured merchant analytics operation update_campaign_budget to "
    "update C102 daily budget to 1200."
)


def _planner_payload(
    action: str,
    arguments: dict[str, Any] | None = None,
    *,
    completed: bool = False,
) -> str:
    return json.dumps(
        {
            "objective": "Increase C102 daily budget to 1200.",
            "next_action": action,
            "arguments": arguments or {},
            "completed": completed,
            "user_visible_reason": (
                "The requested write requires human approval."
                if not completed
                else "The requested work is complete."
            ),
        }
    )


def _model_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def _state(*, role: str = "operator", thread_id: str = "live-a800-thread"):
    return initial_agent_state(
        thread_id=thread_id,
        run_id="live-a800-run",
        user_id="operator-1",
        organization_id="org-1",
        role=role,
        task="Increase C102 daily budget to 1200.",
    )


def _business_state(database_path) -> tuple[float, int, int]:
    with sqlite3.connect(database_path) as connection:
        budget = connection.execute(
            "SELECT daily_budget FROM campaigns WHERE campaign_id = ?", ("C102",)
        ).fetchone()[0]
        audit_count = connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        idempotency_count = connection.execute(
            "SELECT COUNT(*) FROM idempotency_keys"
        ).fetchone()[0]
    return float(budget), int(audit_count), int(idempotency_count)


def _workbench_graph(runtime, model_client):
    orchestrator = WorkbenchGraph(
        model_client=model_client,
        registry=runtime["registry"],
        memory_store=runtime["memory"],
        settings=runtime["settings"],
    )
    return orchestrator, orchestrator.build(checkpointer=InMemorySaver())


def _request_prompt(request: httpx.Request) -> str:
    return json.loads(request.content)["messages"][0]["content"]


@pytest.mark.asyncio
async def test_live_descriptive_write_repairs_then_pauses_and_executes_exactly_once(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="operator")
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = _request_prompt(request)
        prompts.append(prompt)
        if prompt.startswith("Plan this business task"):
            return _model_response(_planner_payload(_LIVE_DESCRIPTIVE_ACTION))
        if prompt.startswith("Repair the following invalid planner response"):
            return _model_response(
                _planner_payload(
                    "update_campaign_budget",
                    {"campaign_id": "C102", "daily_budget": 1200},
                )
            )
        if prompt.startswith("Replan this business task"):
            return _model_response(_planner_payload("finalizer", completed=True))
        if prompt.startswith("Produce a concise answer"):
            return _model_response("The approved budget update completed.")
        raise AssertionError(f"unexpected model prompt: {prompt[:120]}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = VLLMHTTPModelClient(
            base_url="http://model.test/v1",
            model_name="local",
            http_client=http,
        )
        orchestrator, graph = _workbench_graph(runtime, model)
        policy_actions: list[str] = []
        evaluate = orchestrator.policy.evaluate

        def recording_policy(*args, **kwargs):
            policy_actions.append(str(args[1] if len(args) > 1 else kwargs["tool_name"]))
            return evaluate(*args, **kwargs)

        orchestrator.policy.evaluate = recording_policy
        config = {
            "configurable": {"thread_id": "live-a800-thread"},
            "recursion_limit": 30,
        }
        paused = await graph.ainvoke(_state(), config)

        assert "__interrupt__" in paused
        assert paused["planner_repair_count"] == 1
        assert paused["pending_action"]["action"] == "update_campaign_budget"
        assert paused["approval_request"]["action"] == "update_campaign_budget"
        pending_arguments = paused["pending_action"]["arguments"]
        idempotency_key = pending_arguments["idempotency_key"]
        assert re.fullmatch(r"wb-[0-9a-f]{64}", idempotency_key)
        assert paused["approval_request"]["arguments"] == pending_arguments
        assert idempotency_key == deterministic_idempotency_key(
            organization_id="org-1",
            user_id="operator-1",
            thread_id="live-a800-thread",
            run_id="live-a800-run",
            action="update_campaign_budget",
            business_arguments={"campaign_id": "C102", "daily_budget": 1200.0},
        )
        assert _LIVE_DESCRIPTIVE_ACTION not in policy_actions
        assert set(policy_actions) == {"update_campaign_budget"}
        assert _business_state(runtime["database_path"]) == (1000.0, 0, 0)

        final = await graph.ainvoke(
            Command(resume={"decision": "approve"}), config
        )
        assert final["completed"] is True
        assert _business_state(runtime["database_path"]) == (1200.0, 1, 1)

        await graph.ainvoke(Command(resume={"decision": "approve"}), config)
        assert _business_state(runtime["database_path"]) == (1200.0, 1, 1)

    assert len(prompts) == 4
    assert "ALLOWED_ACTION_IDS=" in prompts[0]
    assert "update_campaign_budget" in prompts[1]
    assert "VALIDATION_ERROR=" in prompts[1]
    assert any(
        event.get("event_type") == "planner_repair"
        for event in paused["execution_trace"]
    )


def test_control_plane_idempotency_is_deterministic_and_validates_real_schema():
    common = {
        "organization_id": "org-1",
        "user_id": "operator-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "action": "update_campaign_budget",
        "business_arguments": {"campaign_id": "C102", "daily_budget": 1200.0},
    }
    first = deterministic_idempotency_key(**common)
    replay = deterministic_idempotency_key(**common)
    changed_arguments = deterministic_idempotency_key(
        **{**common, "business_arguments": {"campaign_id": "C102", "daily_budget": 1300.0}}
    )
    changed_run = deterministic_idempotency_key(**{**common, "run_id": "run-2"})

    assert first == replay
    assert first != changed_arguments
    assert first != changed_run
    assert re.fullmatch(r"wb-[0-9a-f]{64}", first)
    assert not any(character.isspace() for character in first)
    validated = UpdateCampaignBudgetInput.model_validate(
        {**common["business_arguments"], "idempotency_key": first}
    )
    assert validated.campaign_id == "C102"
    assert validated.daily_budget == 1200.0


@pytest.mark.asyncio
async def test_repeated_identical_semantic_failure_stops_locally_without_recursion(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="operator")
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = _request_prompt(request)
        prompts.append(prompt)
        if prompt.startswith(("Plan this business task", "Repair the following")):
            return _model_response(_planner_payload(_LIVE_DESCRIPTIVE_ACTION))
        raise AssertionError("planner_stuck must use the deterministic local finalizer")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = VLLMHTTPModelClient(
            base_url="http://model.test/v1",
            model_name="local",
            http_client=http,
        )
        orchestrator, graph = _workbench_graph(runtime, model)
        policy_actions: list[str] = []
        evaluate = orchestrator.policy.evaluate

        def recording_policy(*args, **kwargs):
            policy_actions.append(str(args[1] if len(args) > 1 else kwargs["tool_name"]))
            return evaluate(*args, **kwargs)

        orchestrator.policy.evaluate = recording_policy
        result = await graph.ainvoke(
            _state(thread_id="planner-stuck-thread"),
            {
                "configurable": {"thread_id": "planner-stuck-thread"},
                "recursion_limit": 12,
            },
        )

    assert len(prompts) == 2
    assert policy_actions == []
    assert result["termination_reason"] == "planner_stuck"
    assert result["completed"] is False
    assert result["final_answer"] == PLANNER_FAILURE_ANSWER
    assert result["tool_call_count"] == 0
    assert result["planner_repair_count"] == 1
    assert len(result["planner_failure_signatures"]) == 2
    assert len(set(result["planner_failure_signatures"])) == 1
    assert "__interrupt__" not in result
    assert _business_state(runtime["database_path"]) == (1000.0, 0, 0)
    assert any(error.get("code") == "planner_stuck" for error in result["errors"])


def _apply_node_update(state: AgentState, update: dict[str, Any]) -> AgentState:
    additive_fields = {
        "execution_trace",
        "tool_results",
        "sources",
        "errors",
    }
    merged = dict(state)
    for key, value in update.items():
        if key in additive_fields:
            merged[key] = [*state.get(key, []), *value]
        else:
            merged[key] = value
    return merged


@pytest.mark.asyncio
async def test_repeated_identical_policy_denial_uses_planner_stuck_safeguard(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="viewer")
    orchestrator = WorkbenchGraph(
        model_client=runtime["model"],
        registry=runtime["registry"],
        memory_store=runtime["memory"],
        settings=runtime["settings"],
    )
    denied_state = _state(role="viewer", thread_id="policy-denial-thread")
    denied_state.update(
        {
            "next_action": "update_campaign_budget",
            "action_arguments": {
                "campaign_id": "C102",
                "daily_budget": 1200,
                "idempotency_key": "direct-policy-defense-key",
            },
        }
    )

    first = await orchestrator.policy_and_route(denied_state)
    assert first["route"] == "merge_evidence"
    assert first["tool_results"][0]["policy_code"] == "role_denied"
    second = await orchestrator.policy_and_route(
        _apply_node_update(denied_state, first)
    )

    assert second["route"] == "finalizer"
    assert second["termination_reason"] == "planner_stuck"
    assert second["completed"] is False
    assert _business_state(runtime["database_path"]) == (1000.0, 0, 0)
