from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest

from apps.enterprise_agent_workbench.model_client import (
    FakeModelClient,
    PlannerDecision,
    PlannerParseError,
    VLLMHTTPModelClient,
    WorkbenchModelClient,
    extract_json_object,
    parse_planner_decision,
)


def _decision(**overrides):
    payload = {
        "objective": "Analyze campaign C102",
        "next_action": "merchant_analytics",
        "arguments": {"campaign_id": "C102"},
        "completed": False,
        "user_visible_reason": "Metrics are needed.",
    }
    payload.update(overrides)
    return payload


def test_extracts_fenced_planner_json_and_rejects_missing_fields():
    raw = "Result:\n```json\n" + json.dumps(_decision()) + "\n```"
    assert extract_json_object(raw)["next_action"] == "merchant_analytics"
    assert parse_planner_decision(raw).arguments == {"campaign_id": "C102"}
    with pytest.raises(ValueError, match="invalid planner fields"):
        parse_planner_decision('{"objective": "x"}')


def test_fresh_package_import_enables_actual_strict_msgpack_constant():
    environment = os.environ.copy()
    environment.pop("LANGGRAPH_STRICT_MSGPACK", None)
    script = """
import json
import os
import apps.enterprise_agent_workbench.api
from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED
print(json.dumps({
    "environment": os.environ.get("LANGGRAPH_STRICT_MSGPACK"),
    "strict_constant": STRICT_MSGPACK_ENABLED,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "environment": "true",
        "strict_constant": True,
    }


@pytest.mark.asyncio
async def test_fake_client_uses_queue_then_deterministic_fallback():
    queued = PlannerDecision.model_validate(_decision(next_action="campaign_read"))
    client = FakeModelClient([queued])
    assert isinstance(client, WorkbenchModelClient)
    assert (await client.plan("Inspect C102")).next_action == "campaign_read"
    fallback = await client.replan("Compare C102 ROI", {})
    assert fallback.next_action == "compare_periods"
    assert client.calls == ["plan", "replan"]


@pytest.mark.asyncio
async def test_http_client_repairs_invalid_json_exactly_once():
    responses = iter(["not json", json.dumps(_decision())])
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.plan("Analyze C102", {"sources": []})

    assert result.next_action == "merchant_analytics"
    assert len(calls) == 2
    assert "Repair the following invalid planner response" in calls[1]["messages"][0]["content"]
    assert calls[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_http_client_raises_explicit_error_after_single_repair():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "still invalid"}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        with pytest.raises(PlannerParseError) as caught:
            await client.plan("Analyze C102")

    assert calls == 2
    assert caught.value.code == "planner_parse_error"
    assert caught.value.repair_attempts == 1


@pytest.mark.asyncio
async def test_concurrent_planner_repairs_use_task_local_budget_and_diagnostics():
    valid = json.dumps(_decision())

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        if "INVALID_RESPONSE=invalid-allowed" in prompt:
            content = valid
        elif "TASK=allowed-task" in prompt:
            content = "invalid-allowed"
        elif "TASK=denied-task" in prompt:
            content = "invalid-denied"
        else:  # pragma: no cover - protects the deterministic fixture contract
            raise AssertionError(f"unexpected prompt: {prompt[:100]}")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    denied_has_set_budget = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )

        async def allowed_call():
            client.repair_allowed = True
            await denied_has_set_budget.wait()
            decision = await client.plan("allowed-task")
            return decision, client.last_planner_repairs

        async def denied_call():
            client.repair_allowed = False
            denied_has_set_budget.set()
            with pytest.raises(PlannerParseError) as caught:
                await client.plan("denied-task")
            return caught.value, client.last_planner_repairs

        (allowed_decision, allowed_repairs), (denied_error, denied_repairs) = (
            await asyncio.gather(allowed_call(), denied_call())
        )

    assert allowed_decision.next_action == "merchant_analytics"
    assert allowed_repairs == 1
    assert denied_error.repair_attempts == 0
    assert denied_repairs == 0


@pytest.mark.asyncio
async def test_http_client_honors_exhausted_run_level_repair_budget():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "invalid"}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        client.repair_allowed = False
        with pytest.raises(PlannerParseError) as caught:
            await client.plan("Analyze C102")

    assert calls == 1
    assert caught.value.repair_attempts == 0


@pytest.mark.asyncio
async def test_synthesis_and_memory_summary_do_not_require_provider_packages():
    client = FakeModelClient(
        synthesized_answer="ROI declined [S1].", memory_summary="Prefers concise ROI reports."
    )
    assert await client.synthesize("Why?", []) == "ROI declined [S1]."
    assert (
        await client.summarize_memory([{"role": "user", "content": "Be concise"}])
        == "Prefers concise ROI reports."
    )
