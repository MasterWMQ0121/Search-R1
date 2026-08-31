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
    PlannerSemanticError,
    PlannerSemanticValidationError,
    PlannerStuckError,
    VLLMHTTPModelClient,
    WorkbenchModelClient,
    extract_json_object,
    parse_planner_decision,
    validate_planner_decision,
)


def _decision(**overrides):
    payload = {
        "objective": "Analyze campaign C102",
        "next_action": "get_campaign",
        "arguments": {"campaign_id": "C102"},
        "completed": False,
        "user_visible_reason": "Metrics are needed.",
    }
    payload.update(overrides)
    return payload


def _planner_catalog():
    return [
        {
            "name": "get_campaign",
            "description": "Read one campaign.",
            "category": "business_read",
            "risk_level": "low",
            "approval_required": False,
            "side_effecting": False,
            "arguments": {
                "campaign_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "update_campaign_budget",
            "description": "Change a campaign daily budget.",
            "category": "business_write",
            "risk_level": "high",
            "approval_required": True,
            "side_effecting": True,
            "arguments": {
                "campaign_id": {"type": "string", "required": True},
                "daily_budget": {
                    "type": "number",
                    "required": True,
                    "exclusiveMinimum": 0,
                    "maximum": 1_000_000,
                },
            },
        },
    ]


def _planner_context(**overrides):
    context = {"available_tools": _planner_catalog()}
    context.update(overrides)
    return context


def test_extracts_fenced_planner_json_and_rejects_missing_fields():
    raw = "Result:\n```json\n" + json.dumps(_decision()) + "\n```"
    assert extract_json_object(raw)["next_action"] == "get_campaign"
    assert parse_planner_decision(raw).arguments == {"campaign_id": "C102"}
    with pytest.raises(ValueError, match="invalid planner fields"):
        parse_planner_decision('{"objective": "x"}')


def test_planner_semantics_require_exact_action_and_completion_consistency():
    context = _planner_context()
    read = PlannerDecision.model_validate(_decision())
    assert validate_planner_decision(read, context) is read

    with pytest.raises(PlannerSemanticValidationError, match="exactly equal"):
        validate_planner_decision(
            PlannerDecision.model_validate(
                _decision(next_action="Run get_campaign for C102")
            ),
            context,
        )
    with pytest.raises(PlannerSemanticValidationError, match="if and only if"):
        validate_planner_decision(
            PlannerDecision.model_validate(_decision(completed=True)), context
        )
    with pytest.raises(PlannerSemanticValidationError, match="if and only if"):
        validate_planner_decision(
            PlannerDecision.model_validate(
                _decision(next_action="finalizer", arguments={}, completed=False)
            ),
            context,
        )
    finalizer = PlannerDecision.model_validate(
        _decision(next_action="finalizer", arguments={}, completed=True)
    )
    assert validate_planner_decision(finalizer, context) is finalizer


def test_planner_semantics_validate_types_and_compact_constraints():
    context = _planner_context()
    exact_write = PlannerDecision.model_validate(
        _decision(
            next_action="update_campaign_budget",
            arguments={"campaign_id": "C102", "daily_budget": 1200},
        )
    )
    assert validate_planner_decision(exact_write, context) is exact_write

    with pytest.raises(PlannerSemanticValidationError, match="must be > 0"):
        validate_planner_decision(
            PlannerDecision.model_validate(
                _decision(
                    next_action="update_campaign_budget",
                    arguments={"campaign_id": "C102", "daily_budget": 0},
                )
            ),
            context,
        )
    with pytest.raises(PlannerSemanticValidationError, match="must be number"):
        validate_planner_decision(
            PlannerDecision.model_validate(
                _decision(
                    next_action="update_campaign_budget",
                    arguments={"campaign_id": "C102", "daily_budget": True},
                )
            ),
            context,
        )


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
        result = await client.plan("Analyze C102", _planner_context(sources=[]))

    assert result.next_action == "get_campaign"
    assert len(calls) == 2
    assert "Repair the following invalid planner response" in calls[1]["messages"][0]["content"]
    assert calls[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_live_descriptive_action_is_semantically_repaired_to_exact_tool_id():
    bad = _decision(
        objective="Increase C102 daily budget to 1200.",
        next_action=(
            "Run the structured merchant analytics operation "
            "update_campaign_budget to update C102 daily budget to 1200."
        ),
        arguments={},
        user_visible_reason="A campaign update is needed.",
    )
    fixed = _decision(
        objective="Increase C102 daily budget to 1200.",
        next_action="update_campaign_budget",
        arguments={"campaign_id": "C102", "daily_budget": 1200},
        user_visible_reason="The requested write requires human approval.",
    )
    responses = iter([json.dumps(bad), json.dumps(fixed)])
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
        result = await client.plan(
            "Increase C102 daily budget to 1200.", _planner_context()
        )

    assert result.next_action == "update_campaign_budget"
    assert result.arguments == {"campaign_id": "C102", "daily_budget": 1200}
    assert client.last_planner_repairs == 1
    assert len(client.last_planner_failure_signatures) == 1
    assert client.last_planner_failure_signatures[0].startswith("unknown_action:")
    repair_prompt = calls[1]["messages"][0]["content"]
    assert "descriptions are invalid" in repair_prompt
    assert '"name":"update_campaign_budget"' in repair_prompt
    assert '"name":"get_campaign"' not in repair_prompt
    assert "Do not provide idempotency_key" in repair_prompt


@pytest.mark.asyncio
async def test_replanner_binds_missing_campaign_id_without_llm_repair():
    invalid = _decision(
        objective="Explain why campaign ROI declined.",
        next_action="get_campaign",
        arguments={},
    )
    responses = iter([json.dumps(invalid)])
    prompts = []
    prior_result = {
        "tool_name": "roi_anomaly_detection",
        "status": "ok",
        "arguments": {
            "campaign_id": "C102",
            "start_date": "2025-01-01",
            "idempotency_key": "must-not-leak-control-key",
        },
        "source_ids": ["CAMPAIGN-METRICS-C102"],
        "output": {
            "operation": "roi_anomaly_detection",
            "campaign_id": "C102",
            "rows": [
                {
                    "campaign_id": "C102",
                    "description": "full-tool-output-must-not-leak",
                }
            ],
            "hidden_reasoning": "private-chain-must-not-leak",
            "api_key": "sk-supersecret999",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.replan(
            "Explain why C102 ROI declined.",
            _planner_context(prior_tool_results=[prior_result]),
        )

    assert result.arguments == {"campaign_id": "C102"}
    assert client.last_planner_repairs == 0
    assert client.last_planner_failure_signatures == []
    assert client.last_grounded_argument_bindings == ["campaign_id"]
    assert len(prompts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "invented_campaign_id"),
    [
        ("Explain why ROI declined.", "C102"),
        ("Explain why C102 ROI declined.", "C999"),
    ],
)
async def test_repair_rejects_campaign_id_not_grounded_in_task_or_routing_context(
    task, invented_campaign_id
):
    invalid = _decision(
        next_action="get_campaign",
        arguments={},
    )
    repaired = _decision(
        objective="Explain why campaign ROI declined.",
        next_action="get_campaign",
        arguments={"campaign_id": invented_campaign_id},
    )
    responses = iter([json.dumps(invalid), json.dumps(repaired)])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        with pytest.raises(
            PlannerSemanticError,
            match="ungrounded business arguments: arguments.campaign_id",
        ):
            await client.replan(task, _planner_context())

    assert calls == 2
    assert client.last_planner_repairs == 1
    assert len(client.last_planner_failure_signatures) == 2


@pytest.mark.asyncio
async def test_repair_prompt_sanitizes_invalid_response_and_task_payloads():
    invalid = _decision(
        objective="Read C102 with api_key=sk-supersecret999",
        arguments={
            "campaign_id": "C102",
            "idempotency_key": "model-control-key-must-not-leak",
        },
        user_visible_reason="hidden_reasoning=private-chain-must-not-leak",
    )
    repaired = _decision(arguments={"campaign_id": "C102"})
    responses = iter([json.dumps(invalid), json.dumps(repaired)])
    prompts = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.plan(
            "Read C102; token=task-secret-must-not-leak", _planner_context()
        )

    assert result.arguments == {"campaign_id": "C102"}
    repair_prompt = prompts[1]
    assert "[REDACTED]" in repair_prompt
    invalid_line = next(
        line for line in repair_prompt.splitlines() if line.startswith("INVALID_RESPONSE=")
    )
    assert "idempotency_key" not in invalid_line
    assert len(invalid_line.removeprefix("INVALID_RESPONSE=")) <= 2_000
    for forbidden in (
        "sk-supersecret999",
        "model-control-key-must-not-leak",
        "private-chain-must-not-leak",
        "task-secret-must-not-leak",
    ):
        assert forbidden not in repair_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_arguments, expected_error",
    [
        ({}, "missing required arguments"),
        (
            {
                "campaign_id": "C102",
                "daily_budget": 1200,
                "objective": "nested planner field",
            },
            "unexpected arguments: objective",
        ),
        (
            {"campaign_id": "C102", "daily_budget": "1200"},
            "arguments.daily_budget must be number",
        ),
        (
            {
                "campaign_id": "C102",
                "daily_budget": 1200,
                "idempotency_key": "model-supplied-key",
            },
            "control-plane metadata",
        ),
    ],
)
async def test_semantic_argument_errors_share_the_single_repair_budget(
    invalid_arguments, expected_error
):
    invalid = _decision(
        next_action="update_campaign_budget", arguments=invalid_arguments
    )
    repaired = _decision(
        next_action="update_campaign_budget",
        arguments={"campaign_id": "C102", "daily_budget": 1200},
    )
    responses = iter([json.dumps(invalid), json.dumps(repaired)])
    prompts = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.plan(
            "Update C102 daily budget to 1200", _planner_context()
        )

    assert result.next_action == "update_campaign_budget"
    assert result.arguments == {"campaign_id": "C102", "daily_budget": 1200}
    assert len(prompts) == 2
    assert expected_error in prompts[1]


@pytest.mark.asyncio
async def test_caller_tool_schema_validation_shares_the_single_repair_budget():
    first = _decision(next_action="get_campaign", arguments={"campaign_id": "bad"})
    repaired = _decision(
        next_action="get_campaign", arguments={"campaign_id": "C102"}
    )
    responses = iter([json.dumps(first), json.dumps(repaired)])
    prompts = []

    def validate_real_tool_schema(decision):
        if decision.arguments.get("campaign_id") != "C102":
            raise ValueError("campaign_id failed the actual tool schema")
        return decision

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.plan(
            "Read C102",
            _planner_context(_planner_decision_validator=validate_real_tool_schema),
        )

    assert result.arguments == {"campaign_id": "C102"}
    assert client.last_planner_repairs == 1
    assert len(prompts) == 2
    assert "actual tool schema" in prompts[1]
    assert "validate_real_tool_schema" not in prompts[0]


@pytest.mark.asyncio
async def test_repair_grounding_ignores_defaults_added_by_caller_validation():
    invalid = _decision(arguments={})
    repaired = _decision(arguments={"campaign_id": "C102"})
    responses = iter([json.dumps(invalid), json.dumps(repaired)])
    context = _planner_context()
    context["available_tools"][0]["arguments"]["top_k"] = {
        "type": "integer",
        "required": False,
    }

    def add_tool_default(decision):
        return decision.model_copy(
            update={"arguments": {**decision.arguments, "top_k": 3}}
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.plan(
            "Read C102",
            {**context, "_planner_decision_validator": add_tool_default},
        )

    assert result.arguments == {"campaign_id": "C102", "top_k": 3}
    assert client.last_planner_repairs == 1


@pytest.mark.asyncio
async def test_repeated_identical_semantic_failure_is_planner_stuck():
    descriptive = _decision(
        next_action="Please run update_campaign_budget now", arguments={}
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(descriptive)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        with pytest.raises(PlannerStuckError) as caught:
            await client.plan("Update C102", _planner_context())

    assert calls == 2
    assert caught.value.code == "planner_stuck"
    assert caught.value.repair_attempts == 1
    assert len(client.last_planner_failure_signatures) == 2
    assert len(set(client.last_planner_failure_signatures)) == 1
    assert "Please run" not in client.last_planner_failure_signatures[0]


@pytest.mark.asyncio
async def test_http_client_fails_closed_without_caller_catalog():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        with pytest.raises(PlannerSemanticError) as caught:
            await client.plan("Analyze C102")

    assert calls == 0
    assert caught.value.code == "planner_semantic_error"
    assert caught.value.repair_attempts == 0


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
            await client.plan("Analyze C102", _planner_context())

    assert calls == 2
    assert caught.value.code == "planner_parse_error"
    assert caught.value.repair_attempts == 1


@pytest.mark.asyncio
async def test_concurrent_planner_repairs_use_task_local_budget_and_diagnostics():
    valid = json.dumps(_decision())

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        if prompt.startswith("Repair the following invalid planner response"):
            content = valid
        elif "TASK=allowed-task C102" in prompt:
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
            decision = await client.plan("allowed-task C102", _planner_context())
            return decision, client.last_planner_repairs

        async def denied_call():
            client.repair_allowed = False
            denied_has_set_budget.set()
            with pytest.raises(PlannerParseError) as caught:
                await client.plan("denied-task", _planner_context())
            return caught.value, client.last_planner_repairs

        (allowed_decision, allowed_repairs), (denied_error, denied_repairs) = (
            await asyncio.gather(allowed_call(), denied_call())
        )

    assert allowed_decision.next_action == "get_campaign"
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
            await client.plan("Analyze C102", _planner_context())

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
