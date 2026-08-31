from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from apps.enterprise_agent_workbench.graph import WorkbenchGraph
from apps.enterprise_agent_workbench.model_client import (
    FakeModelClient,
    PlannerDecision,
    PlannerSemanticError,
    VLLMHTTPModelClient,
    bind_grounded_identity_arguments,
)
from apps.enterprise_agent_workbench.observability import RuntimeObservability
from apps.enterprise_agent_workbench.state import initial_agent_state
from apps.enterprise_agent_workbench.tools.campaign_api import (
    CampaignCurrentStateInput,
)


def _decision(
    *, action: str = "get_campaign", arguments: dict[str, Any] | None = None
) -> PlannerDecision:
    return PlannerDecision(
        objective="Explain why campaign ROI declined.",
        next_action=action,
        arguments=arguments or {},
        completed=False,
        user_visible_reason="Inspect the grounded campaign data.",
    )


def _catalog(*required_fields: str) -> list[dict[str, Any]]:
    return [
        {
            "name": "get_campaign",
            "description": "Read one campaign.",
            "arguments": {
                field: {"type": "string", "required": True}
                for field in required_fields
            },
        }
    ]


def _context(
    prior_tool_results: list[dict[str, Any]],
    *required_fields: str,
) -> dict[str, Any]:
    return {
        "available_tools": _catalog(*(required_fields or ("campaign_id",))),
        "prior_tool_results": prior_tool_results,
    }


def _result(
    arguments: dict[str, Any],
    *,
    status: str = "ok",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": "roi_anomaly_detection",
        "status": status,
        "arguments": arguments,
        "output": output or {},
    }


@pytest.mark.asyncio
async def test_live_replanner_binds_grounded_campaign_id_before_real_schema_validation():
    prior_tool_results = [
        _result(
            {
                "campaign_id": "C102",
                "current_days": 7,
                "reference_days": 7,
            }
        )
    ]
    model_output = _decision().model_dump(mode="json")
    requests: list[dict[str, Any]] = []
    validated_arguments: list[dict[str, Any]] = []

    def validate_real_tool_schema(decision: PlannerDecision) -> PlannerDecision:
        arguments = CampaignCurrentStateInput.model_validate(
            decision.arguments
        ).model_dump(mode="json")
        validated_arguments.append(arguments)
        return decision.model_copy(update={"arguments": arguments})

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(model_output)}}]},
        )

    context = _context(prior_tool_results)
    context["_planner_decision_validator"] = validate_real_tool_schema
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        client.repair_allowed = False
        decision = await client.replan(
            "Explain why C102 ROI declined.",
            context,
        )

    assert decision.arguments == {"campaign_id": "C102"}
    assert validated_arguments == [{"campaign_id": "C102"}]
    assert client.last_planner_repairs == 0
    assert client.last_planner_failure_signatures == []
    assert client.last_grounded_argument_bindings == ["campaign_id"]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_graph_fallback_binds_custom_client_and_emits_value_free_metric(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="viewer")
    model = FakeModelClient([_decision()])
    telemetry = RuntimeObservability()
    workbench = WorkbenchGraph(
        model_client=model,
        registry=runtime["registry"],
        memory_store=runtime["memory"],
        settings=runtime["settings"],
        observability=telemetry,
    )
    state = initial_agent_state(
        thread_id="binding-thread",
        run_id="binding-run",
        user_id="binding-user",
        organization_id="org-1",
        tenant_id="org-1",
        role="viewer",
        task="Explain why C102 ROI declined.",
    )
    state["tool_results"] = [
        _result(
            {
                "campaign_id": "C102",
                "current_days": 7,
                "reference_days": 7,
            }
        )
    ]

    update = await workbench.replanner(state)

    assert update["next_action"] == "get_campaign"
    assert update["action_arguments"] == {"campaign_id": "C102"}
    assert update.get("errors", []) == []
    assert telemetry.metrics.value(
        "grounded_argument_bindings",
        {"tenant_id": "org-1", "argument_name": "campaign_id"},
    ) == 1
    assert "C102" not in telemetry.prometheus_text()


@pytest.mark.parametrize(
    "prior_tool_results",
    [
        [],
        [_result({"campaign_id": "C102"}, status="error")],
        [_result({}, output={"campaign_id": "C102"})],
        [{**_result({"campaign_id": "C102"}), "tool_name": ""}],
    ],
    ids=[
        "no-prior-id",
        "unsuccessful-result",
        "output-only",
        "noncanonical-tool-envelope",
    ],
)
def test_binder_ignores_missing_unvalidated_or_output_only_candidates(
    prior_tool_results: list[dict[str, Any]],
):
    decision = _decision()

    bound = bind_grounded_identity_arguments(
        decision, _context(prior_tool_results)
    )

    assert bound.arguments == {}


def test_binder_fails_closed_for_ambiguous_prior_identity_values():
    decision = _decision()
    context = _context(
        [
            _result({"campaign_id": "C102"}),
            _result({"campaign_id": "C999"}),
        ]
    )

    assert bind_grounded_identity_arguments(decision, context).arguments == {}


def test_binder_deduplicates_identical_candidates_by_stable_json_value():
    decision = _decision(action="get_campaign_ids")
    catalog = [
        {
            "name": "get_campaign_ids",
            "description": "Read campaigns.",
            "arguments": {
                "campaign_ids": {"type": "array", "required": True},
            },
        }
    ]
    context = {
        "available_tools": catalog,
        "prior_tool_results": [
            _result({"campaign_ids": ["C102", "C103"]}),
            _result({"campaign_ids": ["C102", "C103"]}),
        ],
    }

    bound = bind_grounded_identity_arguments(decision, context)

    assert bound.arguments == {"campaign_ids": ["C102", "C103"]}


def test_binder_supports_an_exact_required_id_field():
    decision = _decision(arguments={})
    context = _context([_result({"id": "record-102"})], "id")

    bound = bind_grounded_identity_arguments(decision, context)

    assert bound.arguments == {"id": "record-102"}


@pytest.mark.parametrize("explicit_value", ["C999", "", None])
def test_binder_never_overwrites_an_explicit_model_identity(explicit_value: Any):
    decision = _decision(arguments={"campaign_id": explicit_value})
    context = _context([_result({"campaign_id": "C102"})])

    bound = bind_grounded_identity_arguments(decision, context)

    assert bound.arguments == {"campaign_id": explicit_value}


@pytest.mark.parametrize(
    "field",
    [
        "start_date",
        "idempotency_key",
        "tenant_id",
        "tenant_ids",
        "organization_id",
        "organization_ids",
        "org_id",
        "org_ids",
        "user_id",
        "user_ids",
        "thread_id",
        "thread_ids",
        "run_id",
        "run_ids",
        "trace_id",
        "trace_ids",
        "span_id",
        "span_ids",
        "auth_token_id",
        "client_secret_id",
    ],
)
def test_binder_never_carries_non_identity_control_or_secret_fields(field: str):
    decision = _decision()
    context = _context([_result({field: "do-not-bind"})], field)

    bound = bind_grounded_identity_arguments(decision, context)

    assert bound.arguments == {}


@pytest.mark.asyncio
async def test_unresolved_identity_uses_existing_bounded_repair_path():
    model_output = _decision().model_dump(mode="json")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(model_output)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        client.repair_allowed = False
        with pytest.raises(PlannerSemanticError, match="missing required arguments"):
            await client.replan(
                "Explain why ROI declined.",
                _context([_result({}, output={"campaign_id": "C102"})]),
            )

    assert calls == 1
    assert client.last_planner_repairs == 0
    assert client.last_grounded_argument_bindings == []


@pytest.mark.asyncio
async def test_repair_result_is_bound_before_final_schema_validation():
    invalid = _decision(action="Please run get_campaign for the campaign")
    repaired_but_missing_identity = _decision()
    responses = iter(
        [
            json.dumps(invalid.model_dump(mode="json")),
            json.dumps(repaired_but_missing_identity.model_dump(mode="json")),
        ]
    )
    calls = 0

    def validate_real_tool_schema(decision: PlannerDecision) -> PlannerDecision:
        arguments = CampaignCurrentStateInput.model_validate(
            decision.arguments
        ).model_dump(mode="json")
        return decision.model_copy(update={"arguments": arguments})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    context = _context([_result({"campaign_id": "C102"})])
    context["_planner_decision_validator"] = validate_real_tool_schema
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.replan("Explain why C102 ROI declined.", context)

    assert result.arguments == {"campaign_id": "C102"}
    assert calls == 2
    assert client.last_planner_repairs == 1
    assert client.last_grounded_argument_bindings == ["campaign_id"]


@pytest.mark.asyncio
async def test_non_identity_repair_preserves_bounded_sanitized_routing_context():
    catalog = [
        {
            "name": "update_campaign_budget",
            "description": "Update one campaign budget.",
            "arguments": {
                "campaign_id": {"type": "string", "required": True},
                "daily_budget": {"type": "number", "required": True},
            },
        }
    ]
    invalid = _decision(action="update_campaign_budget")
    repaired = _decision(
        action="update_campaign_budget",
        arguments={"campaign_id": "C102", "daily_budget": 1200},
    )
    responses = iter(
        [
            json.dumps(invalid.model_dump(mode="json")),
            json.dumps(repaired.model_dump(mode="json")),
        ]
    )
    prompts: list[str] = []
    prior_result = _result(
        {
            "campaign_id": "C102",
            "start_date": "2025-01-01",
            "idempotency_key": "must-not-leak-control-key",
        },
        output={
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
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    context = {
        "available_tools": catalog,
        "prior_tool_results": [prior_result],
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )
        result = await client.replan("Set C102 daily budget to 1200.", context)

    assert result.arguments == {"campaign_id": "C102", "daily_budget": 1200}
    assert client.last_planner_repairs == 1
    assert len(prompts) == 2
    routing_line = next(
        line
        for line in prompts[1].splitlines()
        if line.startswith("ROUTING_CONTEXT=")
    )
    routing_context = json.loads(routing_line.removeprefix("ROUTING_CONTEXT="))
    previous = routing_context["previous_tools"][0]
    assert previous["tool_name"] == "roi_anomaly_detection"
    assert previous["arguments"] == {
        "campaign_id": "C102",
        "start_date": "2025-01-01",
    }
    assert previous["routing_facts"] == {
        "campaign_id": "C102",
        "operation": "roi_anomaly_detection",
        "rows": [{"campaign_id": "C102"}],
    }
    assert len(json.dumps(routing_context)) <= 4_000
    assert "idempotency_key" not in routing_line
    for forbidden in (
        "must-not-leak-control-key",
        "full-tool-output-must-not-leak",
        "private-chain-must-not-leak",
        "sk-supersecret999",
    ):
        assert forbidden not in prompts[1]


@pytest.mark.asyncio
async def test_binding_diagnostics_are_task_local_and_reset_per_request():
    empty = json.dumps(_decision().model_dump(mode="json"))
    explicit = json.dumps(
        _decision(arguments={"campaign_id": "C999"}).model_dump(mode="json")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        content = empty if "TASK=bind-request" in prompt else explicit
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = VLLMHTTPModelClient(
            base_url="http://model.test/v1", model_name="local", http_client=http
        )

        async def run_bound() -> list[str]:
            await client.replan(
                "bind-request", _context([_result({"campaign_id": "C102"})])
            )
            return client.last_grounded_argument_bindings

        async def run_explicit() -> list[str]:
            await client.replan("explicit-request", _context([]))
            return client.last_grounded_argument_bindings

        bound_fields, explicit_fields = await asyncio.gather(
            run_bound(), run_explicit()
        )

        assert bound_fields == ["campaign_id"]
        assert explicit_fields == []

        await client.replan("explicit-request", _context([]))
        assert client.last_grounded_argument_bindings == []
