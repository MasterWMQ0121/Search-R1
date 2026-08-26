from __future__ import annotations

import httpx
import pytest
from dataclasses import replace
from langgraph.checkpoint.memory import InMemorySaver

from apps.enterprise_agent_workbench.graph import ResearchState, WorkbenchGraph
from apps.enterprise_agent_workbench.model_client import FakeModelClient, PlannerDecision
from apps.enterprise_agent_workbench.state import initial_agent_state
from apps.enterprise_agent_workbench.tool_registry import ToolRegistry
from apps.enterprise_agent_workbench.tracing import TraceEvent
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
    Phase5EvidenceAdapter,
    ResearchSearchTool,
)


def decision(action, arguments=None, *, completed=False):
    return PlannerDecision(
        objective="Complete the requested task safely.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="This is the next visible step.",
    )


def state(role="analyst"):
    return initial_agent_state(
        thread_id="thread-1",
        run_id="run-1",
        user_id="user-1",
        organization_id="org-1",
        role=role,
        task="Show campaign C102 and explain its current budget.",
    )


@pytest.mark.asyncio
async def test_graph_compiles_routes_read_and_terminates(graph_factory):
    runtime = graph_factory(
        [decision("get_campaign", {"campaign_id": "C102"}), decision("finalizer", completed=True)],
        "C102 has the recorded campaign state [BUSINESS:C102:get_campaign].",
    )
    graph = runtime["graph"]
    assert "human_approval_interrupt" in graph.get_graph().nodes
    result = await graph.ainvoke(
        state(), {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 80}
    )
    assert result["completed"] is True
    assert result["termination_reason"] == "completed"
    assert result["tool_call_count"] == 1
    assert result["final_citations"] == ["BUSINESS:C102:get_campaign"]
    assert {item["event_type"] for item in result["execution_trace"]} >= {
        "run_started",
        "planner_decision",
        "policy_allowed",
        "tool_started",
        "tool_completed",
        "final_answer_created",
        "run_completed",
    }
    assert [
        TraceEvent.model_validate(item).sequence
        for item in result["execution_trace"]
    ] == list(range(len(result["execution_trace"])))


@pytest.mark.asyncio
async def test_maximum_step_guard_terminates_without_claiming_completion(graph_factory):
    runtime = graph_factory(
        [decision("get_campaign", {"campaign_id": "C102"})],
        "Stopped at the configured safety boundary.",
        max_graph_steps=2,
    )
    result = await runtime["graph"].ainvoke(
        state(), {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 30}
    )
    assert result["completed"] is False
    assert result["termination_reason"] == "maximum_graph_steps"
    assert result["tool_call_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "arguments", "role"),
    [
        ("enterprise_kb_search", {"query": "budget approval", "top_k": 1}, "viewer"),
        (
            "compare_periods",
            {
                "campaign_id": "C102",
                "previous_start": "2026-08-01",
                "previous_end": "2026-08-07",
                "current_start": "2026-08-08",
                "current_end": "2026-08-14",
            },
            "analyst",
        ),
        ("get_campaign", {"campaign_id": "C102"}, "viewer"),
        ("research_search", {"query": "campaign ROI", "top_k": 1}, "viewer"),
    ],
)
async def test_every_read_route_reaches_a_bounded_finalizer(
    graph_factory, action, arguments, role
):
    runtime = graph_factory(
        [decision(action, arguments), decision("finalizer", completed=True)],
        "The bounded route returned a safe result.",
        role=role,
    )
    value = state(role=role)
    result = await runtime["graph"].ainvoke(
        value,
        {"configurable": {"thread_id": value["thread_id"]}, "recursion_limit": 80},
    )
    assert result["termination_reason"] == "completed"
    assert result["step_count"] <= 18
    assert result["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_tool_failure_is_traced_and_replanning_can_recover(graph_factory):
    runtime = graph_factory([], "unused")
    original = runtime["registry"].get("get_campaign")

    class FlakyHandler:
        calls = 0

        def __call__(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient fixture failure")
            return original.handler(request)

    flaky = FlakyHandler()
    specs = []
    for metadata in runtime["registry"].safe_metadata():
        spec = runtime["registry"].get(metadata["name"])
        specs.append(replace(spec, handler=flaky) if spec.name == "get_campaign" else spec)
    registry = ToolRegistry(specs)
    model = FakeModelClient(
        [
            decision("get_campaign", {"campaign_id": "C102"}),
            decision("get_campaign", {"campaign_id": "C102"}),
            decision("finalizer", completed=True),
        ],
        synthesized_answer="Recovered [BUSINESS:C102:get_campaign].",
    )
    graph = WorkbenchGraph(
        model_client=model,
        registry=registry,
        memory_store=runtime["memory"],
        settings=runtime["settings"],
    ).build(checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        state(),
        {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 80},
    )
    names = [event["event_type"] for event in result["execution_trace"]]
    assert flaky.calls == 2
    assert "tool_failed" in names
    assert names.index("tool_failed") < names.index("tool_completed")
    assert result["completed"] is True


@pytest.mark.asyncio
async def test_research_tool_contract_makes_exactly_one_request():
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "result": [
                    [
                        {
                            "document": {
                                "id": "doc-1",
                                "contents": "Campaign context\nCampaign ROI can decline when conversion falls.",
                            },
                            "score": 1.0,
                        }
                    ]
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = ResearchSearchTool(
            "http://retriever/retrieve",
            Phase5EvidenceAdapter(DeterministicTextTokenizer()),
            client=client,
        )
        output = await tool.search({"query": "campaign ROI decline", "top_k": 1})
    assert request_count == 1
    assert output.failure_status is None
    assert output.compressed_evidence
    assert output.sources[0].source_id == "RESEARCH-doc-1"


def test_research_substate_contract_is_explicit():
    annotations = ResearchState.__annotations__
    assert set(annotations) == {
        "query",
        "arguments",
        "context",
        "tool_result",
        "sources",
        "failure_status",
    }


def test_shared_state_is_primitive_and_complete():
    value = state()
    assert value["messages"] == [{"role": "user", "content": value["task"]}]
    assert value["step_count"] == value["tool_call_count"] == 0
    assert value["pending_action"] is None
    assert value["completed"] is False
