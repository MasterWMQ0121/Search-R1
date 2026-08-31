from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from apps.enterprise_agent_workbench.context_budget import (
    ContextBudgetConfig,
    ContextBudgetManager,
)
from apps.enterprise_agent_workbench.graph import WorkbenchGraph
from apps.enterprise_agent_workbench.model_client import (
    FakeModelClient,
    PlannerDecision,
)
from apps.enterprise_agent_workbench.state import initial_agent_state
from apps.enterprise_agent_workbench.tool_registry import ToolRegistry
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
)


def _decision(
    action: str, arguments: dict[str, Any] | None = None, *, completed: bool = False
) -> PlannerDecision:
    return PlannerDecision(
        objective="Complete the requested task safely.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="Use the validated tool result.",
    )


def _state(*, role: str = "viewer") -> dict[str, Any]:
    return initial_agent_state(
        thread_id="routing-thread",
        run_id="routing-run",
        user_id="routing-user",
        organization_id="org-1",
        tenant_id="org-1",
        role=role,
        task="Show campaign C102 and explain its current budget.",
    )


class _CapturingModelClient(FakeModelClient):
    def __init__(self, decisions: list[PlannerDecision], *, answer: str) -> None:
        super().__init__(decisions, synthesized_answer=answer)
        self.replanner_contexts: list[dict[str, Any]] = []

    async def replan(self, task: str, state_summary: dict[str, Any]) -> PlannerDecision:
        self.replanner_contexts.append(
            {
                "prior_tool_results": deepcopy(
                    state_summary.get("prior_tool_results", [])
                )
            }
        )
        return await super().replan(task, state_summary)


@pytest.mark.asyncio
async def test_compressed_replanner_context_retains_validated_business_arguments(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="viewer")
    model = _CapturingModelClient(
        [
            _decision("get_campaign", {"campaign_id": "C102"}),
            _decision("finalizer", completed=True),
        ],
        answer="C102 has a recorded budget [BUSINESS:C102:get_campaign].",
    )
    budget_manager = ContextBudgetManager(
        DeterministicTextTokenizer(),
        replace(ContextBudgetConfig(), tool_results_budget=128),
    )
    graph = WorkbenchGraph(
        model_client=model,
        registry=runtime["registry"],
        memory_store=runtime["memory"],
        settings=runtime["settings"],
        context_budget_manager=budget_manager,
    ).build(checkpointer=InMemorySaver())

    result = await graph.ainvoke(
        _state(),
        {
            "configurable": {"thread_id": "routing-thread"},
            "recursion_limit": 80,
        },
    )

    stored_result = result["tool_results"][0]
    assert set(stored_result) >= {
        "tool_name",
        "status",
        "arguments",
        "output",
        "source_ids",
    }
    assert stored_result["arguments"] == {"campaign_id": "C102"}

    assert len(model.replanner_contexts) == 1
    replanner_result = model.replanner_contexts[0]["prior_tool_results"][0]
    assert replanner_result["tool_name"] == "get_campaign"
    assert replanner_result["arguments"] == {"campaign_id": "C102"}
    tool_budget = next(
        item
        for item in result["context_budget"]["decisions"]
        if item["component"] == "tool_results"
    )
    assert tool_budget["action"] == "compress"


@pytest.mark.asyncio
async def test_approved_write_routing_arguments_exclude_control_plane_idempotency(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="operator")
    workbench = WorkbenchGraph(
        model_client=runtime["model"],
        registry=runtime["registry"],
        memory_store=runtime["memory"],
        settings=runtime["settings"],
    )
    state = _state(role="operator")
    state.update(
        {
            "pending_action": {
                "action": "update_campaign_budget",
                "arguments": {
                    "campaign_id": "C102",
                    "daily_budget": 1200,
                    "idempotency_key": "routing-envelope-key-0001",
                },
                "reason": "User requested the budget update.",
            },
            "approval_decision": {"decision": "approve"},
        }
    )

    update = await workbench.execute_business_write(state)

    tool_result = update["tool_results"][0]
    assert tool_result["status"] == "ok"
    assert tool_result["arguments"] == {
        "campaign_id": "C102",
        "daily_budget": 1200.0,
    }
    assert "idempotency_key" not in tool_result["arguments"]


@pytest.mark.asyncio
async def test_research_success_envelope_uses_gateway_validated_arguments(
    graph_factory,
):
    runtime = graph_factory([], "unused", role="viewer")
    received_arguments: list[dict[str, Any]] = []

    async def research_handler(request):
        received_arguments.append(request.model_dump(mode="json"))
        return {
            "query": request.query,
            "compressed_evidence": "C102 external evidence.",
            "documents": [
                {
                    "source_id": "RESEARCH-C102",
                    "rank": 1,
                    "title": "C102 Evidence",
                    "document_id": "c102-doc",
                    "score": 1.0,
                }
            ],
            "sources": [
                {
                    "source_id": "RESEARCH-C102",
                    "source_type": "external",
                    "title": "C102 Evidence",
                    "snippet": "C102 external evidence.",
                    "score": 1.0,
                }
            ],
            "retrieval_latency_s": 0.0,
            "compression_latency_s": 0.0,
            "failure_status": None,
            "compression_metrics": {},
        }

    registry = ToolRegistry(
        [
            (
                replace(spec, handler=research_handler)
                if spec.name == "research_search"
                else spec
            )
            for metadata in runtime["registry"].safe_metadata()
            for spec in [runtime["registry"].get(metadata["name"])]
        ]
    )
    workbench = WorkbenchGraph(
        model_client=runtime["model"],
        registry=registry,
        memory_store=runtime["memory"],
        settings=runtime["settings"],
    )
    state = _state()
    state.update(
        {
            "next_action": "research_search",
            "action_arguments": {"query": "  C102 external evidence  "},
        }
    )

    update = await workbench.research_search(state)

    assert received_arguments == [{"query": "C102 external evidence", "top_k": 3}]
    tool_result = update["tool_results"][0]
    assert tool_result["status"] == "ok"
    assert tool_result["arguments"] == received_arguments[0]
    assert tool_result["source_ids"] == ["RESEARCH-C102"]
