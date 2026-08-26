"""Primitive, checkpoint-safe state contract for the workbench graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


Role = Literal["viewer", "analyst", "operator", "admin"]


class AgentState(TypedDict, total=False):
    """State shared by the explicit LangGraph workflow.

    Node outputs are partial dictionaries.  Only fields that represent event
    streams use reducers; plans, counters, approval state, and final output are
    replacements so replay cannot accidentally duplicate them.
    """

    thread_id: str
    run_id: str
    user_id: str
    organization_id: str
    role: Role
    runtime_configuration_fingerprint: str
    task: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    conversation_summary: str
    user_preferences: dict[str, Any]
    plan: list[dict[str, Any]]
    next_action: str | None
    action_arguments: dict[str, Any]
    pending_action: dict[str, Any] | None
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    sources: Annotated[list[dict[str, Any]], operator.add]
    execution_trace: Annotated[list[dict[str, Any]], operator.add]
    approval_request: dict[str, Any] | None
    approval_decision: dict[str, Any] | None
    step_count: int
    tool_call_count: int
    research_search_count: int
    planner_repair_count: int
    planner_failure_signatures: list[str]
    route: str | None
    authorization_route: str | None
    errors: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str | None
    final_citations: list[str]
    citation_coverage: float
    completed: bool
    termination_reason: str | None


def initial_agent_state(
    *,
    thread_id: str,
    run_id: str,
    user_id: str,
    organization_id: str,
    role: Role,
    task: str,
    user_preferences: dict[str, Any] | None = None,
    runtime_configuration_fingerprint: str = "",
) -> AgentState:
    """Return a complete initial state containing only checkpoint-safe values."""

    required = {
        "thread_id": thread_id,
        "run_id": run_id,
        "user_id": user_id,
        "organization_id": organization_id,
        "task": task,
    }
    empty = [name for name, value in required.items() if not value.strip()]
    if empty:
        raise ValueError(f"state identifiers/task must be non-empty: {', '.join(empty)}")

    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "user_id": user_id,
        "organization_id": organization_id,
        "role": role,
        "runtime_configuration_fingerprint": runtime_configuration_fingerprint,
        "task": task,
        "messages": [{"role": "user", "content": task}],
        "conversation_summary": "",
        "user_preferences": dict(user_preferences or {}),
        "plan": [],
        "next_action": None,
        "action_arguments": {},
        "pending_action": None,
        "tool_results": [],
        "sources": [],
        "execution_trace": [],
        "approval_request": None,
        "approval_decision": None,
        "step_count": 0,
        "tool_call_count": 0,
        "research_search_count": 0,
        "planner_repair_count": 0,
        "planner_failure_signatures": [],
        "route": None,
        "authorization_route": None,
        "errors": [],
        "final_answer": None,
        "final_citations": [],
        "citation_coverage": 0.0,
        "completed": False,
        "termination_reason": None,
    }
