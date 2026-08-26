"""Explicit LangGraph orchestration for the enterprise workbench."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import WorkbenchSettings
from .citations import citation_coverage, validate_citations
from .model_client import PlannerParseError, WorkbenchModelClient
from .policy import PolicyEngine
from .state import AgentState
from .tool_registry import ToolRegistry


READ_NODE_ALIASES = {
    "enterprise_kb_search": "enterprise_kb_search",
    "research_search": "research_subgraph",
    "merchant_analytics": "merchant_analytics",
    "campaign_read": "campaign_read",
}
FINAL_ACTIONS = {"finalizer", "finish", "complete", "final_answer"}
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})\]")


class ApprovalResume(BaseModel):
    """Validated human decision supplied through ``Command(resume=...)``."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "edit", "reject"]
    edited_arguments: dict[str, Any] | None = None
    feedback: str | None = Field(default=None, max_length=1_000)


class ResearchState(TypedDict, total=False):
    """Narrow parent/subgraph contract for exactly one research request."""

    query: str
    arguments: dict[str, Any]
    context: dict[str, Any]
    tool_result: dict[str, Any]
    sources: list[dict[str, Any]]
    failure_status: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_summary(value: Any, *, limit: int = 400) -> str:
    text = str(value).replace("\n", " ")
    for marker in ("api_key", "authorization", "password", "secret", "token"):
        text = re.sub(
            rf"(?i)({marker}\s*['\"=:]+)\S+", r"\1[REDACTED]", text
        )
    return text[:limit]


def _trace(
    state: AgentState,
    event_type: str,
    node: str,
    *,
    status: str | None = None,
    tool_name: str | None = None,
    duration_ms: float = 0.0,
    input_summary: Any = "",
    output_summary: Any = "",
    error_type: str | None = None,
    retry_number: int = 0,
) -> dict[str, Any]:
    if status is None:
        if event_type in {"run_started", "node_started", "tool_started"}:
            status = "started"
        elif event_type == "approval_requested":
            status = "paused"
        elif event_type in {"run_failed", "tool_failed"}:
            status = "failed"
        elif event_type == "policy_denied":
            status = "denied"
        else:
            status = "completed"
    return {
        "sequence": -1,
        "timestamp": _utc_now(),
        "thread_id": state.get("thread_id", ""),
        "run_id": state.get("run_id", ""),
        "event_type": event_type,
        "node": node,
        "tool_name": tool_name,
        "duration_ms": max(0.0, float(duration_ms)),
        "status": status,
        "safe_input_summary": {"summary": _safe_summary(input_summary)},
        "safe_output_summary": {"summary": _safe_summary(output_summary)},
        "error_type": error_type,
        "retry_number": retry_number,
    }


def _sources_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    if isinstance(output.get("sources"), list):
        candidates.extend(output["sources"])
    if isinstance(output.get("source"), dict):
        candidates.append(output["source"])
    result = []
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("source_id"), str):
            result.append(dict(candidate))
    return result


class WorkbenchGraph:
    """Build and run the bounded, inspectable workbench StateGraph."""

    def __init__(
        self,
        *,
        model_client: WorkbenchModelClient,
        registry: ToolRegistry,
        memory_store: Any,
        settings: WorkbenchSettings,
    ) -> None:
        self.model_client = model_client
        self.registry = registry
        self.memory_store = memory_store
        self.settings = settings
        self.policy = PolicyEngine(registry)
        self.research_graph = self._build_research_subgraph().compile()

    def _build_research_subgraph(self) -> StateGraph:
        graph = StateGraph(ResearchState)
        graph.add_node("retrieve_compress", self._research_once)
        graph.add_edge(START, "retrieve_compress")
        graph.add_edge("retrieve_compress", END)
        return graph

    def build(self, *, checkpointer: Any = None):
        graph = StateGraph(AgentState)
        graph.add_node("load_context", self.load_context)
        graph.add_node("planner", self.planner)
        graph.add_node("policy_and_route", self.policy_and_route)
        graph.add_node("enterprise_kb_search", self.enterprise_kb_search)
        graph.add_node("research_subgraph", self.research_search)
        graph.add_node("merchant_analytics", self.merchant_analytics)
        graph.add_node("campaign_read", self.campaign_read)
        graph.add_node("propose_business_write", self.propose_business_write)
        graph.add_node("authorization_check", self.authorization_check)
        graph.add_node("human_approval_interrupt", self.human_approval_interrupt)
        graph.add_node("execute_business_write", self.execute_business_write)
        graph.add_node("merge_evidence", self.merge_evidence)
        graph.add_node("memory_summary", self.memory_summary)
        graph.add_node("replanner", self.replanner)
        graph.add_node("finalizer", self.finalizer)
        graph.add_node("citation_validation", self.citation_validation)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "planner")
        graph.add_edge("planner", "policy_and_route")
        graph.add_conditional_edges(
            "policy_and_route",
            self.route_after_policy,
            {
                "enterprise_kb_search": "enterprise_kb_search",
                "research_subgraph": "research_subgraph",
                "merchant_analytics": "merchant_analytics",
                "campaign_read": "campaign_read",
                "propose_business_write": "propose_business_write",
                "merge_evidence": "merge_evidence",
                "finalizer": "finalizer",
            },
        )
        for node in (
            "enterprise_kb_search",
            "research_subgraph",
            "merchant_analytics",
            "campaign_read",
        ):
            graph.add_edge(node, "merge_evidence")
        graph.add_edge("propose_business_write", "authorization_check")
        graph.add_conditional_edges(
            "authorization_check",
            self.route_after_authorization,
            {
                "human_approval_interrupt": "human_approval_interrupt",
                "merge_evidence": "merge_evidence",
            },
        )
        graph.add_conditional_edges(
            "human_approval_interrupt",
            self.route_after_approval,
            {
                "execute_business_write": "execute_business_write",
                "merge_evidence": "merge_evidence",
            },
        )
        graph.add_edge("execute_business_write", "merge_evidence")
        graph.add_edge("merge_evidence", "memory_summary")
        graph.add_edge("memory_summary", "replanner")
        graph.add_edge("replanner", "policy_and_route")
        graph.add_edge("finalizer", "citation_validation")
        graph.add_edge("citation_validation", END)
        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _step(state: AgentState, *events: dict[str, Any]) -> dict[str, Any]:
        base_sequence = len(state.get("execution_trace", []))
        expanded_events: list[dict[str, Any]] = []
        if events:
            run_has_event = any(
                event.get("run_id") == state.get("run_id")
                for event in state.get("execution_trace", [])
                if isinstance(event, dict)
            )
            if not run_has_event:
                run_event = dict(events[0])
                run_event.update(
                    {
                        "event_type": "run_started",
                        "node": "graph",
                        "status": "started",
                        "duration_ms": 0.0,
                        "safe_input_summary": {"summary": "run accepted"},
                        "safe_output_summary": {"summary": ""},
                    }
                )
                expanded_events.append(run_event)
            if events[0].get("event_type") != "node_started":
                node_event = dict(events[0])
                node_event.update(
                    {
                        "event_type": "node_started",
                        "status": "started",
                        "duration_ms": 0.0,
                        "safe_output_summary": {"summary": ""},
                    }
                )
                expanded_events.append(node_event)
        expanded_events.extend(events)
        normalized_events = []
        for offset, event in enumerate(expanded_events):
            event = dict(event)
            event["sequence"] = base_sequence + offset
            normalized_events.append(event)
        return {
            "step_count": int(state.get("step_count", 0)) + 1,
            "execution_trace": normalized_events,
        }

    async def load_context(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        preferences: dict[str, Any] = {}
        getter = getattr(self.memory_store, "list_preferences", None)
        if getter is not None:
            value = getter(state["organization_id"], state["user_id"])
            preferences = await value if asyncio.iscoroutine(value) else value
        event = _trace(
            state,
            "node_completed",
            "load_context",
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary=f"loaded {len(preferences)} explicit preferences",
        )
        return {**self._step(state, event), "user_preferences": preferences}

    def _planner_context(self, state: AgentState) -> dict[str, Any]:
        return {
            "conversation_summary": state.get("conversation_summary", ""),
            "preferences": state.get("user_preferences", {}),
            "available_tools": self.registry.safe_metadata(),
            "prior_tool_results": state.get("tool_results", [])[-6:],
            "errors": state.get("errors", [])[-3:],
            "limits": {
                "remaining_steps": max(
                    0,
                    self.settings.max_graph_steps - int(state.get("step_count", 0)),
                ),
                "remaining_tools": max(
                    0,
                    self.settings.max_tool_calls
                    - int(state.get("tool_call_count", 0)),
                ),
            },
        }

    async def _decide(self, state: AgentState, *, replan: bool) -> dict[str, Any]:
        node = "replanner" if replan else "planner"
        started = time.perf_counter()
        if hasattr(self.model_client, "repair_allowed"):
            self.model_client.repair_allowed = (
                int(state.get("planner_repair_count", 0))
                < self.settings.max_planner_repairs
            )
        try:
            if replan:
                decision = await self.model_client.replan(
                    state["task"], self._planner_context(state)
                )
            else:
                decision = await self.model_client.plan(
                    state["task"], self._planner_context(state)
                )
        except PlannerParseError as error:
            failure = {
                "code": "planner_parse_error",
                "message": str(error),
                "node": node,
            }
            event = _trace(
                state,
                "run_failed",
                node,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=type(error).__name__,
                output_summary=error,
                retry_number=1,
            )
            return {
                **self._step(state, event),
                "errors": [failure],
                "next_action": "finalizer",
                "action_arguments": {},
                "termination_reason": "planner_parse_error",
                "planner_repair_count": int(state.get("planner_repair_count", 0))
                + int(getattr(error, "repair_attempts", 0)),
            }
        except Exception as error:
            failure = {
                "code": "planner_backend_error",
                "message": _safe_summary(error),
                "node": node,
            }
            event = _trace(
                state,
                "run_failed",
                node,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=type(error).__name__,
                output_summary=error,
            )
            return {
                **self._step(state, event),
                "errors": [failure],
                "next_action": "finalizer",
                "action_arguments": {},
                "termination_reason": "planner_backend_error",
            }
        payload = decision.model_dump(mode="json")
        decision_event = _trace(
            state,
            "planner_decision",
            node,
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary={
                "next_action": decision.next_action,
                "completed": decision.completed,
                "reason": decision.user_visible_reason,
            },
        )
        repair_count = int(getattr(self.model_client, "last_planner_repairs", 0))
        events = []
        if repair_count:
            events.append(
                _trace(
                    state,
                    "planner_repair",
                    node,
                    output_summary="accepted one schema-repaired planner response",
                    retry_number=1,
                )
            )
        events.append(decision_event)
        return {
            **self._step(state, *events),
            "plan": [*state.get("plan", []), payload],
            "next_action": decision.next_action,
            "action_arguments": decision.arguments,
            "completed": bool(decision.completed),
            "planner_repair_count": int(state.get("planner_repair_count", 0))
            + repair_count,
        }

    async def planner(self, state: AgentState) -> dict[str, Any]:
        return await self._decide(state, replan=False)

    async def replanner(self, state: AgentState) -> dict[str, Any]:
        return await self._decide(state, replan=True)

    def _action_and_arguments(self, state: AgentState) -> tuple[str, dict[str, Any]]:
        action = str(state.get("next_action") or "")
        arguments = dict(state.get("action_arguments") or {})
        if action in {"merchant_analytics", "campaign_read", "propose_business_write"}:
            operation = arguments.pop("operation", None) or arguments.pop("action", None)
            if isinstance(operation, str) and operation:
                action = operation
        return action, arguments

    async def policy_and_route(self, state: AgentState) -> dict[str, Any]:
        if int(state.get("step_count", 0)) >= self.settings.max_graph_steps:
            event = _trace(
                state,
                "policy_denied",
                "policy_and_route",
                status="denied",
                output_summary="maximum graph-step guard reached",
            )
            return {
                **self._step(state, event),
                "route": "finalizer",
                "termination_reason": "maximum_graph_steps",
                "completed": False,
            }
        action, arguments = self._action_and_arguments(state)
        if state.get("completed") or action in FINAL_ACTIONS:
            event = _trace(
                state,
                "policy_allowed",
                "policy_and_route",
                output_summary="route to finalizer",
            )
            return {**self._step(state, event), "route": "finalizer"}

        decision = self.policy.evaluate(
            state.get("role", "viewer"),
            action,
            arguments,
            {
                "tool_call_count": int(state.get("tool_call_count", 0)),
                "max_tool_calls": self.settings.max_tool_calls,
                "research_search_count": int(state.get("research_search_count", 0)),
                "max_research_searches": self.settings.max_research_searches,
            },
        )
        event = _trace(
            state,
            decision["event"]["event_type"],
            "policy_and_route",
            status=("completed" if decision["allowed"] else "denied"),
            tool_name=action,
            input_summary=arguments,
            output_summary=decision["user_message"],
        )
        if not decision["allowed"]:
            denial = {
                "tool_name": action,
                "status": "denied",
                "policy_code": decision["decision_code"],
                "message": decision["user_message"],
                "sources": [],
            }
            return {
                **self._step(state, event),
                "route": "merge_evidence",
                "tool_results": [denial],
            }
        spec = self.registry.get(action)
        if spec.side_effecting:
            route = "propose_business_write"
        elif spec.category == "knowledge":
            route = "enterprise_kb_search"
        elif spec.category == "research":
            route = "research_subgraph"
        elif spec.category == "analytics":
            route = "merchant_analytics"
        elif spec.category == "business_read":
            route = "campaign_read"
        else:
            route = "merge_evidence"
        return {
            **self._step(state, event),
            "route": route,
            "next_action": action,
            "action_arguments": arguments,
        }

    @staticmethod
    def route_after_policy(state: AgentState) -> str:
        return str(state.get("route", "finalizer"))

    async def _execute_read(self, state: AgentState, node: str) -> dict[str, Any]:
        action, arguments = self._action_and_arguments(state)
        started = time.perf_counter()
        start_event = _trace(
            state,
            "tool_started",
            node,
            tool_name=action,
            input_summary=arguments,
        )
        try:
            output = await asyncio.wait_for(
                self.registry.execute(action, arguments),
                timeout=self.settings.tool_timeout_seconds,
            )
            sources = _sources_from_output(output)
            result = {
                "tool_name": action,
                "status": "ok",
                "output": output,
                "source_ids": [source["source_id"] for source in sources],
            }
            end_event = _trace(
                state,
                "tool_completed",
                node,
                tool_name=action,
                duration_ms=(time.perf_counter() - started) * 1_000,
                output_summary=f"{len(sources)} sources",
            )
            update = {
                **self._step(state, start_event, end_event),
                "tool_results": [result],
                "sources": sources,
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            }
            if self.registry.get(action).category == "research":
                update["research_search_count"] = (
                    int(state.get("research_search_count", 0)) + 1
                )
            return update
        except Exception as error:  # graph converts tool failures into safe evidence
            failure = {
                "tool_name": action,
                "status": "error",
                "error_type": type(error).__name__,
                "message": _safe_summary(error),
                "sources": [],
            }
            end_event = _trace(
                state,
                "tool_failed",
                node,
            status="failed",
                tool_name=action,
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=type(error).__name__,
                output_summary=error,
            )
            return {
                **self._step(state, start_event, end_event),
                "tool_results": [failure],
                "errors": [
                    {
                        "code": "tool_failed",
                        "tool_name": action,
                        "message": _safe_summary(error),
                    }
                ],
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            }

    async def enterprise_kb_search(self, state: AgentState) -> dict[str, Any]:
        return await self._execute_read(state, "enterprise_kb_search")

    async def merchant_analytics(self, state: AgentState) -> dict[str, Any]:
        return await self._execute_read(state, "merchant_analytics")

    async def campaign_read(self, state: AgentState) -> dict[str, Any]:
        return await self._execute_read(state, "campaign_read")

    async def _research_once(self, state: ResearchState) -> dict[str, Any]:
        try:
            output = await self.registry.execute(
                "research_search", state.get("arguments", {"query": state["query"]})
            )
            return {
                "tool_result": output,
                "sources": _sources_from_output(output),
                "failure_status": output.get("failure_status"),
            }
        except Exception as error:
            return {
                "tool_result": {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": _safe_summary(error),
                },
                "sources": [],
                "failure_status": type(error).__name__,
            }

    async def research_search(self, state: AgentState) -> dict[str, Any]:
        action, arguments = self._action_and_arguments(state)
        started = time.perf_counter()
        start_event = _trace(
            state,
            "tool_started",
            "research_subgraph",
            tool_name=action,
            input_summary=arguments,
        )
        substate = await self.research_graph.ainvoke(
            {
                "query": str(arguments.get("query") or state["task"]),
                "arguments": arguments,
                "context": {
                    "thread_id": state["thread_id"],
                    "run_id": state["run_id"],
                },
            }
        )
        failed = bool(substate.get("failure_status"))
        output = substate.get("tool_result", {})
        sources = list(substate.get("sources", []))
        result = {
            "tool_name": "research_search",
            "status": "error" if failed else "ok",
            "output": output,
            "source_ids": [source["source_id"] for source in sources],
        }
        end_event = _trace(
            state,
            "tool_failed" if failed else "tool_completed",
            "research_subgraph",
            status="failed" if failed else "completed",
            tool_name="research_search",
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary=(
                substate.get("failure_status") or f"{len(sources)} compressed sources"
            ),
        )
        update: dict[str, Any] = {
            **self._step(state, start_event, end_event),
            "tool_results": [result],
            "sources": sources,
            "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            "research_search_count": int(state.get("research_search_count", 0)) + 1,
        }
        if failed:
            update["errors"] = [
                {
                    "code": "research_failed",
                    "message": str(substate.get("failure_status")),
                }
            ]
        return update

    async def propose_business_write(self, state: AgentState) -> dict[str, Any]:
        action, arguments = self._action_and_arguments(state)
        pending = {
            "action": action,
            "arguments": arguments,
            "reason": (
                state.get("plan", [{}])[-1].get("user_visible_reason", "")
                if state.get("plan")
                else ""
            ),
        }
        event = _trace(
            state,
            "node_completed",
            "propose_business_write",
            tool_name=action,
            output_summary="write proposal created; no side effect executed",
        )
        return {**self._step(state, event), "pending_action": pending}

    def _policy_for_pending(self, state: AgentState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        return self.policy.evaluate(
            state.get("role", "viewer"),
            str(pending.get("action", "")),
            pending.get("arguments", {}),
            {
                "tool_call_count": int(state.get("tool_call_count", 0)),
                "max_tool_calls": self.settings.max_tool_calls,
                "research_search_count": int(state.get("research_search_count", 0)),
                "max_research_searches": self.settings.max_research_searches,
            },
        )

    async def authorization_check(self, state: AgentState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        decision = self._policy_for_pending(state)
        if not decision["allowed"]:
            result = {
                "tool_name": pending.get("action"),
                "status": "denied",
                "policy_code": decision["decision_code"],
                "message": decision["user_message"],
                "sources": [],
            }
            event = _trace(
                state,
                "policy_denied",
                "authorization_check",
                status="denied",
                tool_name=str(pending.get("action", "")),
                output_summary=decision["user_message"],
            )
            return {
                **self._step(state, event),
                "authorization_route": "merge_evidence",
                "tool_results": [result],
                "pending_action": None,
            }
        try:
            spec = self.registry.get(str(pending.get("action", "")))
            validated_arguments = spec.input_model.model_validate(
                pending.get("arguments", {})
            ).model_dump(mode="json")
        except (KeyError, ValidationError) as error:
            message = f"Write proposal arguments are invalid: {_safe_summary(error)}"
            event = _trace(
                state,
                "policy_denied",
                "authorization_check",
                status="denied",
                tool_name=str(pending.get("action", "")),
                output_summary=message,
            )
            return {
                **self._step(state, event),
                "authorization_route": "merge_evidence",
                "tool_results": [
                    {
                        "tool_name": pending.get("action"),
                        "status": "denied",
                        "policy_code": "invalid_arguments",
                        "message": message,
                        "sources": [],
                    }
                ],
                "errors": [
                    {"code": "invalid_arguments", "message": message}
                ],
                "pending_action": None,
            }
        pending = {**pending, "arguments": validated_arguments}
        before_state: dict[str, Any] = {}
        for result in reversed(state.get("tool_results", [])):
            if result.get("tool_name") in {"get_campaign", "campaign_current_state"}:
                before_state = dict(result.get("output") or {})
                break
        request = {
            "action": pending["action"],
            "arguments": pending.get("arguments", {}),
            "reason": pending.get("reason", "Business write requested."),
            "risk_level": spec.risk_level,
            "requesting_user": state["user_id"],
            "requesting_role": state["role"],
            "before_state": before_state,
            "allowed_decisions": ["approve", "edit", "reject"],
        }
        event = _trace(
            state,
            "approval_requested",
            "authorization_check",
            tool_name=pending["action"],
            output_summary="human approval required before any side effect",
        )
        return {
            **self._step(state, event),
            "authorization_route": "human_approval_interrupt",
            "approval_request": request,
            "pending_action": pending,
        }

    @staticmethod
    def route_after_authorization(state: AgentState) -> str:
        return str(state.get("authorization_route", "merge_evidence"))

    async def human_approval_interrupt(self, state: AgentState) -> dict[str, Any]:
        # This call is intentionally the first non-local operation in the node.
        # LangGraph restarts this node on resume; the business write is a later node.
        raw_decision = interrupt(dict(state["approval_request"] or {}))
        try:
            decision = ApprovalResume.model_validate(raw_decision)
        except ValidationError as error:
            raise ValueError(f"invalid approval decision: {error}") from error

        pending = dict(state.get("pending_action") or {})
        event_type = {
            "approve": "approval_approved",
            "edit": "approval_edited",
            "reject": "approval_rejected",
        }[decision.decision]
        if decision.decision == "edit":
            if decision.edited_arguments is None:
                raise ValueError("edit approval requires edited_arguments")
            spec = self.registry.get(str(pending.get("action", "")))
            validated = spec.input_model.model_validate(decision.edited_arguments)
            original_arguments = dict(pending.get("arguments") or {})
            edited_arguments = validated.model_dump(mode="json")
            for immutable_field in ("campaign_id", "idempotency_key"):
                if (
                    immutable_field in original_arguments
                    and edited_arguments.get(immutable_field)
                    != original_arguments.get(immutable_field)
                ):
                    raise ValueError(
                        f"approval edits cannot change {immutable_field}; "
                        "submit a new action for a different target or identity"
                    )
            policy_decision = self.policy.evaluate(
                state.get("role", "viewer"),
                str(pending.get("action", "")),
                edited_arguments,
                {
                    "tool_call_count": int(state.get("tool_call_count", 0)),
                    "max_tool_calls": self.settings.max_tool_calls,
                    "research_search_count": int(
                        state.get("research_search_count", 0)
                    ),
                    "max_research_searches": self.settings.max_research_searches,
                },
            )
            if not policy_decision["allowed"]:
                raise ValueError(
                    "edited approval arguments are not policy-authorized: "
                    f"{policy_decision['decision_code']}"
                )
            pending["arguments"] = edited_arguments
        event = _trace(
            state,
            event_type,
            "human_approval_interrupt",
            tool_name=str(pending.get("action", "")),
            output_summary=decision.decision,
        )
        return {
            **self._step(state, event),
            "approval_decision": decision.model_dump(mode="json"),
            "pending_action": pending,
        }

    @staticmethod
    def route_after_approval(state: AgentState) -> str:
        decision = (state.get("approval_decision") or {}).get("decision")
        return (
            "execute_business_write"
            if decision in {"approve", "edit"}
            else "merge_evidence"
        )

    async def execute_business_write(self, state: AgentState) -> dict[str, Any]:
        pending = dict(state.get("pending_action") or {})
        action = str(pending.get("action", ""))
        arguments = dict(pending.get("arguments") or {})
        started = time.perf_counter()
        start_event = _trace(
            state,
            "tool_started",
            "execute_business_write",
            tool_name=action,
            input_summary=arguments,
        )
        context = {
            "thread_id": state["thread_id"],
            "user_id": state["user_id"],
            "role": state["role"],
            "authorization_granted": True,
            "approval_decision": (state.get("approval_decision") or {}).get(
                "decision"
            ),
        }
        try:
            output = await self.registry.execute(action, arguments, context)
            result = {
                "tool_name": action,
                "status": "ok",
                "output": output,
                "source_ids": [],
            }
            end_event = _trace(
                state,
                "tool_completed",
                "execute_business_write",
                tool_name=action,
                duration_ms=(time.perf_counter() - started) * 1_000,
                output_summary="approved write committed transactionally",
            )
            return {
                **self._step(state, start_event, end_event),
                "tool_results": [result],
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
                "pending_action": None,
                "approval_request": None,
            }
        except Exception as error:
            event = _trace(
                state,
                "tool_failed",
                "execute_business_write",
                status="failed",
                tool_name=action,
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=type(error).__name__,
                output_summary=error,
            )
            return {
                **self._step(state, start_event, event),
                "tool_results": [
                    {
                        "tool_name": action,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "message": _safe_summary(error),
                    }
                ],
                "errors": [
                    {
                        "code": "business_write_failed",
                        "tool_name": action,
                        "message": _safe_summary(error),
                    }
                ],
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            }

    async def merge_evidence(self, state: AgentState) -> dict[str, Any]:
        decision = state.get("approval_decision") or {}
        additions: list[dict[str, Any]] = []
        if decision.get("decision") == "reject" and state.get("pending_action"):
            additions.append(
                {
                    "tool_name": state["pending_action"].get("action"),
                    "status": "rejected",
                    "message": decision.get("feedback") or "Human reviewer rejected action.",
                    "sources": [],
                }
            )
        event = _trace(
            state,
            "node_completed",
            "merge_evidence",
            output_summary=f"{len(state.get('tool_results', [])) + len(additions)} results",
        )
        update: dict[str, Any] = {**self._step(state, event)}
        if additions:
            update.update(
                {
                    "tool_results": additions,
                    "pending_action": None,
                    "approval_request": None,
                }
            )
        return update

    async def memory_summary(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if len(messages) <= self.settings.summary_message_threshold:
            return self._step(
                state,
                _trace(
                    state,
                    "node_completed",
                    "memory_summary",
                    output_summary="summary threshold not reached",
                ),
            )
        started = time.perf_counter()
        summary = await self.model_client.summarize_memory(messages)
        event = _trace(
            state,
            "node_completed",
            "memory_summary",
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary="conversation summary refreshed",
        )
        return {**self._step(state, event), "conversation_summary": summary}

    async def finalizer(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            answer = await self.model_client.synthesize(
                state["task"],
                [
                    *state.get("tool_results", []),
                    {"available_sources": state.get("sources", [])},
                ],
            )
            event = _trace(
                state,
                "final_answer_created",
                "finalizer",
                duration_ms=(time.perf_counter() - started) * 1_000,
                output_summary=f"answer characters={len(answer)}",
            )
            return {
                **self._step(state, event),
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }
        except Exception as error:
            answer = (
                "The workbench could not complete final synthesis. "
                "Available tool results remain visible in the execution record."
            )
            event = _trace(
                state,
                "run_failed",
                "finalizer",
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=type(error).__name__,
                output_summary=error,
            )
            return {
                **self._step(state, event),
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "errors": [
                    {"code": "finalizer_failed", "message": _safe_summary(error)}
                ],
                "termination_reason": "finalizer_failed",
            }

    async def citation_validation(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("final_answer") or ""
        sources_by_id: dict[str, dict[str, Any]] = {}
        for source in state.get("sources", []):
            source_id = source.get("source_id")
            if source_id and source_id not in sources_by_id:
                sources_by_id[str(source_id)] = source
        try:
            result = validate_citations(answer, list(sources_by_id.values()))
        except (TypeError, ValueError) as error:
            result = None
            validation_error = str(error)
        else:
            validation_error = "; ".join(result.errors)
        if result is None or not result.valid:
            event = _trace(
                state,
                "run_failed",
                "citation_validation",
                status="failed",
                error_type="UnknownCitationError",
                output_summary=validation_error,
            )
            return {
                **self._step(state, event),
                "final_citations": [] if result is None else result.cited_ids,
                "completed": False,
                "termination_reason": "citation_validation_failed",
                "errors": [
                    {
                        "code": "unknown_citation",
                        "unknown_source_ids": (
                            [] if result is None else result.unknown_ids
                        ),
                        "message": validation_error,
                    }
                ],
            }
        event = _trace(
            state,
            "run_completed",
            "citation_validation",
            output_summary=f"validated {len(result.cited_ids)} citation markers",
        )
        coverage = citation_coverage(answer)
        return {
            **self._step(state, event),
            "final_citations": result.cited_ids,
            "citation_coverage": float(coverage["coverage"]),
            "completed": state.get("termination_reason") in {None, "completed"},
            "termination_reason": state.get("termination_reason") or "completed",
        }


__all__ = [
    "ApprovalResume",
    "ResearchState",
    "WorkbenchGraph",
]
