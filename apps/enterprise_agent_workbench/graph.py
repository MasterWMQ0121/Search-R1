"""Explicit LangGraph orchestration for the enterprise workbench."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import WorkbenchSettings
from .citations import citation_coverage, validate_citations
from .context_budget import ContextBudgetConfig, ContextBudgetManager
from .model_client import (
    PlannerDecision,
    PlannerParseError,
    PlannerSemanticValidationError,
    WorkbenchModelClient,
    validate_planner_decision,
)
from .observability import RuntimeObservability
from .state import AgentState
from .tool_gateway import ToolGateway, ToolInvocationContext
from .tool_registry import ToolRegistry
from .tools.research_search import DeterministicTextTokenizer


READ_NODE_ALIASES = {
    "enterprise_kb_search": "enterprise_kb_search",
    "research_search": "research_subgraph",
    "merchant_analytics": "merchant_analytics",
    "campaign_read": "campaign_read",
}
FINAL_ACTIONS = {"finalizer"}
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})\]")
PLANNER_FAILURE_ANSWER = (
    "The workbench stopped safely because the planner could not select a valid "
    "permitted action. No business action was executed."
)
_ROUTING_CONTROL_ARGUMENT_KEYS = frozenset(
    {
        "approval_decision",
        "authorization",
        "authorization_granted",
        "idempotency_key",
        "organization_id",
        "requesting_role",
        "requesting_user",
        "role",
        "run_id",
        "tenant_id",
        "thread_id",
        "user_id",
    }
)
_ROUTING_SECRET_ARGUMENT_KEYS = frozenset(
    {
        "api_key",
        "authorization_header",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


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


def deterministic_idempotency_key(
    *,
    organization_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str,
    thread_id: str,
    run_id: str,
    action: str,
    business_arguments: dict[str, Any],
) -> str:
    """Bind one control-plane write identity to a run and canonical arguments."""

    canonical_tenant = (tenant_id or organization_id or "").strip()
    if not canonical_tenant:
        raise ValueError("tenant_id is required for idempotency")
    if organization_id and organization_id.strip() != canonical_tenant:
        raise ValueError("tenant_id and organization_id must match")
    payload = {
        "schema_version": 1,
        "tenant_id": canonical_tenant,
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "action": action,
        "business_arguments": business_arguments,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "wb-" + hashlib.sha256(canonical).hexdigest()


def _failure_signature(code: str, action: str) -> str:
    normalized = " ".join(str(action).split()).casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"{code}:{digest}"


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
        "tenant_id": state.get("tenant_id", state.get("organization_id", "")),
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


def _safe_routing_arguments(value: Any) -> Any:
    """Remove control-plane and secret fields from Replanner-visible arguments."""

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            is_secret = normalized in _ROUTING_SECRET_ARGUMENT_KEYS or any(
                normalized.endswith(suffix)
                for suffix in (
                    "_api_key",
                    "_credential",
                    "_credentials",
                    "_password",
                    "_secret",
                    "_token",
                )
            )
            if normalized in _ROUTING_CONTROL_ARGUMENT_KEYS or is_secret:
                continue
            safe[str(key)] = _safe_routing_arguments(item)
        return safe
    if isinstance(value, list):
        return [_safe_routing_arguments(item) for item in value]
    return value


class WorkbenchGraph:
    """Build and run the bounded, inspectable workbench StateGraph."""

    def __init__(
        self,
        *,
        model_client: WorkbenchModelClient,
        registry: ToolRegistry,
        memory_store: Any,
        settings: WorkbenchSettings,
        tool_gateway: ToolGateway | None = None,
        context_budget_manager: ContextBudgetManager | None = None,
        observability: RuntimeObservability | None = None,
    ) -> None:
        self.model_client = model_client
        self.registry = registry
        self.memory_store = memory_store
        self.settings = settings
        self.observability = observability or RuntimeObservability()
        self.tool_gateway = tool_gateway or ToolGateway(
            registry, observability=self.observability
        )
        # Compatibility handle for existing policy inspection/tests. Gateway
        # policy evaluation delegates to this same engine instance.
        self.policy = self.tool_gateway.policy_engine
        self.context_budget_manager = context_budget_manager or ContextBudgetManager(
            DeterministicTextTokenizer(),
            ContextBudgetConfig(
                total_context=settings.context_total_tokens,
                generation_reserve=settings.generation_reserve_tokens,
                safety_margin=settings.context_safety_margin_tokens,
                system_budget=settings.system_context_budget,
                tool_catalog_budget=settings.tool_catalog_context_budget,
                memory_budget=settings.memory_context_budget,
                tool_results_budget=settings.tool_results_context_budget,
                current_task_budget=settings.current_task_context_budget,
            ),
        )
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
            value = getter(state["tenant_id"], state["user_id"])
            preferences = await value if asyncio.iscoroutine(value) else value
        event = _trace(
            state,
            "node_completed",
            "load_context",
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary=f"loaded {len(preferences)} explicit preferences",
        )
        return {**self._step(state, event), "user_preferences": preferences}

    def _validate_planner_tool_contract(
        self, tenant_id: str, role: str, decision: PlannerDecision
    ) -> PlannerDecision:
        if decision.next_action == "finalizer":
            return decision
        arguments = self.tool_gateway.validate_planner_arguments(
            tenant_id, role, decision.next_action, decision.arguments
        )
        return decision.model_copy(update={"arguments": arguments})

    def _validated_invocation_arguments(
        self, action: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize a native invocation with the exact registered ToolSpec schema."""

        return (
            self.registry.get(action)
            .input_model.model_validate(arguments)
            .model_dump(mode="json")
        )

    @staticmethod
    def _successful_tool_result(
        *,
        action: str,
        validated_arguments: dict[str, Any],
        output: dict[str, Any],
        source_ids: list[str],
    ) -> dict[str, Any]:
        """Build the stable, safe routing envelope consumed by the Replanner."""

        return {
            "tool_name": action,
            "status": "ok",
            "arguments": _safe_routing_arguments(validated_arguments),
            "output": output,
            "source_ids": list(source_ids),
        }

    def _planner_context(
        self, state: AgentState
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        role = state.get("role", "viewer")
        tenant_id = state.get("tenant_id", state.get("organization_id", ""))
        limits = {
            "remaining_steps": max(
                0,
                self.settings.max_graph_steps - int(state.get("step_count", 0)),
            ),
            "remaining_tools": max(
                0,
                self.settings.max_tool_calls
                - int(state.get("tool_call_count", 0)),
            ),
        }
        task, context, report = self.context_budget_manager.build_planner_context(
            task=state["task"],
            conversation_summary=state.get("conversation_summary", ""),
            preferences=state.get("user_preferences", {}),
            tool_catalog=self.tool_gateway.planner_metadata(tenant_id, role),
            tool_results=state.get("tool_results", [])[-6:],
            errors=state.get("errors", [])[-3:],
            limits=limits,
        )
        context["_planner_decision_validator"] = (
            lambda decision: self._validate_planner_tool_contract(
                tenant_id, role, decision
            )
        )
        report_dict = report.as_dict()
        labels = {"tenant_id": tenant_id}
        self.observability.observe(
            "context_tokens", report.context_tokens, **labels
        )
        self.observability.counter(
            "compressed_tokens", report.compressed_tokens, **labels
        )
        self.observability.observe(
            "budget_headroom", report.budget_headroom, **labels
        )
        self.observability.counter("input_tokens", report.context_tokens, **labels)
        return task, context, report_dict

    async def _call_planner_model(
        self,
        state: AgentState,
        *,
        node: str,
        replan: bool,
        task: str,
        planner_context: dict[str, Any],
    ) -> PlannerDecision:
        tenant_id = state.get("tenant_id", state.get("organization_id", ""))
        started = time.perf_counter()
        try:
            with self.observability.span(
                "Replanner" if replan else "Planner",
                {
                    "tenant.id": tenant_id,
                    "thread.id": state.get("thread_id", ""),
                    "run.id": state.get("run_id", ""),
                },
            ):
                with self.observability.span(
                    "LLM/Replanner" if replan else "LLM/Planner"
                ):
                    if replan:
                        return await self.model_client.replan(task, planner_context)
                    return await self.model_client.plan(task, planner_context)
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            self.observability.observe(
                "replanner.latency" if replan else "planner.latency",
                elapsed,
                tenant_id=tenant_id,
            )
            self.observability.observe(
                "llm.latency", elapsed, tenant_id=tenant_id, operation=node
            )

    async def _decide(self, state: AgentState, *, replan: bool) -> dict[str, Any]:
        node = "replanner" if replan else "planner"
        started = time.perf_counter()
        planner_task, planner_context, budget_report = self._planner_context(state)
        if hasattr(self.model_client, "repair_allowed"):
            self.model_client.repair_allowed = (
                int(state.get("planner_repair_count", 0))
                < self.settings.max_planner_repairs
            )
        try:
            decision = await self._call_planner_model(
                state,
                node=node,
                replan=replan,
                task=planner_task,
                planner_context=planner_context,
            )
        except PlannerParseError as error:
            repair_attempts = int(getattr(error, "repair_attempts", 0))
            if repair_attempts:
                self.observability.counter(
                    "planner.repairs",
                    repair_attempts,
                    tenant_id=state["tenant_id"],
                    operation=node,
                )
            failure_signatures = list(
                getattr(self.model_client, "last_planner_failure_signatures", [])
            )
            prior_signatures = list(state.get("planner_failure_signatures", []))
            error_code = str(getattr(error, "code", "planner_parse_error"))
            if error_code in {"planner_semantic_error", "planner_stuck"} and (
                error_code == "planner_stuck"
                or any(signature in prior_signatures for signature in failure_signatures)
            ):
                error_code = "planner_stuck"
            failure = {
                "code": error_code,
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
                "context_budget": budget_report,
                "errors": [failure],
                "next_action": "finalizer",
                "action_arguments": {},
                "completed": False,
                "termination_reason": error_code,
                "planner_repair_count": int(state.get("planner_repair_count", 0))
                + repair_attempts,
                "planner_failure_signatures": [
                    *prior_signatures,
                    *failure_signatures,
                ],
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
                "context_budget": budget_report,
                "errors": [failure],
                "next_action": "finalizer",
                "action_arguments": {},
                "termination_reason": "planner_backend_error",
            }
        failure_signatures = list(
            getattr(self.model_client, "last_planner_failure_signatures", [])
        )
        prior_signatures = list(state.get("planner_failure_signatures", []))
        if any(signature in prior_signatures for signature in failure_signatures):
            error = PlannerSemanticValidationError(
                "the same Planner semantic failure repeated in this run",
                signature=next(
                    signature
                    for signature in failure_signatures
                    if signature in prior_signatures
                ),
            )
            failure = {
                "code": "planner_stuck",
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
                retry_number=int(
                    getattr(self.model_client, "last_planner_repairs", 0)
                ),
            )
            return {
                **self._step(state, event),
                "context_budget": budget_report,
                "errors": [failure],
                "next_action": "finalizer",
                "action_arguments": {},
                "completed": False,
                "termination_reason": "planner_stuck",
                "planner_repair_count": int(state.get("planner_repair_count", 0))
                + int(getattr(self.model_client, "last_planner_repairs", 0)),
                "planner_failure_signatures": [
                    *prior_signatures,
                    *failure_signatures,
                ],
            }

        try:
            # Defense in depth for test/deterministic clients as well as the live
            # HTTP client, which already performs this validation before repair.
            decision = validate_planner_decision(decision, planner_context)
            action = decision.next_action
            if action == "finalizer":
                action_arguments: dict[str, Any] = {}
            else:
                business_arguments = self.tool_gateway.validate_planner_arguments(
                    state["tenant_id"],
                    state.get("role", "viewer"),
                    action,
                    decision.arguments,
                )
                spec = self.registry.get(action)
                if spec.side_effecting:
                    idempotency_key = deterministic_idempotency_key(
                        tenant_id=state["tenant_id"],
                        user_id=state["user_id"],
                        thread_id=state["thread_id"],
                        run_id=state["run_id"],
                        action=action,
                        business_arguments=business_arguments,
                    )
                    action_arguments = spec.input_model.model_validate(
                        {
                            **business_arguments,
                            "idempotency_key": idempotency_key,
                        }
                    ).model_dump(mode="json")
                else:
                    action_arguments = business_arguments
        except (PlannerSemanticValidationError, ValueError, ValidationError) as error:
            signature = getattr(
                error,
                "signature",
                _failure_signature("planner_semantic_error", decision.next_action),
            )
            error_code = (
                "planner_stuck" if signature in prior_signatures else "planner_semantic_error"
            )
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
                "context_budget": budget_report,
                "errors": [
                    {"code": error_code, "message": str(error), "node": node}
                ],
                "next_action": "finalizer",
                "action_arguments": {},
                "completed": False,
                "termination_reason": error_code,
                "planner_failure_signatures": [
                    *prior_signatures,
                    *failure_signatures,
                    signature,
                ],
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
        self.observability.counter(
            "output_tokens",
            self.context_budget_manager.token_count(payload),
            tenant_id=state["tenant_id"],
            operation=node,
        )
        if repair_count:
            self.observability.counter(
                "planner.repairs",
                repair_count,
                tenant_id=state["tenant_id"],
                operation=node,
            )
        events = []
        if repair_count:
            events.append(
                _trace(
                    state,
                    "planner_repair",
                    node,
                    output_summary=(
                        "accepted one schema/semantic-repaired planner response"
                    ),
                    retry_number=1,
                )
            )
        events.append(decision_event)
        return {
            **self._step(state, *events),
            "context_budget": budget_report,
            "plan": [*state.get("plan", []), payload],
            "next_action": decision.next_action,
            "action_arguments": action_arguments,
            "completed": bool(decision.completed),
            "planner_repair_count": int(state.get("planner_repair_count", 0))
            + repair_count,
            "planner_failure_signatures": [
                *prior_signatures,
                *failure_signatures,
            ],
        }

    async def planner(self, state: AgentState) -> dict[str, Any]:
        return await self._decide(state, replan=False)

    async def replanner(self, state: AgentState) -> dict[str, Any]:
        return await self._decide(state, replan=True)

    def _action_and_arguments(self, state: AgentState) -> tuple[str, dict[str, Any]]:
        action = str(state.get("next_action") or "")
        arguments = dict(state.get("action_arguments") or {})
        return action, arguments

    @staticmethod
    def _tool_context(
        state: AgentState, *, authorized: bool = False
    ) -> ToolInvocationContext:
        approval_decision = (state.get("approval_decision") or {}).get("decision")
        return ToolInvocationContext(
            tenant_id=state["tenant_id"],
            user_id=state["user_id"],
            role=state.get("role", "viewer"),
            thread_id=state["thread_id"],
            run_id=state["run_id"],
            authorization_granted=authorized,
            approval_decision=(str(approval_decision) if approval_decision else None),
        )

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

        decision = self.tool_gateway.evaluate_policy(
            tenant_id=state["tenant_id"],
            role=state.get("role", "viewer"),
            tool_name=action,
            arguments=arguments,
            state_counts={
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
            signature = _failure_signature(decision["decision_code"], action)
            prior_signatures = list(state.get("planner_failure_signatures", []))
            denial = {
                "tool_name": action,
                "status": "denied",
                "policy_code": decision["decision_code"],
                "message": decision["user_message"],
                "sources": [],
            }
            if signature in prior_signatures:
                stuck_event = _trace(
                    state,
                    "run_failed",
                    "policy_and_route",
                    status="failed",
                    tool_name=action,
                    error_type="PlannerStuckError",
                    output_summary=(
                        "identical policy denial repeated; stopped before replanning"
                    ),
                )
                return {
                    **self._step(state, event, stuck_event),
                    "route": "finalizer",
                    "tool_results": [denial],
                    "errors": [
                        {
                            "code": "planner_stuck",
                            "tool_name": action,
                            "message": (
                                "The same policy-denied action repeated in this run."
                            ),
                        }
                    ],
                    "completed": False,
                    "termination_reason": "planner_stuck",
                    "planner_failure_signatures": prior_signatures,
                }
            return {
                **self._step(state, event),
                "route": "merge_evidence",
                "tool_results": [denial],
                "planner_failure_signatures": [*prior_signatures, signature],
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
            validated_arguments = self._validated_invocation_arguments(
                action, arguments
            )
            output = await self.tool_gateway.execute(
                action,
                validated_arguments,
                context=self._tool_context(state),
                state_counts={
                    "tool_call_count": int(state.get("tool_call_count", 0)),
                    "max_tool_calls": self.settings.max_tool_calls,
                    "research_search_count": int(
                        state.get("research_search_count", 0)
                    ),
                    "max_research_searches": self.settings.max_research_searches,
                },
            )
            sources = _sources_from_output(output)
            result = self._successful_tool_result(
                action=action,
                validated_arguments=validated_arguments,
                output=output,
                source_ids=[source["source_id"] for source in sources],
            )
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
            invocation = ToolInvocationContext(
                tenant_id=str(state["context"]["tenant_id"]),
                user_id=str(state["context"]["user_id"]),
                role=str(state["context"]["role"]),
                thread_id=str(state["context"]["thread_id"]),
                run_id=str(state["context"]["run_id"]),
            )
            output = await self.tool_gateway.execute(
                "research_search",
                state.get("arguments", {"query": state["query"]}),
                context=invocation,
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
        try:
            validated_arguments = self._validated_invocation_arguments(
                action, arguments
            )
        except (KeyError, ValidationError) as error:
            # Preserve the existing fail-safe error result without exposing an
            # unvalidated argument object to the Replanner.
            substate = {
                "tool_result": {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": _safe_summary(error),
                },
                "sources": [],
                "failure_status": type(error).__name__,
            }
        else:
            substate = await self.research_graph.ainvoke(
                {
                    "query": str(validated_arguments.get("query") or state["task"]),
                    "arguments": validated_arguments,
                    "context": {
                        "tenant_id": state["tenant_id"],
                        "user_id": state["user_id"],
                        "role": state["role"],
                        "thread_id": state["thread_id"],
                        "run_id": state["run_id"],
                    },
                }
            )
        failed = bool(substate.get("failure_status"))
        output = substate.get("tool_result", {})
        sources = list(substate.get("sources", []))
        result = (
            {
                "tool_name": action,
                "status": "error",
                "output": output,
                "source_ids": [source["source_id"] for source in sources],
            }
            if failed
            else self._successful_tool_result(
                action=action,
                validated_arguments=validated_arguments,
                output=output,
                source_ids=[source["source_id"] for source in sources],
            )
        )
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
        return self.tool_gateway.evaluate_policy(
            tenant_id=state["tenant_id"],
            role=state.get("role", "viewer"),
            tool_name=str(pending.get("action", "")),
            arguments=pending.get("arguments", {}),
            state_counts={
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
            "approval_requested_at": _utc_now(),
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

        wait_seconds = 0.0
        requested_at = state.get("approval_requested_at")
        if requested_at:
            try:
                started_at = datetime.fromisoformat(str(requested_at))
                wait_seconds = max(
                    0.0, (datetime.now(timezone.utc) - started_at).total_seconds()
                )
            except ValueError:
                wait_seconds = 0.0
        self.observability.observe(
            "hitl.wait_time", wait_seconds, tenant_id=state["tenant_id"]
        )

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
            policy_decision = self.tool_gateway.evaluate_policy(
                tenant_id=state["tenant_id"],
                role=state.get("role", "viewer"),
                tool_name=str(pending.get("action", "")),
                arguments=edited_arguments,
                state_counts={
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
            "approval_requested_at": None,
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
        approval_decision = (state.get("approval_decision") or {}).get("decision")
        if approval_decision not in {"approve", "edit"}:
            error = PermissionError(
                "business write execution requires a recorded approve/edit decision"
            )
            event = _trace(
                state,
                "run_failed",
                "execute_business_write",
                status="failed",
                tool_name=action,
                error_type=type(error).__name__,
                output_summary=error,
            )
            return {
                **self._step(state, event),
                "errors": [
                    {
                        "code": "business_write_not_approved",
                        "tool_name": action,
                        "message": str(error),
                    }
                ],
                "completed": False,
                "termination_reason": "business_write_not_approved",
            }
        started = time.perf_counter()
        start_event = _trace(
            state,
            "tool_started",
            "execute_business_write",
            tool_name=action,
            input_summary=arguments,
        )
        try:
            validated_arguments = self._validated_invocation_arguments(
                action, arguments
            )
            output = await self.tool_gateway.execute(
                action,
                validated_arguments,
                context=self._tool_context(state, authorized=True),
                state_counts={
                    "tool_call_count": int(state.get("tool_call_count", 0)),
                    "max_tool_calls": self.settings.max_tool_calls,
                    "research_search_count": int(
                        state.get("research_search_count", 0)
                    ),
                    "max_research_searches": self.settings.max_research_searches,
                },
            )
            result = self._successful_tool_result(
                action=action,
                validated_arguments=validated_arguments,
                output=output,
                source_ids=[],
            )
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
        with self.observability.span("MemorySummary"):
            with self.observability.span("LLM/MemorySummary"):
                summary = await self.model_client.summarize_memory(messages)
        elapsed = max(0.0, time.perf_counter() - started)
        self.observability.observe(
            "llm.latency",
            elapsed,
            tenant_id=state["tenant_id"],
            operation="memory_summary",
        )
        self.observability.counter(
            "input_tokens",
            self.context_budget_manager.token_count(messages),
            tenant_id=state["tenant_id"],
            operation="memory_summary",
        )
        self.observability.counter(
            "output_tokens",
            self.context_budget_manager.token_count(summary),
            tenant_id=state["tenant_id"],
            operation="memory_summary",
        )
        event = _trace(
            state,
            "node_completed",
            "memory_summary",
            duration_ms=(time.perf_counter() - started) * 1_000,
            output_summary="conversation summary refreshed",
        )
        return {**self._step(state, event), "conversation_summary": summary}

    async def finalizer(self, state: AgentState) -> dict[str, Any]:
        with self.observability.span(
            "Finalizer",
            {
                "tenant.id": state.get("tenant_id", ""),
                "thread.id": state.get("thread_id", ""),
                "run.id": state.get("run_id", ""),
            },
        ):
            return await self._finalize(state)

    async def _finalize(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if state.get("termination_reason") in {
            "planner_semantic_error",
            "planner_stuck",
        }:
            answer = PLANNER_FAILURE_ANSWER
            event = _trace(
                state,
                "run_failed",
                "finalizer",
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1_000,
                error_type=str(state.get("termination_reason")),
                output_summary="deterministic local planner failure answer created",
            )
            return {
                **self._step(state, event),
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "completed": False,
            }
        try:
            evidence = [
                *state.get("tool_results", []),
                {"available_sources": state.get("sources", [])},
            ]
            with self.observability.span("LLM/Finalizer"):
                answer = await self.model_client.synthesize(state["task"], evidence)
            elapsed = max(0.0, time.perf_counter() - started)
            self.observability.observe(
                "llm.latency",
                elapsed,
                tenant_id=state["tenant_id"],
                operation="finalizer",
            )
            self.observability.counter(
                "input_tokens",
                self.context_budget_manager.token_count(
                    {"task": state["task"], "evidence": evidence}
                ),
                tenant_id=state["tenant_id"],
                operation="finalizer",
            )
            self.observability.counter(
                "output_tokens",
                self.context_budget_manager.token_count(answer),
                tenant_id=state["tenant_id"],
                operation="finalizer",
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
            self.observability.observe(
                "llm.latency",
                max(0.0, time.perf_counter() - started),
                tenant_id=state["tenant_id"],
                operation="finalizer",
            )
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
        terminal_failure = state.get("termination_reason") not in {None, "completed"}
        event = _trace(
            state,
            "run_failed" if terminal_failure else "run_completed",
            "citation_validation",
            status="failed" if terminal_failure else "completed",
            error_type=(
                str(state.get("termination_reason")) if terminal_failure else None
            ),
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
    "deterministic_idempotency_key",
]
