"""FastAPI and SSE boundary for the Enterprise Agent Workbench."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlsplit, urlunsplit

import aiosqlite
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .citations import SourceRecord
from .config import (
    ENTERPRISE_DOCS_ROOT,
    MERCHANT_SEED_PATH,
    WorkbenchSettings,
)
from .graph import WorkbenchGraph
from .memory import MemoryValidationError, PreferenceMemoryStore
from .model_client import VLLMHTTPModelClient, WorkbenchModelClient
from .policy import PolicyEngine
from .state import AgentState, Role, initial_agent_state
from .tokenizer_runtime import (
    TokenizerRuntime,
    approximate_test_runtime,
    load_tokenizer_runtime,
)
from .tool_registry import ToolRegistry, build_default_registry
from .tools.campaign_api import CampaignAPI, initialize_demo_database
from .tools.enterprise_kb import EnterpriseKnowledgeBase
from .tools.merchant_analytics import MerchantAnalytics
from .tools.research_search import (
    EVIDENCE_COMPRESSOR_POLICY,
    EVIDENCE_COMPRESSOR_VERSION,
    Phase5EvidenceAdapter,
    ResearchSearchTool,
    evidence_compressor_fingerprint,
)
from .tracing import redact_payload


class ThreadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=200)
    organization_id: str = Field(min_length=1, max_length=200)
    role: Role


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=10_000)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "edit", "reject"]
    edited_arguments: dict[str, Any] | None = None
    feedback: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def require_edits(self) -> "ResumeRequest":
        if self.decision == "edit" and self.edited_arguments is None:
            raise ValueError("edited_arguments is required for an edit decision")
        if self.decision != "edit" and self.edited_arguments is not None:
            raise ValueError("edited_arguments is accepted only for an edit decision")
        return self


class PreferencePutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: str = Field(min_length=1, max_length=200)
    value: str | int | float | bool


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    strict_msgpack: bool
    model_process: str
    retriever_process: str
    tool_count: int
    tokenizer_mode: Literal["exact", "approximate_test"]
    tokenizer_path: str
    tokenizer_class: str
    tokenizer_artifact_fingerprint: str
    tokenizer_vocabulary_size: int
    tokenizer_pad_token_id: int | None
    tokenizer_eos_token_id: int | None
    evidence_compressor_policy: str
    evidence_compressor_version: str
    evidence_compressor_fingerprint: str
    max_evidence_token_budget: int
    model_name: str
    retriever_configuration: dict[str, Any]
    runtime_configuration_fingerprint: str


class ToolsResponse(BaseModel):
    tools: list[dict[str, Any]]


class RunAcceptedResponse(BaseModel):
    thread_id: str
    run_id: str
    status: str
    stream_url: str


class ResumeAcceptedResponse(RunAcceptedResponse):
    decision: Literal["approve", "edit", "reject"]


class SafeStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    run_id: str | None = None
    user_id: str | None = None
    organization_id: str | None = None
    role: Role | None = None
    status: str | None = None
    task: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    conversation_summary: str = ""
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    plan: list[Any] = Field(default_factory=list)
    next_action: str | None = None
    action_arguments: dict[str, Any] = Field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    tool_results: list[Any] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    approval_request: dict[str, Any] | None = None
    approval_decision: dict[str, Any] | None = None
    step_count: int = 0
    tool_call_count: int = 0
    research_search_count: int = 0
    planner_repair_count: int = 0
    errors: list[Any] = Field(default_factory=list)
    final_answer: str | None = None
    final_citations: list[str] = Field(default_factory=list)
    citation_coverage: float = 0.0
    completed: bool = False
    termination_reason: str | None = None


class HistoryResponse(BaseModel):
    thread_id: str
    history: list[dict[str, Any]]


class TraceResponse(BaseModel):
    thread_id: str
    events: list[dict[str, Any]]


class MemoryListResponse(BaseModel):
    organization_id: str
    user_id: str
    preferences: dict[str, Any]


class MemoryMutationResponse(BaseModel):
    organization_id: str
    user_id: str
    key: str
    stored: bool | None = None
    deleted: bool | None = None


class ThreadRecord(BaseModel):
    thread_id: str
    user_id: str
    organization_id: str
    role: Role
    created_at: str


class ThreadDirectory:
    """Small persistent directory; checkpoint state remains in LangGraph SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workbench_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create(self, request: ThreadCreateRequest) -> ThreadRecord:
        thread_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO workbench_threads
                    (thread_id, user_id, organization_id, role)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, request.user_id, request.organization_id, request.role),
            )
        return self.get(thread_id)

    def get(self, thread_id: str) -> ThreadRecord:
        row = self.connection.execute(
            """
            SELECT thread_id, user_id, organization_id, role, created_at
            FROM workbench_threads WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        return ThreadRecord.model_validate(dict(row))

    def close(self) -> None:
        self.connection.close()


@dataclass
class RunChannel:
    """Replayable, process-local SSE channel for one run or resume segment."""

    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        async with self.condition:
            self.events.append({"event": event, "data": data})
            self.condition.notify_all()

    async def finish(self) -> None:
        async with self.condition:
            self.finished = True
            self.condition.notify_all()

    async def iterate(self) -> AsyncIterator[dict[str, Any]]:
        index = 0
        while True:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: index < len(self.events) or self.finished
                )
                available = list(self.events[index:])
                index = len(self.events)
                done = self.finished
            for event in available:
                yield event
            if done and index >= len(self.events):
                return


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    try:
        return SourceRecord.model_validate(source).public_dict()
    except (TypeError, ValueError):
        return {
            key: redact_payload(source.get(key))
            for key in (
                "source_id",
                "source_type",
                "title",
                "section",
                "snippet",
                "score",
                "metadata",
            )
            if key in source
        }


def safe_state_projection(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    """Allowlisted public projection; never return an arbitrary checkpoint object."""

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in state.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id", ""))
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        sources.append(_public_source(source))
    messages = []
    for message in state.get("messages", [])[-50:]:
        if isinstance(message, dict) and message.get("role") in {"user", "assistant"}:
            messages.append(
                {
                    "role": message["role"],
                    "content": str(message.get("content", ""))[:4_000],
                }
            )
    return {
        "thread_id": state.get("thread_id"),
        "run_id": state.get("run_id"),
        "user_id": state.get("user_id"),
        "organization_id": state.get("organization_id"),
        "role": state.get("role"),
        "task": state.get("task"),
        "messages": messages,
        "conversation_summary": str(state.get("conversation_summary", ""))[:4_000],
        "user_preferences": redact_payload(state.get("user_preferences", {})),
        "plan": redact_payload(state.get("plan", [])),
        "next_action": state.get("next_action"),
        "action_arguments": redact_payload(state.get("action_arguments", {})),
        "pending_action": redact_payload(state.get("pending_action")),
        "tool_results": redact_payload(state.get("tool_results", [])),
        "sources": sources,
        "approval_request": redact_payload(state.get("approval_request")),
        "approval_decision": redact_payload(state.get("approval_decision")),
        "step_count": int(state.get("step_count", 0)),
        "tool_call_count": int(state.get("tool_call_count", 0)),
        "research_search_count": int(state.get("research_search_count", 0)),
        "planner_repair_count": int(state.get("planner_repair_count", 0)),
        "errors": redact_payload(state.get("errors", [])),
        "final_answer": state.get("final_answer"),
        "final_citations": state.get("final_citations", []),
        "citation_coverage": float(state.get("citation_coverage", 0.0)),
        "completed": bool(state.get("completed", False)),
        "termination_reason": state.get("termination_reason"),
    }


def _sanitized_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname or "configured-host"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _runtime_identity(
    settings: WorkbenchSettings,
    tokenizer_runtime: TokenizerRuntime,
) -> tuple[dict[str, Any], str]:
    compressor_fingerprint = evidence_compressor_fingerprint()
    private_contract = {
        "schema_version": 1,
        "tokenizer_artifact_fingerprint": (
            tokenizer_runtime.artifact_fingerprint
        ),
        "evidence_compressor_policy": EVIDENCE_COMPRESSOR_POLICY,
        "evidence_compressor_version": EVIDENCE_COMPRESSOR_VERSION,
        "evidence_compressor_fingerprint": compressor_fingerprint,
        "max_evidence_token_budget": settings.evidence_token_budget,
        "model_name": settings.model_name,
        "retriever_url": settings.retriever_url,
        "retriever_top_k": settings.retriever_top_k,
    }
    runtime_fingerprint = hashlib.sha256(
        json.dumps(
            private_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        **tokenizer_runtime.safe_metadata(),
        "evidence_compressor_policy": EVIDENCE_COMPRESSOR_POLICY,
        "evidence_compressor_version": EVIDENCE_COMPRESSOR_VERSION,
        "evidence_compressor_fingerprint": compressor_fingerprint,
        "max_evidence_token_budget": settings.evidence_token_budget,
        "model_name": settings.model_name,
        "retriever_configuration": {
            "url": _sanitized_endpoint(settings.retriever_url),
            "top_k": settings.retriever_top_k,
        },
        "runtime_configuration_fingerprint": runtime_fingerprint,
    }
    return metadata, runtime_fingerprint


class WorkbenchService:
    """Coordinates durable graph execution with replayable safe SSE events."""

    def __init__(
        self,
        *,
        graph: Any,
        registry: ToolRegistry,
        memory_store: PreferenceMemoryStore,
        thread_directory: ThreadDirectory,
        settings: WorkbenchSettings,
        runtime_metadata: dict[str, Any],
        runtime_configuration_fingerprint: str,
    ) -> None:
        self.graph = graph
        self.registry = registry
        self.memory_store = memory_store
        self.thread_directory = thread_directory
        self.settings = settings
        self.runtime_metadata = dict(runtime_metadata)
        self.runtime_configuration_fingerprint = runtime_configuration_fingerprint
        self.channels: dict[str, RunChannel] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_graph_steps + 4,
        }

    def _require_compatible_runtime(self, state: dict[str, Any]) -> None:
        stored = state.get("runtime_configuration_fingerprint")
        if stored != self.runtime_configuration_fingerprint:
            raise RuntimeError(
                "thread checkpoint runtime configuration fingerprint mismatch; "
                "start a new thread under the current tokenizer/model/Retriever "
                "configuration"
            )

    async def _publish_update(
        self, channel: RunChannel, update: dict[str, Any]
    ) -> bool:
        interrupted = False
        for node, delta in update.items():
            if node == "__interrupt__":
                interrupts = delta if isinstance(delta, (tuple, list)) else [delta]
                for item in interrupts:
                    value = getattr(item, "value", item)
                    await channel.publish(
                        "approval_request",
                        {
                            "event_type": "approval_requested",
                            "status": "paused",
                            "approval_request": redact_payload(value),
                        },
                    )
                interrupted = True
                continue
            if not isinstance(delta, dict):
                continue
            await channel.publish(
                "node_update",
                {
                    "event_type": "node_completed",
                    "node": node,
                    "status": "completed",
                },
            )
            for event in delta.get("execution_trace", []):
                if isinstance(event, dict):
                    event_type = str(event.get("event_type", "trace"))
                    await channel.publish(event_type, redact_payload(event))
            if isinstance(delta.get("plan"), list) and delta["plan"]:
                plan = delta["plan"][-1]
                if isinstance(plan, dict):
                    await channel.publish(
                        "planner_decision",
                        {
                            "event_type": "planner_decision",
                            "next_action": plan.get("next_action"),
                            "completed": plan.get("completed"),
                            "user_visible_reason": plan.get("user_visible_reason"),
                        },
                    )
            for source in delta.get("sources", []):
                if isinstance(source, dict):
                    await channel.publish("source_created", _public_source(source))
            if delta.get("final_answer"):
                await channel.publish(
                    "final_answer",
                    {
                        "event_type": "final_answer_created",
                        "answer": str(delta["final_answer"]),
                    },
                )
        return interrupted

    async def _drive(self, thread_id: str, graph_input: Any, channel: RunChannel) -> None:
        interrupted = False
        try:
            async for update in self.graph.astream(
                graph_input,
                self.config(thread_id),
                stream_mode="updates",
            ):
                if isinstance(update, dict):
                    interrupted = await self._publish_update(channel, update) or interrupted
            snapshot = await self.graph.aget_state(self.config(thread_id))
            safe_state = safe_state_projection(dict(snapshot.values or {}))
            if snapshot.next or interrupted:
                await channel.publish(
                    "paused",
                    {
                        "event_type": "approval_requested",
                        "status": "paused",
                        "next_nodes": list(snapshot.next),
                        "approval_request": safe_state.get("approval_request"),
                    },
                )
            else:
                await channel.publish(
                    "complete",
                    {
                        "event_type": (
                            "run_completed" if safe_state["completed"] else "run_failed"
                        ),
                        "status": "completed" if safe_state["completed"] else "failed",
                        "final_answer": safe_state.get("final_answer"),
                        "termination_reason": safe_state.get("termination_reason"),
                    },
                )
        except Exception as error:
            safe_message = await self._record_runtime_failure(thread_id, error)
            await channel.publish(
                "error",
                {
                    "event_type": "run_failed",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": safe_message,
                },
            )
        finally:
            await channel.finish()

    async def _record_runtime_failure(
        self, thread_id: str, error: Exception
    ) -> str:
        redacted = redact_payload({"message": str(error)[:1_000]})
        safe_message = str(redacted.get("message", "runtime execution failed"))
        try:
            snapshot = await self.graph.aget_state(self.config(thread_id))
            state = dict(snapshot.values or {})
            if not state:
                return safe_message
            event = {
                "sequence": len(state.get("execution_trace", [])),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "thread_id": str(state.get("thread_id", thread_id)),
                "run_id": str(state.get("run_id", "unknown-run")),
                "event_type": "run_failed",
                "node": "runtime_boundary",
                "tool_name": None,
                "duration_ms": 0.0,
                "status": "failed",
                "safe_input_summary": {"summary": ""},
                "safe_output_summary": {"summary": safe_message},
                "error_type": type(error).__name__,
                "retry_number": 0,
            }
            await self.graph.aupdate_state(
                self.config(thread_id),
                {
                    "errors": [
                        {
                            "code": "runtime_error",
                            "message": safe_message,
                            "node": "runtime_boundary",
                        }
                    ],
                    "execution_trace": [event],
                    "completed": False,
                    "termination_reason": "runtime_error",
                },
                as_node="citation_validation",
            )
        except Exception:
            # The public stream remains redacted even if a failed checkpoint
            # backend cannot durably record the terminal event.
            pass
        return safe_message

    async def start_run(self, thread_id: str, task: str) -> tuple[str, RunChannel]:
        record = self.thread_directory.get(thread_id)
        active = self.tasks.get(thread_id)
        if active is not None and not active.done():
            raise RuntimeError("thread already has an active graph segment")
        snapshot = await self.graph.aget_state(self.config(thread_id))
        if snapshot.values:
            self._require_compatible_runtime(dict(snapshot.values))
        if snapshot.next:
            raise RuntimeError("thread is awaiting approval; resume it before a new run")
        run_id = str(uuid.uuid4())
        if snapshot.values:
            graph_input: AgentState = {
                "run_id": run_id,
                "task": task,
                "messages": [{"role": "user", "content": task}],
                "plan": [],
                "next_action": None,
                "action_arguments": {},
                "pending_action": None,
                "approval_request": None,
                "approval_decision": None,
                "step_count": 0,
                "tool_call_count": 0,
                "research_search_count": 0,
                "planner_repair_count": 0,
                "final_answer": None,
                "final_citations": [],
                "citation_coverage": 0.0,
                "completed": False,
                "termination_reason": None,
            }
        else:
            graph_input = initial_agent_state(
                thread_id=thread_id,
                run_id=run_id,
                user_id=record.user_id,
                organization_id=record.organization_id,
                role=record.role,
                task=task,
                runtime_configuration_fingerprint=(
                    self.runtime_configuration_fingerprint
                ),
            )
        channel = RunChannel(run_id=run_id)
        self.channels[thread_id] = channel
        self.tasks[thread_id] = asyncio.create_task(
            self._drive(thread_id, graph_input, channel)
        )
        return run_id, channel

    async def resume(self, thread_id: str, request: ResumeRequest) -> tuple[str, RunChannel]:
        self.thread_directory.get(thread_id)
        active = self.tasks.get(thread_id)
        if active is not None and not active.done():
            raise RuntimeError("thread already has an active graph segment")
        snapshot = await self.graph.aget_state(self.config(thread_id))
        if snapshot.values:
            self._require_compatible_runtime(dict(snapshot.values))
        if not snapshot.next or "human_approval_interrupt" not in snapshot.next:
            raise RuntimeError("thread is not awaiting a human approval decision")
        if request.decision == "edit":
            state = dict(snapshot.values or {})
            pending = dict(state.get("pending_action") or {})
            action = str(pending.get("action", ""))
            try:
                spec = self.registry.get(action)
                validated = spec.input_model.model_validate(
                    request.edited_arguments or {}
                ).model_dump(mode="json")
            except (KeyError, ValidationError) as error:
                raise ValueError("edited approval arguments are invalid") from error
            original = dict(pending.get("arguments") or {})
            for immutable_field in ("campaign_id", "idempotency_key"):
                if (
                    immutable_field in original
                    and validated.get(immutable_field) != original.get(immutable_field)
                ):
                    raise ValueError(
                        f"approval edits cannot change {immutable_field}; "
                        "submit a new action instead"
                    )
            policy = PolicyEngine(self.registry).evaluate(
                state.get("role", "viewer"),
                action,
                validated,
                {
                    "tool_call_count": int(state.get("tool_call_count", 0)),
                    "max_tool_calls": self.settings.max_tool_calls,
                    "research_search_count": int(
                        state.get("research_search_count", 0)
                    ),
                    "max_research_searches": self.settings.max_research_searches,
                },
            )
            if not policy["allowed"]:
                raise ValueError(
                    "edited approval arguments are not policy-authorized"
                )
        run_id = str((snapshot.values or {}).get("run_id", "")) or str(uuid.uuid4())
        channel = RunChannel(run_id=run_id)
        self.channels[thread_id] = channel
        self.tasks[thread_id] = asyncio.create_task(
            self._drive(
                thread_id,
                Command(resume=request.model_dump(mode="json", exclude_none=True)),
                channel,
            )
        )
        return run_id, channel

    async def shutdown(self) -> None:
        pending = [task for task in self.tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.memory_store.close()
        self.thread_directory.close()


def build_default_service(
    *,
    settings: WorkbenchSettings,
    checkpointer: Any,
    model_client: WorkbenchModelClient | None = None,
    tokenizer: Any | None = None,
    tokenizer_runtime: TokenizerRuntime | None = None,
) -> WorkbenchService:
    if tokenizer is not None and tokenizer_runtime is not None:
        raise ValueError("provide tokenizer or tokenizer_runtime, not both")
    settings.ensure_runtime_directory()
    if not settings.business_db_path.exists():
        initialize_demo_database(
            settings.business_db_path, MERCHANT_SEED_PATH, overwrite=False
        )
    enterprise_kb = EnterpriseKnowledgeBase(ENTERPRISE_DOCS_ROOT)
    if tokenizer_runtime is None:
        tokenizer_runtime = (
            approximate_test_runtime(tokenizer)
            if tokenizer is not None
            else load_tokenizer_runtime(settings)
        )
    tokenizer = tokenizer_runtime.tokenizer
    evidence_adapter = Phase5EvidenceAdapter(
        tokenizer_runtime=tokenizer_runtime,
        max_observation_tokens=settings.evidence_token_budget,
    )
    research = ResearchSearchTool(
        settings.retriever_url,
        evidence_adapter,
        timeout_seconds=settings.tool_timeout_seconds,
    )
    analytics = MerchantAnalytics(settings.business_db_path)
    campaign_api = CampaignAPI(settings.business_db_path)
    registry = build_default_registry(
        enterprise_kb, research, analytics, campaign_api
    )
    memory_store = PreferenceMemoryStore(settings.memory_db_path)
    directory = ThreadDirectory(settings.memory_db_path)
    model_client = model_client or VLLMHTTPModelClient(
        base_url=settings.llm_base_url,
        model_name=settings.model_name,
        timeout_seconds=settings.tool_timeout_seconds * 3,
    )
    graph = WorkbenchGraph(
        model_client=model_client,
        registry=registry,
        memory_store=memory_store,
        settings=settings,
    ).build(checkpointer=checkpointer)
    runtime_metadata, runtime_fingerprint = _runtime_identity(
        settings, tokenizer_runtime
    )
    return WorkbenchService(
        graph=graph,
        registry=registry,
        memory_store=memory_store,
        thread_directory=directory,
        settings=settings,
        runtime_metadata=runtime_metadata,
        runtime_configuration_fingerprint=runtime_fingerprint,
    )


def create_app(service: WorkbenchService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.workbench = service
            yield
            return
        settings = WorkbenchSettings.from_env()
        settings.ensure_runtime_directory()
        async with aiosqlite.connect(str(settings.checkpoint_path)) as connection:
            saver = AsyncSqliteSaver(
                connection,
                serde=JsonPlusSerializer(
                    pickle_fallback=False,
                    allowed_msgpack_modules=None,
                ),
            )
            await saver.setup()
            runtime = build_default_service(settings=settings, checkpointer=saver)
            app.state.workbench = runtime
            try:
                yield
            finally:
                await runtime.shutdown()

    app = FastAPI(
        title="Enterprise Agent Workbench",
        version="0.1.0",
        lifespan=lifespan,
    )

    def runtime(request: Request) -> WorkbenchService:
        return request.app.state.workbench

    def thread_or_404(runtime_service: WorkbenchService, thread_id: str) -> ThreadRecord:
        try:
            return runtime_service.thread_directory.get(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="thread not found") from error

    @app.get("/healthz", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        runtime_service = runtime(request)
        return HealthResponse(
            status="ok",
            checkpoint="sqlite",
            strict_msgpack=True,
            model_process="external",
            retriever_process="external",
            tool_count=len(runtime_service.registry.safe_metadata()),
            **runtime_service.runtime_metadata,
        )

    @app.get("/api/tools", response_model=ToolsResponse)
    async def tools(request: Request) -> ToolsResponse:
        return ToolsResponse(tools=runtime(request).registry.safe_metadata())

    @app.post("/api/threads", status_code=201, response_model=ThreadRecord)
    async def create_thread(
        payload: ThreadCreateRequest, request: Request
    ) -> ThreadRecord:
        record = runtime(request).thread_directory.create(payload)
        return record

    @app.post(
        "/api/threads/{thread_id}/runs",
        status_code=202,
        response_model=RunAcceptedResponse,
    )
    async def create_run(
        thread_id: str, payload: RunCreateRequest, request: Request
    ) -> RunAcceptedResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        try:
            run_id, _ = await runtime_service.start_run(thread_id, payload.task)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            detail = redact_payload({"message": str(error)})["message"]
            raise HTTPException(status_code=422, detail=detail) from error
        return RunAcceptedResponse(
            thread_id=thread_id,
            run_id=run_id,
            status="started",
            stream_url=f"/api/threads/{thread_id}/stream",
        )

    @app.get("/api/threads/{thread_id}/stream")
    async def stream(thread_id: str, request: Request) -> StreamingResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        channel = runtime_service.channels.get(thread_id)
        if channel is None:
            raise HTTPException(status_code=404, detail="thread has no run stream")

        async def event_stream() -> AsyncIterator[str]:
            event_id = 0
            async for item in channel.iterate():
                payload = json.dumps(item["data"], sort_keys=True, separators=(",", ":"))
                yield f"id: {event_id}\nevent: {item['event']}\ndata: {payload}\n\n"
                event_id += 1

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/threads/{thread_id}/state", response_model=SafeStateResponse)
    async def state(thread_id: str, request: Request) -> SafeStateResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        snapshot = await runtime_service.graph.aget_state(
            runtime_service.config(thread_id)
        )
        if not snapshot.values:
            return SafeStateResponse(
                thread_id=thread_id, status="created", completed=False
            )
        return SafeStateResponse.model_validate(
            safe_state_projection(dict(snapshot.values))
        )

    @app.get("/api/threads/{thread_id}/history", response_model=HistoryResponse)
    async def history(thread_id: str, request: Request) -> HistoryResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        records = []
        async for snapshot in runtime_service.graph.aget_state_history(
            runtime_service.config(thread_id), limit=50
        ):
            values = dict(snapshot.values or {})
            records.append(
                {
                    "created_at": (
                        snapshot.created_at.isoformat()
                        if hasattr(snapshot.created_at, "isoformat")
                        else str(snapshot.created_at)
                    ),
                    "next_nodes": list(snapshot.next),
                    "run_id": values.get("run_id"),
                    "step_count": values.get("step_count", 0),
                    "completed": values.get("completed", False),
                    "termination_reason": values.get("termination_reason"),
                }
            )
        return HistoryResponse(thread_id=thread_id, history=records)

    @app.get("/api/threads/{thread_id}/trace", response_model=TraceResponse)
    async def trace(thread_id: str, request: Request) -> TraceResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        snapshot = await runtime_service.graph.aget_state(
            runtime_service.config(thread_id)
        )
        raw_events = (snapshot.values or {}).get("execution_trace", [])
        events = [
            redact_payload(event)
            for event in raw_events[-500:]
            if isinstance(event, dict)
        ]
        return TraceResponse(thread_id=thread_id, events=events)

    @app.post(
        "/api/threads/{thread_id}/resume",
        status_code=202,
        response_model=ResumeAcceptedResponse,
    )
    async def resume(
        thread_id: str, payload: ResumeRequest, request: Request
    ) -> ResumeAcceptedResponse:
        runtime_service = runtime(request)
        thread_or_404(runtime_service, thread_id)
        try:
            run_id, _ = await runtime_service.resume(thread_id, payload)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            detail = redact_payload({"message": str(error)})["message"]
            raise HTTPException(status_code=422, detail=detail) from error
        return ResumeAcceptedResponse(
            thread_id=thread_id,
            run_id=run_id,
            status="resumed",
            decision=payload.decision,
            stream_url=f"/api/threads/{thread_id}/stream",
        )

    @app.get(
        "/api/users/{user_id}/memories", response_model=MemoryListResponse
    )
    async def memories(
        user_id: str,
        request: Request,
        organization_id: str = Query(min_length=1, max_length=200),
    ) -> MemoryListResponse:
        values = runtime(request).memory_store.list_preferences(
            organization_id, user_id
        )
        return MemoryListResponse(
            organization_id=organization_id,
            user_id=user_id,
            preferences=values,
        )

    @app.put(
        "/api/users/{user_id}/memories/{key}",
        response_model=MemoryMutationResponse,
    )
    async def put_memory(
        user_id: str,
        key: str,
        payload: PreferencePutRequest,
        request: Request,
    ) -> MemoryMutationResponse:
        try:
            runtime(request).memory_store.set_preference(
                payload.organization_id, user_id, key, payload.value
            )
        except MemoryValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return MemoryMutationResponse(
            organization_id=payload.organization_id,
            user_id=user_id,
            key=key,
            stored=True,
        )

    @app.delete(
        "/api/users/{user_id}/memories/{key}",
        response_model=MemoryMutationResponse,
    )
    async def delete_memory(
        user_id: str,
        key: str,
        request: Request,
        organization_id: str = Query(min_length=1, max_length=200),
    ) -> MemoryMutationResponse:
        deleted = runtime(request).memory_store.delete_preference(
            organization_id, user_id, key
        )
        return MemoryMutationResponse(
            organization_id=organization_id,
            user_id=user_id,
            key=key,
            deleted=deleted,
        )

    return app


app = create_app()


__all__ = [
    "PreferencePutRequest",
    "ResumeRequest",
    "RunCreateRequest",
    "ThreadCreateRequest",
    "ThreadDirectory",
    "WorkbenchService",
    "app",
    "build_default_service",
    "create_app",
    "safe_state_projection",
]
