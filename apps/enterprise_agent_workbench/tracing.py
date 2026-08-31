"""Safe, ordered execution tracing without hidden reasoning or secrets."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TraceEventType = Literal[
    "run_started",
    "node_started",
    "node_completed",
    "planner_decision",
    "planner_repair",
    "policy_allowed",
    "policy_denied",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "approval_requested",
    "approval_approved",
    "approval_edited",
    "approval_rejected",
    "final_answer_created",
    "run_completed",
    "run_failed",
]
_SENSITIVE_KEY = re.compile(
    r"password|passwd|secret|token|credential|api[_-]?key|authorization|private[_-]?key",
    re.IGNORECASE,
)
_HIDDEN_REASONING_KEY = re.compile(
    r"chain[_-]?of[_-]?thought|hidden[_-]?reason|internal[_-]?reasoning|scratchpad",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE)
_SECRET_TEXT = re.compile(
    r"\bsk-[A-Za-z0-9_-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)
_INLINE_SECRET = re.compile(
    r"(?i)\b(password|passwd|secret|token|credential|api[_-]?key|authorization)"
    r"(\s*[=:]\s*)[^\s,;]+"
)


def redact_payload(value: Any, *, max_depth: int = 4) -> Any:
    """Return a JSON-compatible, bounded projection of trace metadata."""

    if max_depth < 0:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, nested in list(value.items())[:50]:
            key = str(raw_key)[:100]
            if _HIDDEN_REASONING_KEY.search(key):
                continue
            output[key] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(key)
                else redact_payload(nested, max_depth=max_depth - 1)
            )
        return output
    if isinstance(value, (list, tuple)):
        return [redact_payload(item, max_depth=max_depth - 1) for item in value[:50]]
    if isinstance(value, str):
        text = _BEARER.sub("Bearer [REDACTED]", value[:2_000])
        text = _INLINE_SECRET.sub(r"\1\2[REDACTED]", text)
        return _SECRET_TEXT.sub("[REDACTED]", text)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    timestamp: str
    event_type: TraceEventType
    tenant_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    node: str | None = None
    tool_name: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: Literal["started", "completed", "failed", "paused", "denied"]
    safe_input_summary: dict[str, Any] = Field(default_factory=dict)
    safe_output_summary: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    retry_number: int = Field(default=0, ge=0)


class ExecutionTrace:
    """Thread-safe in-process collector whose exported events are primitives."""

    def __init__(
        self, *, thread_id: str, run_id: str, tenant_id: str = "default"
    ) -> None:
        if not thread_id.strip() or not run_id.strip() or not tenant_id.strip():
            raise ValueError("tenant_id, thread_id and run_id must be non-empty")
        self.tenant_id = tenant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self._events: list[TraceEvent] = []
        self._lock = threading.RLock()

    def record(
        self,
        event_type: TraceEventType,
        *,
        status: Literal["started", "completed", "failed", "paused", "denied"],
        node: str | None = None,
        tool_name: str | None = None,
        duration_ms: float | None = None,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        error_type: str | None = None,
        retry_number: int = 0,
    ) -> TraceEvent:
        with self._lock:
            event = TraceEvent(
                sequence=len(self._events),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                tenant_id=self.tenant_id,
                thread_id=self.thread_id,
                run_id=self.run_id,
                node=node,
                tool_name=tool_name,
                duration_ms=duration_ms,
                status=status,
                safe_input_summary=redact_payload(input_summary or {}),
                safe_output_summary=redact_payload(output_summary or {}),
                error_type=error_type,
                retry_number=retry_number,
            )
            self._events.append(event)
            return event

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.model_dump(mode="json") for event in self._events]
