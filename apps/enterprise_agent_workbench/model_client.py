"""Model-client abstraction for an out-of-process OpenAI-compatible vLLM."""

from __future__ import annotations

import json
import os
import hashlib
import re
from collections import deque
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class PlannerDecision(BaseModel):
    """Strict, user-visible planner output; never a hidden reasoning trace."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=1_000)
    next_action: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    completed: bool = False
    user_visible_reason: str = Field(min_length=1, max_length=1_000)

    @field_validator("objective", "next_action", "user_visible_reason")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


# Compatibility for early callers that used the shorter name.
PlanDecision = PlannerDecision


class PlannerParseError(RuntimeError):
    """Raised after the one permitted planner-output repair also fails."""

    def __init__(
        self, first_error: str, repair_error: str, *, repair_attempts: int = 1
    ) -> None:
        super().__init__(
            f"planner_parse_error after {repair_attempts} repair attempt(s): "
            f"initial={first_error}; repair={repair_error}"
        )
        self.code = "planner_parse_error"
        self.repair_attempts = repair_attempts


@runtime_checkable
class WorkbenchModelClient(Protocol):
    async def plan(
        self, task: str, context: dict[str, Any] | None = None
    ) -> PlannerDecision: ...

    async def replan(
        self, task: str, state_summary: dict[str, Any]
    ) -> PlannerDecision: ...

    async def synthesize(
        self, task: str, evidence: Sequence[dict[str, Any]]
    ) -> str: ...

    async def summarize_memory(self, messages: Sequence[dict[str, Any]]) -> str: ...


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from plain or fenced model text."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response contains no complete JSON object")


def parse_planner_decision(text: str) -> PlannerDecision:
    try:
        return PlannerDecision.model_validate(extract_json_object(text))
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(_concise_validation_error(exc)) from exc


def _concise_validation_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
        return "invalid planner fields: " + ", ".join(fields)
    return str(error)[:300]


class VLLMHTTPModelClient:
    """Calls a separately hosted OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip() or not model_name.strip():
            raise ValueError("base_url and model_name must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client = http_client
        # One HTTP client is shared across FastAPI tasks.  These values describe
        # a single planner request/run, so process-wide instance attributes
        # would let concurrent threads overwrite one another's repair budget and
        # diagnostics. ContextVars preserve the existing property API while
        # isolating values per asyncio task.
        self._last_planner_repairs = ContextVar[int](
            f"workbench_last_planner_repairs_{id(self)}", default=0
        )
        self._repair_allowed = ContextVar[bool](
            f"workbench_repair_allowed_{id(self)}", default=True
        )

    @property
    def last_planner_repairs(self) -> int:
        return self._last_planner_repairs.get()

    @last_planner_repairs.setter
    def last_planner_repairs(self, value: int) -> None:
        self._last_planner_repairs.set(int(value))

    @property
    def repair_allowed(self) -> bool:
        return self._repair_allowed.get()

    @repair_allowed.setter
    def repair_allowed(self, value: bool) -> None:
        self._repair_allowed.set(bool(value))

    @classmethod
    def from_env(cls, **kwargs: Any) -> "VLLMHTTPModelClient":
        return cls(
            base_url=os.getenv("WORKBENCH_LLM_BASE_URL", "http://127.0.0.1:8001/v1"),
            model_name=os.getenv("WORKBENCH_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
            **kwargs,
        )

    async def _chat(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        if self._client is not None:
            response = await self._client.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_seconds
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
        response.raise_for_status()
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("model response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model response content is empty")
        return content

    async def _structured_decision(
        self, *, task: str, context: dict[str, Any], mode: str
    ) -> PlannerDecision:
        self.last_planner_repairs = 0
        schema = json.dumps(PlannerDecision.model_json_schema(), sort_keys=True)
        prompt = (
            f"{mode} this business task. Return exactly one JSON object matching the schema. "
            "Provide only a concise user-visible reason; do not include hidden reasoning.\n"
            f"SCHEMA={schema}\nTASK={task}\nCONTEXT={json.dumps(context, sort_keys=True, default=str)}"
        )
        first = await self._chat([{"role": "user", "content": prompt}], max_tokens=700)
        try:
            return parse_planner_decision(first)
        except ValueError as first_error:
            if not self.repair_allowed:
                raise PlannerParseError(
                    str(first_error),
                    "run-level repair budget exhausted",
                    repair_attempts=0,
                ) from first_error
            self.last_planner_repairs = 1
            repair_prompt = (
                "Repair the following invalid planner response. Return exactly one JSON object "
                "matching the provided schema, with no commentary. Do not invent a successful "
                "decision when required fields are unavailable.\n"
                f"SCHEMA={schema}\nINVALID_RESPONSE={first[:4_000]}"
            )
            try:
                repaired = await self._chat(
                    [{"role": "user", "content": repair_prompt}], max_tokens=700
                )
                return parse_planner_decision(repaired)
            except (ValueError, httpx.HTTPError, RuntimeError) as repair_error:
                raise PlannerParseError(str(first_error), str(repair_error)) from repair_error

    async def plan(
        self, task: str, context: dict[str, Any] | None = None
    ) -> PlannerDecision:
        return await self._structured_decision(
            task=task, context=context or {}, mode="Plan"
        )

    async def replan(
        self, task: str, state_summary: dict[str, Any]
    ) -> PlannerDecision:
        return await self._structured_decision(
            task=task, context=state_summary, mode="Replan"
        )

    async def synthesize(
        self, task: str, evidence: Sequence[dict[str, Any]]
    ) -> str:
        prompt = (
            "Produce a concise answer using only the supplied evidence. Cite factual evidence "
            "with exact bracketed source IDs such as [KB-POLICY-001]; do not expose hidden "
            "reasoning.\n"
            f"TASK={task}\nEVIDENCE={json.dumps(list(evidence), sort_keys=True, default=str)}"
        )
        return await self._chat([{"role": "user", "content": prompt}], max_tokens=1_200)

    async def summarize_memory(self, messages: Sequence[dict[str, Any]]) -> str:
        prompt = (
            "Summarize only user-visible conversation facts and explicit safe preferences. "
            "Exclude credentials, tokens, raw tool dumps, and hidden reasoning.\n"
            f"MESSAGES={json.dumps(list(messages), sort_keys=True, default=str)}"
        )
        return await self._chat([{"role": "user", "content": prompt}], max_tokens=500)


class FakeModelClient:
    """Deterministic CPU-test implementation with optional queued decisions."""

    def __init__(
        self,
        decisions: Sequence[PlannerDecision | dict[str, Any]] | None = None,
        *,
        synthesized_answer: str = "Completed using the available evidence.",
        memory_summary: str = "Conversation summary.",
    ) -> None:
        self._decisions = deque(
            PlannerDecision.model_validate(item) for item in (decisions or [])
        )
        self.synthesized_answer = synthesized_answer
        self.memory_summary = memory_summary
        self.calls: list[str] = []
        self.last_planner_repairs = 0

    def _fallback(
        self, task: str, state_summary: dict[str, Any] | None = None
    ) -> PlannerDecision:
        lowered = task.lower()
        campaign_match = re.search(r"\bC\d+\b", task, re.IGNORECASE)
        campaign_id = campaign_match.group(0).upper() if campaign_match else "C102"
        prior_results = (state_summary or {}).get("prior_tool_results", [])
        prior_tools = {
            str(result.get("tool_name"))
            for result in prior_results
            if isinstance(result, dict)
        }
        arguments: dict[str, Any]
        if any(word in lowered for word in ("roi", "compare", "analytics")) and not (
            {"compare_periods", "campaign_performance_summary"} & prior_tools
        ):
            action = "compare_periods"
            arguments = {
                "campaign_id": campaign_id,
                "previous_start": "2026-08-01",
                "previous_end": "2026-08-07",
                "current_start": "2026-08-08",
                "current_end": "2026-08-14",
            }
            reason = "Campaign metrics are needed to answer the request."
        elif any(word in lowered for word in ("policy", "compliant", "permitted")) and (
            "enterprise_kb_search" not in prior_tools
        ):
            action = "enterprise_kb_search"
            arguments = {"query": task, "top_k": 3}
            reason = "The enterprise policy must be consulted."
        elif any(word in lowered for word in ("budget", "status", "current state")) and (
            "get_campaign" not in prior_tools
        ):
            action = "get_campaign"
            arguments = {"campaign_id": campaign_id}
            reason = "Current campaign state is needed."
        elif any(word in lowered for word in ("research", "external", "wikipedia")) and (
            "research_search" not in prior_tools
        ):
            action = "research_search"
            arguments = {"query": task, "top_k": 3}
            reason = "External supporting evidence is needed."
        elif any(word in lowered for word in ("increase", "update")) and (
            "update_campaign_budget" not in prior_tools
        ):
            amount_match = re.search(
                r"(?:to|budget(?:\s+to)?)\s*([0-9]+(?:\.[0-9]+)?)", lowered
            )
            amount = float(amount_match.group(1)) if amount_match else 1200.0
            action = "update_campaign_budget"
            arguments = {
                "campaign_id": campaign_id,
                "daily_budget": amount,
                "idempotency_key": "fake-" + hashlib.sha256(
                    task.encode("utf-8")
                ).hexdigest()[:20],
            }
            reason = "The requested budget change must pass policy and human review."
        elif "pause" in lowered and "pause_campaign" not in prior_tools:
            action = "pause_campaign"
            arguments = {
                "campaign_id": campaign_id,
                "reason": "Requested through the deterministic workbench client.",
                "idempotency_key": "fake-" + hashlib.sha256(
                    task.encode("utf-8")
                ).hexdigest()[:20],
            }
            reason = "The campaign pause must pass policy and human review."
        elif "resume" in lowered and "resume_campaign" not in prior_tools:
            action = "resume_campaign"
            arguments = {
                "campaign_id": campaign_id,
                "reason": "Requested through the deterministic workbench client.",
                "idempotency_key": "fake-" + hashlib.sha256(
                    task.encode("utf-8")
                ).hexdigest()[:20],
            }
            reason = "The campaign resume must pass policy and human review."
        else:
            action = "finalizer"
            arguments = {}
            reason = "No additional tool call is required."
        return PlannerDecision(
            objective=task.strip() or "Complete the task safely.",
            next_action=action,
            arguments=arguments,
            completed=action == "finalizer",
            user_visible_reason=reason,
        )

    def _next(
        self, task: str, state_summary: dict[str, Any] | None = None
    ) -> PlannerDecision:
        return (
            self._decisions.popleft()
            if self._decisions
            else self._fallback(task, state_summary)
        )

    async def plan(
        self, task: str, context: dict[str, Any] | None = None
    ) -> PlannerDecision:
        self.calls.append("plan")
        return self._next(task, context)

    async def replan(
        self, task: str, state_summary: dict[str, Any]
    ) -> PlannerDecision:
        self.calls.append("replan")
        return self._next(task, state_summary)

    async def synthesize(
        self, task: str, evidence: Sequence[dict[str, Any]]
    ) -> str:
        self.calls.append("synthesize")
        return self.synthesized_answer

    async def summarize_memory(self, messages: Sequence[dict[str, Any]]) -> str:
        self.calls.append("summarize_memory")
        return self.memory_summary
