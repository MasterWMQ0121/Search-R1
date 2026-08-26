"""Model-client abstraction for an out-of-process OpenAI-compatible vLLM."""

from __future__ import annotations

import hashlib
import json
import math
import os
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

    code = "planner_parse_error"

    def __init__(
        self, first_error: str, repair_error: str, *, repair_attempts: int = 1
    ) -> None:
        super().__init__(
            f"{self.code} after {repair_attempts} repair attempt(s): "
            f"initial={first_error}; repair={repair_error}"
        )
        self.repair_attempts = repair_attempts


class PlannerSemanticError(PlannerParseError):
    """Raised when the one repair cannot satisfy the Planner tool contract."""

    code = "planner_semantic_error"


class PlannerStuckError(PlannerSemanticError):
    """Raised when initial and repaired decisions repeat one semantic failure."""

    code = "planner_stuck"


class PlannerSemanticValidationError(ValueError):
    """A concise, safe semantic error suitable for one repair prompt."""

    def __init__(self, message: str, *, signature: str) -> None:
        super().__init__(message)
        self.signature = signature


def _failure_signature(code: str, value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).casefold()
    return f"{code}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


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
        fields = sorted(
            {".".join(str(part) for part in item["loc"]) for item in error.errors()}
        )
        return "invalid planner fields: " + ", ".join(fields)
    return str(error)[:300]


def _planner_catalog(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the caller-provided compact catalog or fail closed."""

    catalog = context.get("available_tools")
    if not isinstance(catalog, list):
        raise PlannerSemanticValidationError(
            "planner context must provide an available_tools catalog",
            signature=_failure_signature("planner_catalog", "missing"),
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(catalog):
        if not isinstance(item, dict):
            raise PlannerSemanticValidationError(
                f"available_tools[{index}] must be an object",
                signature=_failure_signature("planner_catalog", [index, "item"]),
            )
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise PlannerSemanticValidationError(
                f"available_tools[{index}].name must be a canonical non-blank string",
                signature=_failure_signature("planner_catalog", [index, "name"]),
            )
        if name == "finalizer":
            raise PlannerSemanticValidationError(
                "finalizer is reserved and must not appear in available_tools",
                signature=_failure_signature("planner_catalog", "finalizer"),
            )
        if name in seen:
            raise PlannerSemanticValidationError(
                f"available_tools contains duplicate action {name!r}",
                signature=_failure_signature("planner_catalog", ["duplicate", name]),
            )
        if not isinstance(arguments, dict):
            raise PlannerSemanticValidationError(
                f"available_tools[{index}].arguments must be an object",
                signature=_failure_signature("planner_catalog", [index, "arguments"]),
            )
        seen.add(name)
        normalized.append(item)
    return normalized


def _value_matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return False


def _argument_constraint_errors(
    field: str, value: Any, contract: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected_type = contract.get("type")
    if not isinstance(expected_type, str):
        return [f"contract for arguments.{field} has no supported type"]
    required = contract.get("required") is True
    if value is None and not required:
        return errors
    if not _value_matches_json_type(value, expected_type):
        return [f"arguments.{field} must be {expected_type}"]

    enum = contract.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"arguments.{field} must be one of {enum!r}")

    if isinstance(value, str):
        minimum_length = contract.get("minLength")
        maximum_length = contract.get("maxLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(
                f"arguments.{field} must contain at least {minimum_length} characters"
            )
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            errors.append(
                f"arguments.{field} must contain at most {maximum_length} characters"
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        comparisons = (
            ("minimum", lambda candidate, boundary: candidate < boundary, ">="),
            ("maximum", lambda candidate, boundary: candidate > boundary, "<="),
            (
                "exclusiveMinimum",
                lambda candidate, boundary: candidate <= boundary,
                ">",
            ),
            (
                "exclusiveMaximum",
                lambda candidate, boundary: candidate >= boundary,
                "<",
            ),
        )
        for key, invalid, operator in comparisons:
            boundary = contract.get(key)
            if (
                isinstance(boundary, (int, float))
                and not isinstance(boundary, bool)
                and invalid(value, boundary)
            ):
                errors.append(f"arguments.{field} must be {operator} {boundary}")
    return errors


def validate_planner_decision(
    decision: PlannerDecision, context: dict[str, Any]
) -> PlannerDecision:
    """Validate exact action and compact input semantics supplied by the caller."""

    catalog = _planner_catalog(context)
    contracts = {item["name"]: item for item in catalog}
    allowed = sorted([*contracts, "finalizer"])
    if decision.completed != (decision.next_action == "finalizer"):
        raise PlannerSemanticValidationError(
            'completed=true if and only if next_action="finalizer"',
            signature=_failure_signature(
                "completion_contract",
                [decision.next_action, decision.completed],
            ),
        )
    if decision.next_action not in contracts and decision.next_action != "finalizer":
        raise PlannerSemanticValidationError(
            f"next_action must exactly equal one of {allowed!r}; descriptions are invalid",
            signature=_failure_signature(
                "unknown_action", " ".join(decision.next_action.split())
            ),
        )
    if decision.next_action == "finalizer":
        if decision.arguments:
            raise PlannerSemanticValidationError(
                "arguments must be empty when next_action is finalizer",
                signature=_failure_signature(
                    "finalizer_arguments", sorted(decision.arguments)
                ),
            )
        return decision

    argument_contracts = contracts[decision.next_action]["arguments"]
    if "idempotency_key" in decision.arguments:
        raise PlannerSemanticValidationError(
            "arguments.idempotency_key is control-plane metadata and must be omitted",
            signature=_failure_signature(
                "control_plane_argument", [decision.next_action, "idempotency_key"]
            ),
        )
    required = {
        name
        for name, contract in argument_contracts.items()
        if isinstance(contract, dict) and contract.get("required") is True
    }
    supplied = set(decision.arguments)
    missing = sorted(required - supplied)
    extra = sorted(supplied - set(argument_contracts))
    errors: list[str] = []
    if missing:
        errors.append("missing required arguments: " + ", ".join(missing))
    if extra:
        errors.append("unexpected arguments: " + ", ".join(extra))
    for field in sorted(supplied & set(argument_contracts)):
        contract = argument_contracts[field]
        if not isinstance(contract, dict):
            errors.append(f"contract for arguments.{field} must be an object")
            continue
        errors.extend(
            _argument_constraint_errors(field, decision.arguments[field], contract)
        )
    if errors:
        raise PlannerSemanticValidationError(
            "; ".join(errors),
            signature=_failure_signature(
                "invalid_arguments",
                [decision.next_action, sorted(errors)],
            ),
        )
    return decision


def _validate_with_caller_contract(
    decision: PlannerDecision, context: dict[str, Any]
) -> PlannerDecision:
    """Apply compact semantics and the caller's real tool-schema validator."""

    decision = validate_planner_decision(decision, context)
    validator = context.get("_planner_decision_validator")
    if validator is None:
        return decision
    if not callable(validator):
        raise PlannerSemanticValidationError(
            "planner decision validator must be callable",
            signature=_failure_signature("planner_validator", "not_callable"),
        )
    try:
        validated = validator(decision)
    except PlannerSemanticValidationError:
        raise
    except (ValidationError, ValueError, TypeError) as error:
        raise PlannerSemanticValidationError(
            str(error)[:1_000],
            signature=_failure_signature(
                "tool_arguments", [decision.next_action, str(error)[:1_000]]
            ),
        ) from error
    if not isinstance(validated, PlannerDecision):
        raise PlannerSemanticValidationError(
            "planner decision validator must return PlannerDecision",
            signature=_failure_signature("planner_validator", "invalid_return"),
        )
    return validated


def _sample_value(contract: dict[str, Any]) -> Any:
    enum = contract.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    return {
        "string": "C102",
        "number": 1200,
        "integer": 3,
        "boolean": True,
        "array": [],
        "object": {},
    }.get(contract.get("type"), "value")


def _valid_example(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    if not catalog:
        return {
            "objective": "Answer the user's request.",
            "next_action": "finalizer",
            "arguments": {},
            "completed": True,
            "user_visible_reason": "No additional tool call is needed.",
        }
    tool = catalog[0]
    arguments = {
        field: _sample_value(contract)
        for field, contract in tool["arguments"].items()
        if isinstance(contract, dict) and contract.get("required") is True
    }
    return {
        "objective": "Complete the user's request.",
        "next_action": tool["name"],
        "arguments": arguments,
        "completed": False,
        "user_visible_reason": "This exact tool is needed next.",
    }


def _relevant_contracts(
    invalid_response: str, catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select exact/standalone action hints without authorizing a rewrite."""

    try:
        candidate = extract_json_object(invalid_response).get("next_action")
    except ValueError:
        candidate = None
    if not isinstance(candidate, str):
        return catalog
    exact = [item for item in catalog if item["name"] == candidate]
    if exact:
        return exact
    hints = [
        item
        for item in catalog
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(item['name'])}(?![A-Za-z0-9_])",
            candidate,
        )
    ]
    return hints if len(hints) == 1 else catalog


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
        self._last_planner_failure_signatures = ContextVar[tuple[str, ...]](
            f"workbench_planner_failure_signatures_{id(self)}", default=()
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

    @property
    def last_planner_failure_signatures(self) -> list[str]:
        return list(self._last_planner_failure_signatures.get())

    def _record_semantic_failure(
        self, error: PlannerSemanticValidationError
    ) -> None:
        current = self._last_planner_failure_signatures.get()
        self._last_planner_failure_signatures.set((*current, error.signature))

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
        self._last_planner_failure_signatures.set(())
        try:
            catalog = _planner_catalog(context)
        except PlannerSemanticValidationError as error:
            self._record_semantic_failure(error)
            raise PlannerSemanticError(
                str(error),
                "caller must supply a valid role-filtered planner catalog",
                repair_attempts=0,
            ) from error
        allowed_actions = sorted([*(item["name"] for item in catalog), "finalizer"])
        task_context = {
            key: value
            for key, value in context.items()
            if key != "available_tools" and not key.startswith("_")
        }
        schema = json.dumps(PlannerDecision.model_json_schema(), sort_keys=True)
        catalog_json = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
        valid_example = json.dumps(
            _valid_example(catalog), sort_keys=True, separators=(",", ":")
        )
        invalid_example = json.dumps(
            {
                "objective": "Update C102.",
                "next_action": "Run the update tool to change C102.",
                "arguments": {"objective": "Update C102."},
                "completed": False,
                "user_visible_reason": "A write is needed.",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt = (
            f"{mode} this business task. Return exactly one JSON object matching the schema. "
            "next_action is a machine-readable identifier and MUST exactly equal one "
            "catalog name or finalizer. Never put descriptions, verbs, explanations, or "
            "sentences in next_action. Put explanations only in user_visible_reason. "
            "arguments may contain only fields from the selected tool contract; never put "
            "objective, next_action, completed, user_visible_reason, or idempotency_key in "
            "arguments. The orchestrator supplies idempotency_key. completed=true if and "
            "only if next_action=finalizer. Do not include hidden reasoning.\n"
            f"ALLOWED_ACTION_IDS={json.dumps(allowed_actions)}\n"
            f"TOOL_CATALOG={catalog_json}\nSCHEMA={schema}\n"
            f"VALID_EXAMPLE={valid_example}\nINVALID_EXAMPLE={invalid_example}\n"
            f"TASK={task}\n"
            f"CONTEXT={json.dumps(task_context, sort_keys=True, default=str)}"
        )
        first = await self._chat([{"role": "user", "content": prompt}], max_tokens=700)
        try:
            return _validate_with_caller_contract(
                parse_planner_decision(first), context
            )
        except ValueError as first_error:
            if isinstance(first_error, PlannerSemanticValidationError):
                self._record_semantic_failure(first_error)
            if not self.repair_allowed:
                error_class = (
                    PlannerSemanticError
                    if isinstance(first_error, PlannerSemanticValidationError)
                    else PlannerParseError
                )
                raise error_class(
                    str(first_error),
                    "run-level repair budget exhausted",
                    repair_attempts=0,
                ) from first_error
            self.last_planner_repairs = 1
            relevant_catalog = _relevant_contracts(first, catalog)
            repair_prompt = (
                "Repair the following invalid planner response. Return exactly one JSON object "
                "with no commentary. next_action must be exactly one allowed ID; never describe "
                "the tool. arguments must contain only the selected contract fields. Do not "
                "provide idempotency_key; the orchestrator supplies it. completed=true if and "
                "only if next_action=finalizer. Do not invent missing business values.\n"
                f"VALIDATION_ERROR={str(first_error)[:1_000]}\n"
                f"ALLOWED_ACTION_IDS={json.dumps(allowed_actions)}\n"
                "RELEVANT_TOOL_CONTRACTS="
                f"{json.dumps(relevant_catalog, sort_keys=True, separators=(',', ':'))}\n"
                f"SCHEMA={schema}\nINVALID_RESPONSE={first[:4_000]}"
            )
            try:
                repaired = await self._chat(
                    [{"role": "user", "content": repair_prompt}], max_tokens=700
                )
                return _validate_with_caller_contract(
                    parse_planner_decision(repaired), context
                )
            except (ValueError, httpx.HTTPError, RuntimeError) as repair_error:
                if isinstance(repair_error, PlannerSemanticValidationError):
                    self._record_semantic_failure(repair_error)
                semantic_signatures = self.last_planner_failure_signatures
                repeated_semantic_failure = (
                    len(semantic_signatures) >= 2
                    and semantic_signatures[-1] == semantic_signatures[-2]
                )
                error_class = (
                    PlannerStuckError
                    if repeated_semantic_failure
                    else PlannerSemanticError
                    if isinstance(
                        first_error, PlannerSemanticValidationError
                    )
                    or isinstance(repair_error, PlannerSemanticValidationError)
                    else PlannerParseError
                )
                raise error_class(
                    str(first_error), str(repair_error)
                ) from repair_error

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
        self.last_planner_failure_signatures: list[str] = []

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
            }
            reason = "The requested budget change must pass policy and human review."
        elif "pause" in lowered and "pause_campaign" not in prior_tools:
            action = "pause_campaign"
            arguments = {
                "campaign_id": campaign_id,
                "reason": "Requested through the deterministic workbench client.",
            }
            reason = "The campaign pause must pass policy and human review."
        elif "resume" in lowered and "resume_campaign" not in prior_tools:
            action = "resume_campaign"
            arguments = {
                "campaign_id": campaign_id,
                "reason": "Requested through the deterministic workbench client.",
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
