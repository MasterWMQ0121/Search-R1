"""Deterministic context budgeting for Planner and Replanner inputs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


_CONTROL_PLANE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "approval_decision",
        "auth",
        "authorization",
        "authorization_granted",
        "authorization_header",
        "client_secret",
        "credential",
        "credentials",
        "idempotency_key",
        "organization_id",
        "password",
        "refresh_token",
        "requesting_role",
        "requesting_user",
        "role",
        "run_id",
        "secret",
        "span_id",
        "tenant_id",
        "thread_id",
        "trace_id",
        "user_id",
    }
)
_ROUTING_KEYS = frozenset(
    {
        "id",
        "name",
        "operation",
        "source_id",
        "source_ids",
        "status",
        "tool_name",
        "type",
    }
)
_LONG_TEXT_KEYS = frozenset(
    {"content", "description", "evidence", "snippet", "text"}
)
_ARGUMENT_ENVELOPE_KEYS = frozenset(
    {"arguments", "invocation_arguments", "validated_arguments"}
)


class BudgetAction(str, Enum):
    KEEP = "keep"
    COMPRESS = "compress"
    SUMMARIZE = "summarize"
    DROP = "drop"


@dataclass(frozen=True)
class ContextBudgetConfig:
    total_context: int = 8192
    generation_reserve: int = 700
    safety_margin: int = 256
    system_budget: int = 800
    tool_catalog_budget: int = 1800
    memory_budget: int = 800
    tool_results_budget: int = 2000
    current_task_budget: int = 500

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("context budgets must be non-negative integers")
        if self.total_context <= 0:
            raise ValueError("total_context must be positive")
        if self.generation_reserve + self.safety_margin >= self.total_context:
            raise ValueError("generation reserve and safety margin exhaust context")
        allocated = (
            self.system_budget
            + self.tool_catalog_budget
            + self.memory_budget
            + self.tool_results_budget
            + self.current_task_budget
        )
        if allocated > self.available_input_tokens:
            raise ValueError("component budgets exceed available input context")

    @property
    def available_input_tokens(self) -> int:
        return self.total_context - self.generation_reserve - self.safety_margin


@dataclass(frozen=True)
class BudgetDecision:
    component: str
    action: BudgetAction
    budget_tokens: int
    original_tokens: int
    final_tokens: int
    dropped_items: int = 0

    @property
    def compressed_tokens(self) -> int:
        return max(0, self.original_tokens - self.final_tokens)


@dataclass(frozen=True)
class ContextBudgetReport:
    total_context: int
    generation_reserve: int
    safety_margin: int
    available_input_tokens: int
    original_tokens: int
    context_tokens: int
    compressed_tokens: int
    budget_headroom: int
    decisions: tuple[BudgetDecision, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_context": self.total_context,
            "generation_reserve": self.generation_reserve,
            "safety_margin": self.safety_margin,
            "available_input_tokens": self.available_input_tokens,
            "original_tokens": self.original_tokens,
            "context_tokens": self.context_tokens,
            "compressed_tokens": self.compressed_tokens,
            "budget_headroom": self.budget_headroom,
            "decisions": [
                {
                    **asdict(decision),
                    "action": decision.action.value,
                }
                for decision in self.decisions
            ],
        }


class ContextBudgetManager:
    """Apply stable keep/compress/summarize/drop rules without another LLM."""

    def __init__(self, tokenizer: Any, config: ContextBudgetConfig | None = None) -> None:
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("ContextBudgetManager requires an injected tokenizer")
        self.tokenizer = tokenizer
        self.config = config or ContextBudgetConfig()

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    def token_count(self, value: Any) -> int:
        return len(self.tokenizer.encode(self._text(value), add_special_tokens=False))

    def _truncate_text(self, value: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        token_ids = self.tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) <= token_budget:
            return value
        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            return str(
                decode(
                    token_ids[:token_budget],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ).strip()
        # Exact tokenizers used by the live service expose decode. This branch
        # keeps injected test tokenizers deterministic rather than guessing.
        ratio = token_budget / max(1, len(token_ids))
        return value[: max(0, int(len(value) * ratio))].rstrip()

    def apply_component(
        self,
        component: str,
        value: Any,
        token_budget: int,
        *,
        overflow_action: BudgetAction = BudgetAction.COMPRESS,
    ) -> tuple[Any, BudgetDecision]:
        """Public deterministic policy primitive for text-like components."""

        original_tokens = self.token_count(value)
        if original_tokens <= token_budget:
            return value, BudgetDecision(
                component, BudgetAction.KEEP, token_budget, original_tokens, original_tokens
            )
        if token_budget <= 0 or overflow_action == BudgetAction.DROP:
            empty: Any = "" if isinstance(value, str) else [] if isinstance(value, list) else {}
            return empty, BudgetDecision(
                component,
                BudgetAction.DROP,
                token_budget,
                original_tokens,
                0,
                len(value) if isinstance(value, (list, tuple, dict)) else 1,
            )
        compact = self._truncate_text(self._text(value), token_budget)
        final_tokens = self.token_count(compact)
        return compact, BudgetDecision(
            component,
            overflow_action,
            token_budget,
            original_tokens,
            final_tokens,
        )

    def _fit_catalog(
        self, catalog: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], BudgetDecision]:
        budget = self.config.tool_catalog_budget
        original = [dict(item) for item in catalog]
        original_tokens = self.token_count(original)
        if original_tokens <= budget:
            return original, BudgetDecision(
                "tool_catalog", BudgetAction.KEEP, budget, original_tokens, original_tokens
            )
        kept: list[dict[str, Any]] = []
        for item in original:
            candidate = [*kept, item]
            if self.token_count(candidate) > budget:
                continue
            kept.append(item)
        final_tokens = self.token_count(kept)
        return kept, BudgetDecision(
            "tool_catalog",
            BudgetAction.DROP,
            budget,
            original_tokens,
            final_tokens,
            len(original) - len(kept),
        )

    def _fit_memory(
        self, conversation_summary: str, preferences: Mapping[str, Any]
    ) -> tuple[dict[str, Any], BudgetDecision]:
        budget = self.config.memory_budget
        original = {
            "conversation_summary": conversation_summary,
            "preferences": dict(preferences),
        }
        original_tokens = self.token_count(original)
        if original_tokens <= budget:
            return original, BudgetDecision(
                "memory", BudgetAction.KEEP, budget, original_tokens, original_tokens
            )

        compact_preferences: dict[str, Any] = {}
        for key in sorted(preferences):
            candidate = {**compact_preferences, str(key): preferences[key]}
            probe = {"conversation_summary": "", "preferences": candidate}
            if self.token_count(probe) <= budget:
                compact_preferences[str(key)] = preferences[key]
        base = {"conversation_summary": "", "preferences": compact_preferences}
        remaining = max(0, budget - self.token_count(base))
        compact_summary = self._truncate_text(conversation_summary, remaining)
        output = {
            "conversation_summary": compact_summary,
            "preferences": compact_preferences,
        }
        final_tokens = self.token_count(output)
        return output, BudgetDecision(
            "memory",
            BudgetAction.SUMMARIZE,
            budget,
            original_tokens,
            final_tokens,
            max(0, len(preferences) - len(compact_preferences)),
        )

    @staticmethod
    def _normalized_key(key: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")

    @classmethod
    def _is_control_plane_key(cls, key: Any) -> bool:
        normalized = cls._normalized_key(key)
        return (
            normalized in _CONTROL_PLANE_KEYS
            or normalized.endswith("_api_key")
            or normalized.endswith("_credential")
            or normalized.endswith("_credentials")
            or normalized.endswith("_password")
            or normalized.endswith("_secret")
            or normalized.endswith("_token")
        )

    @classmethod
    def _is_routing_key(cls, key: Any) -> bool:
        normalized = cls._normalized_key(key)
        return (
            normalized in _ROUTING_KEYS
            or normalized.endswith("_id")
            or normalized.endswith("_ids")
            or normalized.endswith("_identifier")
            or normalized.endswith("_identifiers")
        )

    @classmethod
    def _is_long_text_key(cls, key: Any) -> bool:
        normalized = cls._normalized_key(key)
        return normalized in _LONG_TEXT_KEYS or any(
            normalized.endswith(f"_{suffix}") for suffix in _LONG_TEXT_KEYS
        )

    @classmethod
    def _sanitize_structured(cls, value: Any) -> Any:
        """Remove control-plane metadata without flattening business payloads."""

        if isinstance(value, Mapping):
            return {
                str(key): cls._sanitize_structured(value[key])
                for key in sorted(value, key=lambda item: str(item))
                if not cls._is_control_plane_key(key)
            }
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_structured(item) for item in value]
        return value

    @staticmethod
    def _structured_placeholder(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {}
        if isinstance(value, (list, tuple)):
            return []
        return None

    def _routing_projection(
        self,
        value: Any,
        *,
        arguments: bool = False,
        max_list_items: int = 8,
        include_nested_lists: bool = True,
    ) -> Any:
        """Extract bounded facts that can safely ground a later route decision."""

        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for raw_key in sorted(value, key=lambda item: str(item)):
                if self._is_control_plane_key(raw_key):
                    continue
                key = str(raw_key)
                item = value[raw_key]
                if self._is_routing_key(key):
                    output[key] = self._sanitize_structured(item)
                    continue
                if arguments and not isinstance(item, (Mapping, list, tuple)):
                    if isinstance(item, str) and self.token_count(item) > 64:
                        item = self._truncate_text(item, 64)
                    output[key] = item
                    continue
                if (
                    isinstance(item, (list, tuple))
                    and not include_nested_lists
                ):
                    continue
                projected = self._routing_projection(
                    item,
                    arguments=arguments,
                    max_list_items=max_list_items,
                    include_nested_lists=include_nested_lists,
                )
                if projected not in ({}, [], None, ""):
                    output[key] = projected
            return output
        if isinstance(value, (list, tuple)):
            if not include_nested_lists:
                return []
            output_list: list[Any] = []
            for item in value[:max_list_items]:
                projected = self._routing_projection(
                    item,
                    arguments=arguments,
                    max_list_items=max_list_items,
                    include_nested_lists=include_nested_lists,
                )
                if projected not in ({}, [], None, ""):
                    output_list.append(projected)
            return output_list
        return value if arguments else None

    def _project_payload(
        self,
        value: Any,
        *,
        max_list_items: int,
        max_text_tokens: int,
        field_name: str = "",
    ) -> Any:
        """Keep JSON structure, identifiers and cheap scalars while bounding payload."""

        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            keys = sorted(
                value,
                key=lambda item: (
                    0 if self._is_routing_key(item) else 1,
                    0 if self._normalized_key(item) == "derived_metrics" else 1,
                    str(item),
                ),
            )
            for raw_key in keys:
                if self._is_control_plane_key(raw_key):
                    continue
                key = str(raw_key)
                item = value[raw_key]
                if self._is_routing_key(key):
                    output[key] = self._sanitize_structured(item)
                    continue
                projected = self._project_payload(
                    item,
                    max_list_items=max_list_items,
                    max_text_tokens=(
                        min(max_text_tokens, 64)
                        if self._is_long_text_key(key)
                        else max_text_tokens
                    ),
                    field_name=key,
                )
                if projected not in ({}, [], None, "") or item is None:
                    output[key] = projected
            return output
        if isinstance(value, (list, tuple)):
            output_list: list[Any] = []
            for item in value[:max_list_items]:
                projected = self._project_payload(
                    item,
                    max_list_items=max_list_items,
                    max_text_tokens=max_text_tokens,
                    field_name=field_name,
                )
                if projected not in ({}, [], None, "") or item is None:
                    output_list.append(projected)
            return output_list
        if isinstance(value, str):
            if self._is_routing_key(field_name):
                return value
            if max_text_tokens <= 0:
                return ""
            return self._truncate_text(value, max_text_tokens)
        return value

    @classmethod
    def _merge_required(cls, candidate: Any, required: Any) -> Any:
        """Overlay the routing projection so payload reduction cannot erase it."""

        if isinstance(required, Mapping):
            merged = dict(candidate) if isinstance(candidate, Mapping) else {}
            for key, value in required.items():
                merged[key] = cls._merge_required(merged.get(key), value)
            return merged
        if isinstance(required, list):
            merged_list = list(candidate) if isinstance(candidate, list) else []
            for index, value in enumerate(required):
                if index < len(merged_list):
                    merged_list[index] = cls._merge_required(merged_list[index], value)
                else:
                    merged_list.append(value)
            return merged_list
        return required

    def _tool_result_routing_envelope(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        sanitized = self._sanitize_structured(result)
        envelope: dict[str, Any] = {}
        for key in sorted(sanitized):
            value = sanitized[key]
            if self._is_routing_key(key):
                envelope[key] = value
            elif key in _ARGUMENT_ENVELOPE_KEYS:
                envelope[key] = self._routing_projection(value, arguments=True)
            elif key == "output" and isinstance(value, (Mapping, list, tuple)):
                # Direct output routing fields (for example ``operation``) are
                # mandatory. Row/list identities are retained by richer
                # projections when they fit, but are lower priority than the
                # validated top-level invocation envelope.
                projected = self._routing_projection(
                    value, include_nested_lists=False
                )
                envelope[key] = (
                    projected
                    if projected not in ({}, [])
                    else self._structured_placeholder(value)
                )
        return envelope

    def _compact_tool_result(
        self, result: Mapping[str, Any], budget: int
    ) -> dict[str, Any]:
        """Compact a result structurally, never by truncating serialized JSON."""

        sanitized = self._sanitize_structured(result)
        if self.token_count(sanitized) <= budget:
            return sanitized

        routing = self._tool_result_routing_envelope(sanitized)
        routing_tokens = self.token_count(routing)
        if routing_tokens > budget:
            raise ValueError(
                "routing-critical tool result metadata exceeds tool_results_budget"
            )

        # Richest fitting projection wins. Every candidate is overlaid with the
        # routing envelope, so reducing rows/text cannot discard entity IDs or
        # validated invocation arguments.
        for max_list_items, max_text_tokens in (
            (16, 128),
            (8, 96),
            (4, 64),
            (2, 32),
            (1, 16),
            (1, 0),
            # Drop row/list payload before compact mapping-based metrics.
            (0, 16),
            (0, 0),
        ):
            projected = self._project_payload(
                sanitized,
                max_list_items=max_list_items,
                max_text_tokens=max_text_tokens,
            )
            candidate = self._merge_required(projected, routing)
            if self.token_count(candidate) <= budget:
                return candidate
        return routing

    def _fit_tool_results(
        self, results: Sequence[Mapping[str, Any]], errors: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], BudgetDecision]:
        budget = self.config.tool_results_budget
        original_results = [dict(item) for item in results]
        original_errors = [dict(item) for item in errors]
        original = {"tool_results": original_results, "errors": original_errors}
        original_tokens = self.token_count(original)
        sanitized_results = [self._sanitize_structured(item) for item in original_results]
        sanitized_errors = [self._sanitize_structured(item) for item in original_errors]
        sanitized = {"tool_results": sanitized_results, "errors": sanitized_errors}
        sanitized_tokens = self.token_count(sanitized)
        if sanitized_tokens <= budget:
            action = BudgetAction.KEEP if sanitized == original else BudgetAction.COMPRESS
            return sanitized_results, sanitized_errors, BudgetDecision(
                "tool_results", action, budget, original_tokens, sanitized_tokens
            )

        kept_results: list[dict[str, Any]] = []
        kept_errors: list[dict[str, Any]] = []
        empty_tokens = self.token_count({"tool_results": [], "errors": []})
        if empty_tokens > budget:
            raise ValueError("tool_results_budget cannot fit the result envelope")

        # Successful results are semantically more valuable to the Replanner
        # than error prose. Keep the newest routing envelope first, then use any
        # remaining budget for structured payload and older results.
        for result in reversed(sanitized_results):
            routing = self._tool_result_routing_envelope(result)
            routing_candidate = {
                "tool_results": [routing, *kept_results],
                "errors": kept_errors,
            }
            if self.token_count(routing_candidate) > budget:
                if not kept_results:
                    raise ValueError(
                        "routing-critical tool result metadata exceeds tool_results_budget"
                    )
                continue

            item_budget = budget
            compact: dict[str, Any] | None = None
            while item_budget >= self.token_count(routing):
                proposed = self._compact_tool_result(result, item_budget)
                candidate = {
                    "tool_results": [proposed, *kept_results],
                    "errors": kept_errors,
                }
                candidate_tokens = self.token_count(candidate)
                if candidate_tokens <= budget:
                    compact = proposed
                    break
                item_budget -= max(1, candidate_tokens - budget)
            kept_results = [compact or routing, *kept_results]

        # Errors are sanitized even when small and only consume headroom left
        # after successful result semantics have been retained.
        for error in reversed(sanitized_errors):
            candidates = [error]
            candidates.extend(
                self._project_payload(
                    error,
                    max_list_items=max_list_items,
                    max_text_tokens=max_text_tokens,
                )
                for max_list_items, max_text_tokens in ((4, 64), (1, 16), (1, 0))
            )
            for compact_error in candidates:
                if not compact_error:
                    continue
                candidate_errors = [compact_error, *kept_errors]
                candidate = {
                    "tool_results": kept_results,
                    "errors": candidate_errors,
                }
                if self.token_count(candidate) <= budget:
                    kept_errors = candidate_errors
                    break
        final = {"tool_results": kept_results, "errors": kept_errors}
        final_tokens = self.token_count(final)
        if final_tokens > budget:
            raise AssertionError("structured tool result compaction exceeded its budget")
        action = BudgetAction.COMPRESS if kept_results else BudgetAction.DROP
        return kept_results, kept_errors, BudgetDecision(
            "tool_results",
            action,
            budget,
            original_tokens,
            final_tokens,
            (len(original_results) - len(kept_results))
            + (len(original_errors) - len(kept_errors)),
        )

    def build_planner_context(
        self,
        *,
        task: str,
        conversation_summary: str,
        preferences: Mapping[str, Any],
        tool_catalog: Sequence[Mapping[str, Any]],
        tool_results: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        limits: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], ContextBudgetReport]:
        task_value, task_decision = self.apply_component(
            "current_task", task, self.config.current_task_budget
        )
        catalog_value, catalog_decision = self._fit_catalog(tool_catalog)
        memory_value, memory_decision = self._fit_memory(
            conversation_summary, preferences
        )
        results_value, errors_value, results_decision = self._fit_tool_results(
            tool_results, errors
        )
        context = {
            "conversation_summary": memory_value["conversation_summary"],
            "preferences": memory_value["preferences"],
            "available_tools": catalog_value,
            "prior_tool_results": results_value,
            "errors": errors_value,
            "limits": dict(limits),
        }
        system_decision = BudgetDecision(
            "system", BudgetAction.KEEP, self.config.system_budget, 0, 0
        )
        decisions = (
            system_decision,
            catalog_decision,
            memory_decision,
            results_decision,
            task_decision,
        )
        original_tokens = sum(item.original_tokens for item in decisions)
        context_tokens = sum(item.final_tokens for item in decisions)
        available = self.config.available_input_tokens
        report = ContextBudgetReport(
            total_context=self.config.total_context,
            generation_reserve=self.config.generation_reserve,
            safety_margin=self.config.safety_margin,
            available_input_tokens=available,
            original_tokens=original_tokens,
            context_tokens=context_tokens,
            compressed_tokens=max(0, original_tokens - context_tokens),
            budget_headroom=max(0, available - context_tokens),
            decisions=decisions,
        )
        return str(task_value), context, report


__all__ = [
    "BudgetAction",
    "BudgetDecision",
    "ContextBudgetConfig",
    "ContextBudgetManager",
    "ContextBudgetReport",
]
