"""Deterministic context budgeting for Planner and Replanner inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


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

    def _compact_tool_result(self, result: Mapping[str, Any], budget: int) -> dict[str, Any]:
        stable = {
            key: result[key]
            for key in ("tool_name", "status", "message", "error_type", "source_ids")
            if key in result
        }
        if "output" in result:
            remaining = max(0, budget - self.token_count(stable))
            stable["output"] = self._truncate_text(self._text(result["output"]), remaining)
        return stable

    def _fit_tool_results(
        self, results: Sequence[Mapping[str, Any]], errors: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], BudgetDecision]:
        budget = self.config.tool_results_budget
        original_results = [dict(item) for item in results]
        original_errors = [dict(item) for item in errors]
        original = {"tool_results": original_results, "errors": original_errors}
        original_tokens = self.token_count(original)
        if original_tokens <= budget:
            return original_results, original_errors, BudgetDecision(
                "tool_results", BudgetAction.KEEP, budget, original_tokens, original_tokens
            )

        kept_results: list[dict[str, Any]] = []
        kept_errors: list[dict[str, Any]] = []
        for error in reversed(original_errors):
            candidate = [error, *kept_errors]
            if self.token_count({"tool_results": kept_results, "errors": candidate}) <= budget:
                kept_errors = candidate
        for result in reversed(original_results):
            per_item_budget = max(64, budget // max(1, min(6, len(original_results))))
            compact = self._compact_tool_result(result, per_item_budget)
            candidate = [compact, *kept_results]
            if self.token_count({"tool_results": candidate, "errors": kept_errors}) <= budget:
                kept_results = candidate
        final = {"tool_results": kept_results, "errors": kept_errors}
        final_tokens = self.token_count(final)
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
