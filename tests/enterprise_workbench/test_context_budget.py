from __future__ import annotations

from apps.enterprise_agent_workbench.context_budget import (
    BudgetAction,
    ContextBudgetConfig,
    ContextBudgetManager,
)
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
)


def test_default_budget_contract_and_deterministic_context_policy():
    manager = ContextBudgetManager(DeterministicTextTokenizer())
    assert manager.config == ContextBudgetConfig()
    assert manager.config.available_input_tokens == 8192 - 700 - 256

    payload = {
        "task": "compare campaign performance " * 200,
        "conversation_summary": "historical summary " * 500,
        "preferences": {f"key-{index}": "value " * 100 for index in range(20)},
        "tool_catalog": [
            {
                "name": f"tool-{index}",
                "description": "bounded tool description " * 40,
                "arguments": {"query": {"type": "string", "required": True}},
            }
            for index in range(30)
        ],
        "tool_results": [
            {
                "tool_name": "research_search",
                "status": "ok",
                "output": {"compressed_evidence": "evidence " * 500},
            }
            for _ in range(12)
        ],
        "errors": [{"code": "retry", "message": "bounded"}] * 5,
        "limits": {"remaining_steps": 10, "remaining_tools": 5},
    }

    first = manager.build_planner_context(**payload)
    second = manager.build_planner_context(**payload)

    assert first == second
    task, context, report = first
    assert task
    assert report.context_tokens <= report.available_input_tokens
    assert report.compressed_tokens > 0
    assert report.budget_headroom >= 0
    actions = {decision.component: decision.action for decision in report.decisions}
    assert actions["current_task"] == BudgetAction.COMPRESS
    assert actions["memory"] == BudgetAction.SUMMARIZE
    assert actions["tool_catalog"] == BudgetAction.DROP
    assert actions["tool_results"] in {BudgetAction.COMPRESS, BudgetAction.DROP}
    assert manager.token_count(context["available_tools"]) <= 1800


def test_public_policy_supports_keep_compress_and_drop():
    manager = ContextBudgetManager(DeterministicTextTokenizer())
    kept, keep = manager.apply_component("memory", "short", 10)
    compressed, compress = manager.apply_component("memory", "word " * 50, 5)
    dropped, drop = manager.apply_component(
        "memory", "word " * 50, 5, overflow_action=BudgetAction.DROP
    )

    assert kept == "short" and keep.action == BudgetAction.KEEP
    assert compressed and compress.action == BudgetAction.COMPRESS
    assert dropped == "" and drop.action == BudgetAction.DROP

