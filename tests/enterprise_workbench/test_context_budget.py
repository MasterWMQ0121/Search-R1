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


def test_under_budget_tool_result_still_strips_nested_control_metadata():
    manager = ContextBudgetManager(DeterministicTextTokenizer())
    _, context, report = manager.build_planner_context(
        task="Inspect C102.",
        conversation_summary="",
        preferences={},
        tool_catalog=[],
        tool_results=[
            {
                "tool_name": "campaign_current_state",
                "status": "ok",
                "arguments": {
                    "campaign_id": "C102",
                    "service_api_key": "hidden",
                    "bearer_token": "hidden",
                    "credentials": {"password": "hidden"},
                },
                "output": {
                    "operation": "campaign_current_state",
                    "campaign_id": "C102",
                    "tenant_id": "tenant-a",
                    "idempotency_key": "hidden",
                },
            }
        ],
        errors=[],
        limits={},
    )

    result = context["prior_tool_results"][0]
    assert result["arguments"] == {"campaign_id": "C102"}
    assert result["output"] == {
        "campaign_id": "C102",
        "operation": "campaign_current_state",
    }
    decision = next(
        item for item in report.decisions if item.component == "tool_results"
    )
    assert decision.action == BudgetAction.COMPRESS


def test_oversized_roi_result_preserves_structured_replanner_semantics():
    manager = ContextBudgetManager(DeterministicTextTokenizer())
    rows = [
        {
            "campaign_id": "C102",
            "date": f"2026-08-{(index % 28) + 1:02d}",
            "spend": 1000.0 + index,
            "revenue": 900.0 + index,
            "roi": 0.9,
            "description": "oversized diagnostic evidence " * 300,
            "idempotency_key": f"never-expose-row-{index}",
        }
        for index in range(100)
    ]
    result = {
        "tool_name": "roi_anomaly_detection",
        "status": "ok",
        "arguments": {
            "campaign_id": "C102",
            "as_of_date": "2026-08-28",
            "current_days": 7,
            "reference_days": 7,
            "idempotency_key": "never-expose-arguments",
        },
        "source_ids": ["ANALYTICS:C102:roi_anomaly_detection"],
        "idempotency_key": "never-expose-result",
        "output": {
            "operation": "roi_anomaly_detection",
            "rows": rows,
            "derived_metrics": {
                "roi_decline_fraction": 0.24,
                "anomaly_detected": True,
                "current_roi": 0.9,
                "reference_roi": 1.18,
            },
            "source_ids": ["ANALYTICS:C102:roi_anomaly_detection"],
            "evidence": "full evidence payload " * 1000,
            "idempotency_key": "never-expose-output",
        },
    }

    _, context, report = manager.build_planner_context(
        task="Explain why C102 ROI declined.",
        conversation_summary="",
        preferences={},
        tool_catalog=[],
        tool_results=[result],
        errors=[],
        limits={"remaining_steps": 8, "remaining_tools": 4},
    )

    compact = context["prior_tool_results"][0]
    assert isinstance(compact, dict)
    assert isinstance(compact["output"], dict)
    assert compact["arguments"]["campaign_id"] == "C102"
    assert compact["source_ids"] == ["ANALYTICS:C102:roi_anomaly_detection"]
    assert compact["output"]["operation"] == "roi_anomaly_detection"
    assert compact["output"]["derived_metrics"] == {
        "anomaly_detected": True,
        "current_roi": 0.9,
        "reference_roi": 1.18,
        "roi_decline_fraction": 0.24,
    }
    assert len(compact["output"].get("rows", [])) < len(rows)
    assert (
        "evidence" not in compact["output"]
        or len(compact["output"]["evidence"]) < len(result["output"]["evidence"])
    )

    def assert_no_idempotency_key(value):
        if isinstance(value, dict):
            assert "idempotency_key" not in value
            for item in value.values():
                assert_no_idempotency_key(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_idempotency_key(item)

    assert_no_idempotency_key(compact)
    assert manager.token_count(
        {"tool_results": context["prior_tool_results"], "errors": context["errors"]}
    ) <= manager.config.tool_results_budget
    assert report.context_tokens <= report.available_input_tokens
    assert report.compressed_tokens > 0
