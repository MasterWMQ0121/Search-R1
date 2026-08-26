from dataclasses import replace
from pathlib import Path

import pytest

from apps.enterprise_agent_workbench.policy import PolicyEngine
from apps.enterprise_agent_workbench.tool_registry import (
    ToolRegistry,
    build_default_registry,
)
from apps.enterprise_agent_workbench.tools.campaign_api import (
    CampaignAPI,
    initialize_demo_database,
)
from apps.enterprise_agent_workbench.tools.enterprise_kb import EnterpriseKnowledgeBase
from apps.enterprise_agent_workbench.tools.merchant_analytics import MerchantAnalytics
from apps.enterprise_agent_workbench.tools.research_search import ResearchSearchTool


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "apps" / "enterprise_agent_workbench"


class _UnusedResearchClient:
    async def post(self, *args, **kwargs):
        raise AssertionError("policy evaluation must never call a tool")


class _UnusedCompressor:
    def compress(self, *args, **kwargs):
        raise AssertionError("policy evaluation must never call a tool")


@pytest.fixture()
def registry(tmp_path):
    database = tmp_path / "merchant.sqlite3"
    initialize_demo_database(
        database, APP_DIR / "fixtures" / "merchant_seed.json"
    )
    research = ResearchSearchTool(
        "http://unused/retrieve",
        _UnusedCompressor(),
        client=_UnusedResearchClient(),
    )
    return build_default_registry(
        EnterpriseKnowledgeBase(APP_DIR / "fixtures" / "enterprise_docs"),
        research,
        MerchantAnalytics(database),
        CampaignAPI(database),
    )


def _evaluate(engine, role, tool, arguments=None, **counts):
    return engine.evaluate(role, tool, arguments or {}, count_overrides=0, **counts)


def test_viewer_can_read_knowledge_and_campaign_but_not_analytics_or_writes(registry):
    policy = PolicyEngine(registry)
    assert policy.evaluate("viewer", "enterprise_kb_search", {"query": "budget"})[
        "allowed"
    ]
    assert policy.evaluate("viewer", "get_campaign", {"campaign_id": "C102"})[
        "allowed"
    ]
    analytics = policy.evaluate(
        "viewer",
        "campaign_performance_summary",
        {"campaign_id": "C102"},
    )
    write = policy.evaluate(
        "viewer",
        "update_campaign_budget",
        {"campaign_id": "C102", "daily_budget": 1200, "idempotency_key": "key-12345"},
    )
    assert analytics["decision_code"] == "role_denied"
    assert write["decision_code"] == "role_denied"
    assert analytics["event"]["event_type"] == "policy_denied"


def test_analyst_can_use_analytics_but_cannot_write(registry):
    policy = PolicyEngine(registry)
    assert policy.evaluate("analyst", "compare_periods", {})["allowed"]
    assert not policy.evaluate(
        "analyst",
        "pause_campaign",
        {"idempotency_key": "pause-key-1"},
    )["allowed"]


@pytest.mark.parametrize("role", ["operator", "admin"])
def test_operator_and_admin_writes_require_idempotency_and_approval(registry, role):
    policy = PolicyEngine(registry)
    missing = policy.evaluate(
        role, "update_campaign_budget", {"campaign_id": "C102", "daily_budget": 1200}
    )
    assert missing["decision_code"] == "missing_idempotency_key"
    decision = policy.evaluate(
        role,
        "update_campaign_budget",
        {
            "campaign_id": "C102",
            "daily_budget": 1200,
            "idempotency_key": f"{role}-budget-1200",
        },
    )
    assert decision["allowed"] is True
    assert decision["requires_approval"] is True
    assert decision["decision_code"] == "approval_required"


def test_disabled_unknown_and_invalid_role_are_denied(registry):
    disabled = replace(registry.get("enterprise_kb_search"), enabled=False)
    disabled_registry = ToolRegistry(
        [disabled]
        + [
            spec
            for name in {item["name"] for item in registry.safe_metadata()}
            if name != disabled.name
            for spec in [registry.get(name)]
        ]
    )
    policy = PolicyEngine(disabled_registry)
    assert policy.evaluate("viewer", disabled.name, {"query": "budget"})[
        "decision_code"
    ] == "tool_disabled"
    assert policy.evaluate("viewer", "does_not_exist", {})["decision_code"] == "unknown_tool"
    assert policy.evaluate("superuser", "get_campaign", {})["decision_code"] == "unknown_role"


def test_tool_and_research_budgets_are_hard_denials(registry):
    policy = PolicyEngine(registry)
    exhausted = policy.evaluate(
        "analyst",
        "compare_periods",
        {},
        {"tool_call_count": 4, "max_tool_calls": 4},
    )
    research_exhausted = policy.evaluate(
        "viewer",
        "research_search",
        {"query": "x"},
        {
            "tool_call_count": 1,
            "max_tool_calls": 4,
            "research_search_count": 2,
            "max_research_searches": 2,
        },
    )
    assert exhausted["decision_code"] == "tool_budget_exhausted"
    assert research_exhausted["decision_code"] == "research_budget_exhausted"


def test_policy_evaluation_never_executes_the_requested_handler(registry):
    policy = PolicyEngine(registry)
    decision = policy.evaluate(
        "viewer", "research_search", {"query": "safe policy-only check"}
    )
    assert decision["allowed"] is True
    assert decision["event"]["event_type"] == "policy_allowed"
