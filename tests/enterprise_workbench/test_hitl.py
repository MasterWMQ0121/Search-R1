from __future__ import annotations

import sqlite3

import pytest
from langgraph.types import Command

from apps.enterprise_agent_workbench.model_client import PlannerDecision
from apps.enterprise_agent_workbench.state import initial_agent_state


def decision(action, arguments=None, *, completed=False):
    return PlannerDecision(
        objective="Handle the requested campaign action.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="The requested write needs review.",
    )


def write_state(role="operator", thread_id="write-thread"):
    return initial_agent_state(
        thread_id=thread_id,
        run_id="write-run",
        user_id="operator-1",
        organization_id="org-1",
        role=role,
        task="Increase C102 budget to 1200.",
    )


def budget_and_audits(path):
    with sqlite3.connect(path) as connection:
        budget = connection.execute(
            "SELECT daily_budget FROM campaigns WHERE campaign_id = ?", ("C102",)
        ).fetchone()[0]
        audits = connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    return budget, audits


@pytest.mark.asyncio
async def test_write_interrupts_before_side_effect_and_approve_resumes_once(graph_factory):
    runtime = graph_factory(
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": 1200,
                    "idempotency_key": "thread-write-001",
                },
            ),
            decision("finalizer", completed=True),
        ],
        "The approved local campaign budget update completed.",
        role="operator",
    )
    config = {"configurable": {"thread_id": "write-thread"}, "recursion_limit": 80}
    first = await runtime["graph"].ainvoke(write_state(), config)
    assert "__interrupt__" in first
    assert budget_and_audits(runtime["database_path"]) == (1000.0, 0)

    final = await runtime["graph"].ainvoke(
        Command(resume={"decision": "approve"}), config
    )
    assert final["completed"] is True
    assert budget_and_audits(runtime["database_path"]) == (1200.0, 1)

    await runtime["graph"].ainvoke(
        Command(resume={"decision": "approve"}), config
    )
    assert budget_and_audits(runtime["database_path"]) == (1200.0, 1)


@pytest.mark.asyncio
async def test_edit_executes_only_validated_edited_arguments(graph_factory):
    runtime = graph_factory(
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": 1500,
                    "idempotency_key": "thread-edit-001",
                },
            ),
            decision("finalizer", completed=True),
        ],
        "The edited action completed.",
        role="operator",
    )
    config = {"configurable": {"thread_id": "write-thread"}, "recursion_limit": 80}
    await runtime["graph"].ainvoke(write_state(), config)
    await runtime["graph"].ainvoke(
        Command(
            resume={
                "decision": "edit",
                "edited_arguments": {
                    "campaign_id": "C102",
                    "daily_budget": 1200,
                    "idempotency_key": "thread-edit-001",
                },
            }
        ),
        config,
    )
    assert budget_and_audits(runtime["database_path"]) == (1200.0, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edited_arguments,error_field",
    [
        (
            {
                "campaign_id": "C101",
                "daily_budget": 1200,
                "idempotency_key": "thread-edit-immutable",
            },
            "campaign_id",
        ),
        (
            {
                "campaign_id": "C102",
                "daily_budget": 1200,
                "idempotency_key": "different-edit-key",
            },
            "idempotency_key",
        ),
    ],
)
async def test_edit_cannot_change_action_target_or_idempotency_identity(
    graph_factory, edited_arguments, error_field
):
    original = {
        "campaign_id": "C102",
        "daily_budget": 1500,
        "idempotency_key": "thread-edit-immutable",
    }
    runtime = graph_factory(
        [decision("update_campaign_budget", original)],
        "This action must remain pending.",
        role="operator",
    )
    config = {"configurable": {"thread_id": "write-thread"}, "recursion_limit": 80}
    await runtime["graph"].ainvoke(write_state(), config)

    with pytest.raises(ValueError, match=error_field):
        await runtime["graph"].ainvoke(
            Command(
                resume={
                    "decision": "edit",
                    "edited_arguments": edited_arguments,
                }
            ),
            config,
        )

    assert budget_and_audits(runtime["database_path"]) == (1000.0, 0)


@pytest.mark.asyncio
async def test_reject_and_viewer_denial_never_write(graph_factory):
    arguments = {
        "campaign_id": "C102",
        "daily_budget": 1200,
        "idempotency_key": "thread-reject-001",
    }
    runtime = graph_factory(
        [decision("update_campaign_budget", arguments), decision("finalizer", completed=True)],
        "The reviewer rejected the action.",
        role="operator",
    )
    config = {"configurable": {"thread_id": "write-thread"}, "recursion_limit": 80}
    await runtime["graph"].ainvoke(write_state(), config)
    result = await runtime["graph"].ainvoke(
        Command(resume={"decision": "reject", "feedback": "Keep current budget."}),
        config,
    )
    assert budget_and_audits(runtime["database_path"]) == (1000.0, 0)
    assert any(item["status"] == "rejected" for item in result["tool_results"])

    denied = graph_factory(
        [decision("update_campaign_budget", arguments), decision("finalizer", completed=True)],
        "Viewer permission prevented the action.",
        role="viewer",
    )
    viewer_result = await denied["graph"].ainvoke(
        write_state(role="viewer", thread_id="viewer-thread"),
        {"configurable": {"thread_id": "viewer-thread"}, "recursion_limit": 80},
    )
    assert "__interrupt__" not in viewer_result
    assert budget_and_audits(denied["database_path"]) == (1000.0, 0)
    assert any(item.get("status") == "denied" for item in viewer_result["tool_results"])


@pytest.mark.asyncio
async def test_invalid_write_arguments_are_denied_before_interrupt(graph_factory):
    runtime = graph_factory(
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": "not-a-number",
                    "idempotency_key": "invalid-write-001",
                },
            ),
            decision("finalizer", completed=True),
        ],
        "The invalid proposal was not executed.",
        role="operator",
    )
    result = await runtime["graph"].ainvoke(
        write_state(),
        {"configurable": {"thread_id": "write-thread"}, "recursion_limit": 80},
    )
    assert "__interrupt__" not in result
    assert budget_and_audits(runtime["database_path"]) == (1000.0, 0)
    assert any(error.get("code") == "invalid_arguments" for error in result["errors"])


@pytest.mark.asyncio
async def test_deterministic_fake_drives_complete_mixed_demo_workflow(graph_factory):
    runtime = graph_factory(
        [],
        "The local mock workflow completed.",
        role="operator",
    )
    task = (
        "Analyze why Campaign C102's ROI declined, check the budget policy, "
        "and increase the daily budget to 1200 if compliant."
    )
    initial = initial_agent_state(
        thread_id="mixed-thread",
        run_id="mixed-run",
        user_id="operator-1",
        organization_id="org-1",
        role="operator",
        task=task,
    )
    config = {"configurable": {"thread_id": "mixed-thread"}, "recursion_limit": 80}
    paused = await runtime["graph"].ainvoke(initial, config)
    assert "__interrupt__" in paused
    assert [item["tool_name"] for item in paused["tool_results"]] == [
        "compare_periods",
        "enterprise_kb_search",
        "get_campaign",
    ]
    assert budget_and_audits(runtime["database_path"]) == (1000.0, 0)

    final = await runtime["graph"].ainvoke(
        Command(resume={"decision": "approve"}), config
    )
    assert final["completed"] is True
    assert budget_and_audits(runtime["database_path"]) == (1200.0, 1)
