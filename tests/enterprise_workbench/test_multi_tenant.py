from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from apps.enterprise_agent_workbench.api import build_default_service, create_app
from apps.enterprise_agent_workbench.config import WorkbenchSettings
from apps.enterprise_agent_workbench.model_client import FakeModelClient, PlannerDecision
from apps.enterprise_agent_workbench.tool_gateway import TenantToolPolicy
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
)


def decision(action, arguments=None, *, completed=False):
    return PlannerDecision(
        objective="Validate tenant isolation.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="Tenant isolation test.",
    )


def tenant_service(tmp_path, policy=None):
    settings = replace(
        WorkbenchSettings.from_env(),
        data_dir=tmp_path,
        max_graph_steps=18,
        max_tool_calls=8,
    )
    service = build_default_service(
        settings=settings,
        checkpointer=InMemorySaver(),
        model_client=FakeModelClient(
            [
                decision(
                    "update_campaign_budget",
                    {"campaign_id": "C102", "daily_budget": 1200},
                ),
                decision("finalizer", completed=True),
            ],
            synthesized_answer="Tenant scoped run.",
        ),
        tokenizer=DeterministicTextTokenizer(),
        tenant_tool_policy=policy,
    )
    app = create_app(service)
    app.state.workbench = service
    return app, service


@pytest.mark.asyncio
async def test_cross_tenant_thread_resume_inspect_and_memory_are_isolated(tmp_path):
    app, service = tenant_service(tmp_path)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/threads",
                json={
                    "tenant_id": "tenant-b",
                    "user_id": "shared-user",
                    "role": "operator",
                },
            )
            thread_id = created.json()["thread_id"]
            await client.put(
                "/api/users/shared-user/memories/preferred_kpi",
                json={"tenant_id": "tenant-b", "value": "ROAS"},
            )
            tenant_a_memory = await client.get(
                "/api/users/shared-user/memories",
                params={"tenant_id": "tenant-a"},
            )
            assert tenant_a_memory.json()["preferences"] == {}

            started = await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"tenant_id": "tenant-b", "task": "Increase C102 budget."},
            )
            assert started.status_code == 202
            await client.get(
                f"/api/threads/{thread_id}/stream",
                params={"tenant_id": "tenant-b"},
            )

            inspect = await client.get(
                f"/api/threads/{thread_id}/state",
                params={"tenant_id": "tenant-a"},
            )
            assert inspect.status_code == 404
            history = await client.get(
                f"/api/threads/{thread_id}/history",
                params={"tenant_id": "tenant-a"},
            )
            assert history.status_code == 404
            resume = await client.post(
                f"/api/threads/{thread_id}/resume",
                json={"tenant_id": "tenant-a", "decision": "approve"},
            )
            assert resume.status_code == 404

            own_state = await client.get(
                f"/api/threads/{thread_id}/state",
                params={"tenant_id": "tenant-b"},
            )
            assert own_state.status_code == 200
            assert own_state.json()["tenant_id"] == "tenant-b"
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_tenant_policy_hides_and_denies_tools(tmp_path):
    policy = TenantToolPolicy()
    policy.set_policy("tenant-a", deny={"get_campaign"})
    app, service = tenant_service(tmp_path, policy)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            hidden = await client.get(
                "/api/tools", params={"tenant_id": "tenant-a", "role": "viewer"}
            )
            visible = await client.get(
                "/api/tools", params={"tenant_id": "tenant-b", "role": "viewer"}
            )
            assert "get_campaign" not in {item["name"] for item in hidden.json()["tools"]}
            assert "get_campaign" in {item["name"] for item in visible.json()["tools"]}

            denied = await client.post(
                "/api/mcp/call",
                json={
                    "tenant_id": "tenant-a",
                    "user_id": "viewer",
                    "role": "viewer",
                    "name": "get_campaign",
                    "arguments": {"campaign_id": "C102"},
                },
            )
            assert denied.status_code == 403
    finally:
        await service.shutdown()

