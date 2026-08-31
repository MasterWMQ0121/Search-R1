from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict

from apps.enterprise_agent_workbench.mcp_adapter import MCPToolAdapter
from apps.enterprise_agent_workbench.tool_gateway import (
    ToolAccessDenied,
    ToolGateway,
    ToolInvocationContext,
)
from apps.enterprise_agent_workbench.tool_registry import ToolRegistry, ToolSpec


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: str


def spec(name: str, *, read_only: bool = True) -> ToolSpec:
    async def handler(request, **kwargs):
        del kwargs
        return {"result": f"{name}:{request.query}"}

    return ToolSpec(
        name=name,
        description=f"Read from {name}.",
        input_model=Input,
        output_model=Output,
        category="knowledge" if read_only else "business_write",
        risk_level="low" if read_only else "high",
        required_roles=frozenset({"viewer"}),
        read_only=read_only,
        side_effecting=not read_only,
        approval_required=not read_only,
        timeout_seconds=1.0,
        idempotent=True,
        source_producing=read_only,
        enabled=True,
        handler=handler,
    )


def invocation() -> ToolInvocationContext:
    return ToolInvocationContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="viewer",
        thread_id="thread-a",
        run_id="run-a",
    )


@pytest.mark.asyncio
async def test_mcp_manifest_and_two_read_only_calls_use_gateway():
    adapter = MCPToolAdapter(
        ToolGateway(ToolRegistry([spec("kb_search"), spec("campaign_read")]))
    )

    manifest = adapter.list_tools(tenant_id="tenant-a", role="viewer")
    assert [item["name"] for item in manifest["tools"]] == [
        "campaign_read",
        "kb_search",
    ]
    assert manifest["tools"][0]["inputSchema"]["properties"]["query"]
    assert manifest["tools"][0]["_meta"]["workbench/toolVersion"] == "v1"

    for name in ("kb_search", "campaign_read"):
        result = await adapter.call_tool(
            name, {"query": "C102"}, context=invocation()
        )
        assert result["isError"] is False
        assert result["structuredContent"]["result"] == f"{name}:C102"
        assert json.loads(result["content"][0]["text"])["result"]


@pytest.mark.asyncio
async def test_mcp_compatibility_adapter_refuses_write_tools():
    adapter = MCPToolAdapter(ToolGateway(ToolRegistry([spec("write", read_only=False)])))
    assert adapter.list_tools(tenant_id="tenant-a", role="viewer") == {"tools": []}
    with pytest.raises(ToolAccessDenied):
        await adapter.call_tool("write", {"query": "x"}, context=invocation())

