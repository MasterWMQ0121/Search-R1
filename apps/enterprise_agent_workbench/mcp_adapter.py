"""Small MCP-compatible manifest/call adapter over the native ToolGateway."""

from __future__ import annotations

import json
from typing import Any

from .tool_gateway import ToolAccessDenied, ToolGateway, ToolInvocationContext


class MCPToolAdapter:
    """Compatibility adapter, not a complete or certified MCP server."""

    def __init__(self, gateway: ToolGateway, *, read_only_only: bool = True) -> None:
        self.gateway = gateway
        self.read_only_only = read_only_only

    def list_tools(self, *, tenant_id: str, role: str) -> dict[str, Any]:
        tools = []
        for item in self.gateway.list_tools(tenant_id, role):
            if self.read_only_only and not item["read_only"]:
                continue
            tools.append(
                {
                    "name": item["name"],
                    "description": item["description"],
                    "inputSchema": item["input_schema"],
                    "annotations": {
                        "readOnlyHint": bool(item["read_only"]),
                        "destructiveHint": bool(item["side_effecting"]),
                        "idempotentHint": bool(item["idempotent"]),
                    },
                    "_meta": {
                        "workbench/toolVersion": item["version"],
                        "workbench/category": item["category"],
                    },
                }
            )
        return {"tools": tools}

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolInvocationContext,
        version: str | None = None,
    ) -> dict[str, Any]:
        spec = self.gateway.registry.get(name)
        if self.read_only_only and not spec.read_only:
            raise ToolAccessDenied("MCP compatibility adapter exposes read-only tools")
        output = await self.gateway.execute(
            name,
            arguments,
            context=context,
            version=version,
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        output,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                }
            ],
            "structuredContent": output,
            "isError": False,
        }


__all__ = ["MCPToolAdapter"]
