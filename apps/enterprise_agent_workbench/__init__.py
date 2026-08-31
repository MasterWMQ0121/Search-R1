"""Enterprise Agent Workbench application package.

The workbench is an application layer around Search-R1.  Its public data
contracts deliberately use JSON-compatible values so LangGraph checkpoints do
not depend on arbitrary Python-object deserialization.
"""

import os


# LangGraph reads this flag while importing its msgpack serializer, not when a
# saver is later constructed.  Package bootstrap therefore must establish the
# safe default before importing any workbench submodule that may import
# LangGraph.  Operators can still set an explicit value before process start.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from .citations import SourceRecord
from .context_budget import ContextBudgetManager
from .mcp_adapter import MCPToolAdapter
from .model_client import PlannerDecision, WorkbenchModelClient
from .observability import RuntimeObservability
from .sdk import AgentRuntimeClient
from .state import AgentState, initial_agent_state
from .tool_gateway import TenantToolPolicy, ToolGateway, ToolInvocationContext

__all__ = [
    "AgentState",
    "AgentRuntimeClient",
    "ContextBudgetManager",
    "MCPToolAdapter",
    "PlannerDecision",
    "SourceRecord",
    "RuntimeObservability",
    "TenantToolPolicy",
    "ToolGateway",
    "ToolInvocationContext",
    "WorkbenchModelClient",
    "initial_agent_state",
]
