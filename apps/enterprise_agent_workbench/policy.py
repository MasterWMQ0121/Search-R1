"""Strict role-based authorization decisions for workbench tools."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    OPERATOR = "operator"
    ADMIN = "admin"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    requires_approval: bool
    decision_code: str
    user_message: str
    event: dict[str, Any]


class PolicyEngine:
    """Evaluate policy without executing a tool or causing side effects."""

    def __init__(self, registry: Any):
        self.registry = registry

    @staticmethod
    def _decision(
        *,
        allowed: bool,
        requires_approval: bool,
        code: str,
        message: str,
        role: str,
        tool_name: str,
    ) -> dict[str, Any]:
        event_type = "policy_allowed" if allowed else "policy_denied"
        return PolicyDecision(
            allowed=allowed,
            requires_approval=requires_approval,
            decision_code=code,
            user_message=message,
            event={
                "event_type": event_type,
                "tool_name": tool_name,
                "role": role,
                "decision_code": code,
                "status": "allowed" if allowed else "denied",
                "user_message": message,
            },
        ).model_dump(mode="json")

    def evaluate(
        self,
        role: Role | str,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        state_counts: Mapping[str, int] | None = None,
        **count_overrides: int,
    ) -> dict[str, Any]:
        """Return an auditable primitive decision; never invoke the tool.

        ``state_counts`` accepts ``tool_call_count``, ``max_tool_calls``,
        ``research_search_count`` and ``max_research_searches``. Keyword count
        overrides are supported so graph nodes can pass their scalar state
        fields directly.
        """

        try:
            canonical_role = Role(role).value
        except ValueError:
            return self._decision(
                allowed=False,
                requires_approval=False,
                code="unknown_role",
                message=f"Role {role!r} is not recognized.",
                role=str(role),
                tool_name=tool_name,
            )

        try:
            spec = self.registry.get(tool_name)
        except KeyError:
            return self._decision(
                allowed=False,
                requires_approval=False,
                code="unknown_tool",
                message=f"Tool {tool_name!r} is not registered.",
                role=canonical_role,
                tool_name=tool_name,
            )
        if not spec.enabled:
            return self._decision(
                allowed=False,
                requires_approval=False,
                code="tool_disabled",
                message=f"Tool {tool_name!r} is currently disabled.",
                role=canonical_role,
                tool_name=tool_name,
            )
        if canonical_role not in spec.required_roles:
            return self._decision(
                allowed=False,
                requires_approval=False,
                code="role_denied",
                message=(
                    f"Role {canonical_role!r} is not permitted to use "
                    f"{tool_name!r}."
                ),
                role=canonical_role,
                tool_name=tool_name,
            )

        counts = dict(state_counts or {})
        counts.update(count_overrides)
        tool_count = int(counts.get("tool_call_count", 0))
        max_tools = int(counts.get("max_tool_calls", 12))
        if max_tools <= 0 or tool_count >= max_tools:
            return self._decision(
                allowed=False,
                requires_approval=False,
                code="tool_budget_exhausted",
                message="The run has reached its tool-call limit.",
                role=canonical_role,
                tool_name=tool_name,
            )
        if spec.category == "research":
            research_count = int(counts.get("research_search_count", 0))
            max_research = int(counts.get("max_research_searches", 2))
            if max_research <= 0 or research_count >= max_research:
                return self._decision(
                    allowed=False,
                    requires_approval=False,
                    code="research_budget_exhausted",
                    message="The run has reached its research-search limit.",
                    role=canonical_role,
                    tool_name=tool_name,
                )

        arguments = dict(arguments or {})
        if spec.side_effecting:
            idempotency_key = arguments.get("idempotency_key")
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                return self._decision(
                    allowed=False,
                    requires_approval=False,
                    code="missing_idempotency_key",
                    message=(
                        "Side-effecting actions require a non-empty "
                        "idempotency key before approval."
                    ),
                    role=canonical_role,
                    tool_name=tool_name,
                )

        return self._decision(
            allowed=True,
            requires_approval=bool(spec.approval_required),
            code="approval_required" if spec.approval_required else "allowed",
            message=(
                "The action is authorized but requires human approval."
                if spec.approval_required
                else "The tool request is authorized."
            ),
            role=canonical_role,
            tool_name=tool_name,
        )
