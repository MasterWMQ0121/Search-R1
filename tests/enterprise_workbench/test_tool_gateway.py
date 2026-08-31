from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict, Field

from apps.enterprise_agent_workbench.observability import RuntimeObservability
from apps.enterprise_agent_workbench.tool_gateway import (
    TenantToolPolicy,
    ToolAccessDenied,
    ToolGateway,
    ToolGatewayError,
    ToolIdempotencyConflict,
    ToolInvocationContext,
    ToolRateLimitExceeded,
)
from apps.enterprise_agent_workbench.tool_registry import (
    RateLimitConfig,
    ToolRegistry,
    ToolSpec,
)


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(ge=0)


class ReadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class WriteInput(ReadInput):
    idempotency_key: str = Field(min_length=8)


def context(
    *, role: str = "viewer", tenant_id: str = "tenant-a", approved: bool = False
) -> ToolInvocationContext:
    return ToolInvocationContext(
        tenant_id=tenant_id,
        user_id="user-1",
        role=role,
        thread_id="thread-1",
        run_id="run-1",
        authorization_granted=approved,
        approval_decision="approve" if approved else None,
    )


def read_spec(name, handler, **overrides):
    values = {
        "name": name,
        "description": "Test read tool.",
        "input_model": ReadInput,
        "output_model": ReadOutput,
        "category": "business_read",
        "risk_level": "low",
        "required_roles": frozenset({"viewer", "admin"}),
        "read_only": True,
        "side_effecting": False,
        "approval_required": False,
        "timeout_seconds": 0.1,
        "idempotent": True,
        "source_producing": False,
        "enabled": True,
        "handler": handler,
    }
    values.update(overrides)
    return ToolSpec(**values)


@pytest.mark.asyncio
async def test_gateway_timeout_retry_and_metrics():
    attempts = 0

    async def flaky(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return {"value": request.value}

    telemetry = RuntimeObservability()
    gateway = ToolGateway(
        ToolRegistry([read_spec("flaky", flaky, max_retries=1)]),
        observability=telemetry,
    )
    assert await gateway.execute("flaky", {"value": 7}, context=context()) == {
        "value": 7
    }
    assert attempts == 2
    assert telemetry.metrics.value(
        "runtime.retries", {"tenant_id": "tenant-a", "tool": "flaky"}
    ) == 1

    async def slow(request):
        await asyncio.sleep(0.05)
        return {"value": request.value}

    timeout_gateway = ToolGateway(
        ToolRegistry([read_spec("slow", slow, timeout_seconds=0.005)])
    )
    with pytest.raises(TimeoutError):
        await timeout_gateway.execute("slow", {"value": 1}, context=context())


@pytest.mark.asyncio
async def test_gateway_rate_limit_rbac_and_tenant_policy_fail_closed():
    async def handler(request):
        return {"value": request.value}

    policy = TenantToolPolicy()
    gateway = ToolGateway(
        ToolRegistry(
            [
                read_spec(
                    "limited",
                    handler,
                    required_roles=frozenset({"admin"}),
                    rate_limit=RateLimitConfig(capacity=1, refill_per_second=0.01),
                )
            ]
        ),
        tenant_policy=policy,
    )
    with pytest.raises(ToolAccessDenied):
        await gateway.execute("limited", {"value": 1}, context=context())

    admin = context(role="admin")
    assert await gateway.execute("limited", {"value": 1}, context=admin) == {
        "value": 1
    }
    with pytest.raises(ToolRateLimitExceeded):
        await gateway.execute("limited", {"value": 2}, context=admin)

    policy.set_policy("tenant-b", deny={"limited"})
    with pytest.raises(ToolAccessDenied):
        await gateway.execute(
            "limited", {"value": 1}, context=context(role="admin", tenant_id="tenant-b")
        )
    assert gateway.list_tools("tenant-b", "admin") == []


@pytest.mark.asyncio
async def test_gateway_idempotency_version_and_audit_hook():
    executions = 0
    audit_events = []

    async def write(request, *, context):
        nonlocal executions
        assert context["tenant_id"] == "tenant-a"
        executions += 1
        return {"value": request.value}

    spec = ToolSpec(
        name="write",
        description="Test write tool.",
        input_model=WriteInput,
        output_model=ReadOutput,
        category="business_write",
        risk_level="high",
        required_roles=frozenset({"admin"}),
        read_only=False,
        side_effecting=True,
        approval_required=True,
        timeout_seconds=0.1,
        idempotent=True,
        source_producing=False,
        enabled=True,
        handler=write,
        version="v2",
    )
    gateway = ToolGateway(ToolRegistry([spec]), audit_hook=audit_events.append)
    approved = context(role="admin", approved=True)
    arguments = {"value": 4, "idempotency_key": "same-key-123"}

    first = await gateway.execute("write", arguments, context=approved, version="v2")
    second = await gateway.execute("write", arguments, context=approved, version="v2")
    assert first == second == {"value": 4}
    assert executions == 1
    assert audit_events[0]["status"] == "completed"
    assert audit_events[0]["tenant_id"] == "tenant-a"

    with pytest.raises(ToolIdempotencyConflict):
        await gateway.execute(
            "write",
            {"value": 5, "idempotency_key": "same-key-123"},
            context=approved,
            version="v2",
        )
    with pytest.raises(ToolAccessDenied):
        await gateway.execute("write", arguments, context=approved, version="v1")


@pytest.mark.asyncio
async def test_audit_hook_failure_never_reexecutes_completed_handler():
    executions = 0

    async def handler(request):
        nonlocal executions
        executions += 1
        return {"value": request.value}

    def broken_audit(event):
        raise RuntimeError(event["status"])

    gateway = ToolGateway(
        ToolRegistry([read_spec("audited", handler, max_retries=3)]),
        audit_hook=broken_audit,
    )
    with pytest.raises(ToolGatewayError, match="audit hook failed"):
        await gateway.execute("audited", {"value": 1}, context=context())
    assert executions == 1
