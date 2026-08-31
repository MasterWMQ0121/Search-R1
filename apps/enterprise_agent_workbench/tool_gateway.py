"""Unified, tenant-aware execution gateway for native and protocol tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from .observability import RuntimeObservability
from .policy import PolicyEngine
from .tool_registry import ToolRegistry, ToolSpec
from .tracing import redact_payload


AuditHook = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class ToolGatewayError(RuntimeError):
    pass


class ToolAccessDenied(PermissionError, ToolGatewayError):
    pass


class ToolRateLimitExceeded(ToolGatewayError):
    pass


class ToolIdempotencyConflict(ToolGatewayError):
    pass


@dataclass(frozen=True)
class ToolInvocationContext:
    tenant_id: str
    user_id: str
    role: str
    thread_id: str
    run_id: str
    authorization_granted: bool = False
    approval_decision: str | None = None

    def __post_init__(self) -> None:
        required = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"tool invocation context is missing: {', '.join(missing)}")

    def business_context(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "role": self.role,
            "authorization_granted": self.authorization_granted,
            "approval_decision": self.approval_decision,
        }


@dataclass(frozen=True)
class _TenantRule:
    allow: frozenset[str] | None
    deny: frozenset[str]


class TenantToolPolicy:
    """In-memory tenant policy. Deny wins and unknown tenant IDs use defaults."""

    def __init__(self) -> None:
        self._rules: dict[str, _TenantRule] = {}
        self._lock = threading.RLock()

    def set_policy(
        self,
        tenant_id: str,
        *,
        allow: set[str] | frozenset[str] | None = None,
        deny: set[str] | frozenset[str] | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        normalized_allow = None if allow is None else frozenset(str(item) for item in allow)
        normalized_deny = frozenset(str(item) for item in (deny or set()))
        with self._lock:
            self._rules[tenant_id] = _TenantRule(normalized_allow, normalized_deny)

    def is_allowed(self, tenant_id: str, tool_name: str) -> bool:
        if not tenant_id.strip():
            return False
        with self._lock:
            rule = self._rules.get(tenant_id)
        if rule is None:
            return True
        if tool_name in rule.deny:
            return False
        return rule.allow is None or tool_name in rule.allow


class _TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self.updated_at)
            self.updated_at = now
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_per_second
            )
            if self.tokens < 1.0:
                return False
            self.tokens -= 1.0
            return True


class ToolGateway:
    """Enforce schema, RBAC, tenant policy, resilience, idempotency and audit."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tenant_policy: TenantToolPolicy | None = None,
        observability: RuntimeObservability | None = None,
        audit_hook: AuditHook | None = None,
    ) -> None:
        self.registry = registry
        self.tenant_policy = tenant_policy or TenantToolPolicy()
        self.observability = observability or RuntimeObservability()
        self.audit_hook = audit_hook
        self.policy_engine = PolicyEngine(registry)
        self._buckets: dict[tuple[str, str], _TokenBucket] = {}
        self._bucket_lock = threading.RLock()
        self._idempotency: dict[
            tuple[str, str, str], tuple[str, dict[str, Any]]
        ] = {}
        self._idempotency_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._idempotency_lock_guard = threading.RLock()

    def _require_tenant_tool(self, tenant_id: str, tool_name: str) -> ToolSpec:
        try:
            spec = self.registry.get(tool_name)
        except KeyError as error:
            raise ToolAccessDenied("tool is not registered") from error
        if not spec.enabled or not self.tenant_policy.is_allowed(tenant_id, tool_name):
            raise ToolAccessDenied("tool is disabled by tenant policy")
        return spec

    def list_tools(self, tenant_id: str, role: str) -> list[dict[str, Any]]:
        if not tenant_id.strip():
            raise ToolAccessDenied("tenant_id is required")
        self.registry._validate_role(role)
        return [
            item
            for item in self.registry.safe_metadata()
            if item["enabled"]
            and role in item["required_roles"]
            and self.tenant_policy.is_allowed(tenant_id, str(item["name"]))
        ]

    def planner_metadata(self, tenant_id: str, role: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self.registry.planner_metadata(role)
            if self.tenant_policy.is_allowed(tenant_id, str(item["name"]))
        ]

    def planner_action_ids(self, tenant_id: str, role: str) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.planner_metadata(tenant_id, role))

    def validate_planner_arguments(
        self, tenant_id: str, role: str, action: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self._require_tenant_tool(tenant_id, action)
        return self.registry.validate_planner_arguments(role, action, arguments)

    def evaluate_policy(
        self,
        *,
        tenant_id: str,
        role: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        state_counts: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        if not self.tenant_policy.is_allowed(tenant_id, tool_name):
            return PolicyEngine._decision(
                allowed=False,
                requires_approval=False,
                code="tenant_tool_denied",
                message="The tool is disabled by tenant policy.",
                role=role,
                tool_name=tool_name,
            )
        return self.policy_engine.evaluate(
            role, tool_name, arguments, state_counts or {}
        )

    def _consume_rate_limit(self, tenant_id: str, spec: ToolSpec) -> None:
        if spec.rate_limit is None:
            return
        key = (tenant_id, spec.name)
        with self._bucket_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(
                    spec.rate_limit.capacity, spec.rate_limit.refill_per_second
                )
                self._buckets[key] = bucket
        if not bucket.consume():
            raise ToolRateLimitExceeded("tool rate limit exceeded")

    def _idempotency_lock(self, key: tuple[str, str, str]) -> asyncio.Lock:
        with self._idempotency_lock_guard:
            return self._idempotency_locks.setdefault(key, asyncio.Lock())

    @staticmethod
    def _request_fingerprint(
        spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        payload = json.dumps(
            {
                "tool": spec.name,
                "version": spec.version,
                "arguments": dict(arguments),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def _audit(self, event: dict[str, Any]) -> None:
        if self.audit_hook is None:
            return
        outcome = self.audit_hook(redact_payload(event))
        if inspect.isawaitable(outcome):
            await outcome

    async def _execute_attempts(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> dict[str, Any]:
        for attempt in range(spec.max_retries + 1):
            if attempt:
                self.observability.counter(
                    "runtime.retries", tenant_id=context.tenant_id, tool=spec.name
                )
            started = time.perf_counter()
            span_name = (
                f"Retriever/{spec.name}"
                if spec.category == "research"
                else f"Tool/{spec.name}"
            )
            try:
                with self.observability.span(
                    span_name,
                    {
                        "tenant.id": context.tenant_id,
                        "thread.id": context.thread_id,
                        "run.id": context.run_id,
                        "tool.name": spec.name,
                        "tool.version": spec.version,
                        "retry.number": attempt,
                    },
                ):
                    output = await self.registry.execute(
                        spec.name,
                        arguments,
                        context.business_context() if spec.side_effecting else None,
                    )
            except Exception as error:
                elapsed = max(0.0, time.perf_counter() - started)
                if attempt >= spec.max_retries or not spec.idempotent:
                    self.observability.observe(
                        "tool.latency",
                        elapsed,
                        tenant_id=context.tenant_id,
                        tool=spec.name,
                    )
                    self.observability.counter(
                        "tool.errors", tenant_id=context.tenant_id, tool=spec.name
                    )
                    try:
                        await self._audit(
                            {
                                "event_type": "tool_gateway_failed",
                                "tenant_id": context.tenant_id,
                                "user_id": context.user_id,
                                "thread_id": context.thread_id,
                                "run_id": context.run_id,
                                "tool_name": spec.name,
                                "tool_version": spec.version,
                                "status": "failed",
                                "retry_number": attempt,
                                "duration_ms": elapsed * 1_000,
                                "error_type": type(error).__name__,
                            }
                        )
                    except Exception as audit_error:
                        raise ToolGatewayError(
                            "tool failure audit hook failed"
                        ) from audit_error
                    raise
                await asyncio.sleep(0)
                continue

            elapsed = max(0.0, time.perf_counter() - started)
            self.observability.observe(
                "tool.latency", elapsed, tenant_id=context.tenant_id, tool=spec.name
            )
            if spec.category == "research":
                self.observability.observe(
                    "retrieval.latency",
                    elapsed,
                    tenant_id=context.tenant_id,
                    tool=spec.name,
                )
            try:
                await self._audit(
                    {
                        "event_type": "tool_gateway_completed",
                        "tenant_id": context.tenant_id,
                        "user_id": context.user_id,
                        "thread_id": context.thread_id,
                        "run_id": context.run_id,
                        "tool_name": spec.name,
                        "tool_version": spec.version,
                        "status": "completed",
                        "retry_number": attempt,
                        "duration_ms": elapsed * 1_000,
                    }
                )
            except Exception as audit_error:
                self.observability.counter(
                    "tool.errors", tenant_id=context.tenant_id, tool=spec.name
                )
                raise ToolGatewayError(
                    "tool completed but its audit hook failed"
                ) from audit_error
            return output
        raise AssertionError("tool retry loop exhausted without outcome")

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolInvocationContext,
        version: str | None = None,
        state_counts: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        spec = self._require_tenant_tool(context.tenant_id, name)
        if version is not None and version != spec.version:
            raise ToolAccessDenied("requested tool version is not available")
        decision = self.evaluate_policy(
            tenant_id=context.tenant_id,
            role=context.role,
            tool_name=name,
            arguments=arguments,
            state_counts=state_counts,
        )
        if not decision["allowed"]:
            raise ToolAccessDenied(str(decision["decision_code"]))
        if spec.approval_required and not (
            context.authorization_granted
            and context.approval_decision in {"approve", "edit"}
        ):
            raise ToolAccessDenied("human approval is required")

        validated = spec.input_model.model_validate(arguments).model_dump(mode="json")
        idempotency_key: tuple[str, str, str] | None = None
        request_fingerprint = self._request_fingerprint(spec, validated)
        if spec.side_effecting:
            raw_key = validated.get("idempotency_key")
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ToolAccessDenied("side-effecting tools require idempotency")
            idempotency_key = (context.tenant_id, spec.name, raw_key)

        self.observability.counter(
            "tool.calls", tenant_id=context.tenant_id, tool=spec.name
        )
        if idempotency_key is None:
            self._consume_rate_limit(context.tenant_id, spec)
            return await self._execute_attempts(spec, validated, context)

        async with self._idempotency_lock(idempotency_key):
            prior = self._idempotency.get(idempotency_key)
            if prior is not None:
                prior_fingerprint, prior_output = prior
                if prior_fingerprint != request_fingerprint:
                    raise ToolIdempotencyConflict(
                        "idempotency key was used for different arguments"
                    )
                return dict(prior_output)
            self._consume_rate_limit(context.tenant_id, spec)
            output = await self._execute_attempts(spec, validated, context)
            self._idempotency[idempotency_key] = (request_fingerprint, dict(output))
            return output


__all__ = [
    "TenantToolPolicy",
    "ToolAccessDenied",
    "ToolGateway",
    "ToolGatewayError",
    "ToolIdempotencyConflict",
    "ToolInvocationContext",
    "ToolRateLimitExceeded",
]
