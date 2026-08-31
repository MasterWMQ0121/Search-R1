"""Small synchronous Python SDK for the FastAPI Agent Runtime boundary."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8010"


class AgentRuntimeClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        endpoint = (
            base_url
            or os.getenv("AGENT_RUNTIME_BASE_URL")
            or os.getenv("WORKBENCH_API_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.base_url = endpoint.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AgentRuntimeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Agent Runtime returned non-object JSON")
        return payload

    def create_thread(self, *, tenant_id: str, user_id: str, role: str) -> dict[str, Any]:
        return self._object(
            self._client.post(
                "/api/threads",
                json={"tenant_id": tenant_id, "user_id": user_id, "role": role},
            )
        )

    def run(self, *, tenant_id: str, thread_id: str, task: str) -> dict[str, Any]:
        return self._object(
            self._client.post(
                f"/api/threads/{thread_id}/runs",
                json={"tenant_id": tenant_id, "task": task},
            )
        )

    def resume(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        decision: str,
        edited_arguments: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tenant_id": tenant_id, "decision": decision}
        if edited_arguments is not None:
            payload["edited_arguments"] = dict(edited_arguments)
        if feedback is not None:
            payload["feedback"] = feedback
        return self._object(
            self._client.post(f"/api/threads/{thread_id}/resume", json=payload)
        )

    def history(self, *, tenant_id: str, thread_id: str) -> dict[str, Any]:
        return self._object(
            self._client.get(
                f"/api/threads/{thread_id}/history",
                params={"tenant_id": tenant_id},
            )
        )

    def inspect_thread(self, *, tenant_id: str, thread_id: str) -> dict[str, Any]:
        return self._object(
            self._client.get(
                f"/api/threads/{thread_id}/state",
                params={"tenant_id": tenant_id},
            )
        )

    def list_tools(self, *, tenant_id: str, role: str) -> list[dict[str, Any]]:
        payload = self._object(
            self._client.get(
                "/api/tools", params={"tenant_id": tenant_id, "role": role}
            )
        )
        tools = payload.get("tools")
        if not isinstance(tools, list):
            raise RuntimeError("Agent Runtime tools response is invalid")
        return [dict(item) for item in tools if isinstance(item, dict)]

    def health(self) -> dict[str, Any]:
        return self._object(self._client.get("/healthz"))

    def metrics(self) -> str:
        response = self._client.get("/metrics")
        response.raise_for_status()
        return response.text


__all__ = ["AgentRuntimeClient", "DEFAULT_BASE_URL"]
