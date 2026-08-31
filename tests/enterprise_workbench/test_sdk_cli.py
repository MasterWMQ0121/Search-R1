from __future__ import annotations

import json

import httpx

from apps.enterprise_agent_workbench.cli import main
from apps.enterprise_agent_workbench.sdk import AgentRuntimeClient


def test_python_sdk_covers_runtime_api_and_tenant_scope():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/api/threads/thread-1/runs":
            return httpx.Response(202, json={"thread_id": "thread-1", "run_id": "run-1"})
        if path == "/api/threads/thread-1/resume":
            return httpx.Response(202, json={"thread_id": "thread-1", "status": "resumed"})
        if path.endswith("/history"):
            return httpx.Response(200, json={"thread_id": "thread-1", "history": []})
        if path.endswith("/state"):
            return httpx.Response(200, json={"thread_id": "thread-1", "tenant_id": "tenant-a"})
        if path == "/api/tools":
            return httpx.Response(200, json={"tools": [{"name": "get_campaign"}]})
        if path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/metrics":
            return httpx.Response(200, text="workbench_run_success 1\n")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    with AgentRuntimeClient(
        "http://runtime.test", transport=httpx.MockTransport(handler)
    ) as client:
        assert client.run(
            tenant_id="tenant-a", thread_id="thread-1", task="inspect"
        )["run_id"] == "run-1"
        assert client.resume(
            tenant_id="tenant-a",
            thread_id="thread-1",
            decision="approve",
        )["status"] == "resumed"
        assert client.history(tenant_id="tenant-a", thread_id="thread-1")["history"] == []
        assert client.inspect_thread(
            tenant_id="tenant-a", thread_id="thread-1"
        )["tenant_id"] == "tenant-a"
        assert client.list_tools(tenant_id="tenant-a", role="viewer")[0]["name"] == "get_campaign"
        assert client.health()["status"] == "ok"
        assert "workbench_run_success" in client.metrics()

    run_body = json.loads(requests[0].content)
    resume_body = json.loads(requests[1].content)
    assert run_body["tenant_id"] == "tenant-a"
    assert resume_body["tenant_id"] == "tenant-a"
    assert all(
        request.url.params.get("tenant_id") == "tenant-a"
        for request in requests[2:5]
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return {"status": "started"}

    def resume(self, **kwargs):
        self.calls.append(("resume", kwargs))
        return {"status": "resumed"}

    def inspect_thread(self, **kwargs):
        self.calls.append(("inspect", kwargs))
        return {"thread_id": kwargs["thread_id"]}

    def list_tools(self, **kwargs):
        self.calls.append(("tools", kwargs))
        return [{"name": "get_campaign"}]


def test_agentctl_run_resume_inspect_and_tools_commands(capsys):
    fake = FakeClient()

    def factory(**kwargs):
        assert "base_url" in kwargs
        return fake

    assert main(
        [
            "run",
            "--tenant-id",
            "tenant-a",
            "--thread-id",
            "thread-1",
            "--task",
            "inspect",
        ],
        client_factory=factory,
    ) == 0
    assert main(
        [
            "resume",
            "--tenant-id",
            "tenant-a",
            "--thread-id",
            "thread-1",
            "--decision",
            "edit",
            "--edited-arguments",
            '{"daily_budget":1200}',
        ],
        client_factory=factory,
    ) == 0
    assert main(
        [
            "threads",
            "inspect",
            "--tenant-id",
            "tenant-a",
            "--thread-id",
            "thread-1",
        ],
        client_factory=factory,
    ) == 0
    assert main(
        ["tools", "list", "--tenant-id", "tenant-a", "--role", "viewer"],
        client_factory=factory,
    ) == 0

    assert [name for name, _ in fake.calls] == ["run", "resume", "inspect", "tools"]
    assert fake.calls[1][1]["edited_arguments"] == {"daily_budget": 1200}
    assert "get_campaign" in capsys.readouterr().out

