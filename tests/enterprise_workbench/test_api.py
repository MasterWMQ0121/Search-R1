from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from apps.enterprise_agent_workbench.api import (
    _runtime_identity,
    build_default_service,
    create_app,
)
from apps.enterprise_agent_workbench.config import WorkbenchSettings
from apps.enterprise_agent_workbench.model_client import FakeModelClient, PlannerDecision
from apps.enterprise_agent_workbench.tokenizer_runtime import TokenizerRuntime
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
)


def decision(action, arguments=None, *, completed=False):
    return PlannerDecision(
        objective="Complete the API task.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="Concise visible planner update.",
    )


def service_for(tmp_path, decisions, answer, *, tokenizer_runtime=None):
    settings = replace(
        WorkbenchSettings.from_env(),
        data_dir=tmp_path,
        max_graph_steps=18,
        max_tool_calls=8,
    )
    model = FakeModelClient(decisions, synthesized_answer=answer)
    service = build_default_service(
        settings=settings,
        checkpointer=InMemorySaver(),
        model_client=model,
        **(
            {"tokenizer_runtime": tokenizer_runtime}
            if tokenizer_runtime is not None
            else {"tokenizer": DeterministicTextTokenizer()}
        ),
    )
    app = create_app(service)
    app.state.workbench = service
    return app, service


def sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payloads.append(json.loads(line.partition(":")[2].strip()))
    return payloads


@pytest.mark.asyncio
async def test_api_validates_projects_state_and_streams_sse(tmp_path):
    app, service = service_for(
        tmp_path,
        [decision("get_campaign", {"campaign_id": "C102"}), decision("finalizer", completed=True)],
        "C102 is active [BUSINESS:C102:get_campaign].",
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["model_process"] == "external"
            assert health.json()["tokenizer_mode"] == "approximate_test"
            assert len(health.json()["runtime_configuration_fingerprint"]) == 64
            tools = await client.get("/api/tools")
            assert tools.status_code == 200
            assert any(item["name"] == "get_campaign" for item in tools.json()["tools"])

            invalid = await client.post(
                "/api/threads",
                json={"user_id": "u", "organization_id": "o", "role": "root"},
            )
            assert invalid.status_code == 422
            created = await client.post(
                "/api/threads",
                json={"user_id": "u", "organization_id": "o", "role": "analyst"},
            )
            thread_id = created.json()["thread_id"]
            started = await client.post(
                f"/api/threads/{thread_id}/runs", json={"task": "Show C102."}
            )
            assert started.status_code == 202
            streamed = await client.get(f"/api/threads/{thread_id}/stream")
            assert streamed.headers["content-type"].startswith("text/event-stream")
            events = sse_payloads(streamed.text)
            assert any(item.get("user_visible_reason") for item in events)
            assert any(item.get("event_type") == "tool_completed" for item in events)
            assert any(item.get("event_type") == "run_completed" for item in events)

            state = (await client.get(f"/api/threads/{thread_id}/state")).json()
            assert state["completed"] is True
            assert state["final_citations"] == ["BUSINESS:C102:get_campaign"]
            assert "execution_trace" not in state
            assert "document_path" not in json.dumps(state["sources"])
            history = (await client.get(f"/api/threads/{thread_id}/history")).json()
            assert history["history"]
            trace = (await client.get(f"/api/threads/{thread_id}/trace")).json()
            assert trace["events"]
            assert "chain_of_thought" not in json.dumps(trace).lower()

            stored = await client.put(
                "/api/users/u/memories/preferred_kpi",
                json={"organization_id": "o", "value": "ROI"},
            )
            assert stored.status_code == 200
            memories = await client.get(
                "/api/users/u/memories", params={"organization_id": "o"}
            )
            assert memories.json()["preferences"] == {"preferred_kpi": "ROI"}
            rejected = await client.put(
                "/api/users/u/memories/api_token",
                json={"organization_id": "o", "value": "secret"},
            )
            assert rejected.status_code == 422
            deleted = await client.delete(
                "/api/users/u/memories/preferred_kpi",
                params={"organization_id": "o"},
            )
            assert deleted.json()["deleted"] is True
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_health_reports_exact_tokenizer_and_compressor_runtime_contract(tmp_path):
    tokenizer = DeterministicTextTokenizer()
    exact_runtime = TokenizerRuntime(
        tokenizer=tokenizer,
        mode="exact",
        configured_path=".../actor/global_step_20",
        tokenizer_class="Qwen2TokenizerFast",
        artifact_fingerprint="a" * 64,
        vocabulary_size=151_665,
        pad_token_id=151_643,
        eos_token_id=151_643,
    )
    app, service = service_for(
        tmp_path,
        [decision("finalizer", completed=True)],
        "Exact tokenizer health check.",
        tokenizer_runtime=exact_runtime,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = (await client.get("/healthz")).json()
        assert health["tokenizer_mode"] == "exact"
        assert health["tokenizer_path"] == ".../actor/global_step_20"
        assert health["tokenizer_class"] == "Qwen2TokenizerFast"
        assert health["tokenizer_artifact_fingerprint"] == "a" * 64
        assert (
            health["evidence_compressor_policy"]
            == "deterministic_query_aware_extractive"
        )
        assert health["evidence_compressor_version"] == "phase5-extractive-v1"
        assert len(health["evidence_compressor_fingerprint"]) == 64
        assert health["max_evidence_token_budget"] == 256
        assert health["model_name"] == "phase3-search-r1"
        assert health["retriever_configuration"] == {
            "url": "http://127.0.0.1:8000/retrieve",
            "top_k": 3,
        }
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_api_interrupt_and_same_thread_resume(tmp_path):
    app, service = service_for(
        tmp_path,
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": 1200,
                },
            ),
            decision("finalizer", completed=True),
        ],
        "The local mock update completed.",
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/threads",
                json={"user_id": "operator", "organization_id": "org", "role": "operator"},
            )
            thread_id = created.json()["thread_id"]
            await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"task": "Increase C102 budget to 1200."},
            )
            first_stream = await client.get(f"/api/threads/{thread_id}/stream")
            assert "approval_request" in first_stream.text
            paused_state = (await client.get(f"/api/threads/{thread_id}/state")).json()
            assert paused_state["approval_request"]["action"] == "update_campaign_budget"
            generated_key = paused_state["pending_action"]["arguments"][
                "idempotency_key"
            ]
            assert generated_key.startswith("wb-")
            assert (
                paused_state["approval_request"]["arguments"]["idempotency_key"]
                == generated_key
            )
            with sqlite3.connect(service.settings.business_db_path) as connection:
                assert connection.execute(
                    "SELECT daily_budget FROM campaigns WHERE campaign_id='C102'"
                ).fetchone()[0] == 1000.0

            resumed = await client.post(
                f"/api/threads/{thread_id}/resume", json={"decision": "approve"}
            )
            assert resumed.status_code == 202
            assert resumed.json()["thread_id"] == thread_id
            second_stream = await client.get(f"/api/threads/{thread_id}/stream")
            assert "run_completed" in second_stream.text
            with sqlite3.connect(service.settings.business_db_path) as connection:
                assert connection.execute(
                    "SELECT daily_budget FROM campaigns WHERE campaign_id='C102'"
                ).fetchone()[0] == 1200.0
                assert connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_tokenizer_runtime_mismatch_rejects_approval_resume(tmp_path):
    tokenizer = DeterministicTextTokenizer()
    original_runtime = TokenizerRuntime(
        tokenizer=tokenizer,
        mode="exact",
        configured_path=".../actor/global_step_20",
        tokenizer_class="Qwen2TokenizerFast",
        artifact_fingerprint="a" * 64,
        vocabulary_size=151_665,
        pad_token_id=151_643,
        eos_token_id=151_643,
    )
    app, service = service_for(
        tmp_path,
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": 1200,
                },
            )
        ],
        "No write should occur.",
        tokenizer_runtime=original_runtime,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/threads",
                json={
                    "user_id": "operator",
                    "organization_id": "org",
                    "role": "operator",
                },
            )
            thread_id = created.json()["thread_id"]
            await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"task": "Increase C102 budget."},
            )
            await client.get(f"/api/threads/{thread_id}/stream")

            changed_runtime = replace(
                original_runtime, artifact_fingerprint="b" * 64
            )
            _, service.runtime_configuration_fingerprint = _runtime_identity(
                service.settings, changed_runtime
            )
            response = await client.post(
                f"/api/threads/{thread_id}/resume", json={"decision": "approve"}
            )

            assert response.status_code == 409
            assert "runtime configuration fingerprint mismatch" in response.json()[
                "detail"
            ]
            with sqlite3.connect(service.settings.business_db_path) as connection:
                assert connection.execute(
                    "SELECT daily_budget FROM campaigns WHERE campaign_id='C102'"
                ).fetchone()[0] == 1000.0
                assert connection.execute(
                    "SELECT COUNT(*) FROM audit_log"
                ).fetchone()[0] == 0
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_api_rejects_approval_edit_that_changes_target(tmp_path):
    app, service = service_for(
        tmp_path,
        [
            decision(
                "update_campaign_budget",
                {
                    "campaign_id": "C102",
                    "daily_budget": 1500,
                },
            )
        ],
        "No write should occur.",
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/threads",
                json={
                    "user_id": "operator",
                    "organization_id": "org",
                    "role": "operator",
                },
            )
            thread_id = created.json()["thread_id"]
            await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"task": "Increase C102 budget."},
            )
            await client.get(f"/api/threads/{thread_id}/stream")
            paused = (await client.get(f"/api/threads/{thread_id}/state")).json()
            generated_key = paused["pending_action"]["arguments"][
                "idempotency_key"
            ]

            response = await client.post(
                f"/api/threads/{thread_id}/resume",
                json={
                    "decision": "edit",
                    "edited_arguments": {
                        "campaign_id": "C101",
                        "daily_budget": 1200,
                        "idempotency_key": generated_key,
                    },
                },
            )

            assert response.status_code == 422
            state = (await client.get(f"/api/threads/{thread_id}/state")).json()
            assert state["approval_request"]["action"] == "update_campaign_budget"
            with sqlite3.connect(service.settings.business_db_path) as connection:
                assert connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_runtime_failure_is_redacted_and_durably_checkpointed(tmp_path):
    app, service = service_for(
        tmp_path,
        [decision("finalizer", completed=True)],
        "The read-only run completed.",
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/threads",
                json={"user_id": "u", "organization_id": "o", "role": "viewer"},
            )
            thread_id = created.json()["thread_id"]
            await client.post(
                f"/api/threads/{thread_id}/runs", json={"task": "Summarize safely."}
            )
            await client.get(f"/api/threads/{thread_id}/stream")

            safe_message = await service._record_runtime_failure(
                thread_id, RuntimeError("backend api_key=do-not-leak failed")
            )

            assert "do-not-leak" not in safe_message
            state = (await client.get(f"/api/threads/{thread_id}/state")).json()
            assert state["termination_reason"] == "runtime_error"
            assert state["errors"][-1]["code"] == "runtime_error"
            assert "do-not-leak" not in json.dumps(state)
            trace = (await client.get(f"/api/threads/{thread_id}/trace")).json()
            assert trace["events"][-1]["event_type"] == "run_failed"
    finally:
        await service.shutdown()
