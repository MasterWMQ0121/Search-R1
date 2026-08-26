from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.enterprise_agent_workbench.tracing import (
    ExecutionTrace,
    TraceEvent,
    redact_payload,
)


def test_trace_events_are_ordered_and_latencies_non_negative():
    trace = ExecutionTrace(thread_id="thread-1", run_id="run-1")
    trace.record("run_started", status="started", node="load_context")
    trace.record("node_completed", status="completed", node="load_context", duration_ms=1.2)
    events = trace.events()
    assert [event["sequence"] for event in events] == [0, 1]
    assert events[0]["timestamp"] <= events[1]["timestamp"]
    assert events[1]["duration_ms"] == 1.2

    with pytest.raises(ValidationError):
        TraceEvent(
            sequence=0,
            timestamp="now",
            event_type="node_completed",
            thread_id="thread-1",
            run_id="run-1",
            status="completed",
            duration_ms=-1,
        )


def test_trace_redacts_secrets_and_omits_hidden_reasoning():
    trace = ExecutionTrace(thread_id="thread-1", run_id="run-1")
    trace.record(
        "tool_completed",
        status="completed",
        tool_name="campaign_read",
        input_summary={
            "campaign_id": "C102",
            "authorization": "Bearer abcdefghijklmnop",
            "chain_of_thought": "private reasoning",
            "nested": {"api_key": "top-secret"},
        },
        output_summary={"message": "used Bearer abcdefghijklmnop safely"},
    )
    event = trace.events()[0]
    safe_input = event["safe_input_summary"]
    assert safe_input["authorization"] == "[REDACTED]"
    assert "chain_of_thought" not in safe_input
    assert safe_input["nested"]["api_key"] == "[REDACTED]"
    assert "Bearer [REDACTED]" in event["safe_output_summary"]["message"]


def test_tool_failure_records_safe_error_type():
    trace = ExecutionTrace(thread_id="thread-1", run_id="run-1")
    trace.record(
        "tool_failed",
        status="failed",
        tool_name="merchant_analytics",
        error_type="TimeoutError",
        retry_number=1,
    )
    event = trace.events()[0]
    assert event["error_type"] == "TimeoutError"
    assert event["retry_number"] == 1


def test_redaction_bounds_large_or_non_json_payloads():
    redacted = redact_payload(
        {
            "items": list(range(100)),
            "object": object(),
            "message": "backend failed with api_key=do-not-leak",
        }
    )
    assert len(redacted["items"]) == 50
    assert redacted["object"] == "<object>"
    assert "do-not-leak" not in redacted["message"]
