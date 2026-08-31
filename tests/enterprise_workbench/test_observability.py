from __future__ import annotations

from apps.enterprise_agent_workbench.observability import (
    METRIC_TYPES,
    ObservabilityConfig,
    RuntimeObservability,
)


def test_metrics_render_prometheus_text_and_cover_runtime_contract():
    telemetry = RuntimeObservability(ObservabilityConfig(enabled=True))
    telemetry.counter("tool.calls", tenant_id="tenant-a", tool="lookup")
    telemetry.observe("tool.latency", 0.125, tenant_id="tenant-a", tool="lookup")
    telemetry.counter("run.success", tenant_id="tenant-a")

    text = telemetry.prometheus_text()

    for name in METRIC_TYPES:
        assert f"workbench_{name.replace('.', '_')}" in text
    assert 'tenant_id="tenant-a"' in text
    assert "# TYPE workbench_tool_latency histogram" in text
    assert "workbench_run_success" in text


def test_spans_preserve_real_parent_child_boundaries_without_collector():
    telemetry = RuntimeObservability(
        ObservabilityConfig(
            enabled=True,
            otlp_endpoint="http://127.0.0.1:4318/v1/traces",
        )
    )
    with telemetry.span("Agent Run", {"tenant.id": "tenant-a"}):
        with telemetry.span("Planner"):
            with telemetry.span("LLM/Planner"):
                pass
        with telemetry.span("Tool/get_campaign"):
            pass

    records = telemetry.span_records()
    by_name = {record["name"]: record for record in records}
    assert by_name["Planner"]["parent"] == "Agent Run"
    assert by_name["LLM/Planner"]["parent"] == "Planner"
    assert by_name["Tool/get_campaign"]["parent"] == "Agent Run"
    assert all(record["duration_s"] >= 0 for record in records)

