"""Dependency-light runtime metrics and optional OpenTelemetry export."""

from __future__ import annotations

import contextvars
import math
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


METRIC_TYPES: dict[str, str] = {
    "agent.run.latency": "histogram",
    "planner.latency": "histogram",
    "replanner.latency": "histogram",
    "llm.latency": "histogram",
    "retrieval.latency": "histogram",
    "tool.latency": "histogram",
    "tool.calls": "counter",
    "tool.errors": "counter",
    "planner.repairs": "counter",
    "grounded_argument_bindings": "counter",
    "runtime.retries": "counter",
    "input_tokens": "counter",
    "output_tokens": "counter",
    "context_tokens": "histogram",
    "compressed_tokens": "counter",
    "budget_headroom": "histogram",
    "hitl.wait_time": "histogram",
    "checkpoint.count": "counter",
    "resume.count": "counter",
    "run.success": "counter",
    "run.failure": "counter",
}

_METRIC_HELP = {
    name: f"Enterprise Agent Runtime {name.replace('.', ' ')}."
    for name in METRIC_TYPES
}
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_TOKEN_BUCKETS = (128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0)
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "workbench_current_span", default=None
)


def _prometheus_name(name: str) -> str:
    return "workbench_" + name.replace(".", "_").replace("-", "_")


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels_text(labels: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None) -> str:
    items = list(labels)
    if extra is not None:
        items.append(extra)
    if not items:
        return ""
    return "{" + ",".join(f'{key}="{_escape_label(value)}"' for key, value in items) + "}"


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    service_name: str = "enterprise-agent-runtime"
    otlp_endpoint: str | None = None


class MetricStore:
    """Thread-safe Prometheus text exporter with no collector dependency."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]
        ] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(name: str, labels: Mapping[str, Any] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        if name not in METRIC_TYPES:
            raise KeyError(f"unknown runtime metric: {name}")
        normalized = tuple(
            sorted((str(key), str(value)) for key, value in (labels or {}).items())
        )
        return name, normalized

    def increment(
        self, name: str, amount: float = 1.0, labels: Mapping[str, Any] | None = None
    ) -> None:
        if METRIC_TYPES.get(name) != "counter":
            raise ValueError(f"metric {name!r} is not a counter")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("counter increments must be finite and non-negative")
        key = self._key(name, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def observe(
        self, name: str, value: float, labels: Mapping[str, Any] | None = None
    ) -> None:
        if METRIC_TYPES.get(name) != "histogram":
            raise ValueError(f"metric {name!r} is not a histogram")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("histogram observations must be finite and non-negative")
        key = self._key(name, labels)
        buckets = _LATENCY_BUCKETS if "latency" in name or "wait_time" in name else _TOKEN_BUCKETS
        with self._lock:
            record = self._histograms.setdefault(
                key,
                {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": {boundary: 0 for boundary in buckets},
                },
            )
            record["count"] += 1
            record["sum"] += value
            for boundary in buckets:
                if value <= boundary:
                    record["buckets"][boundary] += 1

    def value(self, name: str, labels: Mapping[str, Any] | None = None) -> float:
        key = self._key(name, labels)
        with self._lock:
            if METRIC_TYPES[name] == "counter":
                return self._values.get(key, 0.0)
            return float(self._histograms.get(key, {}).get("sum", 0.0))

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            values = dict(self._values)
            histograms = {
                key: {
                    "count": record["count"],
                    "sum": record["sum"],
                    "buckets": dict(record["buckets"]),
                }
                for key, record in self._histograms.items()
            }
        for name in METRIC_TYPES:
            prom_name = _prometheus_name(name)
            metric_type = METRIC_TYPES[name]
            lines.extend(
                [
                    f"# HELP {prom_name} {_METRIC_HELP[name]}",
                    f"# TYPE {prom_name} {metric_type}",
                ]
            )
            if metric_type == "counter":
                matching = [
                    (labels, value)
                    for (sample_name, labels), value in values.items()
                    if sample_name == name
                ]
                if not matching:
                    matching = [((), 0.0)]
                for labels, value in matching:
                    lines.append(f"{prom_name}{_labels_text(labels)} {value:g}")
                continue
            matching_histograms = [
                (labels, record)
                for (sample_name, labels), record in histograms.items()
                if sample_name == name
            ]
            if not matching_histograms:
                matching_histograms = [
                    (
                        (),
                        {"count": 0, "sum": 0.0, "buckets": {}},
                    )
                ]
            for labels, record in matching_histograms:
                for boundary, count in record["buckets"].items():
                    lines.append(
                        f'{prom_name}_bucket{_labels_text(labels, ("le", str(boundary)))} {count}'
                    )
                lines.append(
                    f'{prom_name}_bucket{_labels_text(labels, ("le", "+Inf"))} {record["count"]}'
                )
                lines.append(f"{prom_name}_sum{_labels_text(labels)} {record['sum']:g}")
                lines.append(f"{prom_name}_count{_labels_text(labels)} {record['count']}")
        return "\n".join(lines) + "\n"


class RuntimeObservability:
    """Metrics plus optional OTLP spans; missing telemetry packages are harmless."""

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        self.config = config or ObservabilityConfig()
        self.metrics = MetricStore()
        self._span_records: list[dict[str, Any]] = []
        self._span_lock = threading.RLock()
        self._otel_tracer: Any = None
        self._tracer_provider: Any = None
        if self.config.enabled:
            self._configure_otel()

    def _configure_otel(self) -> None:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError:
            return
        provider = TracerProvider(
            resource=Resource.create({"service.name": self.config.service_name})
        )
        if self.config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except (ImportError, TypeError, ValueError):
                # Metrics and local span records remain available. Collector
                # configuration must never prevent local runtime startup.
                pass
        self._tracer_provider = provider
        self._otel_tracer = provider.get_tracer(
            "apps.enterprise_agent_workbench", "1.0"
        )

    @staticmethod
    def _attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in (attributes or {}).items():
            if isinstance(value, (str, bool, int, float)) and not (
                isinstance(value, float) and not math.isfinite(value)
            ):
                output[str(key)] = value
        return output

    @contextmanager
    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[None]:
        if not self.config.enabled:
            yield
            return
        parent = _current_span.get()
        token = _current_span.set(name)
        started = time.perf_counter()
        safe_attributes = self._attributes(attributes)
        otel_context = (
            self._otel_tracer.start_as_current_span(name, attributes=safe_attributes)
            if self._otel_tracer is not None
            else nullcontext()
        )
        status = "completed"
        error_type: str | None = None
        try:
            with otel_context:
                yield
        except Exception as error:
            status = "failed"
            error_type = type(error).__name__
            raise
        finally:
            duration = max(0.0, time.perf_counter() - started)
            with self._span_lock:
                self._span_records.append(
                    {
                        "name": name,
                        "parent": parent,
                        "duration_s": duration,
                        "status": status,
                        "error_type": error_type,
                        "attributes": safe_attributes,
                    }
                )
            _current_span.reset(token)

    def span_records(self) -> list[dict[str, Any]]:
        with self._span_lock:
            return [dict(record) for record in self._span_records]

    def counter(
        self, name: str, amount: float = 1.0, **labels: Any
    ) -> None:
        if self.config.enabled:
            self.metrics.increment(name, amount, labels)

    def observe(self, name: str, value: float, **labels: Any) -> None:
        if self.config.enabled:
            self.metrics.observe(name, value, labels)

    def prometheus_text(self) -> str:
        return self.metrics.render_prometheus()

    def shutdown(self) -> None:
        if self._tracer_provider is not None:
            shutdown = getattr(self._tracer_provider, "shutdown", None)
            if callable(shutdown):
                shutdown()


__all__ = [
    "METRIC_TYPES",
    "MetricStore",
    "ObservabilityConfig",
    "RuntimeObservability",
]
