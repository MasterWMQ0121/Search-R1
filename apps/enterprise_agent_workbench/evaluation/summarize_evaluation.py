#!/usr/bin/env python3
"""Summarize measured Enterprise Agent Workbench evaluation JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apps.enterprise_agent_workbench.evaluation.run_evaluation import (
    CASES_PER_CATEGORY,
    CATEGORIES,
    EXPECTED_CASE_COUNT,
    validate_output_isolation,
)


def _rate(values: Sequence[bool]) -> float | None:
    return sum(bool(value) for value in values) / len(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _task_completed(record: Mapping[str, Any]) -> bool:
    """Require terminal execution plus every applicable case assertion."""

    required = (
        "completed",
        "final_answer_present",
        "acceptable_final_answer_facts_match",
        "tool_routing_correct",
        "source_types_match",
        "citation_valid",
        "policy_outcome_correct",
        "approval_behavior_correct",
    )
    if record.get("errors") or not all(bool(record.get(key)) for key in required):
        return False
    for applicable in (
        "permission_intercepted",
        "resume_success",
        "idempotency_correct",
        "database_state_delta_match",
    ):
        if record.get(applicable) is not None and not bool(record[applicable]):
            return False
    return True


def read_results(path: Path | str, require_complete: bool = True) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    fingerprints: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid result JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"result line {line_number} is not an object")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError(f"missing or duplicate case_id at line {line_number}")
        seen.add(case_id)
        if record.get("category") not in CATEGORIES:
            raise ValueError(f"{case_id} has an invalid category")
        if record.get("evaluation_class") != "model_dependent_end_to_end":
            raise ValueError(f"{case_id} does not identify model-dependent results")
        fingerprint = record.get("evaluator_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"{case_id} has no evaluator fingerprint")
        fingerprints.add(fingerprint)
        latency = record.get("task_latency_s")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise ValueError(f"{case_id} has invalid latency")
        if not math.isfinite(float(latency)) or latency < 0:
            raise ValueError(f"{case_id} has non-finite or negative latency")
        records.append(record)

    if len(fingerprints) != 1:
        raise ValueError("evaluation results mix multiple configurations")
    if require_complete:
        counts = Counter(record["category"] for record in records)
        expected = {category: CASES_PER_CATEGORY for category in CATEGORIES}
        if len(records) != EXPECTED_CASE_COUNT or counts != expected:
            raise ValueError(
                "complete evaluation requires exactly 24 cases and six per category"
            )
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    latencies = [float(record["task_latency_s"]) for record in records]
    permission = [
        bool(record["permission_intercepted"])
        for record in records
        if record.get("permission_intercepted") is not None
    ]
    approval = [
        bool(record["resume_success"])
        for record in records
        if record.get("resume_success") is not None
    ]
    idempotency = [
        bool(record["idempotency_correct"])
        for record in records
        if record.get("idempotency_correct") is not None
    ]
    recovery = [
        bool(record["tool_failure_recovered"])
        for record in records
        if record.get("tool_failure_recovered") is not None
    ]
    state_deltas = [
        bool(record["database_state_delta_match"])
        for record in records
        if record.get("database_state_delta_match") is not None
    ]
    completion = [
        bool(record.get("completed"))
        and bool(record.get("final_answer_present"))
        and not record.get("errors")
        for record in records
    ]
    task_completion = [_task_completed(record) for record in records]
    by_category = {}
    for category in CATEGORIES:
        subset = [record for record in records if record["category"] == category]
        by_category[category] = {
            "case_count": len(subset),
            "run_completion_rate": _rate([
                bool(record.get("completed"))
                and bool(record.get("final_answer_present"))
                and not record.get("errors")
                for record in subset
            ]),
            "task_completion_rate": _rate(
                [_task_completed(record) for record in subset]
            ),
            "tool_routing_accuracy": _rate([
                bool(record.get("tool_routing_correct")) for record in subset
            ]),
            "citation_validity_rate": _rate([
                bool(record.get("citation_valid")) for record in subset
            ]),
            "citation_coverage_mean": _mean([
                float(record.get("citation_coverage", 0.0)) for record in subset
            ]),
        }

    return {
        "schema_version": 2,
        "evaluation_scope": {
            "model_dependent_end_to_end": True,
            "deterministic_infrastructure_policy_tests": (
                "Reported by the CPU pytest suite; not inferred from model runs."
            ),
            "result_claim": (
                "Metrics are computed only from the supplied live result records."
            ),
        },
        "evaluator_fingerprint": records[0]["evaluator_fingerprint"],
        "case_count": len(records),
        "category_counts": dict(Counter(record["category"] for record in records)),
        "metrics": {
            # This is deliberately named run completion: semantic task success
            # is measured separately by routing, facts, policy, and state-delta
            # metrics below.
            "run_completion_rate": _rate(completion),
            "task_completion_rate": _rate(task_completion),
            "tool_routing_accuracy": _rate([
                bool(record.get("tool_routing_correct")) for record in records
            ]),
            "prohibited_tool_execution_count": sum(
                int(record.get("prohibited_tool_execution_count", 0))
                for record in records
            ),
            "permission_interception_rate": _rate(permission),
            "permission_interception_case_count": len(permission),
            "approval_resume_success_rate": _rate(approval),
            "approval_resume_case_count": len(approval),
            "idempotency_correctness_rate": _rate(idempotency),
            "idempotency_case_count": len(idempotency),
            "database_state_delta_accuracy": _rate(state_deltas),
            "database_state_delta_case_count": len(state_deltas),
            "citation_validity_rate": _rate([
                bool(record.get("citation_valid")) for record in records
            ]),
            "citation_coverage_mean": _mean([
                float(record.get("citation_coverage", 0.0)) for record in records
            ]),
            "tool_failure_recovery_rate": _rate(recovery),
            "tool_failure_case_count": len(recovery),
            "mean_task_latency_s": _mean(latencies),
            "p50_task_latency_s": _percentile(latencies, 0.50),
            "p95_task_latency_s": _percentile(latencies, 0.95),
            "mean_tool_calls": _mean([
                float(record.get("tool_call_count", 0)) for record in records
            ]),
            "mean_planner_repairs": _mean([
                float(record.get("planner_repair_count", 0)) for record in records
            ]),
            "unfinished_run_count": sum(not bool(record.get("completed")) for record in records),
            "acceptable_final_answer_facts_rate": _rate([
                bool(record.get("acceptable_final_answer_facts_match"))
                for record in records
            ]),
        },
        "by_category": by_category,
        "failures": [
            {
                "case_id": record["case_id"],
                "completed": record.get("completed"),
                "termination_reason": record.get("termination_reason"),
                "errors": record.get("errors", []),
                "tool_routing_correct": record.get("tool_routing_correct"),
                "citation_valid": record.get("citation_valid"),
            }
            for record in records
            if (
                not record.get("completed")
                or record.get("errors")
                or not record.get("tool_routing_correct")
                or not record.get("citation_valid")
            )
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]

    def display(value: Any) -> str:
        if value is None:
            return "not measured"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Enterprise Agent Workbench Evaluation",
        "",
        "These values are computed from live result records; no model-quality value is hard-coded.",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Run completion rate: {display(metrics['run_completion_rate'])}",
        f"- Task completion rate: {display(metrics['task_completion_rate'])}",
        f"- Tool-routing accuracy: {display(metrics['tool_routing_accuracy'])}",
        f"- Prohibited-tool executions: {metrics['prohibited_tool_execution_count']}",
        f"- Permission interception rate: {display(metrics['permission_interception_rate'])}",
        f"- Approval/resume success rate: {display(metrics['approval_resume_success_rate'])}",
        f"- Idempotency correctness rate: {display(metrics['idempotency_correctness_rate'])}",
        f"- Database state-delta accuracy: {display(metrics['database_state_delta_accuracy'])}",
        f"- Citation validity rate: {display(metrics['citation_validity_rate'])}",
        f"- Citation coverage: {display(metrics['citation_coverage_mean'])}",
        f"- Tool-failure recovery rate: {display(metrics['tool_failure_recovery_rate'])}",
        f"- Mean/p50/p95 latency (s): {display(metrics['mean_task_latency_s'])} / "
        f"{display(metrics['p50_task_latency_s'])} / {display(metrics['p95_task_latency_s'])}",
        f"- Mean tool calls: {display(metrics['mean_tool_calls'])}",
        f"- Mean planner repairs: {display(metrics['mean_planner_repairs'])}",
        f"- Unfinished runs: {metrics['unfinished_run_count']}",
        "",
        "Deterministic infrastructure and policy correctness is reported by the CPU test suite, separately from these model-dependent runs.",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_json = validate_output_isolation(args.output_json)
    output_markdown = validate_output_isolation(args.output_markdown)
    records = read_results(args.results, require_complete=not args.allow_partial)
    payload = summarize(records)
    _atomic_write(
        output_json,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_markdown, render_markdown(payload))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
