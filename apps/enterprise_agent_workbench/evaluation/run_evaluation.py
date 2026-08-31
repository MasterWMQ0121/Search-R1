#!/usr/bin/env python3
"""Run the 24-case live Enterprise Agent Workbench evaluation.

This runner records only responses observed from a running FastAPI workbench.
It has no synthetic-success mode and therefore cannot fabricate model quality.
CPU-safe unit tests exercise its deterministic scoring helpers separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from apps.enterprise_agent_workbench.citations import citation_coverage
from verl.utils.reward_score.qa_em import normalize_answer


CATEGORIES = (
    "knowledge_only",
    "analytics_only",
    "mixed_read_only",
    "write_approval_permission",
)
CASES_PER_CATEGORY = 6
EXPECTED_CASE_COUNT = 24
DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_DATA_ROOT = Path(
    os.environ.get("WORKBENCH_DATA_DIR", "~/.search_r1_workbench")
).expanduser()
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "evaluation" / "results.jsonl"
FORBIDDEN_ARTIFACT_DIRECTORIES = {
    "phase4_benchmark_results",
    "phase5_observation_results",
}
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})\]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path | str = DEFAULT_CASES) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluation JSON at line {line_number}") from error
        if not isinstance(case, dict):
            raise ValueError(f"evaluation case {line_number} is not an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"evaluation case {line_number} has no case_id")
        if case_id in seen:
            raise ValueError(f"duplicate evaluation case_id: {case_id}")
        seen.add(case_id)
        if case.get("category") not in CATEGORIES:
            raise ValueError(f"{case_id} has an unsupported category")
        if case.get("evaluation_class") != "model_dependent_end_to_end":
            raise ValueError(f"{case_id} must be marked model-dependent")
        if not isinstance(case.get("task"), str) or not case["task"].strip():
            raise ValueError(f"{case_id} has an empty task")
        identity = case.get("identity")
        if not isinstance(identity, dict) or identity.get("role") not in {
            "viewer",
            "analyst",
            "operator",
            "admin",
        }:
            raise ValueError(f"{case_id} has an invalid identity")
        if not isinstance(case.get("expectations"), dict):
            raise ValueError(f"{case_id} has no expectations object")
        cases.append(case)

    counts = Counter(case["category"] for case in cases)
    expected = {category: CASES_PER_CATEGORY for category in CATEGORIES}
    if len(cases) != EXPECTED_CASE_COUNT or counts != expected:
        raise ValueError(
            f"evaluation must contain exactly 24 cases and six per category; "
            f"found total={len(cases)}, counts={dict(counts)}"
        )
    return cases


def validate_output_isolation(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part in FORBIDDEN_ARTIFACT_DIRECTORIES for part in resolved.parts):
        raise ValueError(
            "workbench evaluation output must not be written into Phase-4 or "
            "Phase-5 result directories"
        )
    return resolved


def citation_audit(answer: str, sources: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_ids = {
        str(source.get("source_id"))
        for source in sources
        if isinstance(source.get("source_id"), str)
    }
    cited = set(CITATION_PATTERN.findall(answer or ""))
    unknown = sorted(cited.difference(source_ids))
    coverage_result = citation_coverage(answer or "")
    return {
        "valid": not unknown,
        "cited_source_ids": sorted(cited),
        "unknown_citation_ids": unknown,
        "evidence_dependent_section_count": coverage_result[
            "evidence_dependent_sections"
        ],
        "cited_evidence_dependent_section_count": coverage_result[
            "cited_evidence_sections"
        ],
        "coverage": coverage_result["coverage"],
        "approximation": coverage_result["method"],
    }


def acceptable_facts_match(answer: str, facts: list[str]) -> bool:
    normalized_answer = normalize_answer(answer or "")
    return all(normalize_answer(str(fact)) in normalized_answer for fact in facts)


def _event_name(event: Mapping[str, Any]) -> str:
    value = event.get("event", event.get("event_type", ""))
    return str(value)


def _tool_name(event: Mapping[str, Any]) -> str | None:
    value = event.get("tool_name", event.get("tool"))
    return str(value) if isinstance(value, str) and value else None


def score_trace(
    trace: list[Mapping[str, Any]],
    tool_categories: Mapping[str, str],
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    completed_tools = [
        _tool_name(event)
        for event in trace
        if _event_name(event) == "tool_completed" and _tool_name(event)
    ]
    started_tools = [
        _tool_name(event)
        for event in trace
        if _event_name(event) == "tool_started" and _tool_name(event)
    ]
    actual_categories = sorted(
        {tool_categories[name] for name in completed_tools if name in tool_categories}
    )
    expected_categories = set(expectations.get("tool_categories", []))
    prohibited = set(expectations.get("prohibited_tools", []))
    # A tool invocation is an execution attempt as soon as it starts. Counting
    # only completions would incorrectly treat a prohibited tool that failed
    # after starting (and may already have had a side effect) as intercepted.
    prohibited_count = sum(
        name in prohibited or tool_categories.get(name) in prohibited
        for name in started_tools
    )

    names = [_event_name(event) for event in trace]
    expected_approval = expectations.get("approval_behavior", "none")
    approval_requested = "approval_requested" in names
    if expected_approval == "none":
        approval_behavior_match = not approval_requested
    elif expected_approval == "permission_denied":
        approval_behavior_match = not approval_requested and "policy_denied" in names
    else:
        expected_event = {
            "approve": "approval_approved",
            "edit": "approval_edited",
            "reject": "approval_rejected",
        }[expected_approval]
        approval_behavior_match = approval_requested and expected_event in names

    expected_policy = expectations.get("policy_outcome")
    policy_match = (
        "policy_denied" in names
        if expected_policy == "denied"
        else "policy_denied" not in names
    )
    permission_intercepted = None
    if expected_policy == "denied":
        permission_intercepted = "policy_denied" in names and prohibited_count == 0

    failures = [index for index, name in enumerate(names) if name == "tool_failed"]
    recovery = None
    if failures:
        recovery = any(
            any(later in {"tool_completed", "final_answer_created", "run_completed"}
                for later in names[index + 1 :])
            for index in failures
        )

    return {
        "tools_started": started_tools,
        "tools_completed": completed_tools,
        "tool_categories": actual_categories,
        "tool_routing_correct": expected_categories.issubset(actual_categories)
        and prohibited_count == 0,
        "prohibited_tool_execution_count": prohibited_count,
        "policy_outcome_correct": policy_match,
        "permission_intercepted": permission_intercepted,
        "approval_requested": approval_requested,
        "approval_behavior_correct": approval_behavior_match,
        "tool_failure_recovered": recovery,
        "tool_call_count": len(started_tools),
        "planner_repair_count": names.count("planner_repair"),
        "event_count": len(trace),
    }


class LiveWorkbenchClient:
    def __init__(self, base_url: str, timeout_seconds: float):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)
        self._thread_tenants: dict[str, str] = {}

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _object(response: Any, label: str) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"{label} returned non-object JSON")
        return payload

    def tool_categories(self) -> dict[str, str]:
        response = self.client.get(
            "/api/tools", params={"tenant_id": "evaluation", "role": "admin"}
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("tools", [])
        if not isinstance(payload, list):
            raise RuntimeError("tool registry response is not a list")
        return {
            str(tool["name"]): str(tool["category"])
            for tool in payload
            if isinstance(tool, dict) and "name" in tool and "category" in tool
        }

    def runtime_descriptor(self) -> dict[str, Any]:
        """Return stable, public server metadata safe to bind into a run."""

        health = self._object(self.client.get("/healthz"), "health check")
        response = self.client.get(
            "/api/tools", params={"tenant_id": "evaluation", "role": "admin"}
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("tools", [])
        if not isinstance(payload, list):
            raise RuntimeError("tool registry response is not a list")
        tools = [dict(tool) for tool in payload if isinstance(tool, dict)]
        tools.sort(key=lambda tool: str(tool.get("name", "")))
        return {"health": health, "tools": tools}

    def create_thread(self, identity: Mapping[str, Any]) -> str:
        tenant_id = str(
            identity.get("tenant_id") or identity.get("organization_id") or ""
        )
        if not tenant_id:
            raise ValueError("evaluation identity requires tenant_id")
        request_identity = {
            **dict(identity),
            "tenant_id": tenant_id,
            "organization_id": tenant_id,
        }
        payload = self._object(
            self.client.post("/api/threads", json=request_identity), "create thread"
        )
        thread_id = payload.get("thread_id")
        if thread_id is None and isinstance(payload.get("thread"), dict):
            thread_id = payload["thread"].get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise RuntimeError("create thread response has no thread_id")
        self._thread_tenants[thread_id] = tenant_id
        return thread_id

    def start_run(self, thread_id: str, task: str) -> dict[str, Any]:
        return self._object(
            self.client.post(
                f"/api/threads/{thread_id}/runs",
                json={"tenant_id": self._thread_tenants[thread_id], "task": task},
            ),
            "start run",
        )

    def state(self, thread_id: str) -> dict[str, Any]:
        payload = self._object(
            self.client.get(
                f"/api/threads/{thread_id}/state",
                params={"tenant_id": self._thread_tenants[thread_id]},
            ),
            "thread state",
        )
        nested = payload.get("state")
        return dict(nested) if isinstance(nested, dict) else payload

    def trace(self, thread_id: str) -> list[dict[str, Any]]:
        response = self.client.get(
            f"/api/threads/{thread_id}/trace",
            params={"tenant_id": self._thread_tenants[thread_id]},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("events", payload.get("trace", []))
        if not isinstance(payload, list):
            raise RuntimeError("trace response does not contain a list")
        return [dict(event) for event in payload if isinstance(event, dict)]

    def resume(self, thread_id: str, decision: Mapping[str, Any]) -> dict[str, Any]:
        return self._object(
            self.client.post(
                f"/api/threads/{thread_id}/resume",
                json={
                    **dict(decision),
                    "tenant_id": self._thread_tenants[thread_id],
                },
            ),
            "resume run",
        )


def _terminal_or_paused(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("completed")
        or state.get("approval_request")
        or state.get("termination_reason") in {
            "completed",
            "failed",
            "max_steps",
            "max_tool_calls",
            "planner_parse_error",
        }
    )


def wait_for_state(
    client: LiveWorkbenchClient,
    thread_id: str,
    timeout_seconds: float,
    poll_seconds: float = 0.25,
    *,
    allow_paused: bool = True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = client.state(thread_id)
        if state.get("completed") or state.get("termination_reason"):
            return state
        if allow_paused and state.get("approval_request"):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"thread {thread_id} did not pause or complete in time")
        time.sleep(poll_seconds)


def _sources(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = state.get("sources", [])
    if not isinstance(sources, list):
        return []
    return [dict(source) for source in sources if isinstance(source, dict)]


def _source_types(sources: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(source.get("source_type"))
        for source in sources
        if isinstance(source.get("source_type"), str)
    }


def database_state_delta_matches(
    state: Mapping[str, Any],
    expected_delta: Mapping[str, Any] | None,
    tool_categories: Mapping[str, str],
) -> bool | None:
    """Check expected local-mock state only from recorded write outputs."""

    if expected_delta is None:
        return None
    results = state.get("tool_results", [])
    writes = [
        result
        for result in results
        if isinstance(result, Mapping)
        and result.get("status") == "ok"
        and tool_categories.get(str(result.get("tool_name"))) == "business_write"
    ] if isinstance(results, list) else []
    if not expected_delta:
        return not writes
    for result in writes:
        output = result.get("output", {})
        after = output.get("after_state", {}) if isinstance(output, Mapping) else {}
        if isinstance(after, Mapping) and all(
            after.get(key) == value for key, value in expected_delta.items()
        ):
            return True
    return False


def run_case(
    client: LiveWorkbenchClient,
    case: Mapping[str, Any],
    tool_categories: Mapping[str, str],
    timeout_seconds: float,
    evaluator_fingerprint: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    expectations = case["expectations"]
    thread_id: str | None = None
    run_id: str | None = None
    errors: list[str] = []
    state: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    resume_success: bool | None = None
    idempotency_correct: bool | None = None

    try:
        thread_id = client.create_thread(case["identity"])
        start_payload = client.start_run(thread_id, case["task"])
        value = start_payload.get("run_id")
        run_id = str(value) if value is not None else None
        state = wait_for_state(client, thread_id, timeout_seconds)

        approval = expectations.get("approval_decision")
        if isinstance(approval, dict):
            if not isinstance(state.get("approval_request"), dict):
                errors.append("expected approval interrupt was not observed")
                resume_success = False
            else:
                approval_payload = dict(approval)
                if approval_payload.get("decision") == "edit":
                    request_arguments = state["approval_request"].get("arguments", {})
                    edits = approval_payload.get("edited_arguments", {})
                    if not isinstance(request_arguments, dict) or not isinstance(edits, dict):
                        raise ValueError("edit approval arguments must be JSON objects")
                    approval_payload["edited_arguments"] = {
                        **request_arguments,
                        **edits,
                    }
                client.resume(thread_id, approval_payload)
                state = wait_for_state(
                    client, thread_id, timeout_seconds, allow_paused=False
                )
                resume_success = bool(state.get("completed"))
                if expectations.get("repeat_resume"):
                    trace_before_retry = client.trace(thread_id)
                    completed_before = sum(
                        _event_name(event) == "tool_completed"
                        and tool_categories.get(_tool_name(event)) == "business_write"
                        for event in trace_before_retry
                    )
                    try:
                        client.resume(thread_id, approval)
                    except Exception:
                        pass
                    trace_after_retry = client.trace(thread_id)
                    completed_after = sum(
                        _event_name(event) == "tool_completed"
                        and tool_categories.get(_tool_name(event)) == "business_write"
                        for event in trace_after_retry
                    )
                    idempotency_correct = completed_after == completed_before
        trace = client.trace(thread_id)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        if thread_id:
            try:
                state = client.state(thread_id)
                trace = client.trace(thread_id)
            except Exception as inspection_error:
                errors.append(
                    f"state_inspection_failed:{type(inspection_error).__name__}"
                )

    trace_metrics = score_trace(trace, tool_categories, expectations)
    answer = str(state.get("final_answer") or "")
    sources = _sources(state)
    citations = citation_audit(answer, sources)
    expected_source_types = set(expectations.get("source_types", []))
    source_types_match = expected_source_types.issubset(_source_types(sources))
    state_delta_match = database_state_delta_matches(
        state,
        expectations.get("expected_database_state_delta"),
        tool_categories,
    )
    latency = time.perf_counter() - started
    if not math.isfinite(latency) or latency < 0:
        raise AssertionError("evaluation latency must be finite and non-negative")

    return {
        "schema_version": 1,
        "evaluator_fingerprint": evaluator_fingerprint,
        "case_id": case["case_id"],
        "category": case["category"],
        "evaluation_class": case["evaluation_class"],
        "thread_id": thread_id,
        "run_id": run_id,
        "role": case["identity"]["role"],
        "completed": bool(state.get("completed")),
        "termination_reason": state.get("termination_reason"),
        "task_latency_s": latency,
        "final_answer_present": bool(answer.strip()),
        "acceptable_final_answer_facts_match": acceptable_facts_match(
            answer, expectations.get("acceptable_final_answer_facts", [])
        ),
        "expected_source_types": sorted(expected_source_types),
        "observed_source_types": sorted(_source_types(sources)),
        "source_types_match": source_types_match,
        "database_state_delta_match": state_delta_match,
        "source_count": len(sources),
        "citation_valid": citations["valid"],
        "citation_coverage": citations["coverage"],
        "citation_audit": citations,
        "resume_success": resume_success,
        "idempotency_correct": idempotency_correct,
        "errors": errors,
        **trace_metrics,
    }


def _append_result(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_existing(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid result JSON at line {line_number}") from error
        if record.get("evaluator_fingerprint") != fingerprint:
            raise ValueError("existing evaluation was produced by a different configuration")
        case_id = record.get("case_id")
        if case_id in records:
            raise ValueError(f"duplicate existing result for {case_id}")
        records[case_id] = record
    return records


def validate_live_evaluation_contract(
    run_config_identity: str,
    fresh_isolated_database_confirmed: bool,
) -> str:
    """Require the caller to bind external services and isolated mutable data."""

    identity = run_config_identity.strip()
    if not identity:
        raise ValueError(
            "--run-config-identity must identify the app revision, model "
            "checkpoint, Retriever/index configuration, and fixture seed"
        )
    if not fresh_isolated_database_confirmed:
        raise ValueError(
            "live evaluation requires --fresh-isolated-database-confirmed; "
            "use a freshly seeded database dedicated to this evaluation "
            "identity (or the unchanged database when resuming that same run)"
        )
    return identity


def evaluator_fingerprint(
    case_path: Path,
    base_url: str,
    run_config_identity: str,
    runtime_descriptor: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {
            "schema_version": 2,
            "cases_sha256": sha256_file(case_path),
            "base_url": base_url.rstrip("/"),
            "run_config_identity": run_config_identity,
            "runtime_descriptor": runtime_descriptor,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--run-config-identity",
        required=True,
        help=(
            "Immutable identity covering app revision, model checkpoint, "
            "Retriever/index configuration, fixture seed, and isolated run."
        ),
    )
    parser.add_argument(
        "--fresh-isolated-database-confirmed",
        action="store_true",
        help=(
            "Confirm that the server uses a freshly seeded database dedicated "
            "to this run, or its unchanged database when resuming the same run."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be finite and positive")
    case_path = Path(args.cases).expanduser().resolve()
    output_path = validate_output_isolation(args.output)
    cases = load_cases(case_path)
    run_config_identity = validate_live_evaluation_contract(
        args.run_config_identity,
        args.fresh_isolated_database_confirmed,
    )

    client = LiveWorkbenchClient(args.api_base_url, args.timeout_seconds)
    try:
        runtime_descriptor = client.runtime_descriptor()
        fingerprint = evaluator_fingerprint(
            case_path,
            args.api_base_url,
            run_config_identity,
            runtime_descriptor,
        )
        if args.overwrite and output_path.exists():
            output_path.unlink()
        existing = _read_existing(output_path, fingerprint)
        unexpected = sorted(set(existing).difference(case["case_id"] for case in cases))
        if unexpected:
            raise ValueError(f"result file contains unknown cases: {unexpected}")
        categories = client.tool_categories()
        for case in cases:
            if case["case_id"] in existing:
                continue
            record = run_case(
                client,
                case,
                categories,
                args.timeout_seconds,
                fingerprint,
            )
            _append_result(output_path, record)
            print(
                f"[{len(existing) + 1}/{len(cases)}] {case['case_id']}: "
                f"completed={record['completed']}",
                flush=True,
            )
            existing[case["case_id"]] = record
    finally:
        client.close()

    if len(existing) != len(cases):
        raise RuntimeError("evaluation ended without all configured cases")
    print(f"Wrote measured results to {output_path}")


if __name__ == "__main__":
    main()
