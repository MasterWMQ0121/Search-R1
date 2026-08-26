import json
import itertools
from collections import Counter
from pathlib import Path

import pytest

from apps.enterprise_agent_workbench.evaluation import run_evaluation as runner
from apps.enterprise_agent_workbench.evaluation import summarize_evaluation as summary


def test_case_catalog_has_exactly_six_cases_in_each_required_category():
    cases = runner.load_cases()
    assert len(cases) == 24
    assert Counter(case["category"] for case in cases) == {
        category: 6 for category in runner.CATEGORIES
    }
    assert len({case["case_id"] for case in cases}) == 24
    assert {case["evaluation_class"] for case in cases} == {
        "model_dependent_end_to_end"
    }


def test_case_catalog_contains_no_answer_reward_or_ground_truth_payloads():
    raw = runner.DEFAULT_CASES.read_text(encoding="utf-8").lower()
    assert '"answer"' not in raw
    assert '"ground_truth"' not in raw
    assert '"reward"' not in raw


def test_case_catalog_uses_fixture_dates_and_only_supported_analytics_tasks():
    cases = {case["case_id"]: case for case in runner.load_cases()}
    date_sensitive = [
        case
        for case in cases.values()
        if case["category"] in {"analytics_only", "mixed_read_only"}
        and any(
            term in case["task"].lower()
            for term in ("performance", "roi", "funnel", "channel", "anomaly")
        )
    ]
    assert date_sensitive
    assert all("2026-08-" in case["task"] for case in date_sensitive)
    assert "refund" not in cases["analytics-06"]["task"].lower()
    assert "performance summary" in cases["analytics-06"]["task"].lower()


def test_citation_audit_rejects_unknown_ids_and_measures_coverage():
    valid = runner.citation_audit(
        "Campaign ROI declined [S1].\n\nBudget policy requires review [S2].",
        [{"source_id": "S1"}, {"source_id": "S2"}],
    )
    assert valid["valid"] is True
    assert valid["coverage"] == 1.0

    invalid = runner.citation_audit(
        "Campaign ROI declined [S9].", [{"source_id": "S1"}]
    )
    assert invalid["valid"] is False
    assert invalid["unknown_citation_ids"] == ["S9"]

    structured = runner.citation_audit(
        "Current state [BUSINESS:C102:get_campaign].",
        [{"source_id": "BUSINESS:C102:get_campaign"}],
    )
    assert structured["valid"] is True
    assert structured["cited_source_ids"] == ["BUSINESS:C102:get_campaign"]


def test_trace_scoring_detects_routing_denial_and_prohibited_execution():
    categories = {
        "merchant_analytics": "analytics",
        "update_campaign_budget": "business_write",
    }
    allowed = runner.score_trace(
        [
            {"event": "tool_started", "tool_name": "merchant_analytics"},
            {"event": "tool_completed", "tool_name": "merchant_analytics"},
        ],
        categories,
        {
            "tool_categories": ["analytics"],
            "prohibited_tools": ["update_campaign_budget"],
            "approval_behavior": "none",
            "policy_outcome": "allowed",
        },
    )
    assert allowed["tool_routing_correct"] is True
    assert allowed["prohibited_tool_execution_count"] == 0

    denied = runner.score_trace(
        [{"event": "policy_denied", "tool_name": "update_campaign_budget"}],
        categories,
        {
            "tool_categories": [],
            "prohibited_tools": ["update_campaign_budget"],
            "approval_behavior": "permission_denied",
            "policy_outcome": "denied",
        },
    )
    assert denied["permission_intercepted"] is True
    assert denied["approval_behavior_correct"] is True

    failed_after_start = runner.score_trace(
        [
            {"event": "tool_started", "tool_name": "update_campaign_budget"},
            {"event": "tool_failed", "tool_name": "update_campaign_budget"},
        ],
        categories,
        {
            "tool_categories": [],
            "prohibited_tools": ["update_campaign_budget"],
            "approval_behavior": "none",
            "policy_outcome": "allowed",
        },
    )
    assert failed_after_start["prohibited_tool_execution_count"] == 1
    assert failed_after_start["tool_routing_correct"] is False


def test_live_evaluation_contract_and_fingerprint_bind_external_runtime(tmp_path):
    assert runner.validate_live_evaluation_contract(" config-v1 ", True) == "config-v1"
    with pytest.raises(ValueError, match="run-config-identity"):
        runner.validate_live_evaluation_contract(" ", True)
    with pytest.raises(ValueError, match="fresh-isolated-database-confirmed"):
        runner.validate_live_evaluation_contract("config-v1", False)

    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")
    descriptor = {"health": {"status": "ok"}, "tools": [{"name": "tool-a"}]}
    baseline = runner.evaluator_fingerprint(
        cases, "http://127.0.0.1:8010", "config-v1", descriptor
    )
    assert baseline != runner.evaluator_fingerprint(
        cases, "http://127.0.0.1:8010", "config-v2", descriptor
    )
    assert baseline != runner.evaluator_fingerprint(
        cases,
        "http://127.0.0.1:8010",
        "config-v1",
        {"health": {"status": "ok"}, "tools": [{"name": "tool-b"}]},
    )


def test_evaluation_resume_rejects_tokenizer_fingerprint_mismatch(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")

    def descriptor(tokenizer_fingerprint):
        return {
            "health": {
                "status": "ok",
                "tokenizer_mode": "exact",
                "tokenizer_artifact_fingerprint": tokenizer_fingerprint,
                "evidence_compressor_fingerprint": "c" * 64,
                "model_name": "phase3-search-r1",
                "retriever_url": "http://127.0.0.1:8000/retrieve",
            },
            "tools": [{"name": "research_search"}],
        }

    original = runner.evaluator_fingerprint(
        cases,
        "http://127.0.0.1:8010",
        "config-v1",
        descriptor("a" * 64),
    )
    changed = runner.evaluator_fingerprint(
        cases,
        "http://127.0.0.1:8010",
        "config-v1",
        descriptor("b" * 64),
    )
    assert original != changed

    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"case_id": "knowledge-01", "evaluator_fingerprint": original})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different configuration"):
        runner._read_existing(results, changed)


def test_post_resume_wait_ignores_stale_approval_state():
    class SequenceClient:
        def __init__(self):
            self.states = itertools.chain(
                [
                    {"approval_request": {"action": "update_campaign_budget"}},
                    {"approval_request": {"action": "update_campaign_budget"}},
                    {"approval_request": None, "completed": True},
                ]
            )

        def state(self, thread_id):
            assert thread_id == "thread-1"
            return next(self.states)

    result = runner.wait_for_state(
        SequenceClient(),
        "thread-1",
        timeout_seconds=1.0,
        poll_seconds=0.0,
        allow_paused=False,
    )
    assert result["completed"] is True


def test_database_state_delta_uses_recorded_business_write_output():
    categories = {"update_campaign_budget": "business_write"}
    state = {
        "tool_results": [
            {
                "tool_name": "update_campaign_budget",
                "status": "ok",
                "output": {
                    "after_state": {"campaign_id": "C102", "daily_budget": 1200}
                },
            }
        ]
    }
    assert runner.database_state_delta_matches(
        state, {"campaign_id": "C102", "daily_budget": 1200}, categories
    )
    assert not runner.database_state_delta_matches(state, {}, categories)
    assert runner.database_state_delta_matches({"tool_results": []}, {}, categories)


def _result(case_id, category, index):
    is_denial = category == "write_approval_permission" and index % 2 == 0
    is_approval = category == "write_approval_permission" and not is_denial
    return {
        "schema_version": 1,
        "evaluator_fingerprint": "a" * 64,
        "case_id": case_id,
        "category": category,
        "evaluation_class": "model_dependent_end_to_end",
        "completed": True,
        "termination_reason": "completed",
        "task_latency_s": float(index + 1),
        "final_answer_present": True,
        "acceptable_final_answer_facts_match": True,
        "source_types_match": True,
        "tool_routing_correct": True,
        "policy_outcome_correct": True,
        "approval_behavior_correct": True,
        "prohibited_tool_execution_count": 0,
        "permission_intercepted": True if is_denial else None,
        "resume_success": True if is_approval else None,
        "idempotency_correct": True if case_id == "write_approval_permission-1" else None,
        "database_state_delta_match": True if is_approval or is_denial else None,
        "citation_valid": True,
        "citation_coverage": 1.0,
        "tool_failure_recovered": None,
        "tool_call_count": 2,
        "planner_repair_count": 0,
        "errors": [],
    }


def test_summary_computes_metrics_only_from_supplied_records(tmp_path):
    records = []
    for category in runner.CATEGORIES:
        for index in range(6):
            records.append(_result(f"{category}-{index + 1}", category, index))
    payload = summary.summarize(records)
    assert payload["case_count"] == 24
    assert payload["metrics"]["run_completion_rate"] == 1.0
    assert payload["metrics"]["task_completion_rate"] == 1.0
    assert all(
        category["run_completion_rate"] == 1.0
        for category in payload["by_category"].values()
    )
    assert all(
        category["task_completion_rate"] == 1.0
        for category in payload["by_category"].values()
    )
    assert payload["metrics"]["tool_routing_accuracy"] == 1.0
    assert payload["metrics"]["prohibited_tool_execution_count"] == 0
    assert payload["metrics"]["unfinished_run_count"] == 0
    assert payload["metrics"]["mean_tool_calls"] == 2.0
    assert "pytest" in payload["evaluation_scope"][
        "deterministic_infrastructure_policy_tests"
    ]

    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    loaded = summary.read_results(result_path)
    assert len(loaded) == 24


def test_task_completion_requires_case_assertions_beyond_terminal_run():
    records = [
        _result(f"{category}-{index + 1}", category, index)
        for category in runner.CATEGORIES
        for index in range(6)
    ]
    records[0]["acceptable_final_answer_facts_match"] = False

    payload = summary.summarize(records)

    assert payload["metrics"]["run_completion_rate"] == 1.0
    assert payload["metrics"]["task_completion_rate"] == 23 / 24


def test_complete_summary_rejects_partial_results(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        json.dumps(_result("knowledge_only-1", "knowledge_only", 0)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 24"):
        summary.read_results(path)
