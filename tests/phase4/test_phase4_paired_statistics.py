import copy
import hashlib
import inspect
import json

import pytest

from experiments.phase4_benchmark import paired_statistics as stats


def _result_sets(nq_count=4, hotpotqa_count=4, outcome=None):
    if outcome is None:
        outcome = lambda _mode, _source, _index: 0
    result_sets = {mode: [] for mode in stats.PAIRED_MODES}
    for source, count in (("nq", nq_count), ("hotpotqa", hotpotqa_count)):
        for index in range(count):
            uid = f"{source}:{index}"
            for mode in stats.PAIRED_MODES:
                record = {
                    "mode": mode,
                    "uid": uid,
                    "data_source": source,
                    "question": f"question-{uid}",
                    "ground_truth": [f"answer-{uid}"],
                    "exact_match": int(outcome(mode, source, index)),
                    "run_config": {
                        "eval_sha256": "eval-sha",
                        "eval_manifest_sha256": "manifest-sha",
                    },
                }
                if mode in ("base_search", "search_rl"):
                    record["search_retrieval_failure_count"] = 0
                result_sets[mode].append(record)
    return result_sets


def _joined(outcome=None, nq_count=4, hotpotqa_count=4):
    return stats.join_paired_results(
        _result_sets(nq_count, hotpotqa_count, outcome)
    )


def test_join_aligns_by_uid_instead_of_file_order():
    result_sets = _result_sets()
    result_sets["static_rag"].reverse()
    result_sets["base_search"] = result_sets["base_search"][2:] + result_sets["base_search"][:2]

    joined = stats.join_paired_results(result_sets)

    assert [row["uid"] for row in joined] == sorted(
        record["uid"] for record in result_sets["direct"]
    )
    assert all(set(row["records"]) == set(stats.PAIRED_MODES) for row in joined)


def test_join_rejects_nonidentical_uid_sets():
    result_sets = _result_sets()
    result_sets["search_rl"].pop()

    with pytest.raises(ValueError, match="nonidentical UID set"):
        stats.join_paired_results(result_sets)


def test_join_rejects_duplicate_uids():
    result_sets = _result_sets()
    result_sets["base_search"].append(copy.deepcopy(result_sets["base_search"][0]))

    with pytest.raises(ValueError, match="duplicate UID"):
        stats.join_paired_results(result_sets)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("question", "different question", "mismatched question"),
        ("ground_truth", ["different answer"], "mismatched ground truth"),
        ("data_source", "hotpotqa", "mismatched data source"),
    ],
)
def test_join_rejects_paired_identity_mismatches(field, replacement, message):
    result_sets = _result_sets()
    result_sets["base_search"][0][field] = replacement

    with pytest.raises(ValueError, match=message):
        stats.join_paired_results(result_sets)


def test_join_rejects_different_eval_artifact_bindings():
    result_sets = _result_sets()
    result_sets["static_rag"][0]["run_config"]["eval_sha256"] = "other-eval"

    with pytest.raises(ValueError, match="same eval manifest/parquet hashes"):
        stats.join_paired_results(result_sets)


def test_supplied_manifest_binding_and_uid_set_are_verified(tmp_path):
    result_sets = _result_sets()
    manifest = {
        "selected_row_count": 8,
        "selected_source_rows": [
            {"uid": record["uid"]} for record in result_sets["direct"]
        ],
        "output": {"sha256": "eval-sha"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for records in result_sets.values():
        for record in records:
            record["run_config"]["eval_manifest_sha256"] = manifest_sha

    audit = stats.validate_joined_against_manifest(
        stats.join_paired_results(result_sets), manifest_path
    )

    assert audit["sha256"] == manifest_sha
    assert audit["eval_parquet_sha256"] == "eval-sha"
    assert audit["selected_uid_count"] == 8
    assert audit["uid_set_matches"] is True


def test_supplied_manifest_uid_mismatch_is_rejected(tmp_path):
    result_sets = _result_sets()
    manifest = {
        "selected_row_count": 8,
        "selected_source_rows": [
            {"uid": record["uid"]} for record in result_sets["direct"]
        ],
        "output": {"sha256": "eval-sha"},
    }
    manifest["selected_source_rows"][0]["uid"] = "nq:missing"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for records in result_sets.values():
        for record in records:
            record["run_config"]["eval_manifest_sha256"] = manifest_sha

    with pytest.raises(ValueError, match="UIDs do not match"):
        stats.validate_joined_against_manifest(
            stats.join_paired_results(result_sets), manifest_path
        )


def test_paired_bootstrap_is_reproducible():
    joined = _joined(
        lambda mode, source, index: (
            (index % 3 == 0) if mode == "search_rl"
            else (index % 4 == 0) if mode == "base_search"
            else 0
        ),
        nq_count=12,
        hotpotqa_count=12,
    )

    first = stats.paired_bootstrap(
        joined, "search_rl", "base_search", samples=2_000, seed=42
    )
    second = stats.paired_bootstrap(
        joined, "search_rl", "base_search", samples=2_000, seed=42
    )

    assert first == second


def test_overall_bootstrap_preserves_the_source_mixture():
    joined = _joined(
        lambda mode, source, _index: (
            mode == "search_rl" and source == "nq"
        ) or (
            mode == "base_search" and source == "hotpotqa"
        ),
    )

    result = stats.paired_bootstrap(
        joined, "search_rl", "base_search", samples=500, seed=42
    )

    assert result["difference_pp"] == 0.0
    assert result["bootstrap_ci_95_pp"] == [0.0, 0.0]
    assert result["interval_method"] == "source-stratified paired percentile"


@pytest.mark.parametrize(
    ("search_outcome", "base_outcome", "difference", "includes_zero"),
    [
        (1, 0, 100.0, False),
        (0, 1, -100.0, False),
        (1, 1, 0.0, True),
    ],
)
def test_bootstrap_known_positive_negative_and_zero_differences(
    search_outcome, base_outcome, difference, includes_zero
):
    joined = _joined(
        lambda mode, _source, _index: (
            search_outcome if mode == "search_rl"
            else base_outcome if mode == "base_search"
            else 0
        )
    )

    result = stats.paired_bootstrap(
        joined, "search_rl", "base_search", samples=200, seed=42
    )

    assert result["difference_pp"] == difference
    assert result["bootstrap_ci_95_pp"] == [difference, difference]
    assert result["ci_includes_zero"] is includes_zero


def test_mcnemar_no_discordance_has_p_value_one():
    joined = _joined(
        lambda mode, _source, index: index % 2
        if mode in ("search_rl", "base_search") else 0
    )

    result = stats.exact_mcnemar(joined, "search_rl", "base_search")

    assert result == {
        "mode_a": "search_rl",
        "mode_b": "base_search",
        "orientation": "search_rl minus base_search",
        "both_correct": 4,
        "a_only_correct": 0,
        "b_only_correct": 0,
        "both_incorrect": 4,
        "discordant_pair_count": 0,
        "exact_two_sided_p_value": 1.0,
    }


def test_mcnemar_symmetric_discordance_has_p_value_one():
    joined = _joined(
        lambda mode, _source, index: (
            mode == "search_rl" and index % 2 == 0
        ) or (
            mode == "base_search" and index % 2 == 1
        )
    )

    result = stats.exact_mcnemar(joined, "search_rl", "base_search")

    assert result["a_only_correct"] == 4
    assert result["b_only_correct"] == 4
    assert result["exact_two_sided_p_value"] == 1.0


@pytest.mark.parametrize(
    ("a_wins", "b_wins", "expected_p"),
    [
        (6, 0, 0.03125),
        (5, 1, 0.21875),
    ],
)
def test_mcnemar_matches_known_exact_binomial_values(a_wins, b_wins, expected_p):
    joined = _joined(nq_count=a_wins + b_wins, hotpotqa_count=1)
    for position, row in enumerate(
        row for row in joined if row["data_source"] == "nq"
    ):
        row["outcomes"]["search_rl"] = int(position < a_wins)
        row["outcomes"]["base_search"] = int(position >= a_wins)

    result = stats.exact_mcnemar(
        joined, "search_rl", "base_search", source="nq"
    )

    assert result["a_only_correct"] == a_wins
    assert result["b_only_correct"] == b_wins
    assert result["discordant_pair_count"] == a_wins + b_wins
    assert result["exact_two_sided_p_value"] == expected_p


def test_analysis_has_primary_secondary_and_all_source_outputs():
    report = stats.analyze_paired_results(
        _result_sets(), bootstrap_samples=200, seed=42
    )

    assert report["primary"]["comparison"] == "search_rl_vs_base_search"
    assert report["primary"]["classification"] == "primary"
    assert report["primary"]["orientation"] == "EM(search_rl) - EM(base_search)"
    assert all(key in report["primary"] for key in ("overall", "nq", "hotpotqa"))
    assert set(report["secondary"]) == {
        comparison[0] for comparison in stats.SECONDARY_COMPARISONS
    }
    assert all(
        comparison["classification"] == "secondary_exploratory"
        for comparison in report["secondary"].values()
    )
    assert "not independently confirmatory" in (
        report["interpretation_guardrails"]["secondary"]
    )


def test_ready_report_requires_64_balanced_error_free_rows():
    report = stats.analyze_paired_results(
        _result_sets(32, 32), bootstrap_samples=100, seed=42
    )

    assert report["readiness"]["quality_claim_ready"] is True
    assert report["readiness"]["inference_claim_ready"] is True
    assert report["quality_claim_ready"] is True
    assert report["inference_claim_ready"] is True
    assert report["readiness"]["warning"] is None
    assert report["configuration"]["eval_parquet_sha256"] == "eval-sha"
    assert report["configuration"]["eval_manifest_sha256"] == "manifest-sha"
    assert "n = 64 (32 NQ / 32 HotpotQA)" in report["primary"]["interpretation"]
    assert "sparse binary" in report["primary"]["interpretation"]


@pytest.mark.parametrize("failure_kind", ["retrieval", "evaluation"])
def test_failed_rows_are_retained_but_claim_readiness_is_false(failure_kind):
    result_sets = _result_sets(32, 32)
    if failure_kind == "retrieval":
        result_sets["base_search"][0]["search_retrieval_failure_count"] = 1
        result_sets["base_search"][0]["evaluation_error"] = "retriever failed"
    else:
        result_sets["direct"][0]["evaluation_error"] = "evaluation failed"

    report = stats.analyze_paired_results(
        result_sets, bootstrap_samples=100, seed=42
    )

    assert report["readiness"]["row_count"] == 64
    assert report["readiness"]["quality_claim_ready"] is False
    assert report["readiness"]["inference_claim_ready"] is False
    assert "not ready" in report["readiness"]["warning"]
    assert "no quality or inferential claim" in report["primary"]["interpretation"]


def test_default_configuration_and_markdown_are_auditable():
    signature = inspect.signature(stats.analyze_paired_results)
    assert signature.parameters["bootstrap_samples"].default == 10_000
    assert signature.parameters["seed"].default == 42
    assert signature.parameters["confidence_level"].default == 0.95
    report = stats.analyze_paired_results(
        _result_sets(32, 32), bootstrap_samples=50, seed=42
    )

    markdown = stats.render_markdown(report)

    assert "# Phase-4 paired statistics" in markdown
    assert "search_rl_vs_base_search" in markdown
    assert "Secondary comparisons" in markdown
    assert "Exact McNemar p" in markdown


def test_cli_defaults_to_json_and_markdown_artifacts(tmp_path):
    json_path, markdown_path = stats.output_paths(tmp_path)

    assert json_path == tmp_path / "paired_statistics.json"
    assert markdown_path == tmp_path / "paired_statistics.md"

    custom_json, custom_markdown = stats.output_paths(
        tmp_path,
        tmp_path / "custom.json",
        tmp_path / "custom.md",
    )
    assert custom_json == tmp_path / "custom.json"
    assert custom_markdown == tmp_path / "custom.md"
