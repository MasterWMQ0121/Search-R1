#!/usr/bin/env python3
"""Summarize the three-condition Phase-5 observation-context ablation."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from experiments.phase4_benchmark.paired_statistics import (
    analyze_comparison,
    join_result_sets,
)
from experiments.phase4_benchmark.prepare_eval_data import sha256_file
from experiments.phase4_benchmark.run_benchmark import (
    read_result_file,
    run_config_fingerprint,
)
from experiments.phase4_benchmark.summarize_results import (
    latency_summary,
    quality_summary,
    search_agent_metrics,
)
from experiments.phase5_observation_context.run_context_ablation import (
    CONDITIONS,
    CONDITION_LIMITS,
    MAX_RESPONSE_LENGTH,
    MAX_START_LENGTH,
    MAX_TURNS,
    RETRIEVER_TOPK,
    SEED,
    read_phase5_result_file,
    validate_results_directory_isolation,
)


SCHEMA_VERSION = 1
SOURCES = ("nq", "hotpotqa")
PRIMARY_COMPARISON = (
    "compressed_256_vs_raw_256_baseline",
    "compressed_256",
    "raw_256_baseline",
)
SECONDARY_COMPARISONS = (
    ("raw_512_vs_raw_256_baseline", "raw_512", "raw_256_baseline"),
    ("compressed_256_vs_raw_512", "compressed_256", "raw_512"),
)


def _mean(values):
    return statistics.fmean(values) if values else None


def _flatten(records, key):
    flattened = []
    for record in records:
        values = record.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a per-retrieval list")
        flattened.extend(values)
    return flattened


def _single_run_binding(records, label):
    configurations = {
        json.dumps(record["run_config"], sort_keys=True): record["run_config"]
        for record in records
    }
    fingerprints = {record["run_fingerprint"] for record in records}
    if len(configurations) != 1 or len(fingerprints) != 1:
        raise ValueError(f"{label} mixes multiple run configurations")
    run_config = next(iter(configurations.values()))
    fingerprint = next(iter(fingerprints))
    if fingerprint != run_config_fingerprint(run_config):
        raise ValueError(f"{label} has an invalid run fingerprint")
    return run_config, fingerprint


def adapt_raw_256_baseline(records):
    """Create a read-only, in-memory Phase-5 view of Phase-4 Search-RL rows."""

    adapted = []
    for record in records:
        retrieval_count = record["number_of_successful_retrievals"]
        raw_lengths = list(
            record["retrieved_observation_lengths_before_truncation"]
        )
        retained_lengths = list(
            record["retained_observation_lengths_after_truncation"]
        )
        if len(raw_lengths) != retrieval_count or len(retained_lengths) != retrieval_count:
            raise ValueError(
                f"raw_256_baseline telemetry is misaligned for {record['uid']}"
            )
        baseline_view = dict(record)
        baseline_view.update({
            "mode": "raw_256_baseline",
            "observation_policy": "raw_256_baseline",
            "raw_retrieved_observation_tokens": raw_lengths,
            "policy_output_observation_tokens": list(raw_lengths),
            "retained_observation_tokens": retained_lengths,
            "policy_compression_ratio": [1.0] * retrieval_count,
            "post_policy_truncation_count": record["observation_truncation_count"],
            "documents_returned": [RETRIEVER_TOPK] * retrieval_count,
            "documents_represented": [RETRIEVER_TOPK] * retrieval_count,
            "sentences_considered": [0] * retrieval_count,
            "sentences_selected": [0] * retrieval_count,
            "evidence_compression_latency_s": 0.0,
            "zero_overlap_fallback_used": False,
            "partial_sentence_fallback_used": False,
            "selected_document_ranks": [
                list(range(1, RETRIEVER_TOPK + 1))
                for _ in range(retrieval_count)
            ],
            "selected_sentence_identifiers": [[] for _ in range(retrieval_count)],
        })
        adapted.append(baseline_view)
    return adapted


def validate_and_load_baseline(
    baseline_path,
    manifest,
    manifest_path,
    search_model,
    *,
    expected_count=64,
):
    """Validate and adapt the immutable Phase-4 baseline without writing it."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    before_sha256 = sha256_file(baseline_path)
    records = read_result_file(baseline_path, "search_rl")
    if len(records) != expected_count:
        raise ValueError(
            f"raw_256_baseline must contain {expected_count} rows; found {len(records)}"
        )
    run_config, fingerprint = _single_run_binding(records, "raw_256_baseline")
    required = {
        "model_path": str(search_model),
        "prompt_contract": "phase4-benchmark-v1",
        "seed": SEED,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.20,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": 256,
        "max_prompt_length": 1408,
        "max_turns": MAX_TURNS,
        "retriever_topk": RETRIEVER_TOPK,
    }
    for key, expected in required.items():
        if run_config.get(key) != expected:
            raise ValueError(
                f"raw_256_baseline has unexpected {key}: "
                f"{run_config.get(key)!r}; expected {expected!r}"
            )
    manifest_sha256 = sha256_file(manifest_path)
    eval_sha256 = manifest.get("output", {}).get("sha256")
    if run_config.get("eval_sha256") != eval_sha256:
        raise ValueError("raw_256_baseline is bound to a different eval parquet")
    if run_config.get("eval_manifest_sha256") != manifest_sha256:
        raise ValueError("raw_256_baseline is bound to a different eval manifest")
    retrieval_failures = sum(
        record["search_retrieval_failure_count"] for record in records
    )
    evaluation_errors = sum(bool(record.get("evaluation_error")) for record in records)
    if retrieval_failures:
        raise ValueError("raw_256_baseline contains Retriever failures")
    if evaluation_errors:
        raise ValueError("raw_256_baseline contains evaluation errors")
    if sha256_file(baseline_path) != before_sha256:
        raise RuntimeError("raw_256_baseline changed during read-only validation")
    return adapt_raw_256_baseline(records), {
        "validated": True,
        "condition": "raw_256_baseline",
        "path": str(baseline_path),
        "sha256": before_sha256,
        "row_count": len(records),
        "run_fingerprint": fingerprint,
        "eval_parquet_sha256": eval_sha256,
        "eval_manifest_sha256": manifest_sha256,
        "retrieval_failure_count": retrieval_failures,
        "evaluation_error_count": evaluation_errors,
    }


def validate_manifest_binding(joined, manifest, manifest_path, expected_count=64):
    """Bind the strict paired join to the supplied Phase-4 manifest and UID set."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_sha256 = sha256_file(manifest_path)
    eval_sha256 = manifest.get("output", {}).get("sha256")
    if manifest.get("seed") != SEED:
        raise ValueError("Phase-5 requires the Phase-4 seed-42 eval manifest")
    if manifest.get("selected_row_count") != expected_count:
        raise ValueError(f"Phase-5 manifest must declare exactly {expected_count} rows")
    if manifest.get("data_source_counts") != {"nq": 32, "hotpotqa": 32}:
        raise ValueError("Phase-5 manifest must declare 32 NQ and 32 HotpotQA rows")
    if not manifest.get("non_overlap_audit", {}).get("passed"):
        raise ValueError("Phase-5 manifest non-overlap audit did not pass")
    if manifest.get("non_overlap_audit", {}).get("uid_overlap"):
        raise ValueError("Phase-5 manifest contains excluded UID overlap")
    if manifest.get("non_overlap_audit", {}).get("source_position_overlap"):
        raise ValueError("Phase-5 manifest contains excluded source-position overlap")
    expected_uids = [
        entry.get("uid") for entry in manifest.get("selected_source_rows", [])
    ]
    if len(expected_uids) != expected_count or len(set(expected_uids)) != expected_count:
        raise ValueError("Phase-5 manifest has missing or duplicate selected UIDs")
    actual_uids = {row["uid"] for row in joined}
    if set(expected_uids) != actual_uids:
        raise ValueError("Phase-5 result UIDs do not match the eval manifest")
    for row in joined:
        for mode, record in row["records"].items():
            run_config = record["run_config"]
            if run_config.get("eval_sha256") != eval_sha256:
                raise ValueError(f"{mode} is bound to a different eval parquet")
            if run_config.get("eval_manifest_sha256") != manifest_sha256:
                raise ValueError(f"{mode} is bound to a different eval manifest")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "eval_parquet_sha256": eval_sha256,
        "selected_uid_count": len(expected_uids),
        "uid_set_matches": True,
    }


def validate_phase5_bindings(result_sets, baseline_audit, search_model):
    """Verify the two new conditions share the baseline model and runtime contract."""

    baseline_config, _ = _single_run_binding(
        result_sets["raw_256_baseline"], "raw_256_baseline"
    )
    for condition in ("raw_512", "compressed_256"):
        run_config, _ = _single_run_binding(result_sets[condition], condition)
        required = {
            "model_path": str(search_model),
            "prompt_contract": "phase4-benchmark-v1",
            "experiment_contract": "phase5-observation-context-v1",
            "seed": SEED,
            "greedy": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.20,
            "max_start_length": MAX_START_LENGTH,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "max_turns": MAX_TURNS,
            "retriever_topk": RETRIEVER_TOPK,
            **CONDITION_LIMITS[condition],
        }
        for key, expected in required.items():
            if run_config.get(key) != expected:
                raise ValueError(f"{condition} has unexpected {key}")
        if run_config.get("retriever_url") != baseline_config.get("retriever_url"):
            raise ValueError(f"{condition} does not use the baseline Retriever")
        if run_config.get("attention_backend") != baseline_config.get("attention_backend"):
            raise ValueError(f"{condition} does not use the baseline attention backend")
        if run_config.get("raw_retriever_format") != (
            "LLMGenerationManager._passages2string"
        ):
            raise ValueError(f"{condition} does not use the baseline raw formatting")
        expected_compressor = None
        if condition == "compressed_256":
            from experiments.phase5_observation_context import evidence_compressor

            expected_compressor = {
                "name": "deterministic_query_aware_extractive",
                "version": "phase5-extractive-v1",
                "source_sha256": sha256_file(Path(evidence_compressor.__file__)),
                "wrapped_token_budget": 256,
                "bm25_k1": evidence_compressor.BM25_K1,
                "bm25_b": evidence_compressor.BM25_B,
                "query_coverage_weight": (
                    evidence_compressor.QUERY_COVERAGE_WEIGHT
                ),
                "title_coverage_weight": (
                    evidence_compressor.TITLE_COVERAGE_WEIGHT
                ),
                "rank_prior_weight": evidence_compressor.RANK_PRIOR_WEIGHT,
                "retrieval_score_prior_weight": (
                    evidence_compressor.RETRIEVAL_SCORE_PRIOR_WEIGHT
                ),
                "selection_priority": "relevance/sqrt(sentence_token_count)",
            }
        if run_config.get("compressor") != expected_compressor:
            raise ValueError(f"{condition} has an unexpected compressor contract")
        if run_config.get("training_max_obs_length") != 256:
            raise ValueError(f"{condition} has an unexpected training observation cap")
        if run_config.get("inference_context_distribution_shift") != (
            condition == "raw_512"
        ):
            raise ValueError(f"{condition} has an inconsistent distribution shift")
        if Path(run_config.get("baseline_path", "")).expanduser().resolve() != Path(
            baseline_audit["path"]
        ).expanduser().resolve():
            raise ValueError(f"{condition} references a different baseline path")
        if run_config.get("baseline_sha256") != baseline_audit["sha256"]:
            raise ValueError(f"{condition} is not bound to the immutable baseline")
        if (
            run_config.get("baseline_run_fingerprint")
            != baseline_audit["run_fingerprint"]
        ):
            raise ValueError(f"{condition} is not bound to the baseline run config")


def context_summary(records):
    raw = _flatten(records, "raw_retrieved_observation_tokens")
    policy = _flatten(records, "policy_output_observation_tokens")
    retained = _flatten(records, "retained_observation_tokens")
    ratios = _flatten(records, "policy_compression_ratio")
    represented = _flatten(records, "documents_represented")
    selected = _flatten(records, "sentences_selected")
    if not (len(raw) == len(policy) == len(retained) == len(ratios)):
        raise ValueError("Phase-5 context telemetry is misaligned")
    return {
        "retrieved_observation_count": len(raw),
        "post_policy_truncation_count": sum(
            record["post_policy_truncation_count"] for record in records
        ),
        "fraction_trajectories_with_post_policy_truncation": _mean([
            record["post_policy_truncation_count"] > 0 for record in records
        ]),
        "raw_observation_tokens_mean": _mean(raw),
        "raw_observation_tokens_max": max(raw) if raw else None,
        "policy_output_tokens_mean": _mean(policy),
        "policy_output_tokens_max": max(policy) if policy else None,
        "retained_observation_tokens_mean": _mean(retained),
        "retained_observation_tokens_max": max(retained) if retained else None,
        "mean_compression_ratio": _mean(ratios),
        "mean_tokens_saved": _mean([
            raw_count - policy_count
            for raw_count, policy_count in zip(raw, policy)
        ]),
        "mean_documents_represented": _mean(represented),
        "mean_sentences_selected": _mean(selected),
        "zero_overlap_fallback_trajectory_rate": _mean([
            bool(record["zero_overlap_fallback_used"]) for record in records
        ]),
        "partial_sentence_fallback_trajectory_rate": _mean([
            bool(record["partial_sentence_fallback_used"]) for record in records
        ]),
    }


def summarize_condition(records):
    """Aggregate quality, latency, agent, and context metrics for one condition."""

    return {
        "quality": quality_summary(records),
        "latency": {
            "end_to_end": latency_summary([
                record["end_to_end_latency_s"] for record in records
            ]),
            "generation": latency_summary([
                record["generation_latency_s"] for record in records
            ]),
            "retrieval": latency_summary([
                record["retrieval_latency_s"] for record in records
            ]),
            "compression": latency_summary([
                record["evidence_compression_latency_s"] for record in records
            ]),
        },
        "agent": search_agent_metrics(records),
        "context": context_summary(records),
    }


def _delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def descriptive_comparison(summaries, name, mode_a, mode_b):
    a = summaries[mode_a]
    b = summaries[mode_b]
    return {
        "comparison": name,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "orientation": f"{mode_a} minus {mode_b}",
        "overall_em_delta_pp": 100.0 * _delta(
            a["quality"]["overall_em"], b["quality"]["overall_em"]
        ),
        "nq_em_delta_pp": 100.0 * _delta(
            a["quality"]["nq_em"], b["quality"]["nq_em"]
        ),
        "hotpotqa_em_delta_pp": 100.0 * _delta(
            a["quality"]["hotpotqa_em"], b["quality"]["hotpotqa_em"]
        ),
        "end_to_end_latency_mean_s_delta": _delta(
            a["latency"]["end_to_end"]["mean_s"],
            b["latency"]["end_to_end"]["mean_s"],
        ),
        "post_policy_truncation_fraction_delta": _delta(
            a["context"]["fraction_trajectories_with_post_policy_truncation"],
            b["context"]["fraction_trajectories_with_post_policy_truncation"],
        ),
        "finish_ratio_delta": _delta(
            a["agent"]["finish_ratio"], b["agent"]["finish_ratio"]
        ),
        "valid_action_ratio_delta": _delta(
            a["agent"]["valid_action_ratio"], b["agent"]["valid_action_ratio"]
        ),
        "valid_search_ratio_delta": _delta(
            a["agent"]["valid_search_ratio"], b["agent"]["valid_search_ratio"]
        ),
    }


def engineering_readiness(result_sets, baseline_audit, expected_count=64):
    counts = {mode: len(records) for mode, records in result_sets.items()}
    source_counts = Counter(
        record["data_source"] for record in result_sets["raw_256_baseline"]
    )
    retrieval_failures = {
        mode: sum(record["search_retrieval_failure_count"] for record in records)
        for mode, records in result_sets.items()
    }
    evaluation_errors = {
        mode: sum(bool(record.get("evaluation_error")) for record in records)
        for mode, records in result_sets.items()
    }
    uid_sets = {
        mode: {record["uid"] for record in records}
        for mode, records in result_sets.items()
    }
    artifact_bindings = {
        (
            record["run_config"].get("eval_sha256"),
            record["run_config"].get("eval_manifest_sha256"),
        )
        for records in result_sets.values()
        for record in records
    }
    checks = {
        "all_conditions_have_exactly_64_rows": all(
            count == expected_count for count in counts.values()
        ),
        "source_balance_is_32_nq_32_hotpotqa": (
            source_counts == Counter({"nq": 32, "hotpotqa": 32})
        ),
        "identical_uid_sets": len({frozenset(uids) for uids in uid_sets.values()}) == 1,
        "same_eval_manifest_and_parquet_hashes": len(artifact_bindings) == 1,
        "no_retriever_failures": not any(retrieval_failures.values()),
        "no_evaluation_errors": not any(evaluation_errors.values()),
        "immutable_baseline_validated": bool(baseline_audit.get("validated")),
        "compressed_observations_within_256_token_budget": all(
            token_count <= 256
            for record in result_sets["compressed_256"]
            for token_count in record["policy_output_observation_tokens"]
        ),
    }
    ready = all(checks.values())
    return {
        "engineering_pass": ready,
        "quality_claim_ready": ready,
        "checks": checks,
        "paired_row_count": len(result_sets["raw_256_baseline"]),
        "row_counts": counts,
        "source_counts": {source: source_counts.get(source, 0) for source in SOURCES},
        "retrieval_failure_counts": retrieval_failures,
        "evaluation_error_counts": evaluation_errors,
        "warning": None if ready else (
            "Phase-5 engineering readiness failed. Descriptive and paired output "
            "is retained for audit, but no quality claim should be made."
        ),
    }


def compression_assessment(summaries):
    """Report measured compression mechanics without imposing a quality claim."""

    baseline_fraction = summaries["raw_256_baseline"]["context"][
        "fraction_trajectories_with_post_policy_truncation"
    ]
    compressed_fraction = summaries["compressed_256"]["context"][
        "fraction_trajectories_with_post_policy_truncation"
    ]
    return {
        "wrapped_observation_budget_tokens": 256,
        "compressed_policy_output_max_tokens": summaries["compressed_256"][
            "context"
        ]["policy_output_tokens_max"],
        "compressed_observations_within_budget": (
            summaries["compressed_256"]["context"]["policy_output_tokens_max"]
            is None
            or summaries["compressed_256"]["context"]["policy_output_tokens_max"]
            <= 256
        ),
        "baseline_post_policy_truncation_fraction": baseline_fraction,
        "compressed_post_policy_truncation_fraction": compressed_fraction,
        "post_policy_truncation_fraction_delta": _delta(
            compressed_fraction, baseline_fraction
        ),
        "post_policy_truncation_lower_than_baseline": (
            compressed_fraction is not None
            and baseline_fraction is not None
            and compressed_fraction < baseline_fraction
        ),
        "policy_retrieval_contract": (
            "The observation policy consumes each existing Top-3 result; it does "
            "not issue an additional Retriever call."
        ),
    }


def paired_statistics_report(
    joined,
    readiness,
    manifest_audit,
    *,
    bootstrap_samples=10_000,
    seed=42,
    confidence_level=0.95,
):
    comparisons = {}
    for classification, specs in (
        ("primary", (PRIMARY_COMPARISON,)),
        ("secondary", SECONDARY_COMPARISONS),
    ):
        for name, mode_a, mode_b in specs:
            comparisons[name] = analyze_comparison(
                joined,
                name,
                mode_a,
                mode_b,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                confidence_level=confidence_level,
                classification=classification,
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "confidence_level": confidence_level,
            "overall_bootstrap": "source-stratified paired percentile",
            "source_bootstrap": "within-source paired percentile",
            "mcnemar": "exact two-sided binomial on discordant pairs",
            "manifest_audit": manifest_audit,
        },
        "readiness": readiness,
        "primary": comparisons[PRIMARY_COMPARISON[0]],
        "secondary": {
            name: comparisons[name] for name, _a, _b in SECONDARY_COMPARISONS
        },
    }


def neutral_interpretation(summaries, statistics_report, readiness):
    """Apply the preregistered neutral interpretation rules to measured results."""

    if not readiness["quality_claim_ready"]:
        return {
            "conclusion": readiness["warning"],
            "guardrail": "No Phase-5 quality claim is ready.",
        }
    primary = statistics_report["primary"]["overall"]
    compressed_delta = primary["difference_pp"]
    raw_delta = statistics_report["secondary"][
        "raw_512_vs_raw_256_baseline"
    ]["overall"]["difference_pp"]
    compressed_vs_raw512 = statistics_report["secondary"][
        "compressed_256_vs_raw_512"
    ]["overall"]["difference_pp"]
    ci = primary["bootstrap_ci_95_pp"]
    observations = []
    if compressed_delta > 0:
        observations.append(
            "compressed_256 has a higher measured EM than raw_256_baseline "
            f"by {compressed_delta:.6g} percentage points; the 95% paired-bootstrap "
            f"CI is [{ci[0]:.6g}, {ci[1]:.6g}]. This run-specific association "
            "does not establish production-wide causality."
        )
    if raw_delta > 0 and compressed_delta <= 0:
        observations.append(
            "raw_512 improves the measured point estimate while compressed_256 "
            "does not; context capacity appears more important than the current "
            "compression heuristic for this checkpoint and evaluation."
        )
    compressed_tokens = summaries["compressed_256"]["context"][
        "policy_output_tokens_mean"
    ]
    raw512_tokens = summaries["raw_512"]["context"]["policy_output_tokens_mean"]
    if (
        compressed_vs_raw512 >= 0
        and compressed_tokens is not None
        and raw512_tokens is not None
        and compressed_tokens < raw512_tokens
    ):
        observations.append(
            "compressed_256 matches or improves the raw_512 measured EM point "
            "estimate while using fewer policy-output tokens, a fixed-budget "
            "efficiency signal specific to this run."
        )
    if raw_delta <= 0 and compressed_delta <= 0:
        observations.append(
            "Neither alternative improves the measured EM point estimate. "
            "Baseline truncation may be common, but this is insufficient evidence "
            "that increasing or compressing context improves this checkpoint."
        )
    if not observations:
        observations.append(
            "The measured condition differences are descriptive and mixed; inspect "
            "their paired intervals and exact McNemar results before interpretation."
        )
    return {
        "conclusion": " ".join(observations),
        "guardrail": (
            "These deterministic 64-example results are checkpoint-, Retriever-, "
            "and dataset-specific; no broad causal or production claim is implied."
        ),
    }


def build_reports(
    result_sets,
    baseline_audit,
    manifest_audit,
    *,
    bootstrap_samples=10_000,
    seed=42,
    confidence_level=0.95,
    expected_count=64,
):
    joined = join_result_sets(result_sets, CONDITIONS)
    summaries = {
        condition: summarize_condition(result_sets[condition])
        for condition in CONDITIONS
    }
    readiness = engineering_readiness(
        result_sets, baseline_audit, expected_count=expected_count
    )
    comparisons = {}
    for name, mode_a, mode_b in (PRIMARY_COMPARISON,) + SECONDARY_COMPARISONS:
        comparisons[name] = descriptive_comparison(
            summaries, name, mode_a, mode_b
        )
    statistics_report = paired_statistics_report(
        joined,
        readiness,
        manifest_audit,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence_level=confidence_level,
    )
    interpretation = neutral_interpretation(
        summaries, statistics_report, readiness
    )
    statistics_report["interpretation"] = interpretation
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "phase5_observation_context",
        "readiness": readiness,
        "baseline_audit": baseline_audit,
        "manifest_audit": manifest_audit,
        "conditions": summaries,
        "condition_contracts": {
            "raw_256_baseline": {
                "source": "immutable Phase-4 search_rl artifact",
                "max_obs_length": 256,
                "max_prompt_length": 1408,
            },
            "raw_512": {
                "max_obs_length": 512,
                "max_prompt_length": 1920,
                "max_model_len": 2048,
                "inference_context_distribution_shift": (
                    "The checkpoint was trained with max_obs_length=256."
                ),
            },
            "compressed_256": {
                "max_obs_length": 256,
                "max_prompt_length": 1408,
                "max_model_len": 1536,
                "uses_ground_truth": False,
                "additional_retriever_calls": 0,
            },
        },
        "compression_assessment": compression_assessment(summaries),
        "descriptive_comparisons": comparisons,
        "interpretation": interpretation,
    }
    return summary, statistics_report


def _format_number(value):
    return "n/a" if value is None else f"{float(value):.6g}"


def render_paired_markdown(report):
    readiness = report["readiness"]
    lines = [
        "# Phase-5 observation-context paired statistics",
        "",
        f"- Engineering pass: `{str(readiness['engineering_pass']).lower()}`",
        f"- Quality claim ready: `{str(readiness['quality_claim_ready']).lower()}`",
        f"- Paired rows: {readiness['paired_row_count']} "
        f"({readiness['source_counts']['nq']} NQ / "
        f"{readiness['source_counts']['hotpotqa']} HotpotQA)",
    ]
    if readiness["warning"]:
        lines.extend(["", f"> {readiness['warning']}"])
    lines.extend([
        "",
        "| Classification | Comparison | Slice | EM difference (pp) | 95% paired-bootstrap CI | Exact McNemar p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    comparisons = [("primary", report["primary"])] + [
        ("secondary", comparison) for comparison in report["secondary"].values()
    ]
    for classification, comparison in comparisons:
        for slice_name in ("overall", "nq", "hotpotqa"):
            values = comparison[slice_name]
            lower, upper = values["bootstrap_ci_95_pp"]
            lines.append(
                f"| {classification} | {comparison['comparison']} | {slice_name} | "
                f"{_format_number(values['difference_pp'])} | "
                f"[{_format_number(lower)}, {_format_number(upper)}] | "
                f"{_format_number(values['mcnemar']['exact_two_sided_p_value'])} |"
            )
    lines.extend([
        "",
        report["interpretation"]["conclusion"],
        "",
        report["interpretation"]["guardrail"],
        "",
    ])
    return "\n".join(lines)


def write_reports(summary, statistics_report, results_dir):
    """Write only isolated Phase-5 summary/statistics artifacts."""

    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = results_dir / "paired_statistics.json"
    markdown_path = results_dir / "paired_statistics.md"
    summary_path = results_dir / "summary.json"
    statistics_path.write_text(
        json.dumps(statistics_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_paired_markdown(statistics_report), encoding="utf-8")
    summary = dict(summary)
    summary["artifacts"] = {
        "paired_statistics": {
            "path": str(statistics_path),
            "sha256": sha256_file(statistics_path),
        },
        "paired_statistics_markdown": {
            "path": str(markdown_path),
            "sha256": sha256_file(markdown_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary_path,
        "paired_statistics": statistics_path,
        "paired_statistics_markdown": markdown_path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--search-model", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed != 42:
        raise ValueError("Phase-5 paired statistics seed must remain 42")
    manifest_path = Path(args.eval_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_records, baseline_audit = validate_and_load_baseline(
        args.baseline_results,
        manifest,
        manifest_path,
        args.search_model,
    )
    results_dir = validate_results_directory_isolation(
        args.results_dir, args.baseline_results
    )
    result_sets = {
        "raw_256_baseline": baseline_records,
        "raw_512": read_phase5_result_file(results_dir / "raw_512.jsonl", "raw_512"),
        "compressed_256": read_phase5_result_file(
            results_dir / "compressed_256.jsonl", "compressed_256"
        ),
    }
    validate_phase5_bindings(result_sets, baseline_audit, args.search_model)
    joined = join_result_sets(result_sets, CONDITIONS)
    manifest_audit = validate_manifest_binding(joined, manifest, manifest_path)
    summary, statistics_report = build_reports(
        result_sets,
        baseline_audit,
        manifest_audit,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    input_artifacts = {
        "raw_256_baseline": {
            "path": baseline_audit["path"],
            "sha256": baseline_audit["sha256"],
            "row_count": baseline_audit["row_count"],
            "immutable": True,
        },
        "raw_512": {
            "path": str((results_dir / "raw_512.jsonl").resolve()),
            "sha256": sha256_file(results_dir / "raw_512.jsonl"),
            "row_count": len(result_sets["raw_512"]),
        },
        "compressed_256": {
            "path": str((results_dir / "compressed_256.jsonl").resolve()),
            "sha256": sha256_file(results_dir / "compressed_256.jsonl"),
            "row_count": len(result_sets["compressed_256"]),
        },
        "eval_manifest": manifest_audit,
    }
    summary["input_artifacts"] = input_artifacts
    statistics_report["configuration"]["input_artifacts"] = input_artifacts
    paths = write_reports(summary, statistics_report, results_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for label, path in paths.items():
        print(f"Phase-5 {label}: {path}")


if __name__ == "__main__":
    main()
