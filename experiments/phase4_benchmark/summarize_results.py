#!/usr/bin/env python3
"""Aggregate measured Phase-4 JSONL results into one machine-readable summary."""

import argparse
import json
import statistics
from pathlib import Path

from experiments.phase4_benchmark.prepare_eval_data import sha256_file
from experiments.phase4_benchmark.run_benchmark import (
    MODES,
    read_result_file,
    validate_result_record,
)


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"mean_s": None, "p50_s": None, "p95_s": None}
    return {
        "mean_s": statistics.fmean(values),
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
    }


def _quality_slice(records):
    correct = sum(int(record["exact_match"]) for record in records)
    total = len(records)
    return {
        "correct": correct,
        "total": total,
        "em": correct / total if total else None,
    }


def quality_summary(records):
    overall = _quality_slice(records)
    by_source = {
        source: _quality_slice([
            record for record in records if record["data_source"] == source
        ])
        for source in ("nq", "hotpotqa")
    }
    return {
        "overall_em": overall["em"],
        "nq_em": by_source["nq"]["em"],
        "hotpotqa_em": by_source["hotpotqa"]["em"],
        "correct": overall["correct"],
        "total": overall["total"],
        "by_source": by_source,
    }


def _mean(values):
    return statistics.fmean(values) if values else None


def _mean_ratios(numerators, denominators):
    ratios = [
        numerator / denominator
        for numerator, denominator in zip(numerators, denominators)
        if denominator
    ]
    return _mean(ratios)


def static_rag_metrics(records):
    before = [record["retrieved_context_tokens_before_truncation"] for record in records]
    retained = [record["retrieved_context_tokens_retained"] for record in records]
    budgets = {record["static_context_token_budget"] for record in records}
    if len(budgets) != 1:
        raise ValueError("Static-RAG results do not share one context token budget")
    return {
        "retrieval_latency": latency_summary([
            record["retrieval_latency_s"] for record in records
        ]),
        "retrieval_call_count": sum(record["retrieval_call_count"] for record in records),
        "retrieval_calls_per_example": _mean([
            record["retrieval_call_count"] for record in records
        ]),
        "retrieved_context_token_budget": next(iter(budgets)),
        "context_truncation_count": sum(
            raw > kept for raw, kept in zip(before, retained)
        ),
        "fraction_examples_with_context_truncation": _mean([
            raw > kept for raw, kept in zip(before, retained)
        ]),
        "retrieved_context_tokens_before_truncation_mean": _mean(before),
        "retrieved_context_tokens_before_truncation_max": max(before) if before else None,
        "retrieved_context_tokens_retained_mean": _mean(retained),
        "retrieved_context_tokens_retained_max": max(retained) if retained else None,
    }


def search_agent_metrics(records):
    behavior_records = [
        record for record in records if not record["search_retrieval_failure_count"]
    ]
    actions = [record["number_of_actions"] for record in behavior_records]
    valid_actions = [record["number_of_valid_actions"] for record in behavior_records]
    valid_searches = [record["number_of_valid_searches"] for record in behavior_records]
    successful = [
        record["number_of_successful_retrievals"] for record in behavior_records
    ]
    raw_lengths = []
    retained_lengths = []
    excess_tokens = []
    for record in behavior_records:
        raw = record["retrieved_observation_lengths_before_truncation"]
        retained = record["retained_observation_lengths_after_truncation"]
        excess = record["observation_excess_tokens"]
        if not (len(raw) == len(retained) == len(excess)):
            raise ValueError(f"Search-RL observation metrics are misaligned for {record['uid']}")
        raw_lengths.extend(raw)
        retained_lengths.extend(retained)
        excess_tokens.extend(excess)

    return {
        "behavior_trajectory_count": len(behavior_records),
        "behavior_trajectories_excluded_for_retrieval_failure": (
            len(records) - len(behavior_records)
        ),
        "finish_ratio": _mean([
            bool(record["finished"]) for record in behavior_records
        ]),
        "valid_action_ratio": _mean_ratios(valid_actions, actions),
        "valid_search_ratio": _mean_ratios(valid_searches, actions),
        "mean_valid_searches_per_trajectory": _mean(valid_searches),
        "mean_successful_retrievals_per_trajectory": _mean(successful),
        "fraction_trajectories_with_retrieval": _mean([
            count > 0 for count in successful
        ]),
        "mean_number_of_actions": _mean(actions),
        "search_retrieval_failure_count": sum(
            record["search_retrieval_failure_count"] for record in records
        ),
        "generation_latency": latency_summary([
            record["generation_latency_s"] for record in records
        ]),
        "retrieval_latency": latency_summary([
            record["retrieval_latency_s"] for record in records
        ]),
        "observation_truncation_count": sum(
            record["observation_truncation_count"] for record in records
        ),
        "fraction_trajectories_with_observation_truncation": _mean([
            bool(record["trajectory_had_observation_truncation"])
            for record in behavior_records
        ]),
        "retrieved_observation_count": len(raw_lengths),
        "retrieved_observation_tokens_before_truncation_mean": _mean(raw_lengths),
        "retrieved_observation_tokens_before_truncation_max": (
            max(raw_lengths) if raw_lengths else None
        ),
        "retained_observation_tokens_after_truncation_mean": _mean(retained_lengths),
        "retained_observation_tokens_after_truncation_max": (
            max(retained_lengths) if retained_lengths else None
        ),
        "excess_tokens_above_max_obs_length_mean": _mean(excess_tokens),
        "excess_tokens_above_max_obs_length_max": (
            max(excess_tokens) if excess_tokens else None
        ),
    }


def summarize_mode(records, mode):
    quality = quality_summary(records)
    summary = {
        **quality,
        "latency": latency_summary([
            record["end_to_end_latency_s"] for record in records
        ]),
    }
    if mode == "static_rag":
        summary["static_rag"] = static_rag_metrics(records)
    if mode == "search_rl":
        summary["agent"] = search_agent_metrics(records)
    return summary


def _truncation_failure_analysis(records, truncation_predicate, label):
    by_source = {}
    total_incorrect_with_truncation = 0
    total_incorrect = 0
    for source in ("nq", "hotpotqa"):
        source_incorrect = [
            record for record in records
            if record["data_source"] == source and not record["exact_match"]
        ]
        with_truncation = sum(truncation_predicate(record) for record in source_incorrect)
        by_source[source] = {
            "incorrect": len(source_incorrect),
            "incorrect_with_truncation": with_truncation,
            "fraction_incorrect_with_truncation": (
                with_truncation / len(source_incorrect)
                if source_incorrect else None
            ),
        }
        total_incorrect += len(source_incorrect)
        total_incorrect_with_truncation += with_truncation

    if total_incorrect_with_truncation:
        assessment = (
            f"{label} co-occurred with {total_incorrect_with_truncation} incorrect "
            "examples, so it is a plausible contributor to some errors; this "
            "observational benchmark does not establish causality."
        )
    else:
        assessment = (
            f"No incorrect example recorded {label}; this run provides no measured "
            "co-occurrence evidence, but does not prove that truncation is harmless."
        )
    return {
        "incorrect": total_incorrect,
        "incorrect_with_truncation": total_incorrect_with_truncation,
        "fraction_incorrect_with_truncation": (
            total_incorrect_with_truncation / total_incorrect
            if total_incorrect else None
        ),
        "plausible_contributor": total_incorrect_with_truncation > 0,
        "by_source": by_source,
        "assessment": assessment,
    }


def summarize_all(result_sets):
    if set(result_sets) != set(MODES):
        raise ValueError(f"summary requires exactly these modes: {MODES}")
    uid_sets = {mode: {record["uid"] for record in records} for mode, records in result_sets.items()}
    baseline_uids = uid_sets["direct"]
    if any(uids != baseline_uids for uids in uid_sets.values()):
        raise ValueError("all Phase-4 modes must contain exactly the same UID set")
    if any(len(records) != len(baseline_uids) for records in result_sets.values()):
        raise ValueError("a Phase-4 mode contains duplicate or missing result rows")
    for mode, records in result_sets.items():
        for record in records:
            validate_result_record(record, mode)
        fingerprints = {record["run_fingerprint"] for record in records}
        if len(fingerprints) != 1:
            raise ValueError(f"{mode} results mix multiple run configurations")

    baseline_examples = {
        record["uid"]: (
            record["data_source"], record["question"], tuple(record["ground_truth"])
        )
        for record in result_sets["direct"]
    }
    for mode in ("static_rag", "search_rl"):
        examples = {
            record["uid"]: (
                record["data_source"], record["question"], tuple(record["ground_truth"])
            )
            for record in result_sets[mode]
        }
        if examples != baseline_examples:
            raise ValueError(
                f"{mode} results do not match Direct questions and ground truths"
            )

    summaries = {
        mode: summarize_mode(result_sets[mode], mode)
        for mode in MODES
    }
    direct_em = summaries["direct"]["overall_em"]
    static_em = summaries["static_rag"]["overall_em"]
    search_em = summaries["search_rl"]["overall_em"]
    summaries["comparisons"] = {
        "search_rl_vs_direct_em_pp": (search_em - direct_em) * 100.0,
        "search_rl_vs_static_rag_em_pp": (search_em - static_em) * 100.0,
    }
    summaries["failure_analysis"] = {
        "static_rag_context_truncation": _truncation_failure_analysis(
            result_sets["static_rag"],
            lambda record: (
                record["retrieved_context_tokens_before_truncation"]
                > record["retrieved_context_tokens_retained"]
            ),
            "Static-RAG retrieved-context truncation",
        ),
        "search_rl_observation_truncation": _truncation_failure_analysis(
            result_sets["search_rl"],
            lambda record: bool(record["trajectory_had_observation_truncation"]),
            "Search-RL retrieved-observation truncation",
        ),
    }
    return summaries


def validate_results_against_manifest(result_sets, manifest, manifest_sha256):
    expected_uids = [
        entry.get("uid") for entry in manifest.get("selected_source_rows", [])
    ]
    if not expected_uids or len(expected_uids) != len(set(expected_uids)):
        raise ValueError("evaluation manifest has missing or duplicate selected UIDs")
    expected_eval_sha = manifest.get("output", {}).get("sha256")
    run_configs = {}
    for mode in MODES:
        records = result_sets[mode]
        if [record["uid"] for record in records] != expected_uids:
            raise ValueError(f"{mode} result order/UIDs do not match the eval manifest")
        configs = {
            json.dumps(record["run_config"], sort_keys=True): record["run_config"]
            for record in records
        }
        if len(configs) != 1:
            raise ValueError(f"{mode} results mix multiple run configurations")
        run_config = next(iter(configs.values()))
        if run_config.get("eval_sha256") != expected_eval_sha:
            raise ValueError(f"{mode} run_config is bound to a different eval parquet")
        if run_config.get("eval_manifest_sha256") != manifest_sha256:
            raise ValueError(f"{mode} run_config is bound to a different eval manifest")
        run_configs[mode] = run_config
    return run_configs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    result_paths = {mode: results_dir / f"{mode}.jsonl" for mode in MODES}
    result_sets = {
        mode: read_result_file(path, mode) for mode, path in result_paths.items()
    }
    if any(len(records) != 64 for records in result_sets.values()):
        counts = {mode: len(records) for mode, records in result_sets.items()}
        raise ValueError(f"each primary Phase-4 mode must contain 64 rows: {counts}")
    manifest_path = Path(args.eval_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("seed") != 42
        or manifest.get("selected_row_count") != 64
        or not manifest.get("non_overlap_audit", {}).get("passed")
    ):
        raise ValueError("the Phase-4 evaluation manifest is not the approved audited set")
    manifest_sha = sha256_file(manifest_path)
    run_configs = validate_results_against_manifest(
        result_sets, manifest, manifest_sha
    )
    summary = summarize_all(result_sets)
    summary["benchmark"] = {
        "eval_manifest_path": str(manifest_path),
        "eval_manifest_sha256": manifest_sha,
        "eval_parquet_sha256": manifest.get("output", {}).get("sha256"),
        "seed": 42,
        "question_count": 64,
        "nq_count": 32,
        "hotpotqa_count": 32,
        "greedy": True,
        "max_response_length": 128,
        "models": {mode: run_configs[mode]["model_path"] for mode in MODES},
        "run_fingerprints": {
            mode: result_sets[mode][0]["run_fingerprint"] for mode in MODES
        },
        "static_rag": {
            "retriever_topk": 3,
            "retriever_url": run_configs["static_rag"]["retriever_url"],
            "retrieved_context_token_budget": 512,
            "retrieval_stages_per_example": 1,
        },
        "search_rl": {
            "retriever_topk": 3,
            "retriever_url": run_configs["search_rl"]["retriever_url"],
            "max_turns": 2,
            "max_obs_length": 256,
            "max_prompt_length": 1408,
        },
    }
    retrieval_failure_count = summary["search_rl"]["agent"][
        "search_retrieval_failure_count"
    ]
    summary["benchmark_status"] = {
        "quality_claim_ready": retrieval_failure_count == 0,
        "search_retrieval_failure_count": retrieval_failure_count,
        "warning": (
            None
            if retrieval_failure_count == 0
            else "Search-RL retrieval failures contaminated quality results; rerun before making a quality claim."
        ),
    }
    summary["artifacts"] = {
        mode: {"path": str(path), "sha256": sha256_file(path)}
        for mode, path in result_paths.items()
    }
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output else results_dir / "summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase-4 summary: {output_path}")


if __name__ == "__main__":
    main()
