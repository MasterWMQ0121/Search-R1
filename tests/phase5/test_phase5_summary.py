import hashlib
import json
from pathlib import Path

import pytest

from experiments.phase4_benchmark import run_benchmark as phase4
from experiments.phase5_observation_context import evidence_compressor
from experiments.phase5_observation_context import run_context_ablation as runner
from experiments.phase5_observation_context import summarize_results as summary


def _run_config(condition, *, eval_sha="eval-sha", manifest_sha="manifest-sha"):
    if condition == "raw_256_baseline":
        mode = "search_rl"
        config = phase4._default_test_run_config(mode, "trained-model")
        config["prompt_contract"] = "phase4-benchmark-v1"
    else:
        limits = {
            "raw_512": (512, 1920, 2048),
            "compressed_256": (256, 1408, 1536),
        }[condition]
        config = {
            "schema_version": 1,
            "prompt_contract": "phase4-benchmark-v1",
            "experiment_contract": "phase5-observation-context-v1",
            "mode": condition,
            "observation_policy": condition,
            "observation_policy": condition,
            "model_path": "trained-model",
            "seed": 42,
            "greedy": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.20,
            "attention_backend": "test",
            "max_start_length": 768,
            "max_response_length": 128,
            "max_obs_length": limits[0],
            "max_prompt_length": limits[1],
            "max_model_len": limits[2],
            "max_turns": 2,
            "retriever_url": "test",
            "retriever_topk": 3,
            "eval_sha256": eval_sha,
            "eval_manifest_sha256": manifest_sha,
            "raw_retriever_format": "LLMGenerationManager._passages2string",
            "training_max_obs_length": 256,
            "inference_context_distribution_shift": condition == "raw_512",
            "compressor": (
                {
                    "name": "deterministic_query_aware_extractive",
                    "version": "phase5-extractive-v1",
                    "source_sha256": hashlib.sha256(
                        Path(evidence_compressor.__file__).read_bytes()
                    ).hexdigest(),
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
                if condition == "compressed_256" else None
            ),
            "baseline_path": "/phase4/search_rl.jsonl",
            "baseline_sha256": "baseline-sha",
            "baseline_run_fingerprint": "baseline-fingerprint",
        }
    config["eval_sha256"] = eval_sha
    config["eval_manifest_sha256"] = manifest_sha
    return config


def _record(
    condition,
    uid,
    source,
    exact_match,
    *,
    raw=(480,),
    policy=None,
    retained=None,
    actions=2,
    valid_actions=2,
    valid_searches=1,
    finished=True,
    end_to_end=10.0,
    generation=4.0,
    retrieval=6.0,
    compression=0.0,
    evaluation_error=None,
):
    policy = tuple(raw if policy is None else policy)
    retained = tuple(policy if retained is None else retained)
    retrieval_count = len(policy)
    config = _run_config(condition)
    record = {
        "schema_version": 1,
        "uid": uid,
        "data_source": source,
        "question": f"question-{uid}",
        "ground_truth": [f"answer-{uid}"],
        "mode": condition,
        "model_path": "trained-model",
        "prediction": f"answer-{uid}" if exact_match else "wrong",
        "exact_match": exact_match,
        "end_to_end_latency_s": end_to_end,
        "generation_latency_s": generation,
        "trajectory": "<answer>answer</answer>",
        "run_config": config,
        "run_fingerprint": phase4.run_config_fingerprint(config),
        "number_of_actions": actions,
        "number_of_valid_actions": valid_actions,
        "number_of_valid_searches": valid_searches,
        "number_of_successful_retrievals": retrieval_count,
        "finished": finished,
        "search_retrieval_failure_count": 0,
        "retrieval_latency_s": retrieval,
        "observation_truncation_count": sum(
            kept < produced for produced, kept in zip(policy, retained)
        ),
        "trajectory_had_observation_truncation": any(
            kept < produced for produced, kept in zip(policy, retained)
        ),
        "retrieved_observation_lengths_before_truncation": list(policy),
        "retained_observation_lengths_after_truncation": list(retained),
        "observation_excess_tokens": [
            max(produced - (512 if condition == "raw_512" else 256), 0)
            for produced in policy
        ],
        "observation_policy": condition,
        "raw_retrieved_observation_tokens": list(raw),
        "policy_output_observation_tokens": list(policy),
        "retained_observation_tokens": list(retained),
        "policy_compression_ratio": [
            produced / original if original else 1.0
            for original, produced in zip(raw, policy)
        ],
        "post_policy_truncation_count": sum(
            kept < produced for produced, kept in zip(policy, retained)
        ),
        "documents_returned": [3] * retrieval_count,
        "documents_represented": [3 if condition != "compressed_256" else 2]
        * retrieval_count,
        "sentences_considered": [0 if condition != "compressed_256" else 8]
        * retrieval_count,
        "sentences_selected": [0 if condition != "compressed_256" else 3]
        * retrieval_count,
        "evidence_compression_latency_s": compression,
        "zero_overlap_fallback_used": False,
        "partial_sentence_fallback_used": False,
        "selected_document_ranks": [
            [1, 2] if condition == "compressed_256" else [1, 2, 3]
            for _ in range(retrieval_count)
        ],
        "selected_sentence_identifiers": [
            [
                "document_rank=1;sentence_index=0",
                "document_rank=1;sentence_index=1",
                "document_rank=2;sentence_index=0",
            ] if condition == "compressed_256" else []
            for _ in range(retrieval_count)
        ],
        "evaluation_error": evaluation_error,
    }
    return record


def _result_sets(count_per_source=2):
    sets = {condition: [] for condition in summary.CONDITIONS}
    for source in summary.SOURCES:
        for index in range(count_per_source):
            uid = f"{source}:{index}"
            baseline_outcome = int(index == 0)
            sets["raw_256_baseline"].append(
                _record(
                    "raw_256_baseline",
                    uid,
                    source,
                    baseline_outcome,
                    retained=(256,),
                )
            )
            sets["raw_512"].append(
                _record(
                    "raw_512",
                    uid,
                    source,
                    int(index <= 1),
                    raw=(480,),
                    policy=(480,),
                    retained=(480,),
                    end_to_end=11.0,
                )
            )
            sets["compressed_256"].append(
                _record(
                    "compressed_256",
                    uid,
                    source,
                    int(index <= 1),
                    raw=(480,),
                    policy=(220,),
                    retained=(220,),
                    compression=0.02,
                    end_to_end=10.5,
                )
            )
    return sets


def test_summary_aggregates_quality_latency_agent_and_context_metrics():
    records = [
        _record(
            "compressed_256",
            "nq:1",
            "nq",
            1,
            raw=(500, 300),
            policy=(200, 150),
            retained=(200, 150),
            compression=0.04,
            end_to_end=8.0,
            generation=3.0,
            retrieval=4.0,
        ),
        _record(
            "compressed_256",
            "hotpotqa:1",
            "hotpotqa",
            0,
            raw=(),
            policy=(),
            retained=(),
            actions=1,
            valid_actions=1,
            valid_searches=0,
            end_to_end=12.0,
            generation=5.0,
            retrieval=0.0,
            compression=0.0,
        ),
    ]

    result = summary.summarize_condition(records)

    assert result["quality"]["overall_em"] == 0.5
    assert result["quality"]["nq_em"] == 1.0
    assert result["quality"]["hotpotqa_em"] == 0.0
    assert result["latency"]["end_to_end"]["mean_s"] == 10.0
    assert result["latency"]["compression"]["mean_s"] == 0.02
    assert result["agent"]["finish_ratio"] == 1.0
    context = result["context"]
    assert context["raw_observation_tokens_mean"] == 400.0
    assert context["policy_output_tokens_mean"] == 175.0
    assert context["mean_tokens_saved"] == 225.0
    assert context["mean_compression_ratio"] == pytest.approx(0.45)
    assert context["mean_documents_represented"] == 2.0
    assert context["mean_sentences_selected"] == 3.0


def test_paired_comparison_has_declared_orientation_and_source_breakdown():
    result_sets = _result_sets()
    joined = summary.join_result_sets(result_sets, summary.CONDITIONS)
    readiness = summary.engineering_readiness(
        result_sets, {"validated": True}, expected_count=4
    )

    report = summary.paired_statistics_report(
        joined,
        readiness,
        {"sha256": "manifest-sha"},
        bootstrap_samples=100,
    )

    assert report["primary"]["orientation"] == (
        "EM(compressed_256) - EM(raw_256_baseline)"
    )
    assert report["primary"]["mode_a"] == "compressed_256"
    assert report["primary"]["mode_b"] == "raw_256_baseline"
    assert report["primary"]["nq"]["sample_size"] == 2
    assert report["primary"]["hotpotqa"]["sample_size"] == 2
    assert report["configuration"]["bootstrap_samples"] == 100
    assert report["configuration"]["bootstrap_seed"] == 42


def test_strict_join_rejects_question_and_hash_mismatches():
    result_sets = _result_sets()
    result_sets["compressed_256"][0]["question"] = "other question"
    with pytest.raises(ValueError, match="mismatched question"):
        summary.join_result_sets(result_sets, summary.CONDITIONS)


def test_summary_accepts_the_exact_runner_binding_and_rejects_policy_drift():
    result_sets = _result_sets()
    baseline_fingerprint = result_sets["raw_256_baseline"][0]["run_fingerprint"]
    for condition in ("raw_512", "compressed_256"):
        for record in result_sets[condition]:
            record["run_config"]["baseline_run_fingerprint"] = baseline_fingerprint
            record["run_fingerprint"] = phase4.run_config_fingerprint(
                record["run_config"]
            )
    baseline_audit = {
        "path": "/phase4/search_rl.jsonl",
        "sha256": "baseline-sha",
        "run_fingerprint": baseline_fingerprint,
    }

    summary.validate_phase5_bindings(
        result_sets, baseline_audit, "trained-model"
    )
    for condition in ("raw_512", "compressed_256"):
        for record in result_sets[condition]:
            runner.validate_phase5_result(record, condition)

    for record in result_sets["compressed_256"]:
        record["run_config"]["compressor"]["source_sha256"] = "different-policy"
        record["run_fingerprint"] = phase4.run_config_fingerprint(
            record["run_config"]
        )
    with pytest.raises(ValueError, match="compressor contract"):
        summary.validate_phase5_bindings(
            result_sets, baseline_audit, "trained-model"
        )

    result_sets = _result_sets()
    result_sets["raw_512"][0]["run_config"]["eval_sha256"] = "other"
    with pytest.raises(ValueError, match="same eval manifest/parquet hashes"):
        summary.join_result_sets(result_sets, summary.CONDITIONS)


def test_engineering_readiness_requires_64_balanced_error_free_rows():
    result_sets = _result_sets(count_per_source=32)
    ready = summary.engineering_readiness(result_sets, {"validated": True})
    assert ready["engineering_pass"] is True
    assert ready["source_counts"] == {"nq": 32, "hotpotqa": 32}

    result_sets["compressed_256"][0]["evaluation_error"] = "failure"
    not_ready = summary.engineering_readiness(result_sets, {"validated": True})
    assert not_ready["engineering_pass"] is False
    assert not_ready["checks"]["no_evaluation_errors"] is False
    assert "no quality claim" in not_ready["warning"]


def test_writes_three_isolated_artifacts_with_neutral_measured_interpretation(tmp_path):
    result_sets = _result_sets(count_per_source=32)
    summary_report, paired_report = summary.build_reports(
        result_sets,
        {"validated": True, "sha256": "baseline-sha"},
        {"sha256": "manifest-sha"},
        bootstrap_samples=100,
    )

    paths = summary.write_reports(summary_report, paired_report, tmp_path)

    assert set(paths) == {
        "summary",
        "paired_statistics",
        "paired_statistics_markdown",
    }
    written_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    written_paired = json.loads(
        paths["paired_statistics"].read_text(encoding="utf-8")
    )
    markdown = paths["paired_statistics_markdown"].read_text(encoding="utf-8")
    assert written_summary["readiness"]["engineering_pass"] is True
    assert written_paired["primary"]["classification"] == "primary"
    assert "compressed_256_vs_raw_256_baseline" in markdown
    assert "raw_512_vs_raw_256_baseline" in markdown
    assert "production-wide causality" in markdown
    assert "95.3125" not in markdown
    assert written_summary["artifacts"]["paired_statistics"]["sha256"] == (
        hashlib.sha256(paths["paired_statistics"].read_bytes()).hexdigest()
    )


def _phase4_baseline_record(uid, source, config):
    return {
        "schema_version": 1,
        "uid": uid,
        "data_source": source,
        "question": f"question-{uid}",
        "ground_truth": [f"answer-{uid}"],
        "mode": "search_rl",
        "model_path": "trained-model",
        "prediction": "wrong",
        "exact_match": 0,
        "end_to_end_latency_s": 1.0,
        "generation_latency_s": 0.5,
        "trajectory": "<answer>wrong</answer>",
        "run_config": config,
        "run_fingerprint": phase4.run_config_fingerprint(config),
        "number_of_actions": 2,
        "number_of_valid_actions": 2,
        "number_of_valid_searches": 1,
        "number_of_successful_retrievals": 1,
        "finished": True,
        "search_retrieval_failure_count": 0,
        "retrieval_latency_s": 0.5,
        "observation_truncation_count": 1,
        "trajectory_had_observation_truncation": True,
        "retrieved_observation_lengths_before_truncation": [480],
        "retained_observation_lengths_after_truncation": [256],
        "observation_excess_tokens": [224],
    }


def test_baseline_is_adapted_in_memory_and_never_rewritten(tmp_path):
    manifest = {"output": {"sha256": "eval-sha"}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = _run_config(
        "raw_256_baseline", eval_sha="eval-sha", manifest_sha=manifest_sha
    )
    records = [
        _phase4_baseline_record("nq:1", "nq", config),
        _phase4_baseline_record("hotpotqa:1", "hotpotqa", config),
    ]
    baseline_path = tmp_path / "search_rl.jsonl"
    baseline_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    before = baseline_path.read_bytes()

    adapted, audit = summary.validate_and_load_baseline(
        baseline_path,
        manifest,
        manifest_path,
        "trained-model",
        expected_count=2,
    )

    assert baseline_path.read_bytes() == before
    assert audit["validated"] is True
    assert audit["sha256"] == hashlib.sha256(before).hexdigest()
    assert [record["mode"] for record in adapted] == [
        "raw_256_baseline",
        "raw_256_baseline",
    ]
    assert adapted[0]["raw_retrieved_observation_tokens"] == [480]
    assert adapted[0]["policy_output_observation_tokens"] == [480]
    assert adapted[0]["retained_observation_tokens"] == [256]
    assert adapted[0]["policy_compression_ratio"] == [1.0]


def test_cli_contract_contains_required_phase5_inputs():
    args = summary.parse_args([
        "--baseline-results", "baseline.jsonl",
        "--results-dir", "results",
        "--eval-manifest", "manifest.json",
        "--search-model", "trained-model",
    ])
    assert args.baseline_results == "baseline.jsonl"
    assert args.results_dir == "results"
    assert args.eval_manifest == "manifest.json"
    assert args.search_model == "trained-model"
