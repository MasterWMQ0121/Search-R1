#!/usr/bin/env python3
"""Run the isolated Phase-5 Search-RL observation-context ablation."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import MethodType

from experiments.phase4_benchmark.prepare_eval_data import sha256_file
from experiments.phase4_benchmark.run_benchmark import (
    DEFAULT_RETRIEVER_URL,
    DEFAULT_SEARCH_MODEL,
    VLLMGenerator,
    evaluate_search_agent,
    load_eval_examples,
    read_result_file,
    resume_plan,
    run_config_fingerprint,
)


CONDITIONS = ("raw_256_baseline", "raw_512", "compressed_256")
RUNNABLE_CONDITIONS = ("raw_512", "compressed_256")
SCHEMA_VERSION = 1
MAX_START_LENGTH = 768
MAX_RESPONSE_LENGTH = 128
MAX_TURNS = 2
RETRIEVER_TOPK = 3
SEED = 42
GPU_MEMORY_UTILIZATION = 0.20
CONDITION_LIMITS = {
    "raw_256_baseline": {
        "max_obs_length": 256,
        "max_prompt_length": 1408,
        "max_model_len": 1536,
    },
    "raw_512": {
        "max_obs_length": 512,
        "max_prompt_length": 1920,
        "max_model_len": 2048,
    },
    "compressed_256": {
        "max_obs_length": 256,
        "max_prompt_length": 1408,
        "max_model_len": 1536,
    },
}

PHASE4_AGENT_FIELDS = {
    "number_of_actions",
    "number_of_valid_actions",
    "number_of_valid_searches",
    "number_of_successful_retrievals",
    "finished",
    "search_retrieval_failure_count",
    "retrieval_latency_s",
    "observation_truncation_count",
    "trajectory_had_observation_truncation",
    "retrieved_observation_lengths_before_truncation",
    "retained_observation_lengths_after_truncation",
    "observation_excess_tokens",
}
PHASE5_CONTEXT_FIELDS = {
    "observation_policy",
    "raw_retrieved_observation_tokens",
    "policy_output_observation_tokens",
    "retained_observation_tokens",
    "policy_compression_ratio",
    "post_policy_truncation_count",
    "documents_returned",
    "documents_represented",
    "sentences_considered",
    "sentences_selected",
    "evidence_compression_latency_s",
    "zero_overlap_fallback_used",
    "partial_sentence_fallback_used",
    "selected_document_ranks",
    "selected_sentence_identifiers",
}
COMMON_FIELDS = {
    "schema_version",
    "uid",
    "data_source",
    "question",
    "ground_truth",
    "mode",
    "model_path",
    "prediction",
    "exact_match",
    "end_to_end_latency_s",
    "generation_latency_s",
    "trajectory",
    "run_config",
    "run_fingerprint",
}
REQUIRED_RESULT_FIELDS = COMMON_FIELDS | PHASE4_AGENT_FIELDS | PHASE5_CONTEXT_FIELDS


def _single_run_config(records, label):
    configurations = {
        json.dumps(record["run_config"], sort_keys=True): record["run_config"]
        for record in records
    }
    fingerprints = {record["run_fingerprint"] for record in records}
    if len(configurations) != 1 or len(fingerprints) != 1:
        raise ValueError(f"{label} mixes multiple run configurations")
    return next(iter(configurations.values())), next(iter(fingerprints))


def validate_results_directory_isolation(output_dir, baseline_path):
    """Reject the Phase-4 artifact tree as a Phase-5 write target."""

    output_dir = Path(output_dir).expanduser().resolve()
    baseline_path = Path(baseline_path).expanduser().resolve()
    phase4_dir = baseline_path.parent
    if output_dir == phase4_dir or phase4_dir in output_dir.parents:
        raise ValueError(
            "Phase-5 outputs must not be written into the Phase-4 artifact "
            "directory or any of its descendants"
        )
    return output_dir


def validate_baseline_artifact(
    baseline_path,
    examples,
    manifest,
    manifest_path,
    expected_model=DEFAULT_SEARCH_MODEL,
    expected_count=64,
):
    """Validate the immutable Phase-4 Search-RL baseline without rewriting it."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    before_sha256 = sha256_file(baseline_path)
    records = read_result_file(baseline_path, "search_rl")
    if len(records) != expected_count:
        raise ValueError(
            f"raw_256_baseline must contain {expected_count} rows; found {len(records)}"
        )
    if len(examples) != expected_count:
        raise ValueError(
            f"evaluation data must contain {expected_count} rows; found {len(examples)}"
        )
    pending, records_by_uid = resume_plan(examples, records)
    if pending or len(records_by_uid) != expected_count:
        raise ValueError("raw_256_baseline does not cover the exact Phase-4 UID set")
    if [record["uid"] for record in records] != [example.uid for example in examples]:
        raise ValueError("raw_256_baseline row order does not match Phase-4 evaluation data")

    run_config, run_fingerprint = _single_run_config(
        records, "raw_256_baseline"
    )
    required_config = {
        "model_path": str(expected_model),
        "seed": SEED,
        "greedy": True,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": 256,
        "max_prompt_length": 1408,
        "max_turns": MAX_TURNS,
        "retriever_topk": RETRIEVER_TOPK,
    }
    for key, expected in required_config.items():
        if run_config.get(key) != expected:
            raise ValueError(
                f"raw_256_baseline has unexpected {key}: "
                f"{run_config.get(key)!r}; expected {expected!r}"
            )
    manifest_path = Path(manifest_path).expanduser().resolve()
    expected_eval_sha = manifest.get("output", {}).get("sha256")
    expected_manifest_sha = sha256_file(manifest_path)
    if run_config.get("eval_sha256") != expected_eval_sha:
        raise ValueError("raw_256_baseline is bound to a different eval parquet")
    if run_config.get("eval_manifest_sha256") != expected_manifest_sha:
        raise ValueError("raw_256_baseline is bound to a different eval manifest")
    retrieval_failures = sum(
        record["search_retrieval_failure_count"] for record in records
    )
    evaluation_errors = sum(bool(record.get("evaluation_error")) for record in records)
    if retrieval_failures:
        raise ValueError(
            f"raw_256_baseline contains {retrieval_failures} Retriever failures"
        )
    if evaluation_errors:
        raise ValueError(
            f"raw_256_baseline contains {evaluation_errors} evaluation errors"
        )
    after_sha256 = sha256_file(baseline_path)
    if after_sha256 != before_sha256:
        raise RuntimeError("raw_256_baseline changed during read-only validation")
    return records, {
        "condition": "raw_256_baseline",
        "path": str(baseline_path),
        "sha256": before_sha256,
        "row_count": len(records),
        "run_fingerprint": run_fingerprint,
        "eval_parquet_sha256": expected_eval_sha,
        "eval_manifest_sha256": expected_manifest_sha,
        "retrieval_failure_count": retrieval_failures,
        "evaluation_error_count": evaluation_errors,
        "validated": True,
    }


class ObservationPolicyManager:
    """Install one raw or compressed formatter behind the existing Retriever call."""

    def __init__(self, condition, tokenizer, compressor=None):
        if condition not in RUNNABLE_CONDITIONS:
            raise ValueError(f"unsupported runnable condition: {condition!r}")
        if condition == "compressed_256" and compressor is None:
            raise ValueError("compressed_256 requires an evidence compressor")
        self.condition = condition
        self.tokenizer = tokenizer
        self.compressor = compressor
        self.events = []

    @staticmethod
    def _raw_event(passages):
        ranks = list(range(1, len(passages) + 1))
        return {
            "documents_returned": len(passages),
            "documents_represented": len(passages),
            "sentences_considered": 0,
            "sentences_selected": 0,
            "zero_overlap_fallback_used": False,
            "partial_sentence_fallback_used": False,
            "selected_document_ranks": ranks,
            "selected_sentence_identifiers": [],
            "evidence_compression_latency_s": 0.0,
        }

    def configure_manager(self, manager):
        """Replace formatting only; the timed ``_batch_search`` is called once."""

        def policy_batch_search(_manager, queries):
            payload = _manager._batch_search(queries)
            result_sets = payload.get("result")
            if not isinstance(result_sets, list) or len(result_sets) != len(queries):
                raise RuntimeError("Phase-5 Retriever returned invalid query cardinality")
            observations = []
            for query, passages in zip(queries, result_sets):
                if not isinstance(passages, list):
                    raise RuntimeError("Phase-5 Retriever passages must be a list")
                raw_observation = _manager._passages2string(passages)
                if self.condition == "raw_512":
                    event = self._raw_event(passages)
                    policy_content = raw_observation
                else:
                    compression_started = time.perf_counter()
                    compressed = self.compressor.compress(
                        query, passages, raw_observation=raw_observation
                    )
                    compression_latency = time.perf_counter() - compression_started
                    if not isinstance(compressed, dict) or "content" not in compressed:
                        raise RuntimeError(
                            "EvidenceCompressor.compress must return a mapping with content"
                        )
                    policy_content = str(compressed["content"])
                    event = {
                        key: compressed[key]
                        for key in (
                            "raw_retrieved_observation_tokens",
                            "policy_output_observation_tokens",
                            "documents_returned",
                            "documents_represented",
                            "sentences_considered",
                            "sentences_selected",
                            "zero_overlap_fallback_used",
                            "partial_sentence_fallback_used",
                            "selected_document_ranks",
                            "selected_sentence_identifiers",
                        )
                    }
                    event["evidence_compression_latency_s"] = compression_latency
                event["query"] = str(query)
                event["raw_observation"] = raw_observation
                event["policy_content"] = policy_content
                self.events.append(event)
                observations.append(policy_content)
            return observations

        manager.batch_search = MethodType(policy_batch_search, manager)


def _aligned_list(record, key, expected_length):
    values = record.get(key)
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"{key} must contain one entry per successful retrieval")
    return values


def phase5_result_from_search_result(
    search_result, condition, run_config, policy_events
):
    """Re-label one shared Search Agent result and attach context-policy telemetry."""

    if condition not in RUNNABLE_CONDITIONS:
        raise ValueError(f"unsupported runnable condition: {condition!r}")
    record = dict(search_result)
    retrieval_count = int(record["number_of_successful_retrievals"])
    if len(policy_events) != retrieval_count:
        raise ValueError(
            "policy telemetry does not align with successful Search Agent retrievals"
        )
    policy_lengths = list(
        _aligned_list(
            record,
            "retrieved_observation_lengths_before_truncation",
            retrieval_count,
        )
    )
    retained_lengths = list(
        _aligned_list(
            record,
            "retained_observation_lengths_after_truncation",
            retrieval_count,
        )
    )
    if condition == "raw_512":
        raw_lengths = list(policy_lengths)
    else:
        raw_lengths = [
            int(event["raw_retrieved_observation_tokens"])
            for event in policy_events
        ]
        compressor_policy_lengths = [
            int(event["policy_output_observation_tokens"])
            for event in policy_events
        ]
        if compressor_policy_lengths != policy_lengths:
            raise ValueError(
                "compressor policy-output token counts disagree with manager telemetry"
            )
    ratios = [
        (policy / raw if raw else (1.0 if policy == 0 else None))
        for raw, policy in zip(raw_lengths, policy_lengths)
    ]
    if any(ratio is None for ratio in ratios):
        raise ValueError("a nonempty policy observation cannot have zero raw tokens")

    record.update({
        "schema_version": SCHEMA_VERSION,
        "mode": condition,
        "observation_policy": condition,
        "run_config": run_config,
        "run_fingerprint": run_config_fingerprint(run_config),
        "raw_retrieved_observation_tokens": raw_lengths,
        "policy_output_observation_tokens": policy_lengths,
        "retained_observation_tokens": retained_lengths,
        "policy_compression_ratio": ratios,
        "post_policy_truncation_count": sum(
            retained < policy
            for policy, retained in zip(policy_lengths, retained_lengths)
        ),
        "documents_returned": [
            int(event["documents_returned"]) for event in policy_events
        ],
        "documents_represented": [
            int(event["documents_represented"]) for event in policy_events
        ],
        "sentences_considered": [
            int(event["sentences_considered"]) for event in policy_events
        ],
        "sentences_selected": [
            int(event["sentences_selected"]) for event in policy_events
        ],
        "evidence_compression_latency_s": sum(
            float(event["evidence_compression_latency_s"])
            for event in policy_events
        ),
        "zero_overlap_fallback_used": any(
            bool(event["zero_overlap_fallback_used"]) for event in policy_events
        ),
        "partial_sentence_fallback_used": any(
            bool(event["partial_sentence_fallback_used"]) for event in policy_events
        ),
        "selected_document_ranks": [
            list(event["selected_document_ranks"]) for event in policy_events
        ],
        "selected_sentence_identifiers": [
            list(event["selected_sentence_identifiers"]) for event in policy_events
        ],
    })
    return record


def build_condition_run_config(
    condition,
    model_path,
    retriever_url,
    manifest,
    manifest_path,
    baseline_audit,
    attention_backend="",
):
    if condition not in RUNNABLE_CONDITIONS:
        raise ValueError(f"unsupported runnable condition: {condition!r}")
    limits = CONDITION_LIMITS[condition]
    compressor_config = None
    if condition == "compressed_256":
        from experiments.phase5_observation_context import evidence_compressor

        compressor_config = {
            "name": "deterministic_query_aware_extractive",
            "version": "phase5-extractive-v1",
            "source_sha256": sha256_file(Path(evidence_compressor.__file__)),
            "wrapped_token_budget": 256,
            "bm25_k1": evidence_compressor.BM25_K1,
            "bm25_b": evidence_compressor.BM25_B,
            "query_coverage_weight": evidence_compressor.QUERY_COVERAGE_WEIGHT,
            "title_coverage_weight": evidence_compressor.TITLE_COVERAGE_WEIGHT,
            "rank_prior_weight": evidence_compressor.RANK_PRIOR_WEIGHT,
            "retrieval_score_prior_weight": (
                evidence_compressor.RETRIEVAL_SCORE_PRIOR_WEIGHT
            ),
            "selection_priority": "relevance/sqrt(sentence_token_count)",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_contract": "phase4-benchmark-v1",
        "experiment_contract": "phase5-observation-context-v1",
        "mode": condition,
        "observation_policy": condition,
        "eval_sha256": manifest["output"]["sha256"],
        "eval_manifest_sha256": sha256_file(manifest_path),
        "baseline_path": baseline_audit["path"],
        "baseline_sha256": baseline_audit["sha256"],
        "baseline_run_fingerprint": baseline_audit["run_fingerprint"],
        "model_path": str(model_path),
        "seed": SEED,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "attention_backend": attention_backend,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": limits["max_obs_length"],
        "max_prompt_length": limits["max_prompt_length"],
        "max_model_len": limits["max_model_len"],
        "max_turns": MAX_TURNS,
        "retriever_url": retriever_url,
        "retriever_topk": RETRIEVER_TOPK,
        "raw_retriever_format": "LLMGenerationManager._passages2string",
        "training_max_obs_length": 256,
        "inference_context_distribution_shift": condition == "raw_512",
        "compressor": compressor_config,
    }


def _non_negative_number(value, label, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    if integer:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{label} must be finite and non-negative")


def validate_phase5_result(record, expected_condition=None):
    """Validate the isolated Phase-5 per-example schema."""

    if not isinstance(record, dict):
        raise ValueError("Phase-5 result row must be a JSON object")
    condition = record.get("mode")
    if condition not in RUNNABLE_CONDITIONS:
        raise ValueError(f"unexpected Phase-5 condition: {condition!r}")
    if expected_condition is not None and condition != expected_condition:
        raise ValueError(f"unexpected Phase-5 condition: {condition!r}")
    missing = REQUIRED_RESULT_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Phase-5 result is missing fields: {sorted(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Phase-5 result schema version")
    if record.get("data_source") not in ("nq", "hotpotqa"):
        raise ValueError("Phase-5 result has an unsupported data source")
    if not str(record.get("uid", "")).startswith(f"{record['data_source']}:"):
        raise ValueError("Phase-5 result UID is not source-aware")
    if not isinstance(record.get("ground_truth"), list) or not record["ground_truth"]:
        raise ValueError("Phase-5 ground truth must be a nonempty list")
    if record.get("prediction") is not None and not isinstance(
        record["prediction"], str
    ):
        raise ValueError("Phase-5 prediction must be text or null")
    run_config = record.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Phase-5 run_config must be a mapping")
    if record.get("run_fingerprint") != run_config_fingerprint(run_config):
        raise ValueError("Phase-5 run_fingerprint does not match run_config")
    if run_config.get("mode") != condition or record["observation_policy"] != condition:
        raise ValueError("Phase-5 condition fields are inconsistent")
    if run_config.get("model_path") != record.get("model_path"):
        raise ValueError("Phase-5 model binding is inconsistent")
    limits = CONDITION_LIMITS[condition]
    for key, expected in {
        "seed": SEED,
        "greedy": True,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_turns": MAX_TURNS,
        "retriever_topk": RETRIEVER_TOPK,
        **limits,
    }.items():
        if run_config.get(key) != expected:
            raise ValueError(f"Phase-5 run_config has unexpected {key}")
    if not run_config.get("eval_sha256") or not run_config.get("eval_manifest_sha256"):
        raise ValueError("Phase-5 result is not bound to evaluation hashes")
    if not run_config.get("baseline_sha256"):
        raise ValueError("Phase-5 result is not bound to the immutable baseline")
    if run_config.get("training_max_obs_length") != 256:
        raise ValueError("Phase-5 result has an unexpected training observation cap")
    if run_config.get("inference_context_distribution_shift") != (
        condition == "raw_512"
    ):
        raise ValueError("Phase-5 distribution-shift declaration is inconsistent")
    compressor_config = run_config.get("compressor")
    if condition == "compressed_256":
        if (
            not isinstance(compressor_config, dict)
            or compressor_config.get("version") != "phase5-extractive-v1"
            or compressor_config.get("wrapped_token_budget") != 256
            or not compressor_config.get("source_sha256")
        ):
            raise ValueError("compressed_256 is not bound to its compressor policy")
    elif compressor_config is not None:
        raise ValueError("raw_512 must not declare a compressor policy")
    if record.get("exact_match") not in (0, 1):
        raise ValueError("Phase-5 exact_match must be binary")
    for key in (
        "end_to_end_latency_s",
        "generation_latency_s",
        "retrieval_latency_s",
        "evidence_compression_latency_s",
    ):
        _non_negative_number(record.get(key), key)
    count_keys = (
        "number_of_actions",
        "number_of_valid_actions",
        "number_of_valid_searches",
        "number_of_successful_retrievals",
        "search_retrieval_failure_count",
        "observation_truncation_count",
        "post_policy_truncation_count",
    )
    for key in count_keys:
        _non_negative_number(record.get(key), key, integer=True)
    retrieval_count = record["number_of_successful_retrievals"]
    aligned_keys = (
        "retrieved_observation_lengths_before_truncation",
        "retained_observation_lengths_after_truncation",
        "observation_excess_tokens",
        "raw_retrieved_observation_tokens",
        "policy_output_observation_tokens",
        "retained_observation_tokens",
        "policy_compression_ratio",
        "documents_returned",
        "documents_represented",
        "sentences_considered",
        "sentences_selected",
        "selected_document_ranks",
        "selected_sentence_identifiers",
    )
    for key in aligned_keys:
        _aligned_list(record, key, retrieval_count)
    for key in ("documents_returned", "documents_represented", "sentences_considered", "sentences_selected"):
        for value in record[key]:
            _non_negative_number(value, key, integer=True)
    if any(
        represented > returned
        for represented, returned in zip(
            record["documents_represented"], record["documents_returned"]
        )
    ):
        raise ValueError("represented documents cannot exceed returned documents")
    if not all(
        isinstance(ranks, list) and all(
            isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
            for rank in ranks
        )
        for ranks in record["selected_document_ranks"]
    ):
        raise ValueError("selected document ranks must be positive-integer lists")
    if not all(
        isinstance(identifiers, list) and all(
            isinstance(identifier, str) and identifier
            for identifier in identifiers
        )
        for identifiers in record["selected_sentence_identifiers"]
    ):
        raise ValueError("selected sentence identifiers must be nonempty-text lists")
    for returned, represented, selected, ranks, identifiers in zip(
        record["documents_returned"],
        record["documents_represented"],
        record["sentences_selected"],
        record["selected_document_ranks"],
        record["selected_sentence_identifiers"],
    ):
        if returned != RETRIEVER_TOPK:
            raise ValueError("successful retrievals must contain exactly top-k documents")
        if len(ranks) != represented or len(set(ranks)) != len(ranks):
            raise ValueError("selected document ranks must match documents represented")
        if len(identifiers) != selected:
            raise ValueError(
                "selected sentence identifiers must match sentences selected"
            )
    policy = record["policy_output_observation_tokens"]
    retained = record["retained_observation_tokens"]
    raw = record["raw_retrieved_observation_tokens"]
    if record["retrieved_observation_lengths_before_truncation"] != policy:
        raise ValueError("manager and policy-output token telemetry disagree")
    if record["retained_observation_lengths_after_truncation"] != retained:
        raise ValueError("manager and retained token telemetry disagree")
    expected_excess = [
        max(length - limits["max_obs_length"], 0) for length in policy
    ]
    if record["observation_excess_tokens"] != expected_excess:
        raise ValueError("manager observation-excess telemetry is inconsistent")
    for index, (raw_count, policy_count, retained_count, ratio) in enumerate(
        zip(raw, policy, retained, record["policy_compression_ratio"])
    ):
        for label, value in (
            ("raw", raw_count), ("policy", policy_count), ("retained", retained_count)
        ):
            _non_negative_number(value, f"{label} token count {index}", integer=True)
        if retained_count > policy_count:
            raise ValueError("Phase-5 context token telemetry is inconsistent")
        expected_ratio = policy_count / raw_count if raw_count else 1.0
        if not math.isclose(float(ratio), expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Phase-5 policy compression ratio is inconsistent")
        if condition == "raw_512" and (
            raw_count != policy_count or not math.isclose(float(ratio), 1.0)
        ):
            raise ValueError("raw_512 must not compress Retriever observations")
        if condition == "compressed_256" and policy_count > 256:
            raise ValueError("compressed_256 wrapped observation exceeds 256 tokens")
    expected_post_policy_truncations = sum(
        kept < produced for kept, produced in zip(retained, policy)
    )
    if record["post_policy_truncation_count"] != expected_post_policy_truncations:
        raise ValueError("post-policy truncation telemetry is inconsistent")
    if record["observation_truncation_count"] != expected_post_policy_truncations:
        raise ValueError("manager observation-truncation telemetry is inconsistent")
    if record["trajectory_had_observation_truncation"] != (
        expected_post_policy_truncations > 0
    ):
        raise ValueError("trajectory observation-truncation flag is inconsistent")
    if record["number_of_valid_actions"] > record["number_of_actions"]:
        raise ValueError("valid actions cannot exceed actions")
    if record["number_of_valid_searches"] > record["number_of_valid_actions"]:
        raise ValueError("valid searches cannot exceed valid actions")
    if retrieval_count > record["number_of_valid_searches"]:
        raise ValueError("successful retrievals cannot exceed valid searches")
    if condition == "raw_512" and record["evidence_compression_latency_s"] != 0:
        raise ValueError("raw_512 must report zero compression latency")
    if condition == "raw_512" and (
        any(record["sentences_considered"])
        or any(record["sentences_selected"])
        or any(record["selected_sentence_identifiers"])
        or any(
            returned != represented
            for returned, represented in zip(
                record["documents_returned"], record["documents_represented"]
            )
        )
        or record["zero_overlap_fallback_used"]
        or record["partial_sentence_fallback_used"]
    ):
        raise ValueError("raw_512 must not report evidence-compression activity")
    if not isinstance(record.get("finished"), bool):
        raise ValueError("finished must be boolean")
    for key in ("zero_overlap_fallback_used", "partial_sentence_fallback_used"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"{key} must be boolean")
    if record["search_retrieval_failure_count"]:
        if record["exact_match"] != 0 or record.get("prediction") is not None:
            raise ValueError("Retriever failure rows must not claim a prediction")
        if not record.get("evaluation_error"):
            raise ValueError("Retriever failure rows must record evaluation_error")
    return record


def read_phase5_result_file(path, condition):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from error
        validate_phase5_result(record, condition)
        if record["uid"] in seen:
            raise ValueError(f"duplicate UID in {path}: {record['uid']}")
        seen.add(record["uid"])
        records.append(record)
    return records


def _append_result(path, record):
    validate_phase5_result(record, record["mode"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_results(path, examples, records_by_uid):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    records_by_uid[example.uid], ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
    temporary.replace(path)


def run_condition_with_resume(
    examples,
    result_path,
    condition,
    evaluate_one,
    overwrite=False,
    expected_run_fingerprint=None,
    immutable_baseline_path=None,
):
    """Persist only Phase-5 files; the Phase-4 baseline is never a write target."""

    if condition not in RUNNABLE_CONDITIONS:
        raise ValueError("raw_256_baseline is validation-only and cannot be run")
    result_path = Path(result_path).expanduser().resolve()
    if (
        immutable_baseline_path is not None
        and result_path == Path(immutable_baseline_path).expanduser().resolve()
    ):
        raise ValueError("Phase-5 output path must not equal the immutable baseline path")
    if overwrite and result_path.exists():
        result_path.unlink()
    pending, records_by_uid = resume_plan(
        examples,
        read_phase5_result_file(result_path, condition),
        expected_run_fingerprint=expected_run_fingerprint,
    )
    for example in pending:
        record = evaluate_one(example)
        if record.get("uid") != example.uid or record.get("mode") != condition:
            raise ValueError("Phase-5 evaluator returned the wrong UID or condition")
        _append_result(result_path, record)
        records_by_uid[example.uid] = record
        print(
            f"[{condition}] persisted {len(records_by_uid)}/{len(examples)}: "
            f"{example.uid}"
        )
    if len(records_by_uid) != len(examples):
        raise RuntimeError("Phase-5 condition ended without all expected results")
    _rewrite_results(result_path, examples, records_by_uid)
    return [records_by_uid[example.uid] for example in examples]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--search-model", default=DEFAULT_SEARCH_MODEL)
    parser.add_argument("--retriever-url", default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("Phase-5 seed must remain 42")
    if args.gpu_memory_utilization != GPU_MEMORY_UTILIZATION:
        raise ValueError("Phase-5 vLLM GPU utilization must remain 0.20")
    examples, manifest = load_eval_examples(args.eval_data, args.eval_manifest)
    _baseline_records, baseline_audit = validate_baseline_artifact(
        args.baseline_results,
        examples,
        manifest,
        args.eval_manifest,
        expected_model=args.search_model,
    )
    if args.condition == "raw_256_baseline":
        print(json.dumps(baseline_audit, indent=2, sort_keys=True))
        print("[raw_256_baseline] immutable Phase-4 artifact validated")
        return

    run_config = build_condition_run_config(
        args.condition,
        args.search_model,
        args.retriever_url,
        manifest,
        Path(args.eval_manifest).resolve(),
        baseline_audit,
        attention_backend=os.environ.get("VLLM_ATTENTION_BACKEND", ""),
    )
    run_fingerprint = run_config_fingerprint(run_config)
    baseline_path = Path(args.baseline_results).expanduser().resolve()
    output_dir = validate_results_directory_isolation(
        args.output_dir, baseline_path
    )
    result_path = output_dir / f"{args.condition}.jsonl"
    if result_path == baseline_path:
        raise ValueError("Phase-5 output path must not equal the immutable baseline path")
    existing = (
        [] if args.overwrite else read_phase5_result_file(result_path, args.condition)
    )
    pending, _completed = resume_plan(
        examples, existing, expected_run_fingerprint=run_fingerprint
    )
    if not pending:
        print(f"[{args.condition}] already complete: {result_path}")
        return

    limits = CONDITION_LIMITS[args.condition]
    print(json.dumps({
        "condition": args.condition,
        "model": args.search_model,
        "baseline": baseline_audit,
        "eval_data": str(Path(args.eval_data).resolve()),
        "remaining_examples": len(pending),
        "greedy": True,
        "seed": SEED,
        "retriever_topk": RETRIEVER_TOPK,
        **limits,
        "result_path": str(result_path),
        "run_fingerprint": run_fingerprint,
    }, indent=2, sort_keys=True))
    generator = VLLMGenerator(
        model_path=args.search_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=limits["max_model_len"],
        seed=args.seed,
        max_response_length=MAX_RESPONSE_LENGTH,
    )
    compressor = None
    if args.condition == "compressed_256":
        from experiments.phase5_observation_context.evidence_compressor import (
            EvidenceCompressor,
        )

        compressor = EvidenceCompressor(
            generator.tokenizer, max_observation_tokens=256
        )

    def evaluate_one(example):
        policy = ObservationPolicyManager(
            args.condition, generator.tokenizer, compressor=compressor
        )
        search_result = evaluate_search_agent(
            example,
            generator.tokenizer,
            generator,
            "search_rl",
            args.search_model,
            args.retriever_url,
            RETRIEVER_TOPK,
            MAX_TURNS,
            MAX_START_LENGTH,
            MAX_RESPONSE_LENGTH,
            limits["max_obs_length"],
            limits["max_prompt_length"],
            run_config,
            configure_manager=policy.configure_manager,
        )
        return phase5_result_from_search_result(
            search_result, args.condition, run_config, policy.events
        )

    run_condition_with_resume(
        examples,
        result_path,
        args.condition,
        evaluate_one,
        overwrite=args.overwrite,
        expected_run_fingerprint=run_fingerprint,
        immutable_baseline_path=baseline_path,
    )
    print(f"[{args.condition}] complete: {result_path}")


if __name__ == "__main__":
    main()
