#!/usr/bin/env python3
"""Summarize the contemporary Phase-6 exact versus IVF-PQ Agent comparison."""

import argparse
import hashlib
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
    DEFAULT_SEARCH_MODEL,
    load_eval_examples,
)
from experiments.phase4_benchmark.summarize_results import (
    latency_summary,
    quality_summary,
    search_agent_metrics,
)
from experiments.phase5_observation_context.run_context_ablation import (
    read_phase5_result_file,
)
from experiments.phase5_observation_context.summarize_results import context_summary
from experiments.phase6_retriever_serving.run_agent_retriever_ablation import (
    CONDITIONS,
    MAX_MODEL_LEN,
    MAX_OBS_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_RESPONSE_LENGTH,
    MAX_START_LENGTH,
    MAX_TURNS,
    RETRIEVER_TOPK,
    SEED,
    STAGE_TIMING_FIELDS,
    load_selected_candidate,
    read_phase6_result_file,
    validate_output_isolation,
    validate_phase5_compressed_baseline,
)
from experiments.phase6_retriever_serving.benchmark_server import (
    validate_complete_report,
)


SCHEMA_VERSION = 1
BASELINE_MODE = "phase5_compressed_256_baseline"
PRIMARY_COMPARISON = (
    "ivfpq_selected_vs_flat_exact_replay",
    "ivfpq_selected",
    "flat_exact_replay",
)
DRIFT_COMPARISON = (
    "flat_exact_replay_vs_phase5_compressed_256_baseline",
    "flat_exact_replay",
    BASELINE_MODE,
)
SERVING_EVIDENCE_CHECKS = (
    "calibration_queries_sha_validated",
    "calibration_manifest_sha_validated",
    "calibration_no_heldout_overlap_validated",
    "calibration_answer_targets_excluded",
    "ivfpq_build_manifest_sha_validated",
    "ivfpq_build_source_config_output_validated",
    "index_benchmark_sha_validated",
    "index_benchmark_candidate_binding_validated",
    "server_benchmark_stable_sha256_recorded",
    "server_benchmark_complete",
    "server_benchmark_selected_candidate_sha_validated",
)
HTTP_AGENT_IDENTITY_FIELDS = (
    "corpus_fingerprint",
    "retrieval_encode_batch_size",
)


def _mean(values):
    return statistics.fmean(values) if values else None


def _stable_json_file(path, label):
    """Read one JSON artifact while binding the exact bytes summarized."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    before_sha256 = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    after_sha256 = sha256_file(path)
    if after_sha256 != before_sha256:
        raise RuntimeError(f"{label} changed while it was being summarized")
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, {
        "path": str(path),
        "sha256": before_sha256,
    }


def _load_bound_json(binding, label):
    if not isinstance(binding, dict):
        raise ValueError(f"selected candidate is missing the {label} binding")
    expected_sha256 = binding.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"selected candidate has no {label} SHA256")
    payload, audit = _stable_json_file(binding.get("path", ""), label)
    if audit["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 does not match selected-candidate binding")
    return payload, audit


def _load_bound_text(binding, label):
    if not isinstance(binding, dict):
        raise ValueError(f"selected candidate is missing the {label} binding")
    expected_sha256 = binding.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"selected candidate has no {label} SHA256")
    path = Path(binding.get("path", "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    before_sha256 = sha256_file(path)
    text = path.read_text(encoding="utf-8")
    after_sha256 = sha256_file(path)
    if before_sha256 != after_sha256:
        raise RuntimeError(f"{label} changed while it was being summarized")
    if before_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA256 does not match selected-candidate binding")
    return text, {"path": str(path), "sha256": before_sha256}


def _same_resolved_path(left, right):
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _config_fingerprint(config):
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_calibration_artifacts(manifest, query_rows, query_binding):
    overlap = manifest.get("leakage_overlap_audit")
    if (
        not isinstance(overlap, dict)
        or overlap.get("passed") is not True
        or overlap.get("overlapping_uids") != []
    ):
        raise ValueError("calibration manifest does not pass the held-out overlap audit")
    if manifest.get("answer_targets_included") is not False:
        raise ValueError("calibration manifest does not attest answer-free selection")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("calibration manifest has no query-output binding")
    if output.get("sha256") != query_binding["sha256"] or not _same_resolved_path(
        output.get("path", ""), query_binding["path"]
    ):
        raise ValueError("calibration manifest output differs from selected queries")
    if manifest.get("row_count") != len(query_rows) or not query_rows:
        raise ValueError("calibration query row count does not match its manifest")
    if any(not isinstance(row, dict) for row in query_rows):
        raise ValueError("calibration query rows must be JSON objects")
    uids = [row.get("uid") for row in query_rows]
    if (
        any(not isinstance(uid, str) or not uid for uid in uids)
        or len(uids) != len(set(uids))
        or uids != manifest.get("selected_uids")
    ):
        raise ValueError("calibration query UIDs do not match their manifest")
    forbidden = {"answer", "answers", "ground_truth", "target", "reward_model"}
    if any(forbidden.intersection(row) for row in query_rows):
        raise ValueError("calibration queries contain answer-target fields")


def _validate_build_manifest(manifest, bindings):
    source = manifest.get("source")
    output = manifest.get("output")
    config = manifest.get("config")
    if not all(isinstance(value, dict) for value in (source, output, config)):
        raise ValueError("IVF-PQ build manifest lacks source/config/output metadata")
    flat = bindings.get("flat_index")
    ivfpq = bindings.get("ivfpq_index")
    if not isinstance(flat, dict) or not isinstance(ivfpq, dict):
        raise ValueError("selected candidate lacks Flat/IVF-PQ index bindings")
    if source.get("sha256") != flat.get("sha256"):
        raise ValueError("IVF-PQ build source SHA256 differs from selected Flat index")
    if output.get("sha256") != ivfpq.get("sha256"):
        raise ValueError("IVF-PQ build output SHA256 differs from selected IVF-PQ index")
    path_pairs = (
        (source.get("path"), flat.get("path"), "Flat source"),
        (output.get("path"), ivfpq.get("path"), "IVF-PQ output"),
        (config.get("source_index"), flat.get("path"), "build-config Flat source"),
        (config.get("output_index"), ivfpq.get("path"), "build-config IVF-PQ output"),
    )
    for observed, expected, label in path_pairs:
        if not observed or not expected or not _same_resolved_path(observed, expected):
            raise ValueError(f"{label} path differs from selected-candidate binding")
    if manifest.get("config_fingerprint") != _config_fingerprint(config):
        raise ValueError("IVF-PQ build config fingerprint is stale")
    if manifest.get("source_sha256_unchanged") is not True:
        raise ValueError("IVF-PQ build does not attest an unchanged Flat source")
    checks = manifest.get("verification", {}).get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("IVF-PQ build verification checks did not all pass")
    if manifest.get("sequential_original_id_assignment") is not True:
        raise ValueError("IVF-PQ build does not attest preserved original IDs")


def _validate_index_benchmark(benchmark, candidate, bindings):
    if benchmark.get("benchmark_kind") != "phase6_calibration_only_index_search":
        raise ValueError("selected index benchmark has an unexpected benchmark kind")
    benchmark_bindings = benchmark.get("bindings")
    if not isinstance(benchmark_bindings, dict):
        raise ValueError("selected index benchmark has no artifact bindings")
    for key in (
        "calibration_queries",
        "calibration_manifest",
        "query_embeddings",
        "query_embeddings_manifest",
        "flat_index",
        "ivfpq_index",
        "ivfpq_build_manifest",
    ):
        if benchmark_bindings.get(key) != bindings.get(key):
            raise ValueError(f"index benchmark has a stale {key} binding")
    selection = benchmark.get("candidate_selection")
    if not isinstance(selection, dict):
        raise ValueError("index benchmark has no candidate-selection evidence")
    for key in (
        "candidate_selection_passed",
        "production_ready_for_agent_evaluation",
        "minimum_recall_at_3",
        "selected_candidate",
        "selection_uses_qa_ground_truth",
    ):
        if selection.get(key) != candidate.get(key):
            raise ValueError(f"selected candidate disagrees with index benchmark for {key}")
    if (
        benchmark.get("selection_uses_qa_ground_truth") is not False
        or benchmark.get("query_embeddings_encoded_once") is not True
        or benchmark.get("topk") != 3
    ):
        raise ValueError("index benchmark violates calibration-only benchmark semantics")


def _batch_evidence(configurations):
    fields = (
        "configured_batch_size",
        "warmup_request_count",
        "warmups_excluded",
        "request_count",
        "successful_request_count",
        "error_count",
        "query_count",
        "elapsed_wall_s",
        "queries_per_second",
        "batches_per_second",
        "client_latency",
        "server_stage_latency",
        "cache",
    )
    return {
        str(batch_size): {
            key: configuration.get(key)
            for key in fields
            if key in configuration
        }
        for batch_size, configuration in configurations.items()
    }


def _server_target_evidence(target):
    return {
        "role": target.get("role"),
        "workload": target.get("workload"),
        "identity": target.get("identity"),
        "batch_configurations": _batch_evidence(
            target.get("batch_configurations", {})
        ),
    }


def _selected_http_agent_identities(server_benchmark):
    """Extract the completed cold Flat/IVF-PQ identities used for Agent parity."""

    cold_targets = server_benchmark.get("workloads", {}).get("cold", {})
    identities = {}
    for condition, role in (
        ("flat_exact_replay", "flat_exact"),
        ("ivfpq_selected", "ivfpq_selected"),
    ):
        matches = [
            target
            for target in cold_targets.values()
            if target.get("role") == role
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("identity"), dict):
            raise ValueError(
                f"completed HTTP benchmark must have exactly one cold {role} identity"
            )
        identity = matches[0]["identity"]
        corpus_fingerprint = identity.get("corpus_fingerprint")
        encode_batch_size = identity.get("retrieval_encode_batch_size")
        if not isinstance(corpus_fingerprint, str) or not corpus_fingerprint:
            raise ValueError(
                f"completed HTTP benchmark {role} has no corpus fingerprint"
            )
        if (
            isinstance(encode_batch_size, bool)
            or not isinstance(encode_batch_size, int)
            or encode_batch_size <= 0
        ):
            raise ValueError(
                f"completed HTTP benchmark {role} has an invalid retrieval encode "
                "batch size"
            )
        identities[condition] = {
            field: identity[field] for field in HTTP_AGENT_IDENTITY_FIELDS
        }

    for field in HTTP_AGENT_IDENTITY_FIELDS:
        if identities["flat_exact_replay"][field] != identities[
            "ivfpq_selected"
        ][field]:
            raise ValueError(
                f"completed HTTP Flat and IVF-PQ identities differ for {field}"
            )
    return identities


def _targets_with_roles(server_benchmark, workload, roles):
    targets = server_benchmark.get("workloads", {}).get(workload, {})
    return {
        name: _server_target_evidence(target)
        for name, target in targets.items()
        if target.get("role") in roles
    }


def _serving_evidence_sections(
    manifest,
    index_benchmark,
    server_benchmark,
    calibration_manifest,
    artifact_bindings,
):
    selection = index_benchmark["candidate_selection"]
    flat_results = [
        {
            key: value
            for key, value in record.items()
            if key not in ("result_ids", "result_scores")
        }
        for record in index_benchmark.get("flat_results", [])
    ]
    exact = {
        "classification": "exact_semantic_serving_optimization",
        "definition": (
            "Exact Flat serving evidence isolates CPU thread, native batch, and "
            "opt-in cache behavior without changing retrieved IDs."
        ),
        "offline_flat_results": flat_results,
        "http_thread_batch_sweep": server_benchmark.get(
            "exact_thread_batch_sweep"
        ),
        "cold_targets": _targets_with_roles(
            server_benchmark, "cold", {"exact_sweep", "flat_exact"}
        ),
        "warm_cache_targets": _targets_with_roles(
            server_benchmark, "warm", {"warm_flat"}
        ),
        "server_benchmark_artifact": artifact_bindings["server_benchmark"],
    }
    ann = {
        "classification": "ann_recall_latency_memory_tradeoff",
        "definition": (
            "IVF-PQ evidence combines verified build provenance, calibration-only "
            "Recall@3 selection, HTTP latency/cache behavior, and resource size."
        ),
        "build": {
            "config": manifest.get("config"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "source": manifest.get("source"),
            "output": manifest.get("output"),
            "compression_ratio_source_over_output": manifest.get(
                "compression_ratio_source_over_output"
            ),
            "build_time_s": manifest.get("build_time_s"),
            "preflight": manifest.get("preflight"),
            "verification": manifest.get("verification"),
            "source_sha256_unchanged": manifest.get("source_sha256_unchanged"),
            "sequential_original_id_assignment": manifest.get(
                "sequential_original_id_assignment"
            ),
        },
        "offline_calibration": {
            "dataset_audit": {
                "row_count": calibration_manifest.get("row_count"),
                "source_counts": calibration_manifest.get("source_counts"),
                "leakage_overlap_audit": calibration_manifest.get(
                    "leakage_overlap_audit"
                ),
                "answer_targets_included": calibration_manifest.get(
                    "answer_targets_included"
                ),
            },
            "query_count": index_benchmark.get("query_count"),
            "topk": index_benchmark.get("topk"),
            "index_geometry": index_benchmark.get("index_geometry"),
            "faiss_thread_counts": index_benchmark.get("faiss_thread_counts"),
            "ivfpq_nprobes": index_benchmark.get("ivfpq_nprobes"),
            "warmup_query_count": index_benchmark.get("warmup_query_count"),
            "warmups_excluded": index_benchmark.get("warmups_excluded"),
            "candidate_selection": selection,
        },
        "http": {
            "cold_target": _targets_with_roles(
                server_benchmark, "cold", {"ivfpq_selected"}
            ),
            "warm_cache_target": _targets_with_roles(
                server_benchmark, "warm", {"warm_ivfpq"}
            ),
            "exact_vs_ivfpq_agreement": server_benchmark.get("comparisons"),
        },
        "artifacts": artifact_bindings,
    }
    return exact, ann


def load_serving_evidence(output_dir, selected_candidate_path, candidate=None):
    """Load and content-bind all non-Agent evidence required by Phase 6."""

    candidate_payload, candidate_file = _stable_json_file(
        selected_candidate_path, "selected candidate"
    )
    if candidate is not None and candidate_payload != candidate:
        raise RuntimeError("selected candidate changed between validation and summary")
    bindings = candidate_payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("selected candidate is missing artifact bindings")
    queries_text, queries_file = _load_bound_text(
        bindings.get("calibration_queries"), "calibration queries"
    )
    try:
        query_rows = [
            json.loads(line)
            for line in queries_text.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("calibration queries contain invalid JSONL") from error
    calibration_manifest, calibration_manifest_file = _load_bound_json(
        bindings.get("calibration_manifest"), "calibration manifest"
    )
    _validate_calibration_artifacts(
        calibration_manifest, query_rows, queries_file
    )
    manifest, manifest_file = _load_bound_json(
        bindings.get("ivfpq_build_manifest"), "IVF-PQ build manifest"
    )
    index_benchmark, index_file = _load_bound_json(
        bindings.get("index_benchmark"), "index benchmark"
    )
    _validate_build_manifest(manifest, bindings)
    _validate_index_benchmark(index_benchmark, candidate_payload, bindings)

    server_path = Path(output_dir).expanduser().resolve() / "server_benchmark.json"
    server_benchmark, server_file = _stable_json_file(
        server_path, "server benchmark"
    )
    inputs = server_benchmark.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("server benchmark has no input bindings")
    if inputs.get("selected_candidate_sha256") != candidate_file["sha256"]:
        raise ValueError("server benchmark is bound to a different selected candidate")
    if not _same_resolved_path(
        inputs.get("selected_candidate_path", ""), candidate_file["path"]
    ):
        raise ValueError("server benchmark selected-candidate path is stale")
    if inputs.get("queries_sha256") != bindings.get(
        "calibration_queries", {}
    ).get("sha256"):
        raise ValueError("server benchmark uses different calibration queries")
    if inputs.get("calibration_manifest_sha256") != bindings.get(
        "calibration_manifest", {}
    ).get("sha256"):
        raise ValueError("server benchmark uses a different calibration manifest")
    completion = validate_complete_report(server_benchmark)
    if server_benchmark.get("completion_validation") != completion:
        raise ValueError("server benchmark completion validation is missing or stale")
    selected_http_agent_identities = _selected_http_agent_identities(
        server_benchmark
    )

    artifact_bindings = {
        "selected_candidate": candidate_file,
        "calibration_queries": queries_file,
        "calibration_manifest": calibration_manifest_file,
        "ivfpq_build_manifest": manifest_file,
        "index_benchmark": index_file,
        "server_benchmark": server_file,
    }
    exact, ann = _serving_evidence_sections(
        manifest,
        index_benchmark,
        server_benchmark,
        calibration_manifest,
        artifact_bindings,
    )
    checks = {key: True for key in SERVING_EVIDENCE_CHECKS}
    return {
        "validated": all(checks.values()),
        "checks": checks,
        "bindings": artifact_bindings,
        "selected_http_agent_identities": selected_http_agent_identities,
        "exact_semantic_serving": exact,
        "ann_tradeoff": ann,
    }


def adapt_phase5_baseline(records):
    """Relabel Phase-5 rows in memory only for the strict generic paired join."""

    adapted = []
    for record in records:
        view = dict(record)
        view["mode"] = BASELINE_MODE
        adapted.append(view)
    return adapted


def _single_config(records, label):
    configs = {
        json.dumps(record["run_config"], sort_keys=True): record["run_config"]
        for record in records
    }
    fingerprints = {record["run_fingerprint"] for record in records}
    if len(configs) != 1 or len(fingerprints) != 1:
        raise ValueError(f"{label} mixes run configurations")
    return next(iter(configs.values())), next(iter(fingerprints))


def validate_pair_contract(result_sets, candidate_audit):
    """Require identical Agent semantics and only the intended Retriever delta."""

    flat, _ = _single_config(result_sets["flat_exact_replay"], "Flat replay")
    ann, _ = _single_config(result_sets["ivfpq_selected"], "IVF-PQ")
    same_keys = (
        "prompt_contract",
        "experiment_contract",
        "eval_sha256",
        "eval_manifest_sha256",
        "phase5_baseline_path",
        "phase5_baseline_sha256",
        "phase5_baseline_run_fingerprint",
        "selected_candidate_sha256",
        "candidate_selection_passed",
        "candidate_production_ready_for_agent_evaluation",
        "candidate_override_used",
        "model_path",
        "model_checkpoint_fingerprint",
        "seed",
        "greedy",
        "dtype",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "attention_backend",
        "max_start_length",
        "max_response_length",
        "max_obs_length",
        "max_prompt_length",
        "max_model_len",
        "max_turns",
        "retriever_topk",
        "retriever_return_metrics",
        "retriever_cache_enabled",
        "observation_policy",
    )
    differences = [key for key in same_keys if flat.get(key) != ann.get(key)]
    if differences:
        raise ValueError(f"Agent contracts differ outside Retriever backend: {differences}")
    if flat.get("selected_candidate_sha256") != candidate_audit["sha256"]:
        raise ValueError("Agent results are bound to a different selected candidate")
    flat_context = dict(flat["phase5_context_run_config"])
    ann_context = dict(ann["phase5_context_run_config"])
    flat_context.pop("retriever_url", None)
    ann_context.pop("retriever_url", None)
    if flat_context != ann_context:
        raise ValueError("Flat and IVF-PQ runs do not share Phase-5 compressor semantics")
    flat_server = flat["retriever_server_contract"]
    ann_server = ann["retriever_server_contract"]
    if flat_server.get("index_backend") != "flat":
        raise ValueError("flat_exact_replay is not bound to a Flat index")
    if ann_server.get("index_backend") != "ivfpq":
        raise ValueError("ivfpq_selected is not bound to an IVF-PQ index")
    if flat_server.get("faiss_thread_count") != ann_server.get("faiss_thread_count"):
        raise ValueError("Agent servers use different FAISS thread counts")
    if flat_server.get("faiss_thread_count") != candidate_audit.get(
        "faiss_thread_count"
    ):
        raise ValueError("Agent server threads differ from selected candidate")
    if ann_server.get("nprobe") != candidate_audit.get("nprobe"):
        raise ValueError("Agent IVF-PQ nprobe differs from selected candidate")
    if flat_server.get("index_fingerprint") != candidate_audit.get(
        "flat_index_sha256"
    ):
        raise ValueError("Agent Flat index differs from selected-candidate binding")
    if ann_server.get("index_fingerprint") != candidate_audit.get(
        "ivfpq_index_sha256"
    ):
        raise ValueError("Agent IVF-PQ index differs from selected-candidate binding")
    for key in (
        "index_ntotal",
        "index_dimension",
        "metric_type",
        "metric_type_code",
        "corpus_row_count",
        "model_path",
        "model_fingerprint",
        "corpus_fingerprint",
        "retrieval_encode_batch_size",
        "result_cache_capacity",
        "embedding_cache_capacity",
    ):
        if flat_server.get(key) != ann_server.get(key):
            raise ValueError(f"Agent Retriever contracts differ for {key}")
    if flat_server.get("model_fingerprint") != candidate_audit.get(
        "model_fingerprint"
    ):
        raise ValueError("Agent E5 model differs from calibration embedding binding")
    if flat_server.get("cache_enabled") is not False or ann_server.get(
        "cache_enabled"
    ) is not False:
        raise ValueError("Agent comparison must keep both caches disabled")
    return {"flat": flat, "ivfpq": ann}


def agent_http_identity_crosscheck(pair_contract, serving_evidence):
    """Bind downstream Agent servers to the completed selected HTTP pair."""

    http_identities = serving_evidence.get("selected_http_agent_identities")
    if not isinstance(http_identities, dict):
        http_identities = {}
    values = {}
    checks = {}
    agent_flat = pair_contract["flat"]["retriever_server_contract"]
    agent_ann = pair_contract["ivfpq"]["retriever_server_contract"]
    for field in HTTP_AGENT_IDENTITY_FIELDS:
        field_values = {
            "agent_flat_exact_replay": agent_flat.get(field),
            "agent_ivfpq_selected": agent_ann.get(field),
            "http_flat_exact_replay": http_identities.get(
                "flat_exact_replay", {}
            ).get(field),
            "http_ivfpq_selected": http_identities.get(
                "ivfpq_selected", {}
            ).get(field),
        }
        values[field] = field_values
        if field == "corpus_fingerprint":
            valid = all(
                isinstance(value, str) and bool(value)
                for value in field_values.values()
            )
        else:
            valid = all(
                not isinstance(value, bool)
                and isinstance(value, int)
                and value > 0
                for value in field_values.values()
            )
        checks[f"agent_http_{field}_match"] = valid and len(
            set(field_values.values())
        ) == 1
    return {"checks": checks, "values": values}


def load_server_audit(results_dir, condition, records):
    path = Path(results_dir) / f"{condition}.server_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config, fingerprint = _single_config(records, condition)
    result_path = Path(results_dir) / f"{condition}.jsonl"
    if payload.get("condition") != condition:
        raise ValueError(f"{condition} server audit declares another condition")
    if payload.get("result_sha256") != sha256_file(result_path):
        raise ValueError(f"{condition} server audit has a stale result SHA256")
    if payload.get("run_fingerprint") != fingerprint:
        raise ValueError(f"{condition} server audit has a stale run fingerprint")
    for label in ("starting_health", "ending_health"):
        health = payload.get(label)
        if not isinstance(health, dict):
            raise ValueError(f"{condition} server audit is missing {label}")
        for key, expected in config["retriever_server_contract"].items():
            observed = health.get(key)
            if key == "index_file_size_bytes":
                index_path = Path(health.get("index_path", "")).expanduser().resolve()
                observed = index_path.stat().st_size if index_path.is_file() else None
            if observed != expected:
                raise ValueError(f"{condition} {label} changed stable field {key}")
    return payload


def server_stage_summary(records):
    metrics = [
        metric
        for record in records
        for metric in record["retriever_request_metrics"]
    ]
    return {
        "request_count": len(metrics),
        "query_count": sum(metric["query_count"] for metric in metrics),
        "stage_latency": {
            field: latency_summary([metric[field] for metric in metrics])
            for field in STAGE_TIMING_FIELDS
        },
        "cache_hit_count": sum(metric["cache_hit_count"] for metric in metrics),
        "cache_miss_count": sum(metric["cache_miss_count"] for metric in metrics),
        "embedding_cache_hit_count": sum(
            metric["embedding_cache_hit_count"] for metric in metrics
        ),
        "embedding_cache_miss_count": sum(
            metric["embedding_cache_miss_count"] for metric in metrics
        ),
    }


def summarize_condition(records, server_audit):
    ending_health = server_audit["ending_health"]
    run_config, _ = _single_config(records, records[0]["mode"])
    return {
        "quality": quality_summary(records),
        "latency": {
            "end_to_end": latency_summary(
                [record["end_to_end_latency_s"] for record in records]
            ),
            "generation": latency_summary(
                [record["generation_latency_s"] for record in records]
            ),
            "retrieval_client": latency_summary(
                [record["retrieval_latency_s"] for record in records]
            ),
            "compression": latency_summary(
                [record["evidence_compression_latency_s"] for record in records]
            ),
        },
        "server": server_stage_summary(records),
        "agent": search_agent_metrics(records),
        "context": context_summary(records),
        "resource": {
            "process_rss_bytes": ending_health.get("process_rss_bytes"),
            "index_path": run_config["retriever_server_contract"]["index_path"],
            "index_file_size_bytes": run_config["retriever_server_contract"][
                "index_file_size_bytes"
            ],
            "index_backend": run_config["retriever_server_contract"][
                "index_backend"
            ],
            "faiss_thread_count": run_config["retriever_server_contract"][
                "faiss_thread_count"
            ],
            "nprobe": run_config["retriever_server_contract"]["nprobe"],
        },
    }


def retrieval_id_agreement(joined):
    comparable = []
    diverged_query_count = 0
    missing_alignment_count = 0
    for row in joined:
        flat = row["records"]["flat_exact_replay"]
        ann = row["records"]["ivfpq_selected"]
        flat_queries = flat["retriever_queries"]
        ann_queries = ann["retriever_queries"]
        flat_ids = flat["retrieved_document_ids"]
        ann_ids = ann["retrieved_document_ids"]
        shared = min(len(flat_queries), len(ann_queries))
        missing_alignment_count += abs(len(flat_queries) - len(ann_queries))
        for position in range(shared):
            if flat_queries[position] != ann_queries[position]:
                diverged_query_count += 1
                continue
            exact = flat_ids[position]
            approximate = ann_ids[position]
            comparable.append((exact, approximate))
    recall_at_3 = [
        len(set(exact).intersection(approximate)) / RETRIEVER_TOPK
        for exact, approximate in comparable
    ]
    return {
        "definition": (
            "Agreement is computed only at aligned Agent search positions where "
            "the generated query string is identical across conditions."
        ),
        "comparable_identical_query_count": len(comparable),
        "diverged_query_count": diverged_query_count,
        "missing_alignment_count": missing_alignment_count,
        "recall_at_3_mean": _mean(recall_at_3),
        "top1_agreement_rate": _mean([
            exact[0] == approximate[0] for exact, approximate in comparable
        ]),
        "full_top3_set_agreement_rate": _mean([
            set(exact) == set(approximate) for exact, approximate in comparable
        ]),
    }


def descriptive_delta(summaries, mode_a, mode_b):
    a = summaries[mode_a]
    b = summaries[mode_b]

    def delta(path):
        left = a
        right = b
        for key in path:
            if not isinstance(left, dict) or not isinstance(right, dict):
                return None
            if key not in left or key not in right:
                return None
            left = left[key]
            right = right[key]
        return None if left is None or right is None else left - right

    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "orientation": f"{mode_a} minus {mode_b}",
        "overall_em_delta_pp": 100.0 * delta(("quality", "overall_em")),
        "nq_em_delta_pp": 100.0 * delta(("quality", "nq_em")),
        "hotpotqa_em_delta_pp": 100.0 * delta(("quality", "hotpotqa_em")),
        "end_to_end_mean_s_delta": delta(("latency", "end_to_end", "mean_s")),
        "retrieval_client_mean_s_delta": delta(
            ("latency", "retrieval_client", "mean_s")
        ),
        "server_request_mean_s_delta": delta(
            ("server", "stage_latency", "request_total_s", "mean_s")
        ),
        "server_search_mean_s_delta": delta(
            ("server", "stage_latency", "faiss_search_s", "mean_s")
        ),
        "finish_ratio_delta": delta(("agent", "finish_ratio")),
        "valid_action_ratio_delta": delta(("agent", "valid_action_ratio")),
        "valid_search_ratio_delta": delta(("agent", "valid_search_ratio")),
    }


def build_reports(
    result_sets,
    server_audits,
    baseline_audit,
    candidate_audit,
    serving_evidence,
):
    modes = ("flat_exact_replay", "ivfpq_selected", BASELINE_MODE)
    joined = join_result_sets(result_sets, modes)
    pair_contract = validate_pair_contract(result_sets, candidate_audit)
    identity_crosscheck = agent_http_identity_crosscheck(
        pair_contract, serving_evidence
    )
    contemporary = join_result_sets(
        {
            "flat_exact_replay": result_sets["flat_exact_replay"],
            "ivfpq_selected": result_sets["ivfpq_selected"],
        },
        CONDITIONS,
    )
    summaries = {
        condition: summarize_condition(result_sets[condition], server_audits[condition])
        for condition in CONDITIONS
    }
    baseline_records = result_sets[BASELINE_MODE]
    summaries[BASELINE_MODE] = {
        "quality": quality_summary(baseline_records),
        "latency": {
            "end_to_end": latency_summary(
                [record["end_to_end_latency_s"] for record in baseline_records]
            ),
            "generation": latency_summary(
                [record["generation_latency_s"] for record in baseline_records]
            ),
            "retrieval_client": latency_summary(
                [record["retrieval_latency_s"] for record in baseline_records]
            ),
            "compression": latency_summary(
                [record["evidence_compression_latency_s"] for record in baseline_records]
            ),
        },
        "agent": search_agent_metrics(baseline_records),
        "context": context_summary(baseline_records),
    }
    primary = analyze_comparison(
        joined,
        PRIMARY_COMPARISON[0],
        PRIMARY_COMPARISON[1],
        PRIMARY_COMPARISON[2],
        bootstrap_samples=10_000,
        seed=42,
        confidence_level=0.95,
        classification="primary_downstream_product_comparison",
    )
    drift = analyze_comparison(
        joined,
        DRIFT_COMPARISON[0],
        DRIFT_COMPARISON[1],
        DRIFT_COMPARISON[2],
        bootstrap_samples=10_000,
        seed=42,
        confidence_level=0.95,
        classification="immutable_baseline_drift_audit",
    )
    drift_uids = [
        row["uid"]
        for row in joined
        if row["outcomes"]["flat_exact_replay"]
        != row["outcomes"][BASELINE_MODE]
    ]
    source_counts = Counter(row["data_source"] for row in joined)
    failures = {
        condition: sum(
            record["search_retrieval_failure_count"]
            for record in result_sets[condition]
        )
        for condition in CONDITIONS
    }
    evaluation_errors = {
        condition: sum(bool(record.get("evaluation_error")) for record in records)
        for condition, records in result_sets.items()
    }
    checks = {
        "both_contemporary_conditions_have_64_rows": all(
            len(result_sets[condition]) == 64 for condition in CONDITIONS
        ),
        "immutable_phase5_baseline_has_64_rows": len(baseline_records) == 64,
        "source_balance_is_32_nq_32_hotpotqa": source_counts
        == Counter({"nq": 32, "hotpotqa": 32}),
        "identical_uid_and_eval_hash_bindings": len(joined) == 64,
        "phase5_baseline_sha_validated": bool(baseline_audit.get("validated")),
        "no_retriever_failures": not any(failures.values()),
        "no_evaluation_errors": not any(evaluation_errors.values()),
        "both_retrievers_cache_disabled": all(
            not summaries[condition]["server"]["cache_hit_count"]
            for condition in CONDITIONS
        ),
        "candidate_selection_status_explicit": isinstance(
            candidate_audit.get("candidate_selection_passed"), bool
        ),
        "candidate_selection_passed_without_override": (
            candidate_audit.get("candidate_selection_passed") is True
            and candidate_audit.get("production_ready_for_agent_evaluation") is True
            and candidate_audit.get("override_used") is False
        ),
        "flat_replay_quality_matches_immutable_phase5": not drift_uids,
        "required_serving_evidence_validated": serving_evidence.get("validated")
        is True,
        **identity_crosscheck["checks"],
        **{
            key: serving_evidence.get("checks", {}).get(key) is True
            for key in SERVING_EVIDENCE_CHECKS
        },
    }
    engineering_checks = dict(checks)
    engineering_checks.pop("flat_replay_quality_matches_immutable_phase5")
    engineering_checks.pop("candidate_selection_passed_without_override")
    primary_descriptive = descriptive_delta(
        summaries, "ivfpq_selected", "flat_exact_replay"
    )
    baseline_drift_descriptive = descriptive_delta(
        summaries, "flat_exact_replay", BASELINE_MODE
    )
    retrieval_agreement = retrieval_id_agreement(contemporary)
    baseline_drift = {
        "mismatched_outcome_count": len(drift_uids),
        "mismatched_outcome_uids": drift_uids,
        "quality_consistent": not drift_uids,
    }
    downstream_agent_impact = {
        "classification": "downstream_agent_product_impact",
        "definition": (
            "Paired deterministic Search Agent comparison; this evidence is "
            "separate from exact-server optimization and calibration-only ANN metrics."
        ),
        "conditions": summaries,
        "primary_descriptive_comparison": primary_descriptive,
        "baseline_drift_descriptive_comparison": baseline_drift_descriptive,
        "retrieval_id_agreement": retrieval_agreement,
        "baseline_drift": baseline_drift,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "phase6-retriever-serving-agent-ablation",
        "exact_semantic_serving": serving_evidence["exact_semantic_serving"],
        "ann_tradeoff": serving_evidence["ann_tradeoff"],
        "downstream_agent_impact": downstream_agent_impact,
        # Retain these concise top-level views for existing Phase-6 readers.
        "conditions": summaries,
        "primary_descriptive_comparison": primary_descriptive,
        "baseline_drift_descriptive_comparison": baseline_drift_descriptive,
        "retrieval_id_agreement": retrieval_agreement,
        "baseline_drift": baseline_drift,
        "retriever_identity_crosscheck": identity_crosscheck,
        "readiness": {
            "engineering_pass": all(engineering_checks.values()),
            "product_comparison_ready": all(checks.values()),
            "checks": checks,
            "retriever_failure_counts": failures,
            "evaluation_error_counts": evaluation_errors,
            "source_counts": dict(source_counts),
            "warning": None if all(checks.values()) else (
                "Phase-6 readiness checks failed; results remain descriptive and "
                "must not be labeled quality-preserving."
            ),
        },
        "bindings": {
            "phase5_baseline": baseline_audit,
            "selected_candidate": candidate_audit,
            "serving_evidence": serving_evidence["bindings"],
        },
    }
    paired = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
        "primary": primary,
        "baseline_drift": drift,
        "readiness": summary["readiness"],
    }
    return summary, paired


def render_markdown(summary, paired):
    primary = paired["primary"]
    drift = summary["baseline_drift"]
    lines = [
        "# Phase-6 downstream Agent comparison",
        "",
        "No claim is implied unless every readiness check passes. Downstream Agent EM,",
        "not calibration Recall@3 alone, is authoritative for product quality.",
        "",
        "## Exact semantic serving evidence",
        "",
        "Exact Flat thread, native-batch, and opt-in cache measurements are reported",
        "separately from ANN approximation and downstream Agent behavior.",
        "",
        "## ANN recall/latency/memory tradeoff evidence",
        "",
        "The selected IVF-PQ build, calibration Recall@3, HTTP behavior, and artifact",
        "sizes are reported separately and do not establish downstream quality.",
        "",
        "## Downstream Agent impact",
        "",
        "## Primary: IVF-PQ minus contemporary exact Flat",
        "",
        f"- EM delta: {primary['overall']['difference_pp']:.6f} percentage points",
        "- 95% paired bootstrap CI: "
        f"[{primary['overall']['bootstrap_ci_95_pp'][0]:.6f}, "
        f"{primary['overall']['bootstrap_ci_95_pp'][1]:.6f}]",
        "- Exact two-sided McNemar p-value: "
        f"{primary['overall']['mcnemar']['exact_two_sided_p_value']:.12g}",
        "",
        "## Immutable Phase-5 drift audit",
        "",
        f"- Outcome mismatches: {drift['mismatched_outcome_count']}",
        f"- Quality-consistent replay: {drift['quality_consistent']}",
        "",
        "## Readiness",
        "",
        f"- Engineering pass: {summary['readiness']['engineering_pass']}",
        f"- Product comparison ready: {summary['readiness']['product_comparison_ready']}",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase5-baseline", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--search-model", default=DEFAULT_SEARCH_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = validate_output_isolation(args.output_dir, args.phase5_baseline)
    examples, manifest = load_eval_examples(args.eval_data, args.eval_manifest)
    baseline_records, baseline_audit = validate_phase5_compressed_baseline(
        args.phase5_baseline,
        examples,
        manifest,
        args.eval_manifest,
        expected_model=args.search_model,
    )
    candidate, candidate_audit = load_selected_candidate(
        args.selected_candidate, allow_unqualified=True
    )
    serving_evidence = load_serving_evidence(
        output_dir, args.selected_candidate, candidate
    )
    result_sets = {
        condition: read_phase6_result_file(
            output_dir / f"{condition}.jsonl", condition
        )
        for condition in CONDITIONS
    }
    result_sets[BASELINE_MODE] = adapt_phase5_baseline(baseline_records)
    audits = {
        condition: load_server_audit(output_dir, condition, result_sets[condition])
        for condition in CONDITIONS
    }
    summary, paired = build_reports(
        result_sets,
        audits,
        baseline_audit,
        candidate_audit,
        serving_evidence,
    )
    if sha256_file(args.phase5_baseline) != baseline_audit["sha256"]:
        raise RuntimeError("immutable Phase-5 baseline changed during summarization")
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_dir / "paired_statistics.json",
        json.dumps(paired, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_dir / "paired_statistics.md", render_markdown(summary, paired)
    )
    print(json.dumps(summary["readiness"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
