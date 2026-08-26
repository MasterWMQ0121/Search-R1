#!/usr/bin/env python3
"""Run the contemporary Phase-6 exact-Flat versus IVF-PQ Agent ablation."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import MethodType

import requests

from experiments.phase4_benchmark.prepare_eval_data import sha256_file
from experiments.phase4_benchmark.run_benchmark import (
    DEFAULT_SEARCH_MODEL,
    VLLMGenerator,
    evaluate_search_rl,
    load_eval_examples,
    resume_plan,
    run_config_fingerprint,
)
from experiments.phase5_observation_context.evidence_compressor import (
    EvidenceCompressor,
)
from experiments.phase5_observation_context.run_context_ablation import (
    ObservationPolicyManager,
    phase5_result_from_search_result,
    read_phase5_result_file,
    validate_phase5_result,
)
from experiments.phase6_retriever_serving.optimized_retrieval_server import (
    fingerprint_asset,
)


CONDITIONS = ("flat_exact_replay", "ivfpq_selected")
SCHEMA_VERSION = 1
EXPERIMENT_CONTRACT = "phase6-retriever-serving-agent-v1"
MAX_START_LENGTH = 768
MAX_RESPONSE_LENGTH = 128
MAX_OBS_LENGTH = 256
MAX_PROMPT_LENGTH = 1408
MAX_MODEL_LEN = 1536
MAX_TURNS = 2
RETRIEVER_TOPK = 3
SEED = 42
GPU_MEMORY_UTILIZATION = 0.20
STAGE_TIMING_FIELDS = (
    "request_total_s",
    "query_normalization_s",
    "cache_lookup_s",
    "query_encoding_s",
    "faiss_search_s",
    "document_fetch_s",
    "response_format_s",
)
SERVER_CONTRACT_FIELDS = (
    "index_type",
    "index_backend",
    "index_path",
    "index_file_size_bytes",
    "index_ntotal",
    "index_dimension",
    "metric_type",
    "metric_type_code",
    "corpus_row_count",
    "model_path",
    "faiss_thread_count",
    "nprobe",
    "cache_enabled",
    "result_cache_capacity",
    "embedding_cache_capacity",
    "retrieval_encode_batch_size",
    "index_fingerprint",
    "model_fingerprint",
    "corpus_fingerprint",
    "retrieval_config_fingerprint",
)
PHASE6_FIELDS = {
    "schema_version",
    "retriever_queries",
    "retriever_request_ids",
    "retrieved_document_ids",
    "retriever_request_metrics",
}


def fingerprint_model_checkpoint(model_path):
    """Content-bind one local Qwen checkpoint file or directory tree."""

    return fingerprint_asset(Path(model_path).expanduser().resolve())


def verify_runnable_inputs_unchanged(
    model_path,
    expected_model_fingerprint,
    baseline_path,
    expected_baseline_sha256,
    lifecycle,
):
    """Recheck both immutable inputs after a runnable Phase-6 lifecycle."""

    changes = []
    try:
        observed_model_fingerprint = fingerprint_model_checkpoint(model_path)
    except (OSError, ValueError) as error:
        changes.append(f"Qwen checkpoint could not be re-fingerprinted: {error}")
    else:
        if observed_model_fingerprint != expected_model_fingerprint:
            changes.append("Qwen checkpoint content changed")
    try:
        observed_baseline_sha256 = sha256_file(baseline_path)
    except OSError as error:
        changes.append(f"Phase-5 baseline could not be re-hashed: {error}")
    else:
        if observed_baseline_sha256 != expected_baseline_sha256:
            changes.append("immutable Phase-5 baseline changed")
    if changes:
        raise RuntimeError(f"{' and '.join(changes)} during Phase-6 {lifecycle}")


def _json_response(response, label):
    body = getattr(response, "text", "")[:500]
    status = int(getattr(response, "status_code", 0))
    if not 200 <= status < 300:
        raise RuntimeError(f"{label} failed with status={status}, body={body!r}")
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f"{label} returned invalid JSON: {body!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} response must be a JSON object")
    return payload


def _service_endpoint(retriever_url, endpoint):
    retriever_url = str(retriever_url).rstrip("/")
    if not retriever_url.endswith("/retrieve"):
        raise ValueError("Phase-6 Retriever URL must end in /retrieve")
    return retriever_url[: -len("/retrieve")] + endpoint


def read_server_endpoint(retriever_url, endpoint, timeout_s=30.0, session=requests):
    response = session.get(
        _service_endpoint(retriever_url, endpoint), timeout=timeout_s
    )
    return _json_response(response, f"Retriever {endpoint}")


def _finite_nonnegative(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def validate_request_metrics(metrics, query_count, topk, expected_contract=None):
    """Validate one opt-in server metrics object without changing its values."""

    if not isinstance(metrics, dict):
        raise ValueError("Retriever response is missing opt-in metrics")
    request_id = metrics.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("Retriever metrics require a nonempty request_id")
    if metrics.get("query_count") != query_count:
        raise ValueError("Retriever metrics query_count disagrees with request")
    if metrics.get("topk") != topk:
        raise ValueError("Retriever metrics topk disagrees with request")
    for key in STAGE_TIMING_FIELDS:
        _finite_nonnegative(metrics.get(key), f"Retriever metrics {key}")
    for key in (
        "cache_hit_count",
        "cache_miss_count",
        "embedding_cache_hit_count",
        "embedding_cache_miss_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Retriever metrics {key} must be a non-negative integer")
    result_ids = metrics.get("result_ids")
    if not isinstance(result_ids, list) or len(result_ids) != query_count:
        raise ValueError("Retriever result_ids do not match query cardinality")
    if any(
        not isinstance(row, list)
        or len(row) != topk
        or any(
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
            for identifier in row
        )
        for row in result_ids
    ):
        raise ValueError("Retriever result_ids must be non-negative top-k integer rows")
    if expected_contract is not None:
        for key in ("index_backend", "nprobe", "faiss_thread_count"):
            if metrics.get(key) != expected_contract.get(key):
                raise ValueError(f"Retriever metrics disagree with health for {key}")
        if expected_contract.get("cache_enabled"):
            raise ValueError("Phase-6 Agent comparison requires cache-disabled servers")
        cache_counters = (
            metrics["cache_hit_count"],
            metrics["cache_miss_count"],
            metrics["embedding_cache_hit_count"],
            metrics["embedding_cache_miss_count"],
        )
        if any(cache_counters):
            raise ValueError(
                "cache-disabled Phase-6 Agent run must report zero cache counters"
            )
    return metrics


class MetricsEnabledRetrieverClient:
    """Install the isolated opt-in metrics request before Phase-4 timing."""

    def __init__(self, retriever_url, topk, server_contract, timeout_s=120.0,
                 session=requests):
        self.retriever_url = str(retriever_url)
        self.topk = int(topk)
        self.server_contract = dict(server_contract)
        self.timeout_s = float(timeout_s)
        self.session = session
        self.events = []
        self.post_count = 0

    def configure_manager(self, manager):
        def metrics_batch_search(_manager, queries):
            normalized = [str(query).strip() for query in (queries or [])]
            if not normalized or any(not query for query in normalized):
                raise ValueError("Retriever queries must be non-empty strings")
            payload = {
                "queries": normalized,
                "topk": self.topk,
                "return_scores": True,
                "return_metrics": True,
            }
            self.post_count += 1
            response = self.session.post(
                self.retriever_url, json=payload, timeout=self.timeout_s
            )
            response_payload = _json_response(response, "Phase-6 Retriever request")
            results = response_payload.get("result")
            if not isinstance(results, list) or len(results) != len(normalized):
                raise RuntimeError("Phase-6 Retriever returned invalid query cardinality")
            if any(not isinstance(row, list) or len(row) != self.topk for row in results):
                raise RuntimeError("Phase-6 Retriever did not return exact top-k rows")
            metrics = validate_request_metrics(
                response_payload.get("metrics"),
                len(normalized),
                self.topk,
                self.server_contract,
            )
            self.events.append({
                "queries": normalized,
                "metrics": copy.deepcopy(metrics),
            })
            return response_payload

        manager._batch_search = MethodType(metrics_batch_search, manager)


def _single_run_binding(records, label):
    configs = {
        json.dumps(record["run_config"], sort_keys=True): record["run_config"]
        for record in records
    }
    fingerprints = {record["run_fingerprint"] for record in records}
    if len(configs) != 1 or len(fingerprints) != 1:
        raise ValueError(f"{label} mixes multiple run configurations")
    config = next(iter(configs.values()))
    fingerprint = next(iter(fingerprints))
    if fingerprint != run_config_fingerprint(config):
        raise ValueError(f"{label} has an invalid run fingerprint")
    return config, fingerprint


def validate_phase5_compressed_baseline(
    baseline_path,
    examples,
    manifest,
    manifest_path,
    expected_model=DEFAULT_SEARCH_MODEL,
    expected_count=64,
):
    """Validate the immutable Phase-5 compressed result without writing it."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    before_sha256 = sha256_file(baseline_path)
    records = read_phase5_result_file(baseline_path, "compressed_256")
    if len(records) != expected_count or len(examples) != expected_count:
        raise ValueError(
            f"Phase-6 requires {expected_count} baseline/eval rows; "
            f"found baseline={len(records)}, eval={len(examples)}"
        )
    pending, records_by_uid = resume_plan(examples, records)
    if pending or len(records_by_uid) != expected_count:
        raise ValueError("Phase-5 compressed baseline has a different UID set")
    if [record["uid"] for record in records] != [example.uid for example in examples]:
        raise ValueError("Phase-5 compressed baseline row order differs from eval data")
    config, fingerprint = _single_run_binding(records, "Phase-5 compressed baseline")
    required = {
        "model_path": str(expected_model),
        "prompt_contract": "phase4-benchmark-v1",
        "experiment_contract": "phase5-observation-context-v1",
        "mode": "compressed_256",
        "observation_policy": "compressed_256",
        "seed": SEED,
        "greedy": True,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": MAX_OBS_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "max_turns": MAX_TURNS,
        "retriever_topk": RETRIEVER_TOPK,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Phase-5 compressed baseline has unexpected {key}")
    manifest_path = Path(manifest_path).expanduser().resolve()
    if config.get("eval_sha256") != manifest.get("output", {}).get("sha256"):
        raise ValueError("Phase-5 baseline is bound to a different eval parquet")
    if config.get("eval_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Phase-5 baseline is bound to a different eval manifest")
    retrieval_failures = sum(
        record["search_retrieval_failure_count"] for record in records
    )
    evaluation_errors = sum(bool(record.get("evaluation_error")) for record in records)
    if retrieval_failures or evaluation_errors:
        raise ValueError("Phase-5 baseline contains retrieval/evaluation failures")
    if sha256_file(baseline_path) != before_sha256:
        raise RuntimeError("Phase-5 compressed baseline changed during validation")
    return records, {
        "validated": True,
        "path": str(baseline_path),
        "sha256": before_sha256,
        "row_count": len(records),
        "run_fingerprint": fingerprint,
        "eval_parquet_sha256": config["eval_sha256"],
        "eval_manifest_sha256": config["eval_manifest_sha256"],
        "retrieval_failure_count": retrieval_failures,
        "evaluation_error_count": evaluation_errors,
    }


def validate_output_isolation(output_dir, phase5_baseline_path):
    output_dir = Path(output_dir).expanduser().resolve()
    baseline = Path(phase5_baseline_path).expanduser().resolve()
    phase5_dir = baseline.parent
    if output_dir == phase5_dir or phase5_dir in output_dir.parents:
        raise ValueError(
            "Phase-6 outputs must not be written into the immutable Phase-5 "
            "artifact directory or any descendant"
        )
    return output_dir


def load_selected_candidate(path, allow_unqualified=False):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("selected_candidate"), dict
    ):
        raise ValueError("selected_candidate.json has no selected_candidate object")
    passed = payload.get("candidate_selection_passed") is True
    ready = payload.get("production_ready_for_agent_evaluation") is True
    if not (passed and ready) and not allow_unqualified:
        raise ValueError(
            "ANN candidate did not pass calibration; explicit "
            "--allow-unqualified-candidate is required"
        )
    selected = payload["selected_candidate"]
    if selected.get("index_backend") != "ivfpq":
        raise ValueError("selected candidate must use the ivfpq backend")
    for key in ("nprobe", "faiss_thread_count"):
        value = selected.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"selected candidate has invalid {key}")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("selected candidate is missing artifact bindings")
    for key in ("flat_index", "ivfpq_index"):
        binding = bindings.get(key)
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("sha256"), str)
            or not binding["sha256"]
        ):
            raise ValueError(f"selected candidate is missing {key} SHA256 binding")
    if selected.get("index_sha256") != bindings["ivfpq_index"]["sha256"]:
        raise ValueError("selected IVF-PQ SHA256 disagrees with artifact bindings")
    if payload.get("selection_uses_qa_ground_truth") is not False:
        raise ValueError("selected candidate must declare calibration-only selection")
    embeddings_manifest_binding = bindings.get("query_embeddings_manifest")
    if not isinstance(embeddings_manifest_binding, dict):
        raise ValueError("selected candidate is missing query-embedding provenance")
    embeddings_manifest_path = Path(
        embeddings_manifest_binding.get("path", "")
    ).expanduser().resolve()
    if (
        not embeddings_manifest_path.is_file()
        or embeddings_manifest_binding.get("sha256")
        != sha256_file(embeddings_manifest_path)
    ):
        raise ValueError("selected query-embedding manifest binding is missing or stale")
    embeddings_manifest = json.loads(
        embeddings_manifest_path.read_text(encoding="utf-8")
    )
    model_fingerprint = (
        embeddings_manifest.get("binding", {}).get("model", {}).get("sha256")
    )
    if not isinstance(model_fingerprint, str) or not model_fingerprint:
        raise ValueError("selected query embeddings have no E5 model fingerprint")
    return payload, {
        "path": str(path),
        "sha256": sha256_file(path),
        "candidate_selection_passed": passed,
        "production_ready_for_agent_evaluation": ready,
        "override_used": bool(allow_unqualified and not (passed and ready)),
        "model_fingerprint": model_fingerprint,
        "query_embeddings_manifest_sha256": embeddings_manifest_binding["sha256"],
        "nprobe": selected["nprobe"],
        "faiss_thread_count": selected["faiss_thread_count"],
        "flat_index_sha256": bindings["flat_index"]["sha256"],
        "ivfpq_index_sha256": bindings["ivfpq_index"]["sha256"],
    }


def stable_server_contract(
    health, condition, candidate, *, require_local_index_file=True
):
    if not isinstance(health, dict) or health.get("ready") is not True:
        raise ValueError("Retriever /healthz is not ready")
    missing = [key for key in SERVER_CONTRACT_FIELDS if key not in health]
    if missing:
        raise ValueError(f"Retriever /healthz is missing fields: {missing}")
    expected_backend = "flat" if condition == "flat_exact_replay" else "ivfpq"
    if health.get("index_backend") != expected_backend:
        raise ValueError(f"{condition} requires a {expected_backend} Retriever")
    if health.get("cache_enabled") is not False:
        raise ValueError("Phase-6 downstream Agent comparison requires cache disabled")
    selected = candidate["selected_candidate"]
    if health.get("faiss_thread_count") != selected["faiss_thread_count"]:
        raise ValueError("Retriever FAISS threads differ from selected serving candidate")
    if condition == "ivfpq_selected":
        if health.get("nprobe") != selected["nprobe"]:
            raise ValueError("IVF-PQ Retriever nprobe differs from selected candidate")
        expected_index_sha = candidate["bindings"]["ivfpq_index"]["sha256"]
    else:
        expected_index_sha = candidate["bindings"]["flat_index"]["sha256"]
    if health.get("index_fingerprint") != expected_index_sha:
        raise ValueError("Retriever index fingerprint differs from candidate binding")
    embeddings_manifest_binding = candidate["bindings"]["query_embeddings_manifest"]
    embeddings_manifest_path = Path(
        embeddings_manifest_binding["path"]
    ).expanduser().resolve()
    if embeddings_manifest_binding.get("sha256") != sha256_file(
        embeddings_manifest_path
    ):
        raise ValueError("selected query-embedding manifest binding became stale")
    embeddings_manifest = json.loads(
        embeddings_manifest_path.read_text(encoding="utf-8")
    )
    expected_model_fingerprint = (
        embeddings_manifest.get("binding", {}).get("model", {}).get("sha256")
    )
    if health.get("model_fingerprint") != expected_model_fingerprint:
        raise ValueError(
            "Retriever E5 model fingerprint differs from calibration embeddings"
        )
    contract = {key: copy.deepcopy(health[key]) for key in SERVER_CONTRACT_FIELDS}
    index_path = Path(health["index_path"]).expanduser().resolve()
    if require_local_index_file and not index_path.is_file():
        raise ValueError("Retriever index path is not a readable local file")
    if require_local_index_file and health.get("index_file_size_bytes") != (
        index_path.stat().st_size
    ):
        raise ValueError("Retriever index file size disagrees with the local artifact")
    return contract


def build_run_config(
    condition,
    model_path,
    model_checkpoint_fingerprint,
    retriever_url,
    manifest,
    manifest_path,
    baseline_audit,
    baseline_config,
    server_contract,
    candidate_audit,
    attention_backend="",
):
    context_config = copy.deepcopy(baseline_config)
    context_config["retriever_url"] = str(retriever_url)
    phase5_context_fingerprint = run_config_fingerprint(context_config)
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_contract": "phase4-benchmark-v1",
        "experiment_contract": EXPERIMENT_CONTRACT,
        "mode": condition,
        "observation_policy": "compressed_256",
        "eval_sha256": manifest["output"]["sha256"],
        "eval_manifest_sha256": sha256_file(manifest_path),
        "phase5_baseline_path": baseline_audit["path"],
        "phase5_baseline_sha256": baseline_audit["sha256"],
        "phase5_baseline_run_fingerprint": baseline_audit["run_fingerprint"],
        "phase5_context_run_config": context_config,
        "phase5_context_run_fingerprint": phase5_context_fingerprint,
        "selected_candidate_path": candidate_audit["path"],
        "selected_candidate_sha256": candidate_audit["sha256"],
        "candidate_selection_passed": candidate_audit[
            "candidate_selection_passed"
        ],
        "candidate_production_ready_for_agent_evaluation": candidate_audit[
            "production_ready_for_agent_evaluation"
        ],
        "candidate_override_used": candidate_audit["override_used"],
        "model_path": str(model_path),
        "model_checkpoint_fingerprint": model_checkpoint_fingerprint,
        "seed": SEED,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "attention_backend": attention_backend,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": MAX_OBS_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "max_turns": MAX_TURNS,
        "retriever_url": str(retriever_url),
        "retriever_topk": RETRIEVER_TOPK,
        "retriever_return_metrics": True,
        "retriever_cache_enabled": False,
        "retriever_server_contract": copy.deepcopy(server_contract),
    }


def phase6_result_from_search_result(
    search_result, condition, run_config, policy_events, request_events
):
    context_config = run_config["phase5_context_run_config"]
    context_record = phase5_result_from_search_result(
        search_result,
        "compressed_256",
        context_config,
        policy_events,
    )
    validate_phase5_result(context_record, "compressed_256")
    metrics = [copy.deepcopy(event["metrics"]) for event in request_events]
    queries = [query for event in request_events for query in event["queries"]]
    ids = [row for event in request_events for row in event["metrics"]["result_ids"]]
    retrieval_count = context_record["number_of_successful_retrievals"]
    if len(queries) != retrieval_count or len(ids) != retrieval_count:
        raise ValueError("server telemetry does not align with successful retrievals")
    record = dict(context_record)
    record.update({
        "schema_version": SCHEMA_VERSION,
        "mode": condition,
        "observation_policy": "compressed_256",
        "run_config": run_config,
        "run_fingerprint": run_config_fingerprint(run_config),
        "retriever_queries": queries,
        "retriever_request_ids": [item["request_id"] for item in metrics],
        "retrieved_document_ids": ids,
        "retriever_request_metrics": metrics,
    })
    return validate_phase6_result(record, condition)


def validate_phase6_result(record, expected_condition=None):
    if not isinstance(record, dict):
        raise ValueError("Phase-6 result row must be a JSON object")
    condition = record.get("mode")
    if condition not in CONDITIONS or (
        expected_condition is not None and condition != expected_condition
    ):
        raise ValueError(f"unexpected Phase-6 condition: {condition!r}")
    missing = PHASE6_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Phase-6 result is missing fields: {sorted(missing)}")
    run_config = record.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Phase-6 run_config must be a mapping")
    if record.get("run_fingerprint") != run_config_fingerprint(run_config):
        raise ValueError("Phase-6 run fingerprint does not match run_config")
    required_config = {
        "schema_version": SCHEMA_VERSION,
        "prompt_contract": "phase4-benchmark-v1",
        "experiment_contract": EXPERIMENT_CONTRACT,
        "mode": condition,
        "observation_policy": "compressed_256",
        "model_path": record.get("model_path"),
        "seed": SEED,
        "greedy": True,
        "max_start_length": MAX_START_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "max_obs_length": MAX_OBS_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "max_turns": MAX_TURNS,
        "retriever_topk": RETRIEVER_TOPK,
        "retriever_return_metrics": True,
        "retriever_cache_enabled": False,
    }
    for key, expected in required_config.items():
        if run_config.get(key) != expected:
            raise ValueError(f"Phase-6 run_config has unexpected {key}")
    if not run_config.get("phase5_baseline_sha256"):
        raise ValueError("Phase-6 result is not bound to the Phase-5 baseline")
    if not run_config.get("selected_candidate_sha256"):
        raise ValueError("Phase-6 result is not bound to selected_candidate.json")
    model_checkpoint_fingerprint = run_config.get("model_checkpoint_fingerprint")
    if (
        not isinstance(model_checkpoint_fingerprint, str)
        or len(model_checkpoint_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in model_checkpoint_fingerprint
        )
    ):
        raise ValueError(
            "Phase-6 result is not bound to a canonical Qwen checkpoint fingerprint"
        )
    if not all(
        isinstance(run_config.get(key), bool)
        for key in (
            "candidate_selection_passed",
            "candidate_production_ready_for_agent_evaluation",
            "candidate_override_used",
        )
    ):
        raise ValueError("Phase-6 candidate status fields must be explicit booleans")
    server_contract = run_config.get("retriever_server_contract")
    if not isinstance(server_contract, dict):
        raise ValueError("Phase-6 result has no stable Retriever server contract")
    expected_backend = "flat" if condition == "flat_exact_replay" else "ivfpq"
    if server_contract.get("index_backend") != expected_backend:
        raise ValueError("Phase-6 result has the wrong Retriever backend")
    if server_contract.get("cache_enabled") is not False:
        raise ValueError("Phase-6 Agent result must use a cache-disabled Retriever")

    context_view = dict(record)
    context_view["mode"] = "compressed_256"
    context_view["observation_policy"] = "compressed_256"
    context_view["run_config"] = run_config.get("phase5_context_run_config")
    context_view["run_fingerprint"] = run_config.get(
        "phase5_context_run_fingerprint"
    )
    validate_phase5_result(context_view, "compressed_256")

    retrieval_count = record["number_of_successful_retrievals"]
    queries = record["retriever_queries"]
    ids = record["retrieved_document_ids"]
    metrics = record["retriever_request_metrics"]
    request_ids = record["retriever_request_ids"]
    if not isinstance(queries, list) or len(queries) != retrieval_count:
        raise ValueError("Phase-6 Retriever queries are not retrieval-aligned")
    if any(not isinstance(query, str) or not query for query in queries):
        raise ValueError("Phase-6 Retriever queries must be nonempty text")
    if not isinstance(ids, list) or len(ids) != retrieval_count:
        raise ValueError("Phase-6 document IDs are not retrieval-aligned")
    if not isinstance(metrics, list) or not isinstance(request_ids, list):
        raise ValueError("Phase-6 request telemetry must use lists")
    if request_ids != [metric.get("request_id") for metric in metrics]:
        raise ValueError("Phase-6 request IDs disagree with request metrics")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Phase-6 request IDs must be unique within a trajectory")
    query_cursor = 0
    id_cursor = 0
    for metric in metrics:
        count = metric.get("query_count")
        validate_request_metrics(
            metric, count, RETRIEVER_TOPK, server_contract
        )
        query_cursor += count
        id_cursor += len(metric["result_ids"])
    if query_cursor != retrieval_count or id_cursor != retrieval_count:
        raise ValueError("Phase-6 request metrics do not cover all retrievals")
    flattened_metric_ids = [row for metric in metrics for row in metric["result_ids"]]
    if flattened_metric_ids != ids:
        raise ValueError("Phase-6 document IDs disagree with server metrics")
    return record


def read_phase6_result_file(path, condition):
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
        validate_phase6_result(record, condition)
        if record["uid"] in seen:
            raise ValueError(f"duplicate UID in {path}: {record['uid']}")
        seen.add(record["uid"])
        records.append(record)
    return records


def _append_result(path, record):
    validate_phase6_result(record, record["mode"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _rewrite_results(path, examples, records_by_uid):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(records_by_uid[example.uid], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_condition_with_resume(
    examples,
    result_path,
    condition,
    evaluate_one,
    expected_run_fingerprint,
    overwrite=False,
):
    result_path = Path(result_path).expanduser().resolve()
    if overwrite and result_path.exists():
        result_path.unlink()
    existing = read_phase6_result_file(result_path, condition)
    pending, records_by_uid = resume_plan(
        examples, existing, expected_run_fingerprint=expected_run_fingerprint
    )
    for example in pending:
        record = evaluate_one(example)
        if record.get("uid") != example.uid or record.get("mode") != condition:
            raise ValueError("Phase-6 evaluator returned the wrong UID or condition")
        _append_result(result_path, record)
        records_by_uid[example.uid] = record
        print(
            f"[{condition}] persisted {len(records_by_uid)}/{len(examples)}: "
            f"{example.uid}"
        )
    if len(records_by_uid) != len(examples):
        raise RuntimeError("Phase-6 condition ended without all expected results")
    _rewrite_results(result_path, examples, records_by_uid)
    return [records_by_uid[example.uid] for example in examples]


def finalize_server_audit(
    output_dir,
    result_path,
    condition,
    records,
    run_fingerprint,
    retriever_url,
    candidate,
    server_contract,
    starting_health,
    starting_stats,
):
    """Validate the live server and atomically create/repair the audit sidecar."""

    ending_health = read_server_endpoint(retriever_url, "/healthz")
    if stable_server_contract(ending_health, condition, candidate) != server_contract:
        raise RuntimeError("Retriever stable health contract changed during Agent run")
    ending_stats = read_server_endpoint(retriever_url, "/stats")
    audit_path = Path(output_dir) / f"{condition}.server_audit.json"
    _atomic_json(audit_path, {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "run_fingerprint": run_fingerprint,
        "starting_health": starting_health,
        "ending_health": ending_health,
        "starting_stats": starting_stats,
        "ending_stats": ending_stats,
        "row_count": len(records),
    })
    return audit_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", choices=("validate_baseline",) + CONDITIONS, required=True
    )
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase5-baseline", required=True)
    parser.add_argument("--selected-candidate")
    parser.add_argument("--search-model", default=DEFAULT_SEARCH_MODEL)
    parser.add_argument("--retriever-url")
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unqualified-candidate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != SEED or args.gpu_memory_utilization != GPU_MEMORY_UTILIZATION:
        raise ValueError("Phase-6 seed/GPU utilization must remain 42/0.20")
    examples, manifest = load_eval_examples(args.eval_data, args.eval_manifest)
    baseline_records, baseline_audit = validate_phase5_compressed_baseline(
        args.phase5_baseline,
        examples,
        manifest,
        args.eval_manifest,
        expected_model=args.search_model,
    )
    if args.condition == "validate_baseline":
        print(json.dumps(baseline_audit, indent=2, sort_keys=True))
        return
    if not args.retriever_url:
        raise ValueError("runnable Phase-6 conditions require --retriever-url")
    if not args.selected_candidate:
        raise ValueError("runnable Phase-6 conditions require --selected-candidate")
    model_checkpoint_fingerprint = fingerprint_model_checkpoint(args.search_model)
    output_dir = validate_output_isolation(args.output_dir, args.phase5_baseline)
    candidate, candidate_audit = load_selected_candidate(
        args.selected_candidate,
        allow_unqualified=args.allow_unqualified_candidate,
    )
    health = read_server_endpoint(args.retriever_url, "/healthz")
    server_contract = stable_server_contract(health, args.condition, candidate)
    starting_stats = read_server_endpoint(args.retriever_url, "/stats")
    baseline_config, _ = _single_run_binding(
        baseline_records, "Phase-5 compressed baseline"
    )
    run_config = build_run_config(
        args.condition,
        args.search_model,
        model_checkpoint_fingerprint,
        args.retriever_url,
        manifest,
        Path(args.eval_manifest).resolve(),
        baseline_audit,
        baseline_config,
        server_contract,
        candidate_audit,
        attention_backend=os.environ.get("VLLM_ATTENTION_BACKEND", ""),
    )
    run_fingerprint = run_config_fingerprint(run_config)
    result_path = output_dir / f"{args.condition}.jsonl"
    existing = [] if args.overwrite else read_phase6_result_file(
        result_path, args.condition
    )
    pending, existing_by_uid = resume_plan(
        examples, existing, expected_run_fingerprint=run_fingerprint
    )
    if not pending:
        try:
            _rewrite_results(result_path, examples, existing_by_uid)
            records = [existing_by_uid[example.uid] for example in examples]
            finalize_server_audit(
                output_dir,
                result_path,
                args.condition,
                records,
                run_fingerprint,
                args.retriever_url,
                candidate,
                server_contract,
                health,
                starting_stats,
            )
        finally:
            verify_runnable_inputs_unchanged(
                args.search_model,
                model_checkpoint_fingerprint,
                args.phase5_baseline,
                baseline_audit["sha256"],
                "audit repair",
            )
        print(f"[{args.condition}] already complete; server audit verified: {result_path}")
        return

    print(json.dumps({
        "condition": args.condition,
        "model": args.search_model,
        "model_checkpoint_fingerprint": model_checkpoint_fingerprint,
        "remaining_examples": len(pending),
        "retriever_url": args.retriever_url,
        "retriever_server_contract": server_contract,
        "cache_enabled": False,
        "observation_policy": "compressed_256",
        "max_turns": MAX_TURNS,
        "max_obs_length": MAX_OBS_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "result_path": str(result_path),
        "run_fingerprint": run_fingerprint,
    }, indent=2, sort_keys=True))
    generator = VLLMGenerator(
        model_path=args.search_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
        seed=args.seed,
        max_response_length=MAX_RESPONSE_LENGTH,
    )
    compressor = EvidenceCompressor(
        generator.tokenizer, max_observation_tokens=MAX_OBS_LENGTH
    )

    def evaluate_one(example):
        policy = ObservationPolicyManager(
            "compressed_256", generator.tokenizer, compressor=compressor
        )
        metrics_client = MetricsEnabledRetrieverClient(
            args.retriever_url,
            RETRIEVER_TOPK,
            server_contract,
            timeout_s=args.request_timeout_s,
        )
        search_result = evaluate_search_rl(
            example,
            generator.tokenizer,
            generator,
            args.search_model,
            args.retriever_url,
            RETRIEVER_TOPK,
            MAX_TURNS,
            MAX_START_LENGTH,
            MAX_RESPONSE_LENGTH,
            MAX_OBS_LENGTH,
            MAX_PROMPT_LENGTH,
            run_config,
            configure_manager=policy.configure_manager,
            configure_retriever_client=metrics_client.configure_manager,
        )
        return phase6_result_from_search_result(
            search_result,
            args.condition,
            run_config,
            policy.events,
            metrics_client.events,
        )

    try:
        records = run_condition_with_resume(
            examples,
            result_path,
            args.condition,
            evaluate_one,
            expected_run_fingerprint=run_fingerprint,
            overwrite=args.overwrite,
        )
        finalize_server_audit(
            output_dir,
            result_path,
            args.condition,
            records,
            run_fingerprint,
            args.retriever_url,
            candidate,
            server_contract,
            health,
            starting_stats,
        )
    finally:
        verify_runnable_inputs_unchanged(
            args.search_model,
            model_checkpoint_fingerprint,
            args.phase5_baseline,
            baseline_audit["sha256"],
            "run",
        )
    print(f"[{args.condition}] complete: {result_path}")


if __name__ == "__main__":
    main()
