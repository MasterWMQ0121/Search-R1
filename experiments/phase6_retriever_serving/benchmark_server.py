#!/usr/bin/env python3
"""Benchmark Phase-6 Retriever HTTP servers without mixing workload classes.

The client is deliberately model- and index-agnostic.  A server is benchmarked
in one process configuration at a time and the resulting target record may be
appended to a hash-bound report.  This permits a memory-safe, sequential FAISS
thread sweep on Wiki-18 instead of loading several copies of the Flat index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES = (1, 4, 8, 16, 32)
REQUIRED_EXACT_THREADS = (1, 4, 8, 16)
SERVER_TIMING_KEYS = (
    "request_total_s",
    "query_normalization_s",
    "cache_lookup_s",
    "query_encoding_s",
    "faiss_search_s",
    "document_fetch_s",
    "response_format_s",
)
VALID_ROLES = {
    "exact_sweep",
    "flat_exact",
    "ivfpq_selected",
    "warm_flat",
    "warm_ivfpq",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _finite_nonnegative(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0 <= probability <= 1:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {"mean_s": None, "p50_s": None, "p95_s": None}
    return {
        "mean_s": statistics.fmean(values),
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
    }


def load_calibration_queries(path: Path | str) -> list[dict]:
    path = Path(path).expanduser().resolve()
    rows = []
    seen_uids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("uid")
            question = row.get("question")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"calibration row {line_number} has no UID")
            if uid in seen_uids:
                raise ValueError(f"calibration queries contain duplicate UID: {uid}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"calibration row {uid} has an empty question")
            if any(key in row for key in ("answer", "answers", "ground_truth", "target")):
                raise ValueError(f"calibration row {uid} contains a QA target")
            seen_uids.add(uid)
            rows.append({"uid": uid, "question": question.strip()})
    if not rows:
        raise ValueError("calibration query file is empty")
    return rows


def exact_batches(rows: Sequence[dict], batch_size: int) -> list[list[dict]]:
    """Return exact-size circular batches while covering every row at least once."""

    if not rows:
        raise ValueError("cannot batch no calibration rows")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    number_of_batches = math.ceil(len(rows) / batch_size)
    return [
        [rows[(batch_number * batch_size + offset) % len(rows)] for offset in range(batch_size)]
        for batch_number in range(number_of_batches)
    ]


def _http_json(url: str, *, payload=None, timeout: float = 120.0) -> tuple[dict, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"{url} returned HTTP {status}")
    if not isinstance(result, dict):
        raise ValueError(f"{url} returned a non-object JSON payload")
    return result, elapsed


def _cache_states(health: Mapping) -> dict:
    cache_configuration = health.get("cache_configuration")
    if isinstance(cache_configuration, Mapping):
        states = {}
        for name in ("result", "embedding"):
            cache = cache_configuration.get(name)
            if isinstance(cache, Mapping) and isinstance(cache.get("enabled"), bool):
                states[name] = {
                    "enabled": cache["enabled"],
                    "capacity": cache.get("capacity"),
                }
        if len(states) == 2:
            return states
    cache = health.get("cache")
    if isinstance(cache, Mapping) and isinstance(cache.get("enabled"), bool):
        return {
            "result": {"enabled": cache["enabled"], "capacity": cache.get("capacity")},
            "embedding": {"enabled": cache["enabled"], "capacity": cache.get("capacity")},
        }
    if isinstance(health.get("cache_enabled"), bool):
        return {
            "result": {
                "enabled": health["cache_enabled"],
                "capacity": health.get("result_cache_capacity"),
            },
            "embedding": {
                "enabled": health["cache_enabled"],
                "capacity": health.get("embedding_cache_capacity"),
            },
        }
    raise ValueError("Retriever health payload does not declare cache enabled state")


def _validate_health(health: Mapping, role: str, workload: str) -> dict:
    if health.get("ready") is not True:
        raise ValueError("Retriever is not ready")
    backend = health.get("index_backend", health.get("index_type"))
    if role in {"exact_sweep", "flat_exact", "warm_flat"} and backend != "flat":
        raise ValueError(f"{role} requires a flat backend, received {backend!r}")
    if role in {"ivfpq_selected", "warm_ivfpq"} and backend != "ivfpq":
        raise ValueError(f"{role} requires an ivfpq backend, received {backend!r}")
    cache_states = _cache_states(health)
    cache_enabled = any(state["enabled"] for state in cache_states.values())
    if workload == "cold" and cache_enabled:
        raise ValueError("cold-query workloads require both server caches disabled")
    if workload == "warm" and not cache_enabled:
        raise ValueError("warm-cache workloads require an explicitly cache-enabled server")
    thread_count = health.get("faiss_thread_count")
    if isinstance(thread_count, bool) or not isinstance(thread_count, int) or thread_count <= 0:
        raise ValueError("Retriever health has an invalid FAISS thread count")
    encode_batch_size = health.get("retrieval_encode_batch_size")
    if (
        isinstance(encode_batch_size, bool)
        or not isinstance(encode_batch_size, int)
        or encode_batch_size <= 0
    ):
        raise ValueError("Retriever health has an invalid retrieval encode batch size")
    for key in ("index_fingerprint", "model_fingerprint", "corpus_fingerprint"):
        if not isinstance(health.get(key), str) or not health[key]:
            raise ValueError(f"Retriever health has no content-bound {key}")
    return {
        "backend": backend,
        "faiss_thread_count": thread_count,
        "retrieval_encode_batch_size": encode_batch_size,
        "nprobe": health.get("nprobe"),
        "cache_enabled": cache_enabled,
        "cache": cache_states,
        "server_instance_id": health.get("server_instance_id"),
        "index_path": health.get("index_path"),
        "index_fingerprint": health.get("index_fingerprint"),
        "model_path": health.get("model_path"),
        "model_fingerprint": health.get("model_fingerprint"),
        "corpus_fingerprint": health.get("corpus_fingerprint"),
        "index_ntotal": health.get("index_ntotal"),
        "index_dimension": health.get("index_dimension"),
        "metric_type": health.get("metric_type"),
        "corpus_row_count": health.get("corpus_row_count"),
        "process_rss_bytes": health.get("process_rss_bytes"),
    }


def _validate_retrieve_payload(payload: Mapping, batch: Sequence[dict], topk: int) -> dict:
    results = payload.get("result")
    metrics = payload.get("metrics")
    if not isinstance(results, list) or len(results) != len(batch):
        raise ValueError("Retriever result cardinality does not match query batch")
    if not isinstance(metrics, Mapping):
        raise ValueError("Phase-6 benchmark requires return_metrics response data")
    result_ids = metrics.get("result_ids")
    if not isinstance(result_ids, list) or len(result_ids) != len(batch):
        raise ValueError("Retriever metrics result_ids do not match query batch")
    for query_index, (query_results, query_ids) in enumerate(zip(results, result_ids)):
        if not isinstance(query_results, list) or len(query_results) != topk:
            raise ValueError(f"query {query_index} did not return exactly top-k documents")
        if not isinstance(query_ids, list) or len(query_ids) != topk:
            raise ValueError(f"query {query_index} did not return exactly top-k IDs")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in query_ids):
            raise ValueError(f"query {query_index} returned invalid internal document IDs")
    timings = {
        key: _finite_nonnegative(metrics.get(key), f"metrics.{key}")
        for key in SERVER_TIMING_KEYS
    }
    if metrics.get("query_count") != len(batch) or metrics.get("topk") != topk:
        raise ValueError("Retriever metrics query_count/topk do not match the request")
    for key in (
        "cache_hit_count",
        "cache_miss_count",
        "embedding_cache_hit_count",
        "embedding_cache_miss_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Retriever metrics {key} must be a non-negative integer")
    return {"result_ids": result_ids, "timings": timings, "metrics": dict(metrics)}


def _validate_request_identity(sample: Mapping, identity: Mapping) -> None:
    metrics = sample["metrics"]
    for metric_key, identity_key in (
        ("index_backend", "backend"),
        ("faiss_thread_count", "faiss_thread_count"),
        ("nprobe", "nprobe"),
    ):
        if metrics.get(metric_key) != identity.get(identity_key):
            raise ValueError(
                f"request metric {metric_key} changed from health identity: "
                f"{metrics.get(metric_key)!r} != {identity.get(identity_key)!r}"
            )
    query_count = sample["query_count"]
    result_lookups = metrics["cache_hit_count"] + metrics["cache_miss_count"]
    result_enabled = identity["cache"]["result"]["enabled"]
    if result_enabled and result_lookups != query_count:
        raise ValueError("result-cache hit/miss counters do not cover the request")
    if not result_enabled and result_lookups != 0:
        raise ValueError("disabled result cache reported hit/miss activity")
    embedding_lookups = (
        metrics["embedding_cache_hit_count"]
        + metrics["embedding_cache_miss_count"]
    )
    embedding_enabled = identity["cache"]["embedding"]["enabled"]
    expected_embedding_lookups = metrics["cache_miss_count"] if result_enabled else query_count
    if embedding_enabled and embedding_lookups != expected_embedding_lookups:
        raise ValueError("embedding-cache hit/miss counters do not cover uncached results")
    if not embedding_enabled and embedding_lookups != 0:
        raise ValueError("disabled embedding cache reported hit/miss activity")


def retrieve_batch(
    url: str,
    batch: Sequence[dict],
    *,
    topk: int,
    timeout: float,
) -> dict:
    payload, latency = _http_json(
        url,
        payload={
            "queries": [row["question"] for row in batch],
            "topk": topk,
            "return_scores": True,
            "return_metrics": True,
        },
        timeout=timeout,
    )
    validated = _validate_retrieve_payload(payload, batch, topk)
    validated.update({
        "client_latency_s": latency,
        "uids": [row["uid"] for row in batch],
        "query_count": len(batch),
    })
    return validated


def _record_ids(target: dict, sample: Mapping) -> None:
    for uid, result_ids in zip(sample["uids"], sample["result_ids"]):
        previous = target.setdefault(uid, list(result_ids))
        if previous != list(result_ids):
            raise ValueError(
                f"Retriever returned nondeterministic IDs for {uid}: {previous} vs {result_ids}"
            )


def aggregate_samples(samples: Sequence[Mapping], *, elapsed_wall_s: float) -> dict:
    successful = [sample for sample in samples if sample.get("error") is None]
    client_latencies = [sample["client_latency_s"] for sample in successful]
    query_count = sum(sample["query_count"] for sample in successful)
    stage = {
        key: _distribution([sample["timings"][key] for sample in successful])
        for key in SERVER_TIMING_KEYS
    }
    cache_totals = {
        key: sum(sample["metrics"][key] for sample in successful)
        for key in (
            "cache_hit_count",
            "cache_miss_count",
            "embedding_cache_hit_count",
            "embedding_cache_miss_count",
        )
    }
    result_cache_lookups = (
        cache_totals["cache_hit_count"] + cache_totals["cache_miss_count"]
    )
    embedding_cache_lookups = (
        cache_totals["embedding_cache_hit_count"]
        + cache_totals["embedding_cache_miss_count"]
    )
    return {
        "request_count": len(samples),
        "successful_request_count": len(successful),
        "error_count": len(samples) - len(successful),
        "errors": [sample["error"] for sample in samples if sample.get("error")],
        "query_count": query_count,
        "elapsed_wall_s": elapsed_wall_s,
        "queries_per_second": query_count / elapsed_wall_s if elapsed_wall_s > 0 else None,
        "batches_per_second": len(successful) / elapsed_wall_s if elapsed_wall_s > 0 else None,
        "client_latency": _distribution(client_latencies),
        "server_stage_latency": stage,
        "cache": {
            **cache_totals,
            "result_cache_hit_rate": (
                cache_totals["cache_hit_count"] / result_cache_lookups
                if result_cache_lookups else None
            ),
            "embedding_cache_hit_rate": (
                cache_totals["embedding_cache_hit_count"] / embedding_cache_lookups
                if embedding_cache_lookups else None
            ),
        },
    }


def benchmark_target(
    url: str,
    rows: Sequence[dict],
    *,
    role: str,
    workload: str,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    topk: int = 3,
    repetitions: int = 1,
    warmup_requests: int = 2,
    timeout: float = 120.0,
) -> dict:
    if role not in VALID_ROLES:
        raise ValueError(f"unsupported target role: {role!r}")
    if workload not in {"cold", "warm"}:
        raise ValueError("workload must be 'cold' or 'warm'")
    if repetitions <= 0 or warmup_requests < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    health, _ = _http_json(url.rstrip("/") + "/healthz", timeout=timeout)
    identity = _validate_health(health, role, workload)
    retrieve_url = url.rstrip("/") + "/retrieve"
    stats_before, _ = _http_json(url.rstrip("/") + "/stats", timeout=timeout)

    configurations = {}
    for batch_size in batch_sizes:
        batches = exact_batches(rows, int(batch_size))
        # A warm-cache measurement primes every exact query key.  Cold workloads
        # only warm model/index code paths and have caches disabled by validation.
        warmup_batches = batches if workload == "warm" else [
            batches[index % len(batches)] for index in range(warmup_requests)
        ]
        for batch in warmup_batches:
            warmup_sample = retrieve_batch(
                retrieve_url, batch, topk=topk, timeout=timeout
            )
            _validate_request_identity(warmup_sample, identity)

        samples = []
        ids_by_uid = {}
        started = time.perf_counter()
        for _ in range(repetitions):
            for batch in batches:
                try:
                    sample = retrieve_batch(retrieve_url, batch, topk=topk, timeout=timeout)
                    _validate_request_identity(sample, identity)
                    _record_ids(ids_by_uid, sample)
                    sample["error"] = None
                except Exception as exc:  # Preserve benchmark evidence, then fail completion.
                    sample = {"error": f"{type(exc).__name__}: {exc}"}
                samples.append(sample)
        elapsed_wall_s = time.perf_counter() - started
        aggregate = aggregate_samples(samples, elapsed_wall_s=elapsed_wall_s)
        aggregate.update({
            "configured_batch_size": int(batch_size),
            "warmup_request_count": len(warmup_batches),
            "warmups_excluded": True,
            "result_ids_by_uid": ids_by_uid,
        })
        configurations[str(batch_size)] = aggregate

    stats_after, _ = _http_json(url.rstrip("/") + "/stats", timeout=timeout)
    return {
        "url": url,
        "role": role,
        "workload": workload,
        "identity": identity,
        "health": health,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "batch_configurations": configurations,
    }


def compute_retrieval_agreement(
    exact_ids: Mapping[str, Sequence[int]],
    candidate_ids: Mapping[str, Sequence[int]],
    *,
    k: int = 3,
) -> dict:
    if k != 3:
        raise ValueError("Phase-6 agreement metrics are pre-specified for top-k=3")
    if set(exact_ids) != set(candidate_ids):
        raise ValueError("exact and candidate result-ID UID sets differ")
    if not exact_ids:
        raise ValueError("cannot compare no retrieval IDs")
    recalls = []
    top1 = 0
    full_topk = 0
    invalid = 0
    for uid in sorted(exact_ids):
        exact = list(exact_ids[uid])[:k]
        candidate = list(candidate_ids[uid])[:k]
        if (
            len(exact) != k
            or len(candidate) != k
            or len(set(exact)) != k
            or len(set(candidate)) != k
            or any(value < 0 for value in exact + candidate)
        ):
            invalid += 1
            recalls.append(0.0)
            continue
        recalls.append(len(set(exact) & set(candidate)) / k)
        top1 += int(exact[0] == candidate[0])
        full_topk += int(set(exact) == set(candidate))
    count = len(recalls)
    return {
        "query_count": count,
        "recall_at_3": statistics.fmean(recalls),
        "top1_agreement_rate": top1 / count,
        "full_top3_set_agreement_rate": full_topk / count,
        "invalid_or_missing_result_count": invalid,
    }


def _single_role_target(targets: Mapping, role: str):
    matches = [target for target in targets.values() if target.get("role") == role]
    if len(matches) > 1:
        raise ValueError(f"multiple targets declare the unique comparison role {role}")
    return matches[0] if matches else None


def _retrieval_identity(identity: Mapping, *, include_index: bool = True) -> tuple:
    values = (
        identity.get("model_fingerprint"),
        identity.get("retrieval_encode_batch_size"),
        identity.get("index_ntotal"),
        identity.get("index_dimension"),
        identity.get("metric_type"),
        identity.get("corpus_row_count"),
        identity.get("corpus_fingerprint"),
    )
    return values + ((identity.get("index_fingerprint"),) if include_index else ())


def _validate_selected_pair(report: Mapping, exact_target: Mapping, ann_target: Mapping) -> None:
    exact_identity = exact_target["identity"]
    ann_identity = ann_target["identity"]
    if _retrieval_identity(exact_identity, include_index=False) != _retrieval_identity(
        ann_identity, include_index=False
    ):
        raise ValueError("Flat and IVF-PQ servers do not share model/corpus/index geometry")
    if exact_identity["faiss_thread_count"] != ann_identity["faiss_thread_count"]:
        raise ValueError("Flat and IVF-PQ comparison requires the same FAISS thread count")
    selected = report["inputs"].get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("IVF-PQ HTTP comparison requires selected_candidate.json")
    if ann_identity.get("nprobe") != selected.get("nprobe"):
        raise ValueError("live IVF-PQ nprobe does not match selected_candidate.json")
    if ann_identity.get("faiss_thread_count") != selected.get("faiss_thread_count"):
        raise ValueError("live IVF-PQ thread count does not match selected_candidate.json")
    if ann_identity.get("index_fingerprint") != selected.get("index_sha256"):
        raise ValueError("live IVF-PQ index fingerprint does not match selected_candidate.json")
    if exact_identity.get("index_fingerprint") != selected.get("flat_index_sha256"):
        raise ValueError("live Flat index fingerprint does not match selected_candidate.json")
    for identity in (exact_identity, ann_identity):
        if identity.get("model_fingerprint") != selected.get("model_sha256"):
            raise ValueError("live E5 model fingerprint does not match selected_candidate.json")


def refresh_report_comparisons(report: dict) -> None:
    for workload, roles in (
        ("cold", ("flat_exact", "ivfpq_selected")),
        ("warm", ("warm_flat", "warm_ivfpq")),
    ):
        targets = report["workloads"][workload]
        exact_target = _single_role_target(targets, roles[0])
        candidate_target = _single_role_target(targets, roles[1])
        if exact_target is None or candidate_target is None:
            report["comparisons"].pop(workload, None)
            continue
        _validate_selected_pair(report, exact_target, candidate_target)
        batch_comparisons = {}
        shared_batches = set(exact_target["batch_configurations"]) & set(
            candidate_target["batch_configurations"]
        )
        for batch_size in sorted(shared_batches, key=int):
            exact_config = exact_target["batch_configurations"][batch_size]
            candidate_config = candidate_target["batch_configurations"][batch_size]
            batch_comparisons[batch_size] = compute_retrieval_agreement(
                exact_config["result_ids_by_uid"],
                candidate_config["result_ids_by_uid"],
                k=report["inputs"]["topk"],
            )
        report["comparisons"][workload] = {
            "exact_role": roles[0],
            "candidate_role": roles[1],
            "batch_configurations": batch_comparisons,
        }

    observed_threads = sorted({
        target["identity"]["faiss_thread_count"]
        for target in report["workloads"]["cold"].values()
        if target["role"] == "exact_sweep"
    })
    report["exact_thread_batch_sweep"] = {
        "required_thread_counts": list(REQUIRED_EXACT_THREADS),
        "observed_thread_counts": observed_threads,
        "missing_thread_counts": sorted(set(REQUIRED_EXACT_THREADS) - set(observed_threads)),
        "required_batch_sizes": list(report["inputs"]["batch_sizes"]),
        "warmups_excluded": True,
    }


def _load_selected_candidate(
    path: Path | None,
    queries_path: Path | None = None,
    manifest_path: Path | None = None,
):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_candidate.json has no selected_candidate object")
    nprobe = selected.get("nprobe")
    threads = selected.get("faiss_thread_count")
    bindings = payload.get("bindings", {})
    binding_sha = bindings.get("ivfpq_index", {}).get("sha256")
    flat_sha = bindings.get("flat_index", {}).get("sha256")
    selected_sha = selected.get("index_sha256")
    if (
        isinstance(nprobe, bool)
        or not isinstance(nprobe, int)
        or nprobe <= 0
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
        or not isinstance(binding_sha, str)
        or not binding_sha
        or not isinstance(flat_sha, str)
        or not flat_sha
        or selected_sha != binding_sha
    ):
        raise ValueError("selected_candidate.json has invalid nprobe/thread/index binding")
    calibration_queries_sha = bindings.get("calibration_queries", {}).get("sha256")
    calibration_manifest_sha = bindings.get("calibration_manifest", {}).get("sha256")
    if queries_path is not None and calibration_queries_sha != sha256_file(queries_path):
        raise ValueError("selected_candidate.json is bound to different calibration queries")
    if manifest_path is not None and calibration_manifest_sha != sha256_file(manifest_path):
        raise ValueError("selected_candidate.json is bound to a different calibration manifest")
    embedding_manifest_binding = bindings.get("query_embeddings_manifest", {})
    embedding_manifest_path = Path(
        embedding_manifest_binding.get("path", "")
    ).expanduser().resolve()
    if (
        not embedding_manifest_path.is_file()
        or embedding_manifest_binding.get("sha256") != sha256_file(embedding_manifest_path)
    ):
        raise ValueError("selected query-embedding manifest binding is missing or stale")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    model_sha = embedding_manifest.get("binding", {}).get("model", {}).get("sha256")
    if not isinstance(model_sha, str) or not model_sha:
        raise ValueError("selected query-embedding manifest has no model fingerprint")
    return {
        "candidate_selection_passed": payload.get("candidate_selection_passed"),
        "production_ready_for_agent_evaluation": payload.get(
            "production_ready_for_agent_evaluation"
        ),
        "nprobe": nprobe,
        "faiss_thread_count": threads,
        "index_sha256": binding_sha,
        "flat_index_sha256": flat_sha,
        "model_sha256": model_sha,
        "calibration_queries_sha256": calibration_queries_sha,
        "calibration_manifest_sha256": calibration_manifest_sha,
        "query_embeddings_manifest_sha256": embedding_manifest_binding.get("sha256"),
        "minimum_recall_at_3": payload.get("minimum_recall_at_3"),
        "recall_at_3": selected.get("recall_at_3"),
    }


def _new_report(
    queries_path: Path,
    manifest_path: Path,
    selected_candidate_path: Path | None,
    *,
    topk: int,
    batch_sizes: Sequence[int],
    repetitions: int,
    ordered_uids: Sequence[str] | None = None,
    warmup_requests: int | None = None,
    timeout_s: float | None = None,
) -> dict:
    selected_candidate = _load_selected_candidate(
        selected_candidate_path, queries_path, manifest_path
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "phase6_server_benchmark",
        "inputs": {
            "queries_path": str(queries_path),
            "queries_sha256": sha256_file(queries_path),
            "calibration_manifest_path": str(manifest_path),
            "calibration_manifest_sha256": sha256_file(manifest_path),
            "selected_candidate_path": (
                str(selected_candidate_path) if selected_candidate_path else None
            ),
            "selected_candidate_sha256": (
                sha256_file(selected_candidate_path) if selected_candidate_path else None
            ),
            "selected_candidate": selected_candidate,
            "topk": topk,
            "batch_sizes": list(batch_sizes),
            "repetitions": repetitions,
            "warmup_requests": warmup_requests,
            "timeout_s": timeout_s,
            "query_count": len(ordered_uids) if ordered_uids is not None else None,
            "ordered_uids": list(ordered_uids) if ordered_uids is not None else None,
        },
        "workloads": {"cold": {}, "warm": {}},
        "comparisons": {},
        "exact_thread_batch_sweep": {},
    }


def _validate_append_binding(existing: Mapping, expected: Mapping) -> None:
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("existing server benchmark has an unsupported schema")
    if existing.get("inputs") != expected.get("inputs"):
        raise ValueError("existing server benchmark is bound to different inputs/configuration")
    if not isinstance(existing.get("workloads"), Mapping):
        raise ValueError("existing server benchmark has no workload sections")


def _validate_target_complete(target: Mapping, inputs: Mapping) -> None:
    required_batches = {str(value) for value in inputs["batch_sizes"]}
    configurations = target.get("batch_configurations")
    if not isinstance(configurations, Mapping) or set(configurations) != required_batches:
        raise ValueError(f"{target.get('role')} target has incomplete batch configurations")
    expected_uids = set(inputs["ordered_uids"])
    for batch_size, configuration in configurations.items():
        if configuration.get("error_count") != 0:
            raise ValueError(
                f"{target.get('role')} batch {batch_size} contains request errors"
            )
        if set(configuration.get("result_ids_by_uid", {})) != expected_uids:
            raise ValueError(
                f"{target.get('role')} batch {batch_size} has incomplete calibration UIDs"
            )


def validate_complete_report(report: Mapping) -> dict:
    inputs = report.get("inputs", {})
    if inputs.get("topk") != 3:
        raise ValueError("Phase-6 server benchmark completion requires top-k=3")
    ordered_uids = inputs.get("ordered_uids")
    if (
        not isinstance(ordered_uids, list)
        or not ordered_uids
        or len(ordered_uids) != len(set(ordered_uids))
        or inputs.get("query_count") != len(ordered_uids)
    ):
        raise ValueError("server benchmark has an invalid calibration UID binding")
    cold = report.get("workloads", {}).get("cold", {})
    warm = report.get("workloads", {}).get("warm", {})
    exact = _single_role_target(cold, "flat_exact")
    ann = _single_role_target(cold, "ivfpq_selected")
    warm_exact = _single_role_target(warm, "warm_flat")
    warm_ann = _single_role_target(warm, "warm_ivfpq")
    if None in (exact, ann, warm_exact, warm_ann):
        raise ValueError("server benchmark lacks a cold or warm Flat/IVF-PQ target")
    for target in list(cold.values()) + list(warm.values()):
        _validate_target_complete(target, inputs)

    sweep_targets = [
        target for target in cold.values() if target.get("role") == "exact_sweep"
    ]
    targets_by_thread = {}
    for target in sweep_targets:
        thread_count = target["identity"]["faiss_thread_count"]
        if thread_count in targets_by_thread:
            raise ValueError(f"duplicate exact-sweep target for thread count {thread_count}")
        targets_by_thread[thread_count] = target
    missing_threads = sorted(set(REQUIRED_EXACT_THREADS) - set(targets_by_thread))
    if missing_threads:
        raise ValueError(f"exact thread sweep is incomplete; missing={missing_threads}")
    flat_identity = _retrieval_identity(exact["identity"])
    for target in sweep_targets:
        if _retrieval_identity(target["identity"]) != flat_identity:
            raise ValueError("exact thread-sweep targets do not use one Flat/model/corpus identity")

    _validate_selected_pair(report, exact, ann)
    _validate_selected_pair(report, warm_exact, warm_ann)
    for cold_target, warm_target in ((exact, warm_exact), (ann, warm_ann)):
        if _retrieval_identity(cold_target["identity"]) != _retrieval_identity(
            warm_target["identity"]
        ):
            raise ValueError("warm-cache server identity differs from its cold server")
        if (
            cold_target["identity"]["faiss_thread_count"]
            != warm_target["identity"]["faiss_thread_count"]
            or cold_target["identity"]["nprobe"] != warm_target["identity"]["nprobe"]
        ):
            raise ValueError("warm-cache server thread/nprobe differs from cold server")
        for batch_size in map(str, inputs["batch_sizes"]):
            cold_ids = cold_target["batch_configurations"][batch_size][
                "result_ids_by_uid"
            ]
            warm_ids = warm_target["batch_configurations"][batch_size][
                "result_ids_by_uid"
            ]
            if cold_ids != warm_ids:
                raise ValueError("cache-on retrieval IDs differ from cache-disabled IDs")
    if "cold" not in report.get("comparisons", {}) or "warm" not in report.get(
        "comparisons", {}
    ):
        raise ValueError("server benchmark paired ID comparisons are incomplete")
    return {
        "passed": True,
        "query_count": len(ordered_uids),
        "exact_thread_counts": sorted(targets_by_thread),
        "batch_sizes": list(inputs["batch_sizes"]),
        "cold_and_warm_isolated": True,
        "selected_candidate_bound": True,
    }


def parse_target_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("target must use NAME|ROLE|URL syntax")
    name, role, url = parts
    if role not in VALID_ROLES:
        raise ValueError(f"unsupported target role: {role!r}")
    return name, role, url


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--selected-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="NAME|ROLE|URL; repeat for live server targets",
    )
    parser.add_argument("--workload", choices=("cold", "warm"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.topk != 3:
        raise ValueError("Phase-6 server benchmarks require the pre-specified top-k=3")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(
        value <= 0 for value in args.batch_sizes
    ):
        raise ValueError("batch sizes must be unique positive integers")
    queries_path = args.queries.expanduser().resolve()
    manifest_path = args.calibration_manifest.expanduser().resolve()
    candidate_path = (
        args.selected_candidate.expanduser().resolve() if args.selected_candidate else None
    )
    output_path = args.output.expanduser().resolve()
    rows = load_calibration_queries(queries_path)
    expected = _new_report(
        queries_path,
        manifest_path,
        candidate_path,
        topk=args.topk,
        batch_sizes=args.batch_sizes,
        repetitions=args.repetitions,
        ordered_uids=[row["uid"] for row in rows],
        warmup_requests=args.warmup_requests,
        timeout_s=args.timeout,
    )
    if args.append and output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_append_binding(report, expected)
    elif output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {output_path}")
    else:
        report = expected

    for spec in args.target:
        name, role, url = parse_target_spec(spec)
        allowed_roles = {
            "cold": {"exact_sweep", "flat_exact", "ivfpq_selected"},
            "warm": {"warm_flat", "warm_ivfpq"},
        }[args.workload]
        if role not in allowed_roles:
            raise ValueError(f"role {role} is not valid for the {args.workload} workload")
        if name in report["workloads"][args.workload]:
            raise ValueError(f"target name already exists for {args.workload}: {name}")
        if role != "exact_sweep" and _single_role_target(
            report["workloads"][args.workload], role
        ) is not None:
            raise ValueError(f"comparison role already exists for {args.workload}: {role}")
        report["workloads"][args.workload][name] = benchmark_target(
            url,
            rows,
            role=role,
            workload=args.workload,
            batch_sizes=args.batch_sizes,
            topk=args.topk,
            repetitions=args.repetitions,
            warmup_requests=args.warmup_requests,
            timeout=args.timeout,
        )
        refresh_report_comparisons(report)
        _atomic_write_json(output_path, report)

    refresh_report_comparisons(report)
    if args.require_complete:
        report["completion_validation"] = validate_complete_report(report)
    else:
        report.pop("completion_validation", None)
    _atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
