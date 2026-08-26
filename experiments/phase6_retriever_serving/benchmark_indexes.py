#!/usr/bin/env python3
"""Benchmark exact Flat and IVF-PQ search on calibration-only E5 queries."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.phase4_benchmark.prepare_eval_data import sha256_file


SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_DIR = (
    "/workspace/searchr1-assets/datasets/phase6_retriever_calibration"
)
DEFAULT_FLAT_INDEX = "/workspace/searchr1-assets/wiki18/e5_Flat.index"
DEFAULT_IVFPQ_INDEX = (
    "/workspace/searchr1-assets/wiki18/"
    "e5_IVFPQ_nlist4096_m96_nbits8.index"
)
DEFAULT_IVFPQ_MANIFEST = (
    "/workspace/searchr1-assets/wiki18/"
    "e5_IVFPQ_nlist4096_m96_nbits8.manifest.json"
)
DEFAULT_E5_MODEL = "/workspace/searchr1-assets/models/e5-base-v2"
DEFAULT_OUTPUT_DIR = "/workspace/Search-R1/phase6_retriever_results"
DEFAULT_NPROBES = (4, 8, 16, 32, 64, 128)
DEFAULT_THREAD_COUNTS = (4, 8, 16)
FORBIDDEN_QUERY_KEYS = {
    "answer",
    "answers",
    "ground_truth",
    "reward_model",
    "target",
}


def _require_faiss():
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for index benchmarking; use the existing "
            "Retriever environment with faiss-cpu installed"
        ) from error


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def load_calibration_queries(queries_path, manifest_path):
    queries_path = Path(queries_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not queries_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("calibration queries and manifest must both exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output", {}).get("sha256") != sha256_file(queries_path):
        raise ValueError("calibration queries SHA256 does not match its manifest")
    records = []
    with queries_path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"calibration query row {position} is not an object")
            forbidden = FORBIDDEN_QUERY_KEYS.intersection(record)
            if forbidden:
                raise ValueError(
                    f"calibration query row {position} contains answer fields: "
                    f"{sorted(forbidden)}"
                )
            for key in ("uid", "data_source", "question", "question_sha256"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"calibration query row {position} has invalid {key}")
            if record["question_sha256"] != _question_sha256(record["question"]):
                raise ValueError(f"calibration query row {position} question hash mismatch")
            records.append(record)
    if len(records) != manifest.get("row_count"):
        raise ValueError("calibration query row count does not match its manifest")
    uids = [record["uid"] for record in records]
    hashes = [record["question_sha256"] for record in records]
    if len(uids) != len(set(uids)) or len(hashes) != len(set(hashes)):
        raise ValueError("calibration queries contain duplicate UIDs or questions")
    if uids != manifest.get("selected_uids"):
        raise ValueError("calibration query UID order does not match its manifest")
    if hashes != manifest.get("question_hashes"):
        raise ValueError("calibration question hashes do not match their manifest")
    if not manifest.get("leakage_overlap_audit", {}).get("passed"):
        raise ValueError("calibration held-out overlap audit did not pass")
    if manifest.get("leakage_overlap_audit", {}).get("overlapping_uids"):
        raise ValueError("calibration manifest declares held-out UID overlap")
    return records, manifest


def fingerprint_path(path):
    """Content-hash a local model file/tree without resolving remote assets."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {path}")
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    if not files:
        raise ValueError(f"model path has no files: {path}")
    from experiments.phase6_retriever_serving.optimized_retrieval_server import (
        fingerprint_asset,
    )

    return {
        "path": str(path),
        "sha256": fingerprint_asset(path),
        "file_count": len(files),
        "total_size_bytes": sum(item.stat().st_size for item in files),
    }


def _embedding_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _load_embedding_artifact(path: Path, manifest_path: Path, expected_binding):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("binding") != expected_binding:
        raise ValueError("existing query embeddings are bound to different inputs")
    if manifest.get("artifact", {}).get("sha256") != sha256_file(path):
        raise ValueError("query embedding artifact SHA256 does not match its manifest")
    with np.load(path, allow_pickle=False) as artifact:
        embeddings = np.asarray(artifact["embeddings"], dtype=np.float32, order="C")
        uids = artifact["uids"].tolist()
        question_hashes = artifact["question_hashes"].tolist()
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("query embedding artifact contains invalid embeddings")
    if uids != expected_binding["ordered_uids"]:
        raise ValueError("query embedding UID order does not match its binding")
    if question_hashes != expected_binding["ordered_question_hashes"]:
        raise ValueError("query embedding question hashes do not match its binding")
    if embeddings.shape != tuple(manifest.get("embedding_shape", [])):
        raise ValueError("query embedding shape does not match its manifest")
    return embeddings, manifest


def encode_calibration_queries_once(
    records,
    queries_path,
    queries_manifest_path,
    model_path,
    artifact_path,
    encode_batch_size=32,
    query_max_length=256,
    overwrite=False,
    encoder_factory=None,
    model_fingerprint=None,
):
    """Encode every ordered query once and persist a strict hash-bound artifact."""

    if encode_batch_size <= 0 or query_max_length <= 0:
        raise ValueError("encode batch size and query max length must be positive")
    queries_path = Path(queries_path).expanduser().resolve()
    queries_manifest_path = Path(queries_manifest_path).expanduser().resolve()
    artifact_path = Path(artifact_path).expanduser().resolve()
    manifest_path = _embedding_manifest_path(artifact_path)
    if model_fingerprint is None:
        model_fingerprint = fingerprint_path(model_path)
    binding = {
        "queries_path": str(queries_path),
        "queries_sha256": sha256_file(queries_path),
        "queries_manifest_path": str(queries_manifest_path),
        "queries_manifest_sha256": sha256_file(queries_manifest_path),
        "ordered_uids": [record["uid"] for record in records],
        "ordered_question_hashes": [
            record["question_sha256"] for record in records
        ],
        "model": model_fingerprint,
        "encoder": {
            "model_name": "e5",
            "pooling_method": "mean",
            "query_prefix_semantics": "existing_Encoder_e5_query_prefix",
            "query_max_length": query_max_length,
            "dtype": "float32",
            "device": "cpu",
            "use_fp16": False,
        },
    }
    if artifact_path.exists() or manifest_path.exists():
        if artifact_path.is_file() and manifest_path.is_file() and not overwrite:
            return _load_embedding_artifact(artifact_path, manifest_path, binding)
        if not overwrite:
            raise FileExistsError(
                "partial/stale query embedding artifact exists; pass overwrite=True"
            )
        for path in (artifact_path, manifest_path):
            if path.exists():
                path.unlink()
    if encoder_factory is None:
        from search_r1.search.retrieval_server import Encoder

        encoder_factory = Encoder
    encoder = encoder_factory(
        model_name="e5",
        model_path=str(Path(model_path).expanduser().resolve()),
        pooling_method="mean",
        max_length=query_max_length,
        use_fp16=False,
        device="cpu",
    )
    chunks = []
    encoded_count = 0
    questions = [record["question"] for record in records]
    for start in range(0, len(questions), encode_batch_size):
        batch = questions[start : start + encode_batch_size]
        encoded = np.asarray(encoder.encode(batch, is_query=True), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError("E5 encoder returned invalid query-embedding cardinality")
        if not np.isfinite(encoded).all():
            raise ValueError("E5 encoder returned non-finite query embeddings")
        chunks.append(np.ascontiguousarray(encoded, dtype=np.float32))
        encoded_count += len(batch)
    if encoded_count != len(records):
        raise RuntimeError("not every calibration query was encoded exactly once")
    embeddings = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(artifact_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        uids=np.asarray(binding["ordered_uids"]),
        question_hashes=np.asarray(binding["ordered_question_hashes"]),
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, artifact_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "binding": binding,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "query_count": len(records),
        "each_query_encoded_once": True,
        "artifact": {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return embeddings, manifest


def _latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("latency samples must be a nonempty finite vector")
    if (values < 0).any():
        raise ValueError("latency samples must be nonnegative")
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def process_rss_bytes():
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        for line in proc_status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def benchmark_index_config(
    index,
    embeddings,
    k=3,
    faiss_thread_count=4,
    nprobe=None,
    warmup_query_count=8,
):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != int(index.d):
        raise ValueError("query embeddings do not match index dimension")
    if len(embeddings) == 0 or k <= 0 or faiss_thread_count <= 0:
        raise ValueError("queries, k, and FAISS thread count must be positive")
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must be nonnegative")
    faiss = _require_faiss()
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(faiss_thread_count)
        if nprobe is not None:
            if nprobe <= 0:
                raise ValueError("nprobe must be positive")
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(nprobe))
        for position in range(min(warmup_query_count, len(embeddings))):
            index.search(embeddings[position : position + 1], k)
        ids = []
        latencies = []
        for position in range(len(embeddings)):
            started = time.perf_counter()
            _query_scores, query_ids = index.search(
                embeddings[position : position + 1], k
            )
            latencies.append(time.perf_counter() - started)
            ids.append(query_ids[0].tolist())
    finally:
        faiss.omp_set_num_threads(previous_threads)
    total_search_s = sum(latencies)
    return {
        "faiss_thread_count": int(faiss_thread_count),
        "nprobe": int(nprobe) if nprobe is not None else None,
        "query_count": len(embeddings),
        "warmup_query_count": min(warmup_query_count, len(embeddings)),
        "warmups_excluded": True,
        "latency_unit": "one_query_index.search_call",
        "pure_search_latency_s": _latency_summary(latencies),
        "pure_search_total_s": total_search_s,
        "queries_per_second": len(embeddings) / total_search_s,
        "process_rss_bytes": process_rss_bytes(),
        "result_ids": ids,
    }


def compute_retrieval_agreement(exact_ids, candidate_ids, ntotal, k=3):
    exact = np.asarray(exact_ids, dtype=np.int64)
    candidate = np.asarray(candidate_ids, dtype=np.int64)
    if exact.ndim != 2 or candidate.shape != exact.shape or exact.shape[1] < k:
        raise ValueError("exact and candidate IDs must have identical N x >=k shape")
    exact = exact[:, :k]
    candidate = candidate[:, :k]
    if ((exact < 0) | (exact >= ntotal)).any():
        raise ValueError("exact Flat reference contains invalid result IDs")
    recall_at_1 = []
    recall_at_3 = []
    full_agreement = []
    invalid_rows = 0
    invalid_values = 0
    for reference, approximate in zip(exact, candidate):
        invalid_mask = (approximate < 0) | (approximate >= ntotal)
        invalid_values += int(invalid_mask.sum())
        approximate_valid = [int(value) for value in approximate if 0 <= value < ntotal]
        if invalid_mask.any() or len(set(approximate_valid)) != k:
            invalid_rows += 1
        reference_set = set(int(value) for value in reference)
        approximate_set = set(approximate_valid)
        recall_at_1.append(int(reference[0] == approximate[0]))
        recall_at_3.append(len(reference_set.intersection(approximate_set)) / k)
        full_agreement.append(
            int(len(approximate_set) == k and approximate_set == reference_set)
        )
    return {
        "recall_at_1": float(np.mean(recall_at_1)),
        "recall_at_3": float(np.mean(recall_at_3)),
        "top1_agreement_rate": float(np.mean(recall_at_1)),
        "full_top3_set_agreement_rate": float(np.mean(full_agreement)),
        "invalid_or_missing_result_count": invalid_rows,
        "invalid_or_missing_id_count": invalid_values,
        "recall_at_3_definition": (
            "mean per-query fraction of exact Flat top-3 IDs present in "
            "candidate top-3"
        ),
    }


def _selection_latency(record, key):
    value = record.get("pure_search_latency_s", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate has invalid {key} latency")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"candidate has invalid {key} latency")
    return float(value)


def _candidate_sort_key(record):
    return (
        _selection_latency(record, "p95"),
        _selection_latency(record, "mean"),
        int(record["nprobe"]),
        int(record["faiss_thread_count"]),
    )


def _valid_selection_records(records):
    valid = []
    for record in records:
        recall = record.get("recall_at_3")
        if isinstance(recall, bool) or not isinstance(recall, (int, float)):
            continue
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            continue
        if record.get("invalid_or_missing_result_count") != 0:
            continue
        try:
            _candidate_sort_key(record)
        except (KeyError, TypeError, ValueError):
            continue
        valid.append(record)
    return valid


def _pareto_front(records):
    front = []
    for record in records:
        dominated = False
        for other in records:
            if other is record:
                continue
            no_worse = (
                other["recall_at_3"] >= record["recall_at_3"]
                and _selection_latency(other, "p95")
                <= _selection_latency(record, "p95")
                and _selection_latency(other, "mean")
                <= _selection_latency(record, "mean")
            )
            strictly_better = (
                other["recall_at_3"] > record["recall_at_3"]
                or _selection_latency(other, "p95")
                < _selection_latency(record, "p95")
                or _selection_latency(other, "mean")
                < _selection_latency(record, "mean")
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(record)
    return front


def _public_candidate(record):
    return {
        key: value
        for key, value in record.items()
        if key not in ("result_ids", "result_scores")
    }


def select_candidate(records, min_recall_at_3=0.95):
    """Apply the pre-specified calibration-only threshold and tie breakers."""

    if not math.isfinite(min_recall_at_3) or not 0 <= min_recall_at_3 <= 1:
        raise ValueError("min_recall_at_3 must be finite and in [0, 1]")
    valid = _valid_selection_records(records)
    if not valid:
        raise ValueError("no valid IVF-PQ benchmark configuration is selectable")
    passing = [
        record for record in valid if record["recall_at_3"] >= min_recall_at_3
    ]
    if passing:
        selected = min(passing, key=_candidate_sort_key)
        passed = True
        fallback_reason = None
    else:
        front = _pareto_front(valid)
        selected = min(
            front,
            key=lambda record: (
                -record["recall_at_3"],
                *_candidate_sort_key(record),
            ),
        )
        passed = False
        fallback_reason = (
            "no calibration configuration met the minimum Recall@3; the "
            "reported candidate is not production-ready and downstream Agent "
            "evaluation requires explicit user override"
        )
    return {
        "candidate_selection_passed": passed,
        "production_ready_for_agent_evaluation": passed,
        "minimum_recall_at_3": min_recall_at_3,
        "selection_rule": [
            "require recall_at_3 >= minimum",
            "lowest p95 pure-search latency",
            "lowest mean pure-search latency",
            "lowest nprobe",
            "lowest FAISS thread count",
        ],
        "selected_candidate": _public_candidate(selected),
        "fallback_reason": fallback_reason,
        "selection_uses_qa_ground_truth": False,
    }


def benchmark_indexes(
    queries_path,
    queries_manifest_path,
    model_path,
    embedding_artifact_path,
    flat_index_path,
    ivfpq_index_path,
    output_dir,
    ivfpq_manifest_path=None,
    nprobes=DEFAULT_NPROBES,
    thread_counts=DEFAULT_THREAD_COUNTS,
    min_recall_at_3=0.95,
    encode_batch_size=32,
    query_max_length=256,
    warmup_query_count=8,
    overwrite_embeddings=False,
):
    records, calibration_manifest = load_calibration_queries(
        queries_path, queries_manifest_path
    )
    nprobes = tuple(int(value) for value in nprobes)
    thread_counts = tuple(int(value) for value in thread_counts)
    if (
        not nprobes
        or not thread_counts
        or any(value <= 0 for value in nprobes + thread_counts)
        or len(nprobes) != len(set(nprobes))
        or len(thread_counts) != len(set(thread_counts))
    ):
        raise ValueError("nprobe and thread configurations must be positive and unique")
    embeddings, embedding_manifest = encode_calibration_queries_once(
        records,
        queries_path,
        queries_manifest_path,
        model_path,
        embedding_artifact_path,
        encode_batch_size=encode_batch_size,
        query_max_length=query_max_length,
        overwrite=overwrite_embeddings,
    )
    faiss = _require_faiss()
    flat_path = Path(flat_index_path).expanduser().resolve()
    ivfpq_path = Path(ivfpq_index_path).expanduser().resolve()
    flat_sha256 = sha256_file(flat_path)
    ivfpq_sha256 = sha256_file(ivfpq_path)
    flat_size_bytes = flat_path.stat().st_size
    ivfpq_size_bytes = ivfpq_path.stat().st_size
    flat = faiss.read_index(str(flat_path))
    if int(flat.d) != embeddings.shape[1]:
        raise ValueError("Flat index dimension must match query embeddings")
    if not isinstance(flat, faiss.IndexFlat):
        raise ValueError("exact reference must be a Flat index")
    expected_dimension = int(flat.d)
    expected_ntotal = int(flat.ntotal)
    expected_metric_type = int(flat.metric_type)
    if expected_ntotal < 3:
        raise ValueError("top-3 benchmarking requires indexes with at least three vectors")

    flat_results = []
    exact_ids = None
    for thread_count in thread_counts:
        result = benchmark_index_config(
            flat,
            embeddings,
            k=3,
            faiss_thread_count=int(thread_count),
            warmup_query_count=warmup_query_count,
        )
        result.update({
            "index_backend": "flat",
            "index_path": str(flat_path),
            "index_sha256": flat_sha256,
            "index_file_size_bytes": flat_size_bytes,
        })
        if exact_ids is None:
            exact_ids = result["result_ids"]
        elif result["result_ids"] != exact_ids:
            raise RuntimeError("exact Flat result IDs changed across thread counts")
        flat_results.append(result)

    del flat
    gc.collect()
    ivfpq = faiss.read_index(str(ivfpq_path))
    if int(ivfpq.d) != expected_dimension:
        raise ValueError("Flat/IVF-PQ index dimensions must match")
    if int(ivfpq.ntotal) != expected_ntotal:
        raise ValueError("Flat and IVF-PQ ntotal must match")
    if int(ivfpq.metric_type) != expected_metric_type:
        raise ValueError("Flat and IVF-PQ metrics must match")
    if not isinstance(ivfpq, faiss.IndexIVFPQ):
        raise ValueError("candidate index must be an IndexIVFPQ")
    if any(value > int(ivfpq.nlist) for value in nprobes):
        raise ValueError("nprobe cannot exceed the IVF-PQ nlist")

    ivfpq_results = []
    for nprobe in nprobes:
        for thread_count in thread_counts:
            result = benchmark_index_config(
                ivfpq,
                embeddings,
                k=3,
                faiss_thread_count=int(thread_count),
                nprobe=int(nprobe),
                warmup_query_count=warmup_query_count,
            )
            result.update(
                compute_retrieval_agreement(
                    exact_ids, result["result_ids"], expected_ntotal, k=3
                )
            )
            result.update({
                "index_backend": "ivfpq",
                "index_path": str(ivfpq_path),
                "index_sha256": ivfpq_sha256,
                "index_file_size_bytes": ivfpq_size_bytes,
            })
            ivfpq_results.append(result)
    selection = select_candidate(ivfpq_results, min_recall_at_3)

    queries_path = Path(queries_path).expanduser().resolve()
    queries_manifest_path = Path(queries_manifest_path).expanduser().resolve()
    embedding_artifact_path = Path(embedding_artifact_path).expanduser().resolve()
    embedding_manifest_path = _embedding_manifest_path(embedding_artifact_path)
    bindings = {
        "calibration_queries": {
            "path": str(queries_path), "sha256": sha256_file(queries_path)
        },
        "calibration_manifest": {
            "path": str(queries_manifest_path),
            "sha256": sha256_file(queries_manifest_path),
        },
        "query_embeddings": {
            "path": str(embedding_artifact_path),
            "sha256": sha256_file(embedding_artifact_path),
        },
        "query_embeddings_manifest": {
            "path": str(embedding_manifest_path),
            "sha256": sha256_file(embedding_manifest_path),
        },
        "flat_index": {"path": str(flat_path), "sha256": flat_sha256},
        "ivfpq_index": {
            "path": str(ivfpq_path), "sha256": ivfpq_sha256
        },
        "ivfpq_build_manifest": None,
    }
    if ivfpq_manifest_path is not None:
        ivfpq_manifest_path = Path(ivfpq_manifest_path).expanduser().resolve()
        build_manifest = json.loads(ivfpq_manifest_path.read_text(encoding="utf-8"))
        if build_manifest.get("source", {}).get("sha256") != flat_sha256:
            raise ValueError("IVF-PQ build manifest is bound to a different Flat index")
        if build_manifest.get("output", {}).get("sha256") != ivfpq_sha256:
            raise ValueError("IVF-PQ build manifest is bound to a different IVF-PQ index")
        bindings["ivfpq_build_manifest"] = {
            "path": str(ivfpq_manifest_path),
            "sha256": sha256_file(ivfpq_manifest_path),
        }
    benchmark = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": "phase6_calibration_only_index_search",
        "bindings": bindings,
        "query_count": len(records),
        "query_embeddings_encoded_once": embedding_manifest[
            "each_query_encoded_once"
        ],
        "topk": 3,
        "index_geometry": {
            "dimension": expected_dimension,
            "ntotal": expected_ntotal,
            "metric_type": expected_metric_type,
            "metric": (
                "inner_product"
                if expected_metric_type == int(faiss.METRIC_INNER_PRODUCT)
                else "l2"
                if expected_metric_type == int(faiss.METRIC_L2)
                else f"faiss_metric_{expected_metric_type}"
            ),
        },
        "warmup_query_count": min(warmup_query_count, len(records)),
        "warmups_excluded": True,
        "faiss_thread_counts": [int(value) for value in thread_counts],
        "ivfpq_nprobes": [int(value) for value in nprobes],
        "flat_results": flat_results,
        "exact_flat_top3_ids": exact_ids,
        "ivfpq_results": ivfpq_results,
        "candidate_selection": selection,
        "selection_uses_qa_ground_truth": False,
        "calibration_manifest_seed": calibration_manifest.get("seed"),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    benchmark_path = output_dir / "index_benchmark.json"
    selected_path = output_dir / "selected_candidate.json"
    _write_json_atomic(benchmark_path, benchmark)
    selected = {
        "schema_version": SCHEMA_VERSION,
        **selection,
        "bindings": {
            **bindings,
            "index_benchmark": {
                "path": str(benchmark_path),
                "sha256": sha256_file(benchmark_path),
            },
        },
    }
    _write_json_atomic(selected_path, selected)
    return {
        "index_benchmark_path": str(benchmark_path),
        "index_benchmark_sha256": sha256_file(benchmark_path),
        "selected_candidate_path": str(selected_path),
        "selected_candidate_sha256": sha256_file(selected_path),
        "candidate_selection_passed": selection["candidate_selection_passed"],
    }


def _comma_separated_ints(value: str):
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", default=f"{DEFAULT_CALIBRATION_DIR}/queries.jsonl"
    )
    parser.add_argument(
        "--queries-manifest", default=f"{DEFAULT_CALIBRATION_DIR}/manifest.json"
    )
    parser.add_argument("--model-path", default=DEFAULT_E5_MODEL)
    parser.add_argument(
        "--embedding-artifact",
        default=f"{DEFAULT_OUTPUT_DIR}/calibration_query_embeddings.npz",
    )
    parser.add_argument("--flat-index", default=DEFAULT_FLAT_INDEX)
    parser.add_argument("--ivfpq-index", default=DEFAULT_IVFPQ_INDEX)
    parser.add_argument("--ivfpq-manifest", default=DEFAULT_IVFPQ_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--nprobes", type=_comma_separated_ints, default=DEFAULT_NPROBES
    )
    parser.add_argument(
        "--thread-counts",
        type=_comma_separated_ints,
        default=DEFAULT_THREAD_COUNTS,
    )
    parser.add_argument(
        "--min-recall-at-3",
        type=float,
        default=float(os.environ.get("PHASE6_MIN_RECALL_AT_3", "0.95")),
    )
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--query-max-length", type=int, default=256)
    parser.add_argument("--warmup-query-count", type=int, default=8)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    result = benchmark_indexes(
        queries_path=args.queries,
        queries_manifest_path=args.queries_manifest,
        model_path=args.model_path,
        embedding_artifact_path=args.embedding_artifact,
        flat_index_path=args.flat_index,
        ivfpq_index_path=args.ivfpq_index,
        output_dir=args.output_dir,
        ivfpq_manifest_path=args.ivfpq_manifest,
        nprobes=args.nprobes,
        thread_counts=args.thread_counts,
        min_recall_at_3=args.min_recall_at_3,
        encode_batch_size=args.encode_batch_size,
        query_max_length=args.query_max_length,
        warmup_query_count=args.warmup_query_count,
        overwrite_embeddings=args.overwrite_embeddings,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
