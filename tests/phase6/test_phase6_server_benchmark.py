import json
import math
from pathlib import Path

import pytest

from experiments.phase6_retriever_serving import benchmark_server as bench


def _rows(count=5):
    return [
        {"uid": f"nq:{index}", "question": f"question {index}"}
        for index in range(count)
    ]


def _health(*, backend="flat", threads=8, cache=False, encode_batch_size=32):
    return {
        "ready": True,
        "index_backend": backend,
        "faiss_thread_count": threads,
        "retrieval_encode_batch_size": encode_batch_size,
        "nprobe": None if backend == "flat" else 32,
        "cache_configuration": {
            "result": {"enabled": cache, "capacity": 128 if cache else 0},
            "embedding": {"enabled": cache, "capacity": 128 if cache else 0},
        },
        "index_ntotal": 100,
        "index_dimension": 8,
        "metric_type": "inner_product",
        "corpus_row_count": 100,
        "index_fingerprint": "flat-sha" if backend == "flat" else "ann-sha",
        "model_fingerprint": "model-sha",
        "corpus_fingerprint": "corpus-sha",
        "process_rss_bytes": 1234,
    }


def _retrieve_payload(request_payload, *, ann=False, cache=False, threads=8):
    result = []
    ids = []
    for query in request_payload["queries"]:
        number = int(query.rsplit(" ", 1)[1])
        exact_ids = [number, number + 10, number + 20]
        query_ids = [number, number + 10, 99] if ann else exact_ids
        ids.append(query_ids)
        result.append([
            {"document": {"contents": f"doc-{value}"}, "score": 1.0}
            for value in query_ids
        ])
    timings = {key: 0.001 for key in bench.SERVER_TIMING_KEYS}
    return {
        "result": result,
        "metrics": {
            **timings,
            "result_ids": ids,
            "query_count": len(request_payload["queries"]),
            "topk": request_payload["topk"],
            "cache_hit_count": len(request_payload["queries"]) if cache else 0,
            "cache_miss_count": 0,
            "embedding_cache_hit_count": 0,
            "embedding_cache_miss_count": 0,
            "index_backend": "ivfpq" if ann else "flat",
            "nprobe": 32 if ann else None,
            "faiss_thread_count": threads,
        },
    }


def test_exact_batches_use_native_exact_sizes_and_cover_every_query():
    batches = bench.exact_batches(_rows(5), 4)
    assert [len(batch) for batch in batches] == [4, 4]
    assert {row["uid"] for batch in batches for row in batch} == {
        f"nq:{index}" for index in range(5)
    }


def test_recall_and_id_agreement_definitions():
    exact = {"a": [1, 2, 3], "b": [4, 5, 6]}
    candidate = {"a": [1, 2, 9], "b": [5, 4, 6]}
    result = bench.compute_retrieval_agreement(exact, candidate)
    assert result["recall_at_3"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["top1_agreement_rate"] == 0.5
    assert result["full_top3_set_agreement_rate"] == 0.5
    assert result["invalid_or_missing_result_count"] == 0


def test_benchmark_excludes_warmups_and_aggregates_batch_qps(monkeypatch):
    calls = []

    def fake_http(url, *, payload=None, timeout=120.0):
        calls.append((url, payload))
        if url.endswith("/healthz"):
            return _health(), 0.01
        if url.endswith("/stats"):
            return {"request_count": len(calls)}, 0.01
        return _retrieve_payload(payload), 0.02

    monkeypatch.setattr(bench, "_http_json", fake_http)
    result = bench.benchmark_target(
        "http://retriever",
        _rows(5),
        role="flat_exact",
        workload="cold",
        batch_sizes=(4,),
        repetitions=2,
        warmup_requests=3,
    )
    config = result["batch_configurations"]["4"]
    assert config["warmup_request_count"] == 3
    assert config["warmups_excluded"] is True
    assert config["request_count"] == 4  # ceil(5 / 4) batches * two repetitions
    assert config["query_count"] == 16
    assert config["error_count"] == 0
    assert math.isfinite(config["queries_per_second"])
    assert config["server_stage_latency"]["faiss_search_s"]["mean_s"] == 0.001
    assert config["cache"]["result_cache_hit_rate"] is None
    retrieve_calls = [payload for url, payload in calls if url.endswith("/retrieve")]
    assert len(retrieve_calls) == 3 + 4


def test_cold_and_warm_workloads_are_strictly_isolated(monkeypatch):
    monkeypatch.setattr(
        bench,
        "_http_json",
        lambda *args, **kwargs: (_health(cache=True), 0.0),
    )
    with pytest.raises(ValueError, match="caches disabled"):
        bench.benchmark_target(
            "http://retriever",
            _rows(),
            role="flat_exact",
            workload="cold",
            batch_sizes=(1,),
        )

    monkeypatch.setattr(
        bench,
        "_http_json",
        lambda *args, **kwargs: (_health(cache=False), 0.0),
    )
    with pytest.raises(ValueError, match="cache-enabled"):
        bench.benchmark_target(
            "http://retriever",
            _rows(),
            role="warm_flat",
            workload="warm",
            batch_sizes=(1,),
        )


def test_warm_cache_primes_every_query_and_excludes_that_pass(monkeypatch):
    calls = []

    def fake_http(url, *, payload=None, timeout=120.0):
        if url.endswith("/healthz"):
            return _health(cache=True), 0.01
        if url.endswith("/stats"):
            return {}, 0.01
        calls.append(payload)
        return _retrieve_payload(payload, cache=True), 0.01

    monkeypatch.setattr(bench, "_http_json", fake_http)
    result = bench.benchmark_target(
        "http://retriever",
        _rows(5),
        role="warm_flat",
        workload="warm",
        batch_sizes=(4,),
        repetitions=1,
    )
    config = result["batch_configurations"]["4"]
    assert config["warmup_request_count"] == 2
    assert config["request_count"] == 2
    assert len(calls) == 4
    assert config["cache"]["result_cache_hit_rate"] == 1.0


def test_report_refresh_keeps_workloads_separate_and_tracks_thread_coverage():
    report = {
        "inputs": {
            "topk": 3,
            "batch_sizes": [1, 4, 8, 16, 32],
            "selected_candidate": {
                "nprobe": 32,
                "faiss_thread_count": 8,
                "index_sha256": "ann-sha",
                "flat_index_sha256": "flat-sha",
                "model_sha256": "model-sha",
            },
        },
        "workloads": {"cold": {}, "warm": {}},
        "comparisons": {},
    }
    exact_config = {"result_ids_by_uid": {"u": [1, 2, 3]}}
    ann_config = {"result_ids_by_uid": {"u": [1, 2, 9]}}
    report["workloads"]["cold"] = {
        "thread1": {
            "role": "exact_sweep",
            "identity": {**bench._validate_health(_health(threads=1), "exact_sweep", "cold")},
            "batch_configurations": {"1": exact_config},
        },
        "exact": {
            "role": "flat_exact",
            "identity": bench._validate_health(_health(), "flat_exact", "cold"),
            "batch_configurations": {"1": exact_config},
        },
        "ann": {
            "role": "ivfpq_selected",
            "identity": bench._validate_health(
                _health(backend="ivfpq"), "ivfpq_selected", "cold"
            ),
            "batch_configurations": {"1": ann_config},
        },
    }
    bench.refresh_report_comparisons(report)
    assert report["comparisons"]["cold"]["batch_configurations"]["1"][
        "recall_at_3"
    ] == pytest.approx(2 / 3)
    assert "warm" not in report["comparisons"]
    assert report["exact_thread_batch_sweep"]["observed_thread_counts"] == [1]
    assert report["exact_thread_batch_sweep"]["missing_thread_counts"] == [4, 8, 16]


def test_calibration_loader_rejects_answer_targets(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"uid": "nq:1", "question": "q", "answer": "leak"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains a QA target"):
        bench.load_calibration_queries(path)


def test_append_binding_rejects_changed_workload_configuration(tmp_path):
    queries = tmp_path / "queries.jsonl"
    manifest = tmp_path / "manifest.json"
    queries.write_text(json.dumps({"uid": "nq:1", "question": "q"}) + "\n")
    manifest.write_text("{}")
    first = bench._new_report(
        queries.resolve(), manifest.resolve(), None, topk=3, batch_sizes=(1, 4), repetitions=1
    )
    changed = bench._new_report(
        queries.resolve(), manifest.resolve(), None, topk=3, batch_sizes=(1, 8), repetitions=1
    )
    with pytest.raises(ValueError, match="different inputs/configuration"):
        bench._validate_append_binding(first, changed)

    first["inputs"]["warmup_requests"] = 2
    changed = json.loads(json.dumps(first))
    changed["inputs"]["warmup_requests"] = 3
    with pytest.raises(ValueError, match="different inputs/configuration"):
        bench._validate_append_binding(first, changed)


def _target(role, health, workload, uids, *, candidate=False):
    ids = {
        uid: [index, index + 10, 99 if candidate else index + 20]
        for index, uid in enumerate(uids)
    }
    configurations = {
        str(batch_size): {
            "error_count": 0,
            "result_ids_by_uid": ids,
        }
        for batch_size in bench.DEFAULT_BATCH_SIZES
    }
    return {
        "role": role,
        "identity": bench._validate_health(health, role, workload),
        "batch_configurations": configurations,
    }


def _complete_report():
    uids = ["nq:0", "hotpotqa:1"]
    report = {
        "inputs": {
            "topk": 3,
            "batch_sizes": list(bench.DEFAULT_BATCH_SIZES),
            "ordered_uids": uids,
            "query_count": len(uids),
            "selected_candidate": {
                "nprobe": 32,
                "faiss_thread_count": 8,
                "index_sha256": "ann-sha",
                "flat_index_sha256": "flat-sha",
                "model_sha256": "model-sha",
            },
        },
        "workloads": {"cold": {}, "warm": {}},
        "comparisons": {},
    }
    for threads in bench.REQUIRED_EXACT_THREADS:
        report["workloads"]["cold"][f"sweep-{threads}"] = _target(
            "exact_sweep", _health(threads=threads), "cold", uids
        )
    report["workloads"]["cold"]["exact"] = _target(
        "flat_exact", _health(), "cold", uids
    )
    report["workloads"]["cold"]["ann"] = _target(
        "ivfpq_selected", _health(backend="ivfpq"), "cold", uids, candidate=True
    )
    report["workloads"]["warm"]["exact"] = _target(
        "warm_flat", _health(cache=True), "warm", uids
    )
    report["workloads"]["warm"]["ann"] = _target(
        "warm_ivfpq",
        _health(backend="ivfpq", cache=True),
        "warm",
        uids,
        candidate=True,
    )
    bench.refresh_report_comparisons(report)
    return report


def test_complete_report_requires_full_fair_hash_bound_evidence():
    result = bench.validate_complete_report(_complete_report())
    assert result == {
        "passed": True,
        "query_count": 2,
        "exact_thread_counts": [1, 4, 8, 16],
        "batch_sizes": [1, 4, 8, 16, 32],
        "cold_and_warm_isolated": True,
        "selected_candidate_bound": True,
    }


def test_completion_rejects_partial_errors_even_with_matching_uid_sets():
    report = _complete_report()
    report["workloads"]["cold"]["ann"]["batch_configurations"]["4"][
        "error_count"
    ] = 1
    with pytest.raises(ValueError, match="contains request errors"):
        bench.validate_complete_report(report)


def test_comparison_rejects_thread_or_selected_candidate_mismatch():
    report = _complete_report()
    report["workloads"]["cold"]["ann"]["identity"]["faiss_thread_count"] = 4
    with pytest.raises(ValueError, match="same FAISS thread count"):
        bench.refresh_report_comparisons(report)

    report = _complete_report()
    report["workloads"]["cold"]["ann"]["identity"]["nprobe"] = 64
    with pytest.raises(ValueError, match="does not match selected_candidate"):
        bench.refresh_report_comparisons(report)


def test_duplicate_comparison_roles_are_rejected():
    report = _complete_report()
    report["workloads"]["cold"]["another-exact"] = report["workloads"]["cold"][
        "exact"
    ]
    with pytest.raises(ValueError, match="multiple targets"):
        bench.refresh_report_comparisons(report)


def test_completion_rejects_unrelated_exact_sweep_index():
    report = _complete_report()
    report["workloads"]["cold"]["sweep-4"]["identity"][
        "index_fingerprint"
    ] = "different-flat"
    with pytest.raises(ValueError, match="one Flat/model/corpus identity"):
        bench.validate_complete_report(report)


def test_completion_rejects_exact_ann_encode_batch_mismatch():
    report = _complete_report()
    report["workloads"]["cold"]["ann"]["identity"][
        "retrieval_encode_batch_size"
    ] = 16
    with pytest.raises(ValueError, match="model/corpus/index geometry"):
        bench.validate_complete_report(report)


def test_completion_rejects_cold_warm_encode_batch_mismatch():
    report = _complete_report()
    for name in ("exact", "ann"):
        report["workloads"]["warm"][name]["identity"][
            "retrieval_encode_batch_size"
        ] = 16
    with pytest.raises(ValueError, match="warm-cache server identity"):
        bench.validate_complete_report(report)


def test_completion_rejects_exact_sweep_encode_batch_mismatch():
    report = _complete_report()
    report["workloads"]["cold"]["sweep-4"]["identity"][
        "retrieval_encode_batch_size"
    ] = 16
    with pytest.raises(ValueError, match="one Flat/model/corpus identity"):
        bench.validate_complete_report(report)


def test_comparison_rejects_same_length_but_different_corpus_content():
    report = _complete_report()
    report["workloads"]["cold"]["ann"]["identity"][
        "corpus_fingerprint"
    ] = "different-corpus-sha"
    with pytest.raises(ValueError, match="model/corpus/index geometry"):
        bench.refresh_report_comparisons(report)


def test_selected_candidate_loader_requires_matching_index_hash(tmp_path):
    path = tmp_path / "selected_candidate.json"
    path.write_text(json.dumps({
        "candidate_selection_passed": True,
        "production_ready_for_agent_evaluation": True,
        "minimum_recall_at_3": 0.95,
        "selected_candidate": {
            "nprobe": 32,
            "faiss_thread_count": 8,
            "index_sha256": "record-sha",
            "recall_at_3": 0.96,
        },
        "bindings": {"ivfpq_index": {"sha256": "binding-sha"}},
    }))
    with pytest.raises(ValueError, match="invalid nprobe/thread/index binding"):
        bench._load_selected_candidate(path)


def test_selected_candidate_loader_binds_queries_manifest_model_and_both_indexes(tmp_path):
    queries = tmp_path / "queries.jsonl"
    calibration_manifest = tmp_path / "calibration.json"
    embedding_manifest = tmp_path / "embeddings.manifest.json"
    queries.write_text('{"uid":"nq:1","question":"q"}\n')
    calibration_manifest.write_text("{}")
    embedding_manifest.write_text(json.dumps({
        "binding": {"model": {"sha256": "model-sha"}},
    }))
    path = tmp_path / "selected_candidate.json"
    path.write_text(json.dumps({
        "candidate_selection_passed": True,
        "production_ready_for_agent_evaluation": True,
        "minimum_recall_at_3": 0.95,
        "selected_candidate": {
            "nprobe": 32,
            "faiss_thread_count": 8,
            "index_sha256": "ann-sha",
            "recall_at_3": 0.97,
        },
        "bindings": {
            "calibration_queries": {"sha256": bench.sha256_file(queries)},
            "calibration_manifest": {
                "sha256": bench.sha256_file(calibration_manifest)
            },
            "flat_index": {"sha256": "flat-sha"},
            "ivfpq_index": {"sha256": "ann-sha"},
            "query_embeddings_manifest": {
                "path": str(embedding_manifest),
                "sha256": bench.sha256_file(embedding_manifest),
            },
        },
    }))
    selected = bench._load_selected_candidate(path, queries, calibration_manifest)
    assert selected["flat_index_sha256"] == "flat-sha"
    assert selected["index_sha256"] == "ann-sha"
    assert selected["model_sha256"] == "model-sha"

    queries.write_text('{"uid":"nq:2","question":"changed"}\n')
    with pytest.raises(ValueError, match="different calibration queries"):
        bench._load_selected_candidate(path, queries, calibration_manifest)
