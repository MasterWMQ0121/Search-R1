import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from experiments.phase6_retriever_serving import benchmark_indexes as benchmark


def _write_synthetic_indexes_worker(connection, flat_path, ivfpq_path):
    try:
        import faiss

        rng = np.random.default_rng(9)
        vectors = rng.normal(size=(256, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        flat = faiss.IndexFlatIP(8)
        flat.add(vectors)
        faiss.write_index(flat, flat_path)
        ivfpq = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(8), 8, 4, 2, 4, faiss.METRIC_INNER_PRODUCT
        )
        ivfpq.cp.seed = 42
        ivfpq.pq.cp.seed = 42
        ivfpq.train(vectors[:128])
        ivfpq.add_with_ids(vectors, np.arange(len(vectors), dtype=np.int64))
        faiss.write_index(ivfpq, ivfpq_path)
        connection.send(("ok",))
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
    finally:
        connection.close()


def _write_synthetic_indexes_isolated(flat_path, ivfpq_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_write_synthetic_indexes_worker,
        args=(child, str(flat_path), str(ivfpq_path)),
    )
    process.start()
    child.close()
    process.join(60)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail("isolated FAISS index setup exceeded 60 seconds")
    assert process.exitcode == 0, (
        "isolated FAISS index setup crashed; "
        f"exitcode={process.exitcode}"
    )
    assert parent.poll(), "isolated FAISS index setup returned no result"
    assert parent.recv() == ("ok",)


def _write_queries(tmp_path, include_answer=False):
    records = []
    for index in range(4):
        question = f"Question {index}?"
        record = {
            "schema_version": 1,
            "uid": f"nq:{index}",
            "data_source": "nq",
            "question": question,
            "question_sha256": benchmark._question_sha256(question),
            "source_position": index,
        }
        if include_answer:
            record["ground_truth"] = {"target": ["secret"]}
        records.append(record)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "row_count": len(records),
        "selected_uids": [record["uid"] for record in records],
        "question_hashes": [record["question_sha256"] for record in records],
        "leakage_overlap_audit": {"passed": True, "overlapping_uids": []},
        "output": {"sha256": benchmark.sha256_file(queries)},
    }), encoding="utf-8")
    return records, queries, manifest


def test_calibration_loader_rejects_answer_fields(tmp_path):
    _, queries, manifest = _write_queries(tmp_path, include_answer=True)
    with pytest.raises(ValueError, match="answer fields"):
        benchmark.load_calibration_queries(queries, manifest)


def test_query_embeddings_are_encoded_once_and_hash_bound(tmp_path):
    records, queries, queries_manifest = _write_queries(tmp_path)
    calls = []

    class FakeEncoder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def encode(self, values, is_query):
            assert is_query is True
            calls.extend(values)
            return np.asarray(
                [[len(value), index + 1] for index, value in enumerate(values)],
                dtype=np.float32,
            )

    artifact = tmp_path / "embeddings.npz"
    model_fingerprint = {
        "path": "/local/e5",
        "sha256": "model-sha",
        "file_count": 1,
        "total_size_bytes": 1,
    }
    embeddings, manifest = benchmark.encode_calibration_queries_once(
        records,
        queries,
        queries_manifest,
        "/local/e5",
        artifact,
        encode_batch_size=2,
        encoder_factory=FakeEncoder,
        model_fingerprint=model_fingerprint,
    )
    assert calls == [record["question"] for record in records]
    assert embeddings.shape == (4, 2)
    assert manifest["each_query_encoded_once"] is True
    assert manifest["artifact"]["sha256"] == benchmark.sha256_file(artifact)

    class MustNotInstantiate:
        def __init__(self, **kwargs):
            raise AssertionError("existing bound embeddings should be reused")

    loaded, loaded_manifest = benchmark.encode_calibration_queries_once(
        records,
        queries,
        queries_manifest,
        "/local/e5",
        artifact,
        encode_batch_size=2,
        encoder_factory=MustNotInstantiate,
        model_fingerprint=model_fingerprint,
    )
    np.testing.assert_array_equal(loaded, embeddings)
    assert loaded_manifest == manifest


def test_embedding_artifact_rejects_changed_query_binding(tmp_path):
    records, queries, queries_manifest = _write_queries(tmp_path)

    class FakeEncoder:
        def __init__(self, **kwargs):
            pass

        def encode(self, values, is_query):
            return np.ones((len(values), 2), dtype=np.float32)

    artifact = tmp_path / "embeddings.npz"
    fingerprint = {"path": "x", "sha256": "a", "file_count": 1,
                   "total_size_bytes": 1}
    benchmark.encode_calibration_queries_once(
        records, queries, queries_manifest, "x", artifact,
        encoder_factory=FakeEncoder, model_fingerprint=fingerprint,
    )
    changed = dict(records[0])
    changed["uid"] = "nq:changed"
    with pytest.raises(ValueError, match="different inputs"):
        benchmark.encode_calibration_queries_once(
            [changed] + records[1:], queries, queries_manifest, "x", artifact,
            encoder_factory=FakeEncoder, model_fingerprint=fingerprint,
        )


def test_recall_and_id_agreement_definitions():
    exact = [[1, 2, 3], [4, 5, 6]]
    candidate = [[1, 3, 9], [5, 4, -1]]
    result = benchmark.compute_retrieval_agreement(exact, candidate, ntotal=10, k=3)
    assert result["recall_at_1"] == 0.5
    assert result["top1_agreement_rate"] == 0.5
    assert result["recall_at_3"] == pytest.approx(2 / 3)
    assert result["full_top3_set_agreement_rate"] == 0.0
    assert result["invalid_or_missing_result_count"] == 1
    assert result["invalid_or_missing_id_count"] == 1


def _candidate(nprobe, threads, recall, p95, mean, invalid=0):
    return {
        "index_backend": "ivfpq",
        "nprobe": nprobe,
        "faiss_thread_count": threads,
        "recall_at_1": recall,
        "recall_at_3": recall,
        "top1_agreement_rate": recall,
        "full_top3_set_agreement_rate": recall,
        "invalid_or_missing_result_count": invalid,
        "pure_search_latency_s": {"p95": p95, "p50": mean, "mean": mean},
        "queries_per_second": 1 / mean,
        "index_path": "/ivf.index",
        "index_sha256": "ivf-sha",
        "index_file_size_bytes": 100,
        "result_ids": [[1, 2, 3]],
    }


def test_candidate_selection_applies_exact_tie_break_order():
    records = [
        _candidate(4, 4, 0.96, 0.20, 0.10),
        _candidate(64, 16, 0.96, 0.10, 0.09),
        _candidate(32, 16, 0.96, 0.10, 0.09),
        _candidate(32, 8, 0.96, 0.10, 0.09),
    ]
    selected = benchmark.select_candidate(records, min_recall_at_3=0.95)
    assert selected["candidate_selection_passed"] is True
    assert selected["production_ready_for_agent_evaluation"] is True
    assert selected["selected_candidate"]["nprobe"] == 32
    assert selected["selected_candidate"]["faiss_thread_count"] == 8
    assert "result_ids" not in selected["selected_candidate"]
    assert selected["selection_uses_qa_ground_truth"] is False


def test_threshold_failure_reports_highest_recall_pareto_candidate():
    records = [
        _candidate(64, 8, 0.94, 0.20, 0.10),
        _candidate(8, 4, 0.90, 0.05, 0.03),
        _candidate(4, 4, 0.80, 0.30, 0.20),  # dominated
    ]
    selected = benchmark.select_candidate(records, min_recall_at_3=0.95)
    assert selected["candidate_selection_passed"] is False
    assert selected["production_ready_for_agent_evaluation"] is False
    assert selected["selected_candidate"]["recall_at_3"] == 0.94
    assert "explicit user override" in selected["fallback_reason"]


def test_invalid_result_configuration_is_never_selected():
    records = [
        _candidate(4, 4, 1.0, 0.01, 0.01, invalid=1),
        _candidate(8, 4, 0.96, 0.02, 0.02),
    ]
    selected = benchmark.select_candidate(records, min_recall_at_3=0.95)
    assert selected["selected_candidate"]["nprobe"] == 8


def test_warmups_are_excluded_from_search_samples(monkeypatch):
    pytest.importorskip("faiss")

    class FakeIndex:
        d = 2

        def __init__(self):
            self.calls = 0

        def search(self, queries, k):
            self.calls += 1
            return (
                np.tile(np.asarray([[1.0, 0.5, 0.1]], dtype=np.float32),
                        (len(queries), 1)),
                np.tile(np.asarray([[0, 1, 2]], dtype=np.int64),
                        (len(queries), 1)),
            )

    index = FakeIndex()
    result = benchmark.benchmark_index_config(
        index,
        np.ones((5, 2), dtype=np.float32),
        k=3,
        faiss_thread_count=1,
        warmup_query_count=2,
    )
    assert index.calls == 7
    assert result["warmup_query_count"] == 2
    assert result["warmups_excluded"] is True
    assert result["query_count"] == 5
    assert len(result["result_ids"]) == 5


def test_model_fingerprint_is_content_bound(tmp_path):
    from experiments.phase6_retriever_serving.optimized_retrieval_server import (
        fingerprint_asset,
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("one")
    first = benchmark.fingerprint_path(model)
    assert first["sha256"] == fingerprint_asset(model)
    (model / "config.json").write_text("two")
    second = benchmark.fingerprint_path(model)
    assert first["sha256"] != second["sha256"]


def test_end_to_end_offline_artifacts_bind_selected_candidate(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    records, queries, queries_manifest = _write_queries(tmp_path)
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(256, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    flat_path = tmp_path / "flat.index"
    ivfpq_path = tmp_path / "ivfpq.index"
    _write_synthetic_indexes_isolated(flat_path, ivfpq_path)

    artifact = tmp_path / "embeddings.npz"

    def fake_encode(*args, **kwargs):
        embeddings = np.ascontiguousarray(vectors[: len(records)])
        np.savez_compressed(artifact, embeddings=embeddings)
        artifact_manifest = artifact.with_suffix(".manifest.json")
        artifact_manifest.write_text(json.dumps({"each_query_encoded_once": True}))
        return embeddings, {"each_query_encoded_once": True}

    monkeypatch.setattr(benchmark, "encode_calibration_queries_once", fake_encode)
    result = benchmark.benchmark_indexes(
        queries,
        queries_manifest,
        tmp_path / "unused-model",
        artifact,
        flat_path,
        ivfpq_path,
        tmp_path / "results",
        nprobes=(1, 2),
        thread_counts=(1, 2),
        min_recall_at_3=0.0,
        warmup_query_count=1,
    )
    assert result["candidate_selection_passed"] is True
    selected_path = Path(result["selected_candidate_path"])
    selected = json.loads(selected_path.read_text())
    assert selected["candidate_selection_passed"] is True
    assert selected["selected_candidate"]["index_backend"] == "ivfpq"
    assert selected["selected_candidate"]["index_sha256"] == benchmark.sha256_file(
        ivfpq_path
    )
    assert selected["bindings"]["flat_index"] == {
        "path": str(flat_path.resolve()),
        "sha256": benchmark.sha256_file(flat_path),
    }
    assert selected["bindings"]["ivfpq_index"] == {
        "path": str(ivfpq_path.resolve()),
        "sha256": benchmark.sha256_file(ivfpq_path),
    }
    benchmark_payload = json.loads(
        Path(result["index_benchmark_path"]).read_text()
    )
    assert benchmark_payload["query_embeddings_encoded_once"] is True
    assert len(benchmark_payload["flat_results"]) == 2
    assert len(benchmark_payload["ivfpq_results"]) == 4
