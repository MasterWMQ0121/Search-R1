import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = (
    ROOT
    / "experiments"
    / "phase6_retriever_serving"
    / "optimized_retrieval_server.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase6_optimized_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER = _load_module()


class FakeParameterSpace:
    def set_index_parameter(self, index, name, value):
        assert name == "nprobe"
        index.nprobe = value


class FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    METRIC_L2 = 1
    configured_threads = None
    ParameterSpace = FakeParameterSpace

    @classmethod
    def omp_set_num_threads(cls, value):
        cls.configured_threads = value


class FakeIndexFlatIP:
    def __init__(self):
        self.d = 4
        self.ntotal = 4
        self.metric_type = FakeFaiss.METRIC_INNER_PRODUCT
        self.is_trained = True
        self.vectors = np.eye(4, dtype=np.float32)
        self.search_calls = []

    def search(self, embeddings, topk):
        self.search_calls.append(np.array(embeddings, copy=True))
        scores = np.asarray(embeddings) @ self.vectors.T
        identifiers = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
        return np.take_along_axis(scores, identifiers, axis=1), identifiers


class FakeIndexIVFPQ(FakeIndexFlatIP):
    def __init__(self):
        super().__init__()
        self.nprobe = 1


class FakeEncoder:
    VECTORS = {
        "alpha": (1.0, 0.0, 0.0, 0.0),
        "beta": (0.0, 1.0, 0.0, 0.0),
        "gamma": (0.0, 0.0, 1.0, 0.0),
        "delta": (0.0, 0.0, 0.0, 1.0),
    }

    def __init__(self):
        self.calls = []

    def encode(self, queries):
        self.calls.append(list(queries))
        values = []
        for query in queries:
            values.append(self.VECTORS.get(query, (0.4, 0.3, 0.2, 0.1)))
        return np.asarray(values, dtype=np.float32)


def _corpus():
    return [
        {"id": f"doc-{index}", "contents": f"Title {index}\nEvidence {index}."}
        for index in range(4)
    ]


def _document_loader(corpus, identifiers):
    return [corpus[int(identifier)] for identifier in identifiers]


def _make_engine(
    tmp_path,
    *,
    cache_enabled=False,
    result_capacity=0,
    embedding_capacity=0,
    encode_batch_size=32,
    backend="flat",
    nprobe=None,
    index=None,
    corpus=None,
    encoder=None,
    index_fingerprint="index-sha-a",
    model_fingerprint="model-sha-a",
    corpus_fingerprint="corpus-sha-a",
    **engine_kwargs,
):
    config = SERVER.ServerConfig(
        index_path=tmp_path / "index.faiss",
        corpus_path=tmp_path / "corpus.jsonl",
        model_path=tmp_path / "model",
        topk=3,
        faiss_thread_count=4,
        retrieval_encode_batch_size=encode_batch_size,
        index_backend=backend,
        nprobe=nprobe,
        cache_enabled=cache_enabled,
        result_cache_capacity=result_capacity,
        embedding_cache_capacity=embedding_capacity,
        index_fingerprint=index_fingerprint,
        model_fingerprint=model_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
    )
    return SERVER.OptimizedRetrieverEngine(
        config,
        index=index or (FakeIndexIVFPQ() if backend == "ivfpq" else FakeIndexFlatIP()),
        corpus=_corpus() if corpus is None else corpus,
        encoder=encoder or FakeEncoder(),
        faiss_module=FakeFaiss,
        document_loader=_document_loader,
        rss_reader=lambda: 123456,
        **engine_kwargs,
    )


def test_existing_result_contract_is_preserved_and_metrics_are_opt_in(tmp_path):
    engine = _make_engine(tmp_path)
    client = TestClient(SERVER.create_app(engine))

    scored = client.post("/retrieve", json={
        "queries": ["alpha"], "topk": 2, "return_scores": True
    })
    assert scored.status_code == 200
    assert set(scored.json()) == {"result"}
    assert scored.json()["result"] == [[
        {"document": _corpus()[0], "score": 1.0},
        {"document": _corpus()[1], "score": 0.0},
    ]]

    scoreless = client.post("/retrieve", json={"queries": ["beta"], "topk": 1})
    assert scoreless.status_code == 200
    assert scoreless.json() == {"result": [[_corpus()[1]]]}


def test_optional_metrics_have_stable_stage_and_result_id_contract(tmp_path):
    engine = _make_engine(tmp_path)
    response = TestClient(SERVER.create_app(engine)).post("/retrieve", json={
        "queries": ["beta", "alpha"],
        "topk": 2,
        "return_scores": True,
        "return_metrics": True,
    })
    assert response.status_code == 200
    assert set(response.json()) == {"result", "metrics"}
    metrics = response.json()["metrics"]
    assert metrics["query_count"] == 2
    assert metrics["topk"] == 2
    assert metrics["result_ids"] == [[1, 0], [0, 1]]
    assert metrics["index_backend"] == "flat"
    assert metrics["nprobe"] is None
    assert metrics["faiss_thread_count"] == 4
    assert isinstance(metrics["request_id"], str) and metrics["request_id"]
    for field in SERVER.TIMING_FIELDS:
        assert math.isfinite(metrics[field]) and metrics[field] >= 0


def test_health_and_stats_expose_integrity_configuration_and_cumulative_values(tmp_path):
    engine = _make_engine(
        tmp_path, cache_enabled=True, result_capacity=2, embedding_capacity=3
    )
    client = TestClient(SERVER.create_app(engine))
    health = client.get("/healthz").json()
    assert health["ready"] is True
    assert health["index_type"] == "FakeIndexFlatIP"
    assert health["index_backend"] == "flat"
    assert health["index_file_size_bytes"] is None
    assert health["index_ntotal"] == health["corpus_row_count"] == 4
    assert health["index_dimension"] == 4
    assert health["metric_type"] == "inner_product"
    assert health["metric_type_code"] == 0
    assert health["faiss_thread_count"] == 4
    assert health["nprobe"] is None
    assert health["cache_enabled"] is True
    assert health["result_cache_capacity"] == 2
    assert health["embedding_cache_capacity"] == 3
    assert health["process_rss_bytes"] == 123456
    assert health["index_fingerprint"] == "index-sha-a"
    assert health["model_fingerprint"] == "model-sha-a"
    assert health["corpus_fingerprint"] == "corpus-sha-a"
    assert len(health["retrieval_config_fingerprint"]) == 64
    assert set(health["startup_load_timings_s"]) == {
        "index_load_s", "corpus_load_s", "model_load_s", "startup_total_s"
    }

    client.post("/retrieve", json={"queries": ["alpha", "beta"], "topk": 1})
    client.post("/retrieve", json={"queries": ["alpha"], "topk": 1})
    stats = client.get("/stats").json()
    assert stats["request_count"] == 2
    assert stats["query_count"] == 3
    assert stats["errors"] == 0
    assert stats["result_cache"]["hits"] == 1
    assert stats["result_cache"]["misses"] == 2
    for field in SERVER.TIMING_FIELDS:
        assert stats["timing_sums"][field] >= 0
        assert stats["timing_means"][field] >= 0


@pytest.mark.parametrize(
    "payload",
    (
        {"queries": []},
        {"queries": [" "]},
        {"queries": ["alpha"], "topk": 0},
        {"queries": ["alpha"], "topk": 5},
    ),
)
def test_invalid_requests_fail_before_cache_lookup(payload, tmp_path):
    engine = _make_engine(
        tmp_path, cache_enabled=True, result_capacity=2, embedding_capacity=2
    )
    response = TestClient(SERVER.create_app(engine)).post("/retrieve", json=payload)
    assert response.status_code == 400
    assert engine.result_cache.snapshot()["hits"] == 0
    assert engine.result_cache.snapshot()["misses"] == 0
    assert engine.embedding_cache.snapshot()["hits"] == 0
    assert engine.embedding_cache.snapshot()["misses"] == 0
    assert engine.cumulative_stats()["errors"] == 1


def test_missing_queries_remains_framework_validation_error(tmp_path):
    engine = _make_engine(tmp_path)
    response = TestClient(SERVER.create_app(engine)).post("/retrieve", json={"topk": 2})
    assert response.status_code == 422
    assert engine.cumulative_stats()["errors"] == 1


def test_cache_disabled_does_not_record_hits_misses_or_change_results(tmp_path):
    encoder = FakeEncoder()
    engine = _make_engine(
        tmp_path,
        cache_enabled=False,
        result_capacity=10,
        embedding_capacity=10,
        encoder=encoder,
    )
    first, first_metrics = engine.retrieve(["alpha"], topk=2, return_scores=True)
    second, second_metrics = engine.retrieve(["alpha"], topk=2, return_scores=True)
    assert first == second
    assert encoder.calls == [["alpha"], ["alpha"]]
    assert first_metrics["cache_hit_count"] == first_metrics["cache_miss_count"] == 0
    assert second_metrics["cache_hit_count"] == second_metrics["cache_miss_count"] == 0
    assert engine.result_cache.snapshot()["hits"] == 0
    assert engine.result_cache.snapshot()["misses"] == 0


def test_result_cache_uses_exact_stripped_query_topk_and_preserves_order(tmp_path):
    encoder = FakeEncoder()
    index = FakeIndexFlatIP()
    engine = _make_engine(
        tmp_path,
        cache_enabled=True,
        result_capacity=8,
        embedding_capacity=0,
        encoder=encoder,
        index=index,
    )
    first, first_metrics = engine.retrieve(["  alpha  "], topk=1, return_scores=True)
    second, second_metrics = engine.retrieve(["alpha"], topk=1, return_scores=True)
    _different_topk, topk_metrics = engine.retrieve(["alpha"], topk=2)
    _different_case, case_metrics = engine.retrieve(["Alpha"], topk=1)
    _different_space, space_metrics = engine.retrieve(["alpha  beta"], topk=1)

    assert first == second
    assert encoder.calls[0] == ["alpha"]
    assert first_metrics["cache_miss_count"] == 1
    assert second_metrics["cache_hit_count"] == 1
    assert topk_metrics["cache_miss_count"] == 1
    assert case_metrics["cache_miss_count"] == 1
    assert space_metrics["cache_miss_count"] == 1
    assert len(index.search_calls) == 4


def test_result_cache_eviction_is_bounded_lru(tmp_path):
    engine = _make_engine(
        tmp_path, cache_enabled=True, result_capacity=1, embedding_capacity=0
    )
    for query in ("alpha", "beta", "alpha"):
        engine.retrieve([query], topk=1)
    snapshot = engine.result_cache.snapshot()
    assert snapshot == {
        "enabled": True,
        "capacity": 1,
        "size": 1,
        "hits": 0,
        "misses": 3,
        "evictions": 2,
    }


def test_embedding_cache_skips_encoding_but_not_exact_index_search(tmp_path):
    encoder = FakeEncoder()
    index = FakeIndexFlatIP()
    engine = _make_engine(
        tmp_path,
        cache_enabled=True,
        result_capacity=0,
        embedding_capacity=2,
        encoder=encoder,
        index=index,
    )
    first, first_metrics = engine.retrieve(["alpha"], topk=2)
    second, second_metrics = engine.retrieve(["alpha"], topk=2)
    assert first == second
    assert encoder.calls == [["alpha"]]
    assert len(index.search_calls) == 2
    assert first_metrics["embedding_cache_miss_count"] == 1
    assert second_metrics["embedding_cache_hit_count"] == 1
    assert first_metrics["cache_miss_count"] == second_metrics["cache_miss_count"] == 0


def test_native_query_batches_preserve_input_order_and_cardinality(tmp_path):
    encoder = FakeEncoder()
    engine = _make_engine(tmp_path, encode_batch_size=2, encoder=encoder)
    queries = ["alpha", "beta", "gamma", "delta", "other"]
    results, metrics = engine.retrieve(queries, topk=1)
    assert encoder.calls == [queries[:2], queries[2:4], queries[4:]]
    assert len(results) == len(queries)
    assert [row[0]["id"] for row in results[:4]] == [
        "doc-0", "doc-1", "doc-2", "doc-3"
    ]
    assert metrics["query_count"] == len(queries)


def test_cache_keys_bind_index_model_and_result_affecting_configuration(tmp_path):
    flat_a = _make_engine(tmp_path / "a")
    flat_b = _make_engine(
        tmp_path / "b", index_fingerprint="index-sha-b", model_fingerprint="model-sha-b"
    )
    ivf = _make_engine(tmp_path / "c", backend="ivfpq", nprobe=16)
    assert flat_a._result_cache_key("alpha", 3) != flat_b._result_cache_key("alpha", 3)
    assert flat_a._result_cache_key("alpha", 3) != ivf._result_cache_key("alpha", 3)
    assert flat_a._result_cache_key("alpha", 2) != flat_a._result_cache_key("alpha", 3)
    assert flat_a._embedding_cache_key("alpha") != flat_b._embedding_cache_key("alpha")


def test_ivfpq_applies_nprobe_and_reports_backend(tmp_path):
    index = FakeIndexIVFPQ()
    engine = _make_engine(tmp_path, backend="ivfpq", nprobe=16, index=index)
    assert index.nprobe == 16
    assert engine.health()["nprobe"] == 16
    _results, metrics = engine.retrieve(["alpha"], topk=1)
    assert metrics["index_backend"] == "ivfpq"
    assert metrics["nprobe"] == 16


@pytest.mark.parametrize(
    "config_change,error",
    (
        ({"backend": "flat", "index": FakeIndexIVFPQ()}, "Flat FAISS index"),
        ({"backend": "ivfpq", "nprobe": 4, "index": FakeIndexFlatIP()}, "IndexIVFPQ"),
        ({"corpus": _corpus()[:3]}, "corpus row count"),
    ),
)
def test_index_backend_and_corpus_integrity_are_enforced(config_change, error, tmp_path):
    with pytest.raises(ValueError, match=error):
        _make_engine(tmp_path, **config_change)


def test_untrained_index_and_oversized_configured_topk_are_rejected(tmp_path):
    index = FakeIndexFlatIP()
    index.is_trained = False
    with pytest.raises(ValueError, match="not trained"):
        _make_engine(tmp_path / "untrained", index=index)

    config = SERVER.ServerConfig(
        index_path=tmp_path / "index",
        corpus_path=tmp_path / "corpus",
        model_path=tmp_path / "model",
        topk=5,
        index_fingerprint="index",
        model_fingerprint="model",
        corpus_fingerprint="corpus",
    )
    with pytest.raises(ValueError, match="topk exceeds"):
        SERVER.OptimizedRetrieverEngine(
            config,
            index=FakeIndexFlatIP(),
            corpus=_corpus(),
            encoder=FakeEncoder(),
            faiss_module=FakeFaiss,
            document_loader=_document_loader,
        )


def test_encoder_factory_preserves_existing_cpu_e5_semantics(tmp_path):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeEncoder()

    config = SERVER.ServerConfig(
        index_path=tmp_path / "index",
        corpus_path=tmp_path / "corpus",
        model_path=tmp_path / "model",
        index_fingerprint="index",
        model_fingerprint="model",
        corpus_fingerprint="corpus",
    )
    SERVER.OptimizedRetrieverEngine(
        config,
        index=FakeIndexFlatIP(),
        corpus=_corpus(),
        encoder_factory=factory,
        faiss_module=FakeFaiss,
        document_loader=_document_loader,
    )
    assert captured == {
        "model_name": "e5",
        "model_path": str((tmp_path / "model").resolve()),
        "pooling_method": "mean",
        "max_length": 256,
        "use_fp16": False,
        "device": "cpu",
        "model_revision": None,
    }


def test_production_fingerprint_expectation_cannot_override_asset_identity(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"actual bytes")
    actual = SERVER.fingerprint_asset(asset)
    assert SERVER.OptimizedRetrieverEngine._resolve_asset_fingerprint(
        asset,
        actual,
        dependency_was_injected=False,
        label="index",
    ) == actual
    with pytest.raises(ValueError, match="does not match the asset"):
        SERVER.OptimizedRetrieverEngine._resolve_asset_fingerprint(
            asset,
            "spoofed-sha",
            dependency_was_injected=False,
            label="index",
        )


def test_nonfinite_embeddings_and_invalid_faiss_ids_fail_and_increment_errors(tmp_path):
    encoder = FakeEncoder()
    encoder.encode = lambda queries: np.full((len(queries), 4), np.nan, dtype=np.float32)
    engine = _make_engine(tmp_path / "nan", encoder=encoder)
    with pytest.raises(RuntimeError, match="non-finite embeddings"):
        engine.retrieve(["alpha"], topk=1)
    assert engine.cumulative_stats()["errors"] == 1

    index = FakeIndexFlatIP()
    index.search = lambda embeddings, topk: (
        np.ones((len(embeddings), topk), dtype=np.float32),
        np.full((len(embeddings), topk), -1, dtype=np.int64),
    )
    engine = _make_engine(tmp_path / "ids", index=index)
    with pytest.raises(RuntimeError, match="invalid document ID"):
        engine.retrieve(["alpha"], topk=1)
    assert engine.cumulative_stats()["errors"] == 1


def test_thread_safe_lru_remains_bounded_under_concurrent_access():
    cache = SERVER.ThreadSafeLRU(capacity=8, enabled=True)

    def exercise(number):
        cache.put(number, number)
        cache.get(number)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(exercise, range(100)))
    snapshot = cache.snapshot()
    assert snapshot["size"] <= 8
    assert snapshot["capacity"] == 8
    assert snapshot["evictions"] >= 92


def test_asset_fingerprints_use_contents_not_path_only(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text("same", encoding="utf-8")
    (second / "config.json").write_text("same", encoding="utf-8")
    assert SERVER.fingerprint_asset(first) == SERVER.fingerprint_asset(second)
    (second / "config.json").write_text("changed", encoding="utf-8")
    assert SERVER.fingerprint_asset(first) != SERVER.fingerprint_asset(second)


def test_cli_defaults_to_cpu_safe_disabled_caches_and_accepts_environment(monkeypatch):
    monkeypatch.setenv("PHASE6_INDEX_BACKEND", "ivfpq")
    monkeypatch.setenv("PHASE6_IVF_NPROBE", "32")
    monkeypatch.setenv("PHASE6_FAISS_THREADS", "16")
    monkeypatch.setenv("PHASE6_CACHE_ENABLED", "false")
    args = SERVER.parse_args([])
    config = SERVER.config_from_args(args)
    assert config.index_backend == "ivfpq"
    assert config.nprobe == 32
    assert config.faiss_thread_count == 16
    assert config.cache_enabled is False
    assert not hasattr(config, "faiss_gpu")


def test_process_rss_is_optional_without_psutil(monkeypatch):
    original = Path.read_text

    def fail_for_proc(path, *args, **kwargs):
        if str(path) == "/proc/self/status":
            raise FileNotFoundError
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_for_proc)
    assert SERVER.process_rss_bytes() is None
