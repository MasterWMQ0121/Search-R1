import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase4_benchmark import run_benchmark as phase4
from experiments.phase5_observation_context import run_context_ablation as phase5
from experiments.phase6_retriever_serving import run_agent_retriever_ablation as agent


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments"
    / "phase6_retriever_serving"
    / "run_agent_retriever_ablation.sh"
)
MODEL_CHECKPOINT_FINGERPRINT = "a" * 64


def _example(index=0):
    source = "nq" if index < 32 else "hotpotqa"
    return phase4.BenchmarkExample(
        uid=f"{source}:{index}",
        data_source=source,
        question=f"question-{index}",
        ground_truth={"target": [f"answer-{index}"]},
        search_prompt=f"Question: question-{index}",
    )


def _baseline_audit(tmp_path):
    return {
        "path": str(tmp_path / "phase4" / "search_rl.jsonl"),
        "sha256": "raw-baseline-sha",
        "run_fingerprint": "raw-baseline-fingerprint",
    }


def _context_config(tmp_path, retriever_url="http://old/retrieve"):
    manifest_path = tmp_path / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text("{}", encoding="utf-8")
    return phase5.build_condition_run_config(
        "compressed_256",
        "trained-model",
        retriever_url,
        {"output": {"sha256": "eval-sha"}},
        manifest_path,
        _baseline_audit(tmp_path),
        attention_backend="XFORMERS",
    )


def _candidate(tmp_path, passed=True):
    embedding_manifest = tmp_path / "query_embeddings.manifest.json"
    embedding_manifest.write_text(
        json.dumps({"binding": {"model": {"sha256": "model-sha"}}}),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "candidate_selection_passed": passed,
        "production_ready_for_agent_evaluation": passed,
        "selection_uses_qa_ground_truth": False,
        "selected_candidate": {
            "index_backend": "ivfpq",
            "nprobe": 32,
            "faiss_thread_count": 8,
            "index_sha256": "ivfpq-sha",
        },
        "bindings": {
            "flat_index": {"path": "/flat.index", "sha256": "flat-sha"},
            "ivfpq_index": {"path": "/ivfpq.index", "sha256": "ivfpq-sha"},
            "query_embeddings_manifest": {
                "path": str(embedding_manifest),
                "sha256": hashlib.sha256(embedding_manifest.read_bytes()).hexdigest(),
            },
        },
    }
    path = tmp_path / "selected_candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _server_contract(backend="flat"):
    return {
        "index_type": "IndexFlatIP" if backend == "flat" else "IndexIVFPQ",
        "index_backend": backend,
        "index_path": f"/{backend}.index",
        "index_file_size_bytes": 4096 if backend == "flat" else 1024,
        "index_ntotal": 100,
        "index_dimension": 8,
        "metric_type": "inner_product",
        "metric_type_code": 0,
        "corpus_row_count": 100,
        "model_path": "/e5",
        "faiss_thread_count": 8,
        "nprobe": None if backend == "flat" else 32,
        "cache_enabled": False,
        "result_cache_capacity": 0,
        "embedding_cache_capacity": 0,
        "retrieval_encode_batch_size": 16,
        "index_fingerprint": "flat-sha" if backend == "flat" else "ivfpq-sha",
        "model_fingerprint": "model-sha",
        "corpus_fingerprint": "corpus-sha",
        "retrieval_config_fingerprint": f"config-{backend}",
    }


def _health(backend="flat"):
    return {"ready": True, **_server_contract(backend)}


def _metrics(backend="flat", request_id="request-1"):
    return {
        "request_id": request_id,
        "result_ids": [[1, 2, 3]],
        **{field: 0.01 for field in agent.STAGE_TIMING_FIELDS},
        "query_count": 1,
        "topk": 3,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "embedding_cache_hit_count": 0,
        "embedding_cache_miss_count": 0,
        "index_backend": backend,
        "nprobe": None if backend == "flat" else 32,
        "faiss_thread_count": 8,
    }


def _run_config(tmp_path, condition="flat_exact_replay"):
    manifest_path = tmp_path / "eval-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    baseline_config = _context_config(tmp_path)
    candidate_path, _ = _candidate(tmp_path)
    _, candidate_audit = agent.load_selected_candidate(candidate_path)
    return agent.build_run_config(
        condition,
        "trained-model",
        MODEL_CHECKPOINT_FINGERPRINT,
        f"http://{condition}/retrieve",
        {"output": {"sha256": "eval-sha"}},
        manifest_path,
        {
            "path": str(tmp_path / "phase5" / "compressed_256.jsonl"),
            "sha256": "phase5-sha",
            "run_fingerprint": "phase5-fingerprint",
        },
        baseline_config,
        _server_contract("flat" if condition == "flat_exact_replay" else "ivfpq"),
        candidate_audit,
        attention_backend="XFORMERS",
    )


def _search_result(tmp_path, condition="flat_exact_replay"):
    example = _example()
    run_config = _run_config(tmp_path, condition)
    return {
        "schema_version": 1,
        "uid": example.uid,
        "data_source": example.data_source,
        "question": example.question,
        "ground_truth": example.ground_truth["target"],
        "mode": "search_rl",
        "model_path": "trained-model",
        "prediction": "answer-0",
        "exact_match": 1,
        "end_to_end_latency_s": 2.0,
        "generation_latency_s": 1.0,
        "trajectory": "<answer>answer-0</answer>",
        "run_config": run_config,
        "run_fingerprint": phase4.run_config_fingerprint(run_config),
        "number_of_actions": 2,
        "number_of_valid_actions": 2,
        "number_of_valid_searches": 1,
        "number_of_successful_retrievals": 1,
        "finished": True,
        "search_retrieval_failure_count": 0,
        "retrieval_latency_s": 0.2,
        "observation_truncation_count": 0,
        "trajectory_had_observation_truncation": False,
        "retrieved_observation_lengths_before_truncation": [100],
        "retained_observation_lengths_after_truncation": [100],
        "observation_excess_tokens": [0],
    }, run_config


def _policy_event():
    return {
        "raw_retrieved_observation_tokens": 400,
        "policy_output_observation_tokens": 100,
        "documents_returned": 3,
        "documents_represented": 2,
        "sentences_considered": 8,
        "sentences_selected": 2,
        "evidence_compression_latency_s": 0.01,
        "zero_overlap_fallback_used": False,
        "partial_sentence_fallback_used": False,
        "selected_document_ranks": [1, 2],
        "selected_sentence_identifiers": ["doc1:s0", "doc2:s0"],
    }


class _Response:
    status_code = 200
    text = "ok"

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return _Response(self.payload)


def test_metrics_client_adds_opt_in_without_extra_retriever_call():
    passages = [
        {"document": {"contents": f"Title {rank}\nEvidence"}, "score": 1.0}
        for rank in range(3)
    ]
    session = _Session({"result": [passages], "metrics": _metrics()})
    client = agent.MetricsEnabledRetrieverClient(
        "http://server/retrieve", 3, _server_contract(), session=session
    )
    manager = SimpleNamespace()
    client.configure_manager(manager)

    response = manager._batch_search(["  Generated Query  "])

    assert response["result"] == [passages]
    assert len(session.posts) == 1
    assert client.post_count == 1
    assert session.posts[0][1] == {
        "queries": ["Generated Query"],
        "topk": 3,
        "return_scores": True,
        "return_metrics": True,
    }
    assert client.events[0]["queries"] == ["Generated Query"]
    assert client.events[0]["metrics"]["result_ids"] == [[1, 2, 3]]


def test_phase4_pre_timing_hook_is_additive_and_ordered():
    signature = inspect.signature(phase4.evaluate_search_agent)
    assert signature.parameters["configure_manager"].default is None
    assert signature.parameters["configure_retriever_client"].default is None
    source = inspect.getsource(phase4.evaluate_search_agent)
    assert source.index("configure_retriever_client(manager)") < source.index(
        "original_batch_search = manager._batch_search"
    )
    assert source.index("original_batch_search = manager._batch_search") < source.index(
        "configure_manager(manager)"
    )


def test_phase4_fingerprints_remain_unchanged():
    expected = {
        "direct": "3e8ada978d4ed4cced87b4e7b1f693187bb9ea533592cdd14ad02be73bf44089",
        "static_rag": "50b659cc974342c340bc5eff8c77b1c3b91b9851465f42eb44353c5b744fd587",
        "search_rl": "6a85575a1a0b9eabb04f7a455d9d9aeb15a080926107d838f50c5a35e9b6228e",
    }
    assert {
        mode: phase4.run_config_fingerprint(
            phase4._default_test_run_config(mode, "model")
        )
        for mode in expected
    } == expected


@pytest.mark.parametrize(
    "output",
    ("phase5", "phase5/nested", "phase5/a/b/../c"),
)
def test_phase6_outputs_cannot_target_phase5_tree(tmp_path, output):
    phase5_dir = tmp_path / "phase5"
    baseline = phase5_dir / "compressed_256.jsonl"
    with pytest.raises(ValueError, match="immutable Phase-5"):
        agent.validate_output_isolation(tmp_path / output, baseline)


def test_selected_candidate_requires_calibration_pass_or_explicit_override(tmp_path):
    path, _ = _candidate(tmp_path, passed=False)
    with pytest.raises(ValueError, match="explicit"):
        agent.load_selected_candidate(path)
    _payload, audit = agent.load_selected_candidate(path, allow_unqualified=True)
    assert audit["override_used"] is True


@pytest.mark.parametrize(
    "condition,backend",
    (("flat_exact_replay", "flat"), ("ivfpq_selected", "ivfpq")),
)
def test_health_binding_requires_selected_threads_index_and_cache_off(
    tmp_path, condition, backend
):
    _path, candidate = _candidate(tmp_path)
    contract = agent.stable_server_contract(
        _health(backend), condition, candidate, require_local_index_file=False
    )
    assert contract["index_backend"] == backend
    assert contract["faiss_thread_count"] == 8
    bad = _health(backend)
    bad["cache_enabled"] = True
    with pytest.raises(ValueError, match="cache disabled"):
        agent.stable_server_contract(
            bad, condition, candidate, require_local_index_file=False
        )


def test_health_binding_requires_the_calibration_e5_model_fingerprint(tmp_path):
    _path, candidate = _candidate(tmp_path)
    bad = _health("flat")
    bad["model_fingerprint"] = "different-e5-model"
    with pytest.raises(ValueError, match="calibration embeddings"):
        agent.stable_server_contract(
            bad,
            "flat_exact_replay",
            candidate,
            require_local_index_file=False,
        )


def test_stable_server_contract_records_actual_index_file_size(tmp_path):
    _candidate_path, candidate = _candidate(tmp_path)
    index_path = tmp_path / "flat.index"
    index_path.write_bytes(b"12345678")
    health = _health("flat")
    health["index_path"] = str(index_path)
    health["index_file_size_bytes"] = 8
    contract = agent.stable_server_contract(
        health, "flat_exact_replay", candidate
    )
    assert contract["index_file_size_bytes"] == 8


def test_cache_disabled_request_rejects_every_nonzero_cache_counter():
    for key in (
        "cache_hit_count",
        "cache_miss_count",
        "embedding_cache_hit_count",
        "embedding_cache_miss_count",
    ):
        metrics = _metrics()
        metrics[key] = 1
        with pytest.raises(ValueError, match="zero cache counters"):
            agent.validate_request_metrics(metrics, 1, 3, _server_contract())


def test_phase6_result_reuses_compressed_policy_and_aligns_server_ids(tmp_path):
    search_result, run_config = _search_result(tmp_path)
    record = agent.phase6_result_from_search_result(
        search_result,
        "flat_exact_replay",
        run_config,
        [_policy_event()],
        [{"queries": ["generated query"], "metrics": _metrics()}],
    )

    assert record["observation_policy"] == "compressed_256"
    assert record["policy_output_observation_tokens"] == [100]
    assert record["retriever_queries"] == ["generated query"]
    assert record["retrieved_document_ids"] == [[1, 2, 3]]
    assert record["retriever_request_ids"] == ["request-1"]
    agent.validate_phase6_result(record, "flat_exact_replay")


def test_resume_rejects_a_different_stable_retriever_fingerprint(tmp_path):
    search_result, run_config = _search_result(tmp_path)
    record = agent.phase6_result_from_search_result(
        search_result,
        "flat_exact_replay",
        run_config,
        [_policy_event()],
        [{"queries": ["generated query"], "metrics": _metrics()}],
    )
    result_path = tmp_path / "flat_exact_replay.jsonl"
    result_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    changed = json.loads(json.dumps(run_config))
    changed["retriever_server_contract"]["retrieval_config_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="run configuration"):
        agent.run_condition_with_resume(
            [_example()],
            result_path,
            "flat_exact_replay",
            lambda _example: None,
            expected_run_fingerprint=phase4.run_config_fingerprint(changed),
        )


def test_model_checkpoint_tree_fingerprint_is_content_bound_and_path_independent(
    tmp_path,
):
    first = tmp_path / "checkpoint-a"
    second = tmp_path / "checkpoint-b"
    for checkpoint in (first, second):
        (checkpoint / "nested").mkdir(parents=True)
        (checkpoint / "config.json").write_text("same config", encoding="utf-8")
        (checkpoint / "nested" / "weights.bin").write_bytes(b"same weights")

    first_fingerprint = agent.fingerprint_model_checkpoint(first)
    assert first_fingerprint == agent.fingerprint_model_checkpoint(second)
    assert len(first_fingerprint) == 64

    (second / "nested" / "weights.bin").write_bytes(b"changed weights")
    assert first_fingerprint != agent.fingerprint_model_checkpoint(second)


def test_checkpoint_fingerprint_binds_run_fingerprint_and_resume(tmp_path):
    search_result, run_config = _search_result(tmp_path)
    record = agent.phase6_result_from_search_result(
        search_result,
        "flat_exact_replay",
        run_config,
        [_policy_event()],
        [{"queries": ["generated query"], "metrics": _metrics()}],
    )
    result_path = tmp_path / "flat_exact_replay.jsonl"
    result_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    changed = json.loads(json.dumps(run_config))
    changed["model_checkpoint_fingerprint"] = "b" * 64

    with pytest.raises(ValueError, match="run configuration"):
        agent.run_condition_with_resume(
            [_example()],
            result_path,
            "flat_exact_replay",
            lambda _example: None,
            expected_run_fingerprint=phase4.run_config_fingerprint(changed),
        )


def test_runnable_input_recheck_rejects_changed_checkpoint_tree(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"before")
    baseline = tmp_path / "compressed_256.jsonl"
    baseline.write_text("immutable\n", encoding="utf-8")
    checkpoint_fingerprint = agent.fingerprint_model_checkpoint(checkpoint)
    baseline_fingerprint = agent.sha256_file(baseline)

    agent.verify_runnable_inputs_unchanged(
        checkpoint,
        checkpoint_fingerprint,
        baseline,
        baseline_fingerprint,
        "test lifecycle",
    )
    weights.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="Qwen checkpoint content changed"):
        agent.verify_runnable_inputs_unchanged(
            checkpoint,
            checkpoint_fingerprint,
            baseline,
            baseline_fingerprint,
            "test lifecycle",
        )


def test_validate_baseline_mode_does_not_fingerprint_qwen_checkpoint(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        condition="validate_baseline",
        eval_data=str(tmp_path / "eval.parquet"),
        eval_manifest=str(tmp_path / "eval.manifest.json"),
        phase5_baseline=str(tmp_path / "compressed_256.jsonl"),
        search_model=str(tmp_path / "checkpoint-does-not-need-to-exist"),
        seed=agent.SEED,
        gpu_memory_utilization=agent.GPU_MEMORY_UTILIZATION,
    )
    monkeypatch.setattr(agent, "parse_args", lambda: args)
    monkeypatch.setattr(agent, "load_eval_examples", lambda *_args: ([], {}))
    monkeypatch.setattr(
        agent,
        "validate_phase5_compressed_baseline",
        lambda *_args, **_kwargs: ([], {"validated": True}),
    )
    monkeypatch.setattr(
        agent,
        "fingerprint_asset",
        lambda _path: pytest.fail("validate_baseline must not hash the checkpoint"),
    )

    agent.main()


def test_launcher_contract_keeps_two_urls_and_phase5_baseline_read_only():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "http://127.0.0.1:8100/retrieve" in source
    assert "http://127.0.0.1:8101/retrieve" in source
    assert "phase5_observation_results/compressed_256.jsonl" in source
    assert "phase6_retriever_results/selected_candidate.json" in source
    assert "flat_exact_replay|ivfpq_selected" in source
    assert "git add" not in source


def test_phase5_compressed_baseline_validation_is_read_only(tmp_path):
    examples = [_example(index) for index in range(64)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = {"output": {"sha256": "eval-sha"}}
    config = _context_config(tmp_path)
    records = []
    for example in examples:
        search_result = {
            "schema_version": 1,
            "uid": example.uid,
            "data_source": example.data_source,
            "question": example.question,
            "ground_truth": example.ground_truth["target"],
            "mode": "search_rl",
            "model_path": "trained-model",
            "prediction": "wrong",
            "exact_match": 0,
            "end_to_end_latency_s": 1.0,
            "generation_latency_s": 0.5,
            "trajectory": "<answer>wrong</answer>",
            "run_config": config,
            "run_fingerprint": phase4.run_config_fingerprint(config),
            "number_of_actions": 1,
            "number_of_valid_actions": 1,
            "number_of_valid_searches": 0,
            "number_of_successful_retrievals": 0,
            "finished": True,
            "search_retrieval_failure_count": 0,
            "retrieval_latency_s": 0.0,
            "observation_truncation_count": 0,
            "trajectory_had_observation_truncation": False,
            "retrieved_observation_lengths_before_truncation": [],
            "retained_observation_lengths_after_truncation": [],
            "observation_excess_tokens": [],
        }
        records.append(
            phase5.phase5_result_from_search_result(
                search_result, "compressed_256", config, []
            )
        )
    baseline = tmp_path / "compressed_256.jsonl"
    baseline.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    before = baseline.read_bytes()

    loaded, audit = agent.validate_phase5_compressed_baseline(
        baseline,
        examples,
        manifest,
        manifest_path,
        expected_model="trained-model",
    )

    assert len(loaded) == 64
    assert audit["validated"] is True
    assert audit["sha256"] == hashlib.sha256(before).hexdigest()
    assert baseline.read_bytes() == before


def test_baseline_sha_helper_is_content_based(tmp_path):
    path = tmp_path / "baseline.jsonl"
    path.write_text("immutable\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert agent.sha256_file(path) == expected


def test_completed_result_can_repair_missing_server_audit(tmp_path, monkeypatch):
    candidate_path, candidate = _candidate(tmp_path)
    assert candidate_path.is_file()
    index_path = tmp_path / "flat.index"
    index_path.write_bytes(b"flat-index")
    ending_health = _health("flat")
    ending_health["index_path"] = str(index_path)
    ending_health["index_file_size_bytes"] = index_path.stat().st_size
    server_contract = agent.stable_server_contract(
        ending_health, "flat_exact_replay", candidate
    )
    ending_stats = {"request_count": 64, "cache_hit_count": 0}

    def fake_read_server_endpoint(_url, endpoint):
        if endpoint == "/healthz":
            return ending_health
        if endpoint == "/stats":
            return ending_stats
        raise AssertionError(endpoint)

    monkeypatch.setattr(agent, "read_server_endpoint", fake_read_server_endpoint)
    result_path = tmp_path / "flat_exact_replay.jsonl"
    result_path.write_text('{"uid":"nq:0"}\n', encoding="utf-8")
    audit_path = tmp_path / "flat_exact_replay.server_audit.json"
    assert not audit_path.exists()

    repaired_path = agent.finalize_server_audit(
        tmp_path,
        result_path,
        "flat_exact_replay",
        [{"uid": "nq:0"}],
        "run-fingerprint",
        "http://127.0.0.1:8100/retrieve",
        candidate,
        server_contract,
        ending_health,
        {"request_count": 64, "cache_hit_count": 0},
    )

    assert repaired_path == audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["row_count"] == 1
    assert audit["result_sha256"] == agent.sha256_file(result_path)
    assert audit["ending_stats"] == ending_stats
