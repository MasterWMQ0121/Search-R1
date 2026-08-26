import copy
import inspect
import json

import pytest

from experiments.phase6_retriever_serving import summarize_results as summary


def _server_contract(backend):
    return {
        "index_backend": backend,
        "index_path": f"/{backend}.index",
        "index_ntotal": 100,
        "index_dimension": 8,
        "metric_type": "inner_product",
        "metric_type_code": 0,
        "corpus_row_count": 100,
        "model_path": "/e5",
        "model_fingerprint": "e5-sha",
        "corpus_fingerprint": "corpus-sha",
        "index_fingerprint": "flat-sha" if backend == "flat" else "ivfpq-sha",
        "retrieval_encode_batch_size": 16,
        "faiss_thread_count": 8,
        "nprobe": None if backend == "flat" else 32,
        "cache_enabled": False,
        "index_file_size_bytes": 4096 if backend == "flat" else 1024,
    }


def _run_config(condition):
    backend = "flat" if condition == "flat_exact_replay" else "ivfpq"
    context_config = {
        "compressor": {"version": "phase5-extractive-v1"},
        "max_obs_length": 256,
        "retriever_url": f"http://{backend}/retrieve",
    }
    return {
        "prompt_contract": "phase4-benchmark-v1",
        "experiment_contract": "phase6-retriever-serving-agent-v1",
        "mode": condition,
        "observation_policy": "compressed_256",
        "eval_sha256": "eval-sha",
        "eval_manifest_sha256": "manifest-sha",
        "phase5_baseline_path": "/phase5/compressed_256.jsonl",
        "phase5_baseline_sha256": "phase5-sha",
        "phase5_baseline_run_fingerprint": "phase5-fingerprint",
        "selected_candidate_sha256": "candidate-sha",
        "candidate_selection_passed": True,
        "candidate_production_ready_for_agent_evaluation": True,
        "candidate_override_used": False,
        "model_path": "trained-model",
        "model_checkpoint_fingerprint": "a" * 64,
        "seed": 42,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.20,
        "attention_backend": "XFORMERS",
        "max_start_length": 768,
        "max_response_length": 128,
        "max_obs_length": 256,
        "max_prompt_length": 1408,
        "max_model_len": 1536,
        "max_turns": 2,
        "retriever_topk": 3,
        "retriever_return_metrics": True,
        "retriever_cache_enabled": False,
        "retriever_url": f"http://{backend}/retrieve",
        "phase5_context_run_config": context_config,
        "retriever_server_contract": _server_contract(backend),
    }


def _metric(backend, index):
    return {
        "request_id": f"{backend}-{index}",
        "query_count": 1,
        "topk": 3,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "embedding_cache_hit_count": 0,
        "embedding_cache_miss_count": 0,
        "index_backend": backend,
        "nprobe": None if backend == "flat" else 32,
        "faiss_thread_count": 8,
        "result_ids": [[index * 10, index * 10 + 1, index * 10 + 2]],
        **{field: 0.1 for field in summary.STAGE_TIMING_FIELDS},
    }


def _record(index, condition, exact_match=0):
    source = "nq" if index < 32 else "hotpotqa"
    backend = "flat" if condition == "flat_exact_replay" else "ivfpq"
    metric = _metric(backend, index)
    return {
        "uid": f"{source}:{index}",
        "data_source": source,
        "question": f"question-{index}",
        "ground_truth": [f"answer-{index}"],
        "mode": condition,
        "model_path": "trained-model",
        "prediction": f"prediction-{index}",
        "exact_match": exact_match,
        "end_to_end_latency_s": 2.0 if backend == "flat" else 1.5,
        "generation_latency_s": 1.0,
        "trajectory": "<answer>x</answer>",
        "run_config": _run_config(condition),
        "run_fingerprint": f"fingerprint-{condition}",
        "number_of_actions": 2,
        "number_of_valid_actions": 2,
        "number_of_valid_searches": 1,
        "number_of_successful_retrievals": 1,
        "finished": True,
        "search_retrieval_failure_count": 0,
        "retrieval_latency_s": 0.5 if backend == "flat" else 0.2,
        "observation_truncation_count": 0,
        "trajectory_had_observation_truncation": False,
        "retrieved_observation_lengths_before_truncation": [100],
        "retained_observation_lengths_after_truncation": [100],
        "observation_excess_tokens": [0],
        "observation_policy": "compressed_256",
        "raw_retrieved_observation_tokens": [400],
        "policy_output_observation_tokens": [100],
        "retained_observation_tokens": [100],
        "policy_compression_ratio": [0.25],
        "post_policy_truncation_count": 0,
        "documents_returned": [3],
        "documents_represented": [2],
        "sentences_considered": [8],
        "sentences_selected": [2],
        "evidence_compression_latency_s": 0.01,
        "zero_overlap_fallback_used": False,
        "partial_sentence_fallback_used": False,
        "selected_document_ranks": [[1, 2]],
        "selected_sentence_identifiers": [["doc1:s0", "doc2:s0"]],
        "retriever_queries": [f"query-{index}"],
        "retriever_request_ids": [metric["request_id"]],
        "retrieved_document_ids": [metric["result_ids"][0]],
        "retriever_request_metrics": [metric],
    }


def _baseline_record(index, exact_match=0):
    record = _record(index, "flat_exact_replay", exact_match)
    record["mode"] = summary.BASELINE_MODE
    record["run_config"] = {
        "eval_sha256": "eval-sha",
        "eval_manifest_sha256": "manifest-sha",
    }
    record["run_fingerprint"] = "phase5-fingerprint"
    return record


def _sets(ann_gain=False, drift=False):
    flat = [
        _record(index, "flat_exact_replay", int(index % 4 == 0))
        for index in range(64)
    ]
    ann = [
        _record(
            index,
            "ivfpq_selected",
            int((index % 4 == 0) or (ann_gain and index == 1)),
        )
        for index in range(64)
    ]
    baseline = [
        _baseline_record(
            index,
            int((index % 4 == 0) if not (drift and index == 0) else False),
        )
        for index in range(64)
    ]
    return {
        "flat_exact_replay": flat,
        "ivfpq_selected": ann,
        summary.BASELINE_MODE: baseline,
    }


def _audits():
    return {
        condition: {
            "ending_health": {
                **_server_contract(
                    "flat" if condition == "flat_exact_replay" else "ivfpq"
                ),
                "process_rss_bytes": 1234,
            }
        }
        for condition in summary.CONDITIONS
    }


def _baseline_audit():
    return {"validated": True, "sha256": "phase5-sha"}


def _candidate_audit():
    return {
        "sha256": "candidate-sha",
        "candidate_selection_passed": True,
        "production_ready_for_agent_evaluation": True,
        "override_used": False,
        "nprobe": 32,
        "faiss_thread_count": 8,
        "flat_index_sha256": "flat-sha",
        "ivfpq_index_sha256": "ivfpq-sha",
        "model_fingerprint": "e5-sha",
    }


def _serving_evidence(**check_overrides):
    checks = {key: True for key in summary.SERVING_EVIDENCE_CHECKS}
    checks.update(check_overrides)
    return {
        "validated": all(checks.values()),
        "checks": checks,
        "bindings": {
            "selected_candidate": {"path": "/candidate.json", "sha256": "candidate-sha"},
            "calibration_queries": {"path": "/queries.jsonl", "sha256": "queries-sha"},
            "calibration_manifest": {"path": "/manifest.json", "sha256": "manifest-sha"},
            "ivfpq_build_manifest": {"path": "/build.json", "sha256": "build-sha"},
            "index_benchmark": {"path": "/index.json", "sha256": "index-sha"},
            "server_benchmark": {"path": "/server.json", "sha256": "server-sha"},
        },
        "selected_http_agent_identities": {
            condition: {
                field: _server_contract(
                    "flat" if condition == "flat_exact_replay" else "ivfpq"
                )[field]
                for field in summary.HTTP_AGENT_IDENTITY_FIELDS
            }
            for condition in summary.CONDITIONS
        },
        "exact_semantic_serving": {
            "classification": "exact_semantic_serving_optimization"
        },
        "ann_tradeoff": {
            "classification": "ann_recall_latency_memory_tradeoff"
        },
    }


def test_summary_produces_primary_paired_and_baseline_drift_analysis():
    report, paired = summary.build_reports(
        _sets(ann_gain=True),
        _audits(),
        _baseline_audit(),
        _candidate_audit(),
        _serving_evidence(),
    )

    assert paired["primary"]["mode_a"] == "ivfpq_selected"
    assert paired["primary"]["mode_b"] == "flat_exact_replay"
    assert paired["primary"]["overall"]["bootstrap_samples"] == 10_000
    assert paired["primary"]["overall"]["bootstrap_seed"] == 42
    assert paired["primary"]["overall"]["mcnemar"][
        "exact_two_sided_p_value"
    ] == 1.0
    assert report["baseline_drift"]["quality_consistent"] is True
    assert report["readiness"]["product_comparison_ready"] is True
    assert report["exact_semantic_serving"]["classification"] == (
        "exact_semantic_serving_optimization"
    )
    assert report["ann_tradeoff"]["classification"] == (
        "ann_recall_latency_memory_tradeoff"
    )
    assert report["downstream_agent_impact"]["classification"] == (
        "downstream_agent_product_impact"
    )
    assert all(report["retriever_identity_crosscheck"]["checks"].values())


@pytest.mark.parametrize(
    ("field", "different_value"),
    (
        ("corpus_fingerprint", "different-corpus-sha"),
        ("retrieval_encode_batch_size", 32),
    ),
)
def test_agent_http_identity_mismatch_blocks_readiness(field, different_value):
    evidence = _serving_evidence()
    for identity in evidence["selected_http_agent_identities"].values():
        identity[field] = different_value

    report, _paired = summary.build_reports(
        _sets(), _audits(), _baseline_audit(), _candidate_audit(), evidence
    )

    check = f"agent_http_{field}_match"
    assert report["retriever_identity_crosscheck"]["checks"][check] is False
    assert report["readiness"]["checks"][check] is False
    assert report["readiness"]["engineering_pass"] is False
    assert report["readiness"]["product_comparison_ready"] is False


def test_pair_contract_rejects_search_checkpoint_content_mismatch():
    sets = _sets()
    for record in sets["ivfpq_selected"]:
        record["run_config"] = copy.deepcopy(record["run_config"])
        record["run_config"]["model_checkpoint_fingerprint"] = "different-model-sha"

    with pytest.raises(ValueError, match="model_checkpoint_fingerprint"):
        summary.build_reports(
            sets,
            _audits(),
            _baseline_audit(),
            _candidate_audit(),
            _serving_evidence(),
        )


def test_summary_reports_source_specific_quality_and_latency_stages():
    report, _paired = summary.build_reports(
        _sets(),
        _audits(),
        _baseline_audit(),
        _candidate_audit(),
        _serving_evidence(),
    )
    ann = report["conditions"]["ivfpq_selected"]
    assert ann["quality"]["by_source"]["nq"]["total"] == 32
    assert ann["quality"]["by_source"]["hotpotqa"]["total"] == 32
    assert ann["server"]["request_count"] == 64
    assert ann["server"]["stage_latency"]["query_encoding_s"]["p95_s"] == 0.1
    assert ann["resource"]["faiss_thread_count"] == 8
    assert ann["resource"]["index_file_size_bytes"] == 1024


def test_retrieval_id_agreement_uses_only_identical_generated_queries():
    sets = _sets()
    sets["ivfpq_selected"][0]["retriever_queries"] = ["diverged query"]
    sets["ivfpq_selected"][1]["retrieved_document_ids"] = [[999, 998, 997]]
    sets["ivfpq_selected"][1]["retriever_request_metrics"][0]["result_ids"] = [
        [999, 998, 997]
    ]
    report, _paired = summary.build_reports(
        sets,
        _audits(),
        _baseline_audit(),
        _candidate_audit(),
        _serving_evidence(),
    )
    agreement = report["retrieval_id_agreement"]
    assert agreement["diverged_query_count"] == 1
    assert agreement["comparable_identical_query_count"] == 63
    assert agreement["recall_at_3_mean"] < 1.0


def test_baseline_drift_is_explicit_and_blocks_product_readiness_only():
    report, paired = summary.build_reports(
        _sets(drift=True),
        _audits(),
        _baseline_audit(),
        _candidate_audit(),
        _serving_evidence(),
    )
    assert report["baseline_drift"]["mismatched_outcome_uids"] == ["nq:0"]
    assert report["readiness"]["engineering_pass"] is True
    assert report["readiness"]["product_comparison_ready"] is False
    assert paired["baseline_drift"]["overall"]["mcnemar"][
        "discordant_pair_count"
    ] == 1


def test_unqualified_candidate_override_never_becomes_product_ready():
    candidate = _candidate_audit()
    candidate.update({
        "candidate_selection_passed": False,
        "production_ready_for_agent_evaluation": False,
        "override_used": True,
    })
    sets = _sets()
    for condition in summary.CONDITIONS:
        for record in sets[condition]:
            record["run_config"]["candidate_selection_passed"] = False
            record["run_config"][
                "candidate_production_ready_for_agent_evaluation"
            ] = False
            record["run_config"]["candidate_override_used"] = True
    report, _paired = summary.build_reports(
        sets, _audits(), _baseline_audit(), candidate, _serving_evidence()
    )
    assert report["readiness"]["engineering_pass"] is True
    assert report["readiness"]["checks"][
        "candidate_selection_passed_without_override"
    ] is False
    assert report["readiness"]["product_comparison_ready"] is False


def test_pair_contract_rejects_agent_or_compressor_semantic_changes():
    sets = _sets()
    sets["ivfpq_selected"][0]["run_config"] = copy.deepcopy(
        sets["ivfpq_selected"][0]["run_config"]
    )
    sets["ivfpq_selected"][0]["run_config"]["max_obs_length"] = 512
    with pytest.raises(ValueError, match="mixes run configurations"):
        summary.build_reports(
            sets,
            _audits(),
            _baseline_audit(),
            _candidate_audit(),
            _serving_evidence(),
        )


def test_missing_required_serving_evidence_blocks_engineering_readiness():
    evidence = _serving_evidence(server_benchmark_complete=False)
    report, _paired = summary.build_reports(
        _sets(), _audits(), _baseline_audit(), _candidate_audit(), evidence
    )

    assert report["readiness"]["engineering_pass"] is False
    assert report["readiness"]["checks"]["server_benchmark_complete"] is False


def _write_bound_serving_artifacts(tmp_path):
    flat_path = tmp_path / "flat.index"
    ivfpq_path = tmp_path / "ivfpq.index"
    manifest_path = tmp_path / "ivfpq.manifest.json"
    index_benchmark_path = tmp_path / "index_benchmark.json"
    candidate_path = tmp_path / "selected_candidate.json"
    queries_path = tmp_path / "queries.jsonl"
    calibration_manifest_path = tmp_path / "queries.manifest.json"
    query_rows = [
        {"uid": "nq:1", "data_source": "nq", "question": "question one"},
        {
            "uid": "hotpotqa:2",
            "data_source": "hotpotqa",
            "question": "question two",
        },
    ]
    queries_path.write_text(
        "".join(json.dumps(row) + "\n" for row in query_rows),
        encoding="utf-8",
    )
    calibration_manifest = {
        "row_count": len(query_rows),
        "selected_uids": [row["uid"] for row in query_rows],
        "source_counts": {"nq": 1, "hotpotqa": 1},
        "leakage_overlap_audit": {"passed": True, "overlapping_uids": []},
        "answer_targets_included": False,
        "output": {
            "path": str(queries_path.resolve()),
            "sha256": summary.sha256_file(queries_path),
        },
    }
    calibration_manifest_path.write_text(
        json.dumps(calibration_manifest), encoding="utf-8"
    )
    config = {
        "source_index": str(flat_path.resolve()),
        "output_index": str(ivfpq_path.resolve()),
        "nlist": 16,
        "m": 2,
        "nbits": 8,
    }
    manifest = {
        "config": config,
        "config_fingerprint": summary._config_fingerprint(config),
        "source": {
            "path": str(flat_path.resolve()),
            "sha256": "flat-sha",
            "size_bytes": 4096,
        },
        "output": {
            "path": str(ivfpq_path.resolve()),
            "sha256": "ivfpq-sha",
            "size_bytes": 1024,
        },
        "compression_ratio_source_over_output": 4.0,
        "build_time_s": 12.0,
        "preflight": {"passed": True},
        "verification": {"checks": {"trained": True, "nonempty_search": True}},
        "source_sha256_unchanged": True,
        "sequential_original_id_assignment": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bindings = {
        "calibration_queries": {
            "path": str(queries_path.resolve()),
            "sha256": summary.sha256_file(queries_path),
        },
        "calibration_manifest": {
            "path": str(calibration_manifest_path.resolve()),
            "sha256": summary.sha256_file(calibration_manifest_path),
        },
        "query_embeddings": {
            "path": str((tmp_path / "embeddings.npz").resolve()),
            "sha256": "embeddings-sha",
        },
        "query_embeddings_manifest": {
            "path": str((tmp_path / "embeddings.manifest.json").resolve()),
            "sha256": "embeddings-manifest-sha",
        },
        "flat_index": {"path": str(flat_path.resolve()), "sha256": "flat-sha"},
        "ivfpq_index": {
            "path": str(ivfpq_path.resolve()),
            "sha256": "ivfpq-sha",
        },
        "ivfpq_build_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": summary.sha256_file(manifest_path),
        },
    }
    selected = {
        "index_backend": "ivfpq",
        "index_sha256": "ivfpq-sha",
        "index_file_size_bytes": 1024,
        "nprobe": 8,
        "faiss_thread_count": 4,
        "recall_at_3": 0.97,
        "pure_search_latency_s": {"mean": 0.01, "p95": 0.02},
    }
    selection = {
        "candidate_selection_passed": True,
        "production_ready_for_agent_evaluation": True,
        "minimum_recall_at_3": 0.95,
        "selected_candidate": selected,
        "selection_uses_qa_ground_truth": False,
    }
    index_benchmark = {
        "benchmark_kind": "phase6_calibration_only_index_search",
        "bindings": copy.deepcopy(bindings),
        "query_count": 128,
        "query_embeddings_encoded_once": True,
        "topk": 3,
        "index_geometry": {"dimension": 8, "ntotal": 100},
        "warmup_query_count": 8,
        "warmups_excluded": True,
        "faiss_thread_counts": [1, 4, 8, 16],
        "ivfpq_nprobes": [1, 4, 8],
        "flat_results": [],
        "candidate_selection": copy.deepcopy(selection),
        "selection_uses_qa_ground_truth": False,
    }
    index_benchmark_path.write_text(json.dumps(index_benchmark), encoding="utf-8")
    candidate = {
        **selection,
        "bindings": {
            **bindings,
            "index_benchmark": {
                "path": str(index_benchmark_path.resolve()),
                "sha256": summary.sha256_file(index_benchmark_path),
            },
        },
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    completion = {"passed": True, "selected_candidate_bound": True}
    server_benchmark = {
        "inputs": {
            "selected_candidate_path": str(candidate_path.resolve()),
            "selected_candidate_sha256": summary.sha256_file(candidate_path),
            "queries_sha256": summary.sha256_file(queries_path),
            "calibration_manifest_sha256": summary.sha256_file(
                calibration_manifest_path
            ),
        },
        "workloads": {
            "cold": {
                "exact": {
                    "role": "flat_exact",
                    "identity": _server_contract("flat"),
                },
                "ann": {
                    "role": "ivfpq_selected",
                    "identity": _server_contract("ivfpq"),
                },
            },
            "warm": {},
        },
        "comparisons": {},
        "exact_thread_batch_sweep": {
            "observed_thread_counts": [1, 4, 8, 16],
            "missing_thread_counts": [],
        },
        "completion_validation": completion,
    }
    server_path = tmp_path / "server_benchmark.json"
    server_path.write_text(json.dumps(server_benchmark), encoding="utf-8")
    return {
        "candidate": candidate,
        "candidate_path": candidate_path,
        "calibration_manifest": calibration_manifest,
        "manifest_path": manifest_path,
        "server_path": server_path,
        "completion": completion,
    }


def test_load_serving_evidence_sha_binds_and_separates_all_three_layers(
    tmp_path, monkeypatch
):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    monkeypatch.setattr(
        summary,
        "validate_complete_report",
        lambda _report: artifacts["completion"],
    )

    evidence = summary.load_serving_evidence(
        tmp_path, artifacts["candidate_path"], artifacts["candidate"]
    )

    assert evidence["validated"] is True
    assert all(evidence["checks"].values())
    assert evidence["bindings"]["server_benchmark"]["sha256"] == (
        summary.sha256_file(artifacts["server_path"])
    )
    assert evidence["exact_semantic_serving"]["classification"] == (
        "exact_semantic_serving_optimization"
    )
    assert evidence["ann_tradeoff"]["build"]["source_sha256_unchanged"] is True
    assert evidence["selected_http_agent_identities"]["flat_exact_replay"] == {
        "corpus_fingerprint": "corpus-sha",
        "retrieval_encode_batch_size": 16,
    }
    calibration = evidence["ann_tradeoff"]["offline_calibration"]["dataset_audit"]
    assert calibration["leakage_overlap_audit"]["passed"] is True
    assert calibration["answer_targets_included"] is False


def test_load_serving_evidence_rejects_stale_build_manifest_sha(tmp_path, monkeypatch):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    monkeypatch.setattr(
        summary,
        "validate_complete_report",
        lambda _report: artifacts["completion"],
    )
    artifacts["manifest_path"].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="build manifest SHA256"):
        summary.load_serving_evidence(
            tmp_path, artifacts["candidate_path"], artifacts["candidate"]
        )


def test_load_serving_evidence_rejects_server_candidate_sha_mismatch(
    tmp_path, monkeypatch
):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    server = json.loads(artifacts["server_path"].read_text(encoding="utf-8"))
    server["inputs"]["selected_candidate_sha256"] = "different-candidate"
    artifacts["server_path"].write_text(json.dumps(server), encoding="utf-8")
    monkeypatch.setattr(
        summary,
        "validate_complete_report",
        lambda _report: artifacts["completion"],
    )

    with pytest.raises(ValueError, match="different selected candidate"):
        summary.load_serving_evidence(
            tmp_path, artifacts["candidate_path"], artifacts["candidate"]
        )


def test_load_serving_evidence_rejects_http_pair_identity_mismatch(
    tmp_path, monkeypatch
):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    server = json.loads(artifacts["server_path"].read_text(encoding="utf-8"))
    server["workloads"]["cold"]["ann"]["identity"][
        "retrieval_encode_batch_size"
    ] = 32
    server["completion_validation"] = artifacts["completion"]
    artifacts["server_path"].write_text(json.dumps(server), encoding="utf-8")
    monkeypatch.setattr(
        summary,
        "validate_complete_report",
        lambda _report: artifacts["completion"],
    )

    with pytest.raises(ValueError, match="differ for retrieval_encode_batch_size"):
        summary.load_serving_evidence(
            tmp_path, artifacts["candidate_path"], artifacts["candidate"]
        )


def test_build_manifest_validation_rejects_stale_config_fingerprint(tmp_path):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text(encoding="utf-8"))
    manifest["config"]["nlist"] = 32

    with pytest.raises(ValueError, match="config fingerprint is stale"):
        summary._validate_build_manifest(
            manifest, artifacts["candidate"]["bindings"]
        )


def test_calibration_evidence_rejects_heldout_overlap_or_answer_targets(tmp_path):
    artifacts = _write_bound_serving_artifacts(tmp_path)
    rows = [{"uid": "nq:1"}, {"uid": "hotpotqa:2"}]
    query_binding = artifacts["candidate"]["bindings"]["calibration_queries"]
    manifest = copy.deepcopy(artifacts["calibration_manifest"])
    manifest["leakage_overlap_audit"] = {
        "passed": False,
        "overlapping_uids": ["nq:1"],
    }
    with pytest.raises(ValueError, match="held-out overlap audit"):
        summary._validate_calibration_artifacts(manifest, rows, query_binding)

    manifest = copy.deepcopy(artifacts["calibration_manifest"])
    manifest["answer_targets_included"] = True
    with pytest.raises(ValueError, match="answer-free selection"):
        summary._validate_calibration_artifacts(manifest, rows, query_binding)


def test_summary_source_contains_no_hard_coded_phase6_quality_claim():
    source = inspect.getsource(summary)
    assert "19 / 64" not in source
    assert "29.6875" not in source
    assert "quality-preserving" in source
    assert "Downstream Agent EM" in source
