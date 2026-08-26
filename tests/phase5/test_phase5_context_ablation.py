import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.phase4_benchmark import run_benchmark as phase4
from experiments.phase5_observation_context import run_context_ablation as phase5


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = (
    ROOT / "experiments" / "phase5_observation_context" / "run_context_ablation.sh"
)


def _example(index):
    source = "nq" if index < 32 else "hotpotqa"
    uid = f"{source}:{index}"
    return phase4.BenchmarkExample(
        uid=uid,
        data_source=source,
        question=f"question-{uid}",
        ground_truth={"target": [f"answer-{uid}"]},
        search_prompt=f"<answer>example</answer> Question: question-{uid}",
    )


def _phase4_baseline_record(example, run_config):
    return {
        "schema_version": 1,
        "uid": example.uid,
        "data_source": example.data_source,
        "question": example.question,
        "ground_truth": example.ground_truth["target"],
        "mode": "search_rl",
        "model_path": run_config["model_path"],
        "prediction": None,
        "exact_match": 0,
        "end_to_end_latency_s": 1.0,
        "generation_latency_s": 0.5,
        "trajectory": "<answer>wrong</answer>",
        "run_config": run_config,
        "run_fingerprint": phase4.run_config_fingerprint(run_config),
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


def _write_baseline(tmp_path, model="trained-model"):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "output": {"sha256": "eval-sha"},
        "selected_source_rows": [
            {"uid": _example(index).uid} for index in range(64)
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    run_config = phase4._default_test_run_config("search_rl", model)
    run_config.update({
        "eval_sha256": "eval-sha",
        "eval_manifest_sha256": manifest_sha,
    })
    records = [
        _phase4_baseline_record(_example(index), run_config)
        for index in range(64)
    ]
    baseline_path = tmp_path / "search_rl.jsonl"
    baseline_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return baseline_path, manifest_path, manifest, records


def _baseline_audit(tmp_path):
    return {
        "path": str(tmp_path / "phase4" / "search_rl.jsonl"),
        "sha256": "baseline-sha",
        "run_fingerprint": "baseline-fingerprint",
    }


def _run_config(tmp_path, condition):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return phase5.build_condition_run_config(
        condition,
        "trained-model",
        "http://retriever/retrieve",
        {"output": {"sha256": "eval-sha"}},
        manifest_path,
        _baseline_audit(tmp_path),
        attention_backend="XFORMERS",
    )


def _search_result(example, policy_length, retained_length, run_config):
    excess = max(policy_length - run_config["max_obs_length"], 0)
    truncated = retained_length < policy_length
    return {
        "schema_version": 1,
        "uid": example.uid,
        "data_source": example.data_source,
        "question": example.question,
        "ground_truth": example.ground_truth["target"],
        "mode": "search_rl",
        "model_path": "trained-model",
        "prediction": "answer",
        "exact_match": 1,
        "end_to_end_latency_s": 2.0,
        "generation_latency_s": 1.0,
        "trajectory": "<answer>answer</answer>",
        "run_config": run_config,
        "run_fingerprint": phase4.run_config_fingerprint(run_config),
        "number_of_actions": 2,
        "number_of_valid_actions": 2,
        "number_of_valid_searches": 1,
        "number_of_successful_retrievals": 1,
        "finished": True,
        "search_retrieval_failure_count": 0,
        "retrieval_latency_s": 0.25,
        "observation_truncation_count": int(truncated),
        "trajectory_had_observation_truncation": truncated,
        "retrieved_observation_lengths_before_truncation": [policy_length],
        "retained_observation_lengths_after_truncation": [retained_length],
        "observation_excess_tokens": [excess],
    }


def _policy_event(raw_tokens=None, policy_tokens=None):
    event = {
        "documents_returned": 3,
        "documents_represented": 2,
        "sentences_considered": 8,
        "sentences_selected": 2,
        "zero_overlap_fallback_used": False,
        "partial_sentence_fallback_used": False,
        "selected_document_ranks": [1, 2],
        "selected_sentence_identifiers": [
            "document_rank=1;sentence_index=0",
            "document_rank=2;sentence_index=1",
        ],
        "evidence_compression_latency_s": 0.01,
    }
    if raw_tokens is not None:
        event["raw_retrieved_observation_tokens"] = raw_tokens
    if policy_tokens is not None:
        event["policy_output_observation_tokens"] = policy_tokens
    return event


def test_condition_token_budgets_follow_the_rolling_prompt_formula():
    assert phase5.CONDITION_LIMITS == {
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
    for limits in phase5.CONDITION_LIMITS.values():
        assert limits["max_prompt_length"] == (
            768 + 128 * (2 - 1) + limits["max_obs_length"] * 2
        )
        assert limits["max_model_len"] == limits["max_prompt_length"] + 128


def test_run_configs_bind_distribution_shift_and_exact_compressor_source(tmp_path):
    raw_config = _run_config(tmp_path, "raw_512")
    compressed_config = _run_config(tmp_path, "compressed_256")

    assert raw_config["training_max_obs_length"] == 256
    assert raw_config["inference_context_distribution_shift"] is True
    assert raw_config["compressor"] is None
    assert compressed_config["inference_context_distribution_shift"] is False
    assert compressed_config["compressor"]["version"] == "phase5-extractive-v1"
    compressor_path = (
        ROOT
        / "experiments"
        / "phase5_observation_context"
        / "evidence_compressor.py"
    )
    assert compressed_config["compressor"]["source_sha256"] == hashlib.sha256(
        compressor_path.read_bytes()
    ).hexdigest()


def test_baseline_validation_is_read_only_and_checks_the_exact_contract(tmp_path):
    baseline_path, manifest_path, manifest, records = _write_baseline(tmp_path)
    before_bytes = baseline_path.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()

    validated, audit = phase5.validate_baseline_artifact(
        baseline_path,
        [_example(index) for index in range(64)],
        manifest,
        manifest_path,
        expected_model="trained-model",
    )

    assert validated == records
    assert audit["validated"] is True
    assert audit["row_count"] == 64
    assert audit["retrieval_failure_count"] == 0
    assert audit["evaluation_error_count"] == 0
    assert audit["sha256"] == before_sha
    assert baseline_path.read_bytes() == before_bytes
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == before_sha


def test_baseline_validation_rejects_a_different_checkpoint(tmp_path):
    baseline_path, manifest_path, manifest, _records = _write_baseline(
        tmp_path, model="other-model"
    )

    with pytest.raises(ValueError, match="unexpected model_path"):
        phase5.validate_baseline_artifact(
            baseline_path,
            [_example(index) for index in range(64)],
            manifest,
            manifest_path,
            expected_model="trained-model",
        )


class _FakeManager:
    def __init__(self):
        self.retrieval_calls = 0

    def _batch_search(self, queries):
        self.retrieval_calls += 1
        passages = [
            {
                "document": {
                    "id": f"doc-{rank}",
                    "contents": f"Title {rank}\nEvidence {rank}.",
                },
                "score": 1.0 / rank,
            }
            for rank in range(1, 4)
        ]
        return {"result": [passages for _query in queries]}

    def _passages2string(self, passages):
        return "".join(
            f"Doc {index}(Title: Title {index}) Evidence {index}.\n"
            for index, _passage in enumerate(passages, start=1)
        )


def test_raw_512_formatter_is_byte_identical_and_makes_one_retrieval_call():
    manager = _FakeManager()
    policy = phase5.ObservationPolicyManager("raw_512", tokenizer=object())
    policy.configure_manager(manager)

    output = manager.batch_search(["query"])

    assert output == [
        "Doc 1(Title: Title 1) Evidence 1.\n"
        "Doc 2(Title: Title 2) Evidence 2.\n"
        "Doc 3(Title: Title 3) Evidence 3.\n"
    ]
    assert manager.retrieval_calls == 1
    assert policy.events[0]["evidence_compression_latency_s"] == 0.0
    assert policy.events[0]["documents_represented"] == 3


def test_compressed_formatter_uses_query_and_hits_without_extra_retrieval_or_truth():
    class FakeCompressor:
        def __init__(self):
            self.calls = []

        def compress(self, query, passages, raw_observation=None):
            self.calls.append((query, passages, raw_observation))
            return {
                "content": "Doc 1: selected evidence.",
                "raw_retrieved_observation_tokens": 500,
                "policy_output_observation_tokens": 100,
                "documents_returned": 3,
                "documents_represented": 1,
                "sentences_considered": 6,
                "sentences_selected": 1,
                "zero_overlap_fallback_used": False,
                "partial_sentence_fallback_used": False,
                "selected_document_ranks": [1],
                "selected_sentence_identifiers": [
                    "document_rank=1;sentence_index=0"
                ],
            }

    manager = _FakeManager()
    compressor = FakeCompressor()
    policy = phase5.ObservationPolicyManager(
        "compressed_256", tokenizer=object(), compressor=compressor
    )
    policy.configure_manager(manager)

    assert manager.batch_search(["generated query"]) == [
        "Doc 1: selected evidence."
    ]
    assert manager.retrieval_calls == 1
    assert len(compressor.calls) == 1
    query, passages, raw_observation = compressor.calls[0]
    assert query == "generated query"
    assert len(passages) == 3
    assert raw_observation.startswith("Doc 1(Title: Title 1)")
    assert "ground_truth" not in FakeCompressor.compress.__code__.co_varnames


def test_raw_512_result_schema_reports_no_compression(tmp_path):
    run_config = _run_config(tmp_path, "raw_512")
    search_result = _search_result(_example(0), 500, 500, run_config)
    event = phase5.ObservationPolicyManager._raw_event([{}, {}, {}])

    record = phase5.phase5_result_from_search_result(
        search_result, "raw_512", run_config, [event]
    )

    phase5.validate_phase5_result(record, "raw_512")
    assert record["raw_retrieved_observation_tokens"] == [500]
    assert record["policy_output_observation_tokens"] == [500]
    assert record["policy_compression_ratio"] == [1.0]
    assert record["evidence_compression_latency_s"] == 0.0
    assert record["post_policy_truncation_count"] == 0


def test_raw_512_schema_rejects_false_compression_activity(tmp_path):
    run_config = _run_config(tmp_path, "raw_512")
    search_result = _search_result(_example(0), 500, 500, run_config)
    event = phase5.ObservationPolicyManager._raw_event([{}, {}, {}])
    event["sentences_selected"] = 1
    event["selected_sentence_identifiers"] = ["not-a-raw-metric"]

    record = phase5.phase5_result_from_search_result(
        search_result, "raw_512", run_config, [event]
    )

    with pytest.raises(ValueError, match="must not report evidence-compression"):
        phase5.validate_phase5_result(record, "raw_512")


def test_compressed_result_distinguishes_policy_from_safety_truncation(tmp_path):
    run_config = _run_config(tmp_path, "compressed_256")
    search_result = _search_result(_example(0), 200, 200, run_config)

    record = phase5.phase5_result_from_search_result(
        search_result,
        "compressed_256",
        run_config,
        [_policy_event(raw_tokens=500, policy_tokens=200)],
    )

    phase5.validate_phase5_result(record, "compressed_256")
    assert record["raw_retrieved_observation_tokens"] == [500]
    assert record["policy_output_observation_tokens"] == [200]
    assert record["retained_observation_tokens"] == [200]
    assert record["policy_compression_ratio"] == [0.4]
    assert record["post_policy_truncation_count"] == 0
    assert record["selected_document_ranks"] == [[1, 2]]


def test_schema_rejects_compressed_output_over_256_tokens(tmp_path):
    run_config = _run_config(tmp_path, "compressed_256")
    search_result = _search_result(_example(0), 257, 256, run_config)
    record = phase5.phase5_result_from_search_result(
        search_result,
        "compressed_256",
        run_config,
        [_policy_event(raw_tokens=500, policy_tokens=257)],
    )

    with pytest.raises(ValueError, match="exceeds 256"):
        phase5.validate_phase5_result(record, "compressed_256")


def test_phase5_resume_skips_completed_uids_and_preserves_order(tmp_path):
    examples = [_example(0), _example(1)]
    run_config = _run_config(tmp_path, "raw_512")

    def make_record(example):
        return phase5.phase5_result_from_search_result(
            _search_result(example, 500, 500, run_config),
            "raw_512",
            run_config,
            [phase5.ObservationPolicyManager._raw_event([{}, {}, {}])],
        )

    result_path = tmp_path / "raw_512.jsonl"
    completed = make_record(examples[0])
    result_path.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    evaluated = []

    records = phase5.run_condition_with_resume(
        examples,
        result_path,
        "raw_512",
        lambda example: evaluated.append(example.uid) or make_record(example),
        expected_run_fingerprint=phase4.run_config_fingerprint(run_config),
        immutable_baseline_path=tmp_path / "phase4" / "search_rl.jsonl",
    )

    assert evaluated == [examples[1].uid]
    assert [record["uid"] for record in records] == [
        examples[0].uid,
        examples[1].uid,
    ]
    assert [
        json.loads(line)["uid"]
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ] == [examples[0].uid, examples[1].uid]


def test_writer_refuses_to_target_the_immutable_baseline(tmp_path):
    baseline = tmp_path / "search_rl.jsonl"
    baseline.write_text("immutable\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not equal"):
        phase5.run_condition_with_resume(
            [],
            baseline,
            "raw_512",
            lambda _example: None,
            immutable_baseline_path=baseline,
        )

    assert baseline.read_text(encoding="utf-8") == "immutable\n"


@pytest.mark.parametrize(
    "relative_output",
    (
        ".",
        "phase5",
        "a/b/c",
        "../phase4_benchmark_results/phase5",
    ),
)
def test_phase5_results_directory_rejects_phase4_tree(tmp_path, relative_output):
    phase4_results = tmp_path / "phase4_benchmark_results"
    baseline = phase4_results / "search_rl.jsonl"
    output = phase4_results / relative_output

    with pytest.raises(ValueError, match="or any of its descendants"):
        phase5.validate_results_directory_isolation(output, baseline)


@pytest.mark.parametrize(
    "output_name",
    (
        "phase5_observation_results",
        "phase4_benchmark_results_backup",
        "another_parent/phase5_observation_results",
    ),
)
def test_phase5_results_directory_allows_separate_trees(tmp_path, output_name):
    baseline = tmp_path / "phase4_benchmark_results" / "search_rl.jsonl"
    output = tmp_path / output_name

    assert phase5.validate_results_directory_isolation(output, baseline) == (
        output.resolve()
    )


def test_phase4_baseline_remains_a_valid_read_only_input(tmp_path):
    phase4_results = tmp_path / "phase4_benchmark_results"
    baseline = phase4_results / "search_rl.jsonl"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("immutable baseline\n", encoding="utf-8")

    isolated = tmp_path / "phase5_observation_results"
    assert phase5.validate_results_directory_isolation(isolated, baseline) == (
        isolated.resolve()
    )
    assert baseline.read_text(encoding="utf-8") == "immutable baseline\n"


def test_phase5_results_directory_rejects_symlink_into_phase4(tmp_path):
    phase4_results = tmp_path / "phase4_benchmark_results"
    nested = phase4_results / "nested"
    nested.mkdir(parents=True)
    symlink = tmp_path / "phase5_symlink"
    try:
        symlink.symlink_to(nested, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported in this environment")

    with pytest.raises(ValueError, match="or any of its descendants"):
        phase5.validate_results_directory_isolation(
            symlink, phase4_results / "search_rl.jsonl"
        )


def test_baseline_only_main_never_initializes_vllm_or_writes_results(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        condition="raw_256_baseline",
        eval_data=str(tmp_path / "eval.parquet"),
        eval_manifest=str(tmp_path / "manifest.json"),
        output_dir=str(tmp_path / "results"),
        baseline_results=str(tmp_path / "phase4" / "search_rl.jsonl"),
        search_model="trained-model",
        retriever_url="http://retriever/retrieve",
        gpu_memory_utilization=0.20,
        seed=42,
        overwrite=False,
    )
    monkeypatch.setattr(phase5, "parse_args", lambda: args)
    monkeypatch.setattr(
        phase5, "load_eval_examples", lambda *_args: ([object()] * 64, {})
    )
    monkeypatch.setattr(
        phase5,
        "validate_baseline_artifact",
        lambda *_args, **_kwargs: ([], {"validated": True}),
    )
    monkeypatch.setattr(
        phase5,
        "VLLMGenerator",
        lambda *_args, **_kwargs: pytest.fail("baseline validation loaded vLLM"),
    )

    phase5.main()

    assert not (tmp_path / "results").exists()


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


def test_phase5_launcher_is_syntax_valid_and_never_overwrites_the_baseline():
    subprocess.run(["bash", "-n", str(RUN_SCRIPT)], check=True)
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "run_condition raw_256_baseline" in source
    assert "run_condition raw_512" in source
    assert "run_condition compressed_256" in source
    assert 'if [[ "${OVERWRITE}" == "true" && "${condition}" != "raw_256_baseline" ]]' in source
    assert 'BASELINE_PATH="${PHASE5_BASELINE_PATH:' in source
    assert 'raw_512 obs / prompt / model length: 512 / 1920 / 2048' in source
    assert 'compressed_256 obs / prompt / model length: 256 / 1408 / 1536' in source
