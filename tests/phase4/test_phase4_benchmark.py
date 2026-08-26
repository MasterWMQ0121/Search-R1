import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from experiments.phase4_benchmark import run_benchmark as benchmark
from experiments.phase4_benchmark import summarize_results as summary
import search_r1.llm_agent.generation as agent_generation
from verl import DataProto


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "experiments" / "phase4_benchmark" / "run_benchmark.sh"


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return f"<chat>{messages[0]['content']}</chat><assistant>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in token_ids if int(token_id))

    def __call__(self, texts, **_kwargs):
        encoded = [self.encode(text) for text in texts]
        width = max((len(tokens) for tokens in encoded), default=0)
        return {
            "input_ids": torch.tensor([
                tokens + [self.pad_token_id] * (width - len(tokens))
                for tokens in encoded
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                [1] * len(tokens) + [0] * (width - len(tokens))
                for tokens in encoded
            ], dtype=torch.long),
        }

    def batch_decode(self, token_rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_rows]


class StrictDecodeTokenizer(FakeTokenizer):
    def __init__(self):
        self.decode_inputs = []

    def decode(self, token_ids, skip_special_tokens=False):
        token_ids = list(token_ids)
        if any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("strict tokenizer requires plain Python integer token IDs")
        self.decode_inputs.append(token_ids)
        return "".join(chr(token_id) for token_id in token_ids if token_id)

    def batch_decode(self, token_rows, skip_special_tokens=True):
        decoded = []
        for row in token_rows:
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().tolist()
            else:
                row = list(row)
            if any(type(token_id) is not int for token_id in row):
                raise TypeError("strict tokenizer requires integer token IDs")
            decoded.append("".join(chr(token_id) for token_id in row if token_id))
        return decoded


class FakeGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_batch(self, prompt_token_ids):
        self.calls.append(prompt_token_ids)
        text = self.outputs.pop(0)
        return [benchmark.GenerationOutput(text=text, token_ids=[ord(char) for char in text])]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve_one(self, question):
        self.queries.append(question)
        return benchmark.RetrievalOutput(
            context="Doc 1(Title: Paris) Paris is the capital of France.\n" * 20,
            documents=[
                {"rank": rank, "id": f"paris-{rank}", "title": "Paris", "score": 0.9}
                for rank in range(1, 4)
            ],
            latency_s=0.25,
        )


def _example(uid="nq:7", source="nq", answer="Paris"):
    question = "What is the capital of France?"
    search_prompt = (
        "Use search and answer inside tags. For example, "
        f"<answer> Beijing </answer>. Question: {question}"
    )
    return benchmark.BenchmarkExample(
        uid=uid,
        data_source=source,
        question=question,
        ground_truth={"target": [answer]},
        search_prompt=search_prompt,
    )


def _record(mode, uid, source, exact_match, latency, **updates):
    run_config = benchmark._default_test_run_config(mode, "model")
    record = {
        "schema_version": 1,
        "uid": uid,
        "data_source": source,
        "question": f"question-{uid}",
        "ground_truth": ["answer"],
        "mode": mode,
        "model_path": "model",
        "prediction": "answer" if exact_match else "wrong",
        "exact_match": exact_match,
        "end_to_end_latency_s": latency,
        "generation_latency_s": latency / 2,
        "trajectory": "<answer>answer</answer>",
        "run_config": run_config,
        "run_fingerprint": benchmark.run_config_fingerprint(run_config),
    }
    if mode == "static_rag":
        record.update({
            "retrieval_latency_s": 0.25,
            "retrieval_call_count": 1,
            "retrieved_documents": [
                {"rank": rank, "id": f"doc-{rank}", "title": "title", "score": 1.0}
                for rank in range(1, 4)
            ],
            "static_context_token_budget": 512,
            "retrieved_context_tokens_before_truncation": 400,
            "retrieved_context_tokens_retained": 400,
        })
    if mode in benchmark.AGENT_MODES:
        record.update({
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
        })
    record.update(updates)
    return record


def _search_output(response_ids, valid_length=None):
    if not isinstance(response_ids, torch.Tensor):
        response_ids = torch.tensor(response_ids)
    if response_ids.ndim != 1:
        raise ValueError("test response IDs must be one-dimensional")
    if valid_length is None:
        valid_length = response_ids.shape[0]
    prompts = torch.tensor([[11, 12]], dtype=torch.long)
    response_mask = torch.zeros((1, response_ids.shape[0]), dtype=torch.long)
    response_mask[:, :valid_length] = 1
    return DataProto.from_dict({
        "prompts": prompts,
        "responses": response_ids.unsqueeze(0),
        "attention_mask": torch.cat(
            [torch.ones_like(prompts), response_mask], dim=1
        ),
    })


def test_base_search_is_supported_and_existing_fingerprints_are_unchanged():
    assert benchmark.MODES == (
        "direct",
        "static_rag",
        "base_search",
        "search_rl",
    )
    assert benchmark.AGENT_MODES == ("base_search", "search_rl")
    assert benchmark.MODE_RESULT_FIELDS["base_search"] == (
        benchmark.MODE_RESULT_FIELDS["search_rl"]
    )

    expected_existing_fingerprints = {
        "direct": "3e8ada978d4ed4cced87b4e7b1f693187bb9ea533592cdd14ad02be73bf44089",
        "static_rag": "50b659cc974342c340bc5eff8c77b1c3b91b9851465f42eb44353c5b744fd587",
        "search_rl": "6a85575a1a0b9eabb04f7a455d9d9aeb15a080926107d838f50c5a35e9b6228e",
    }
    assert {
        mode: benchmark.run_config_fingerprint(
            benchmark._default_test_run_config(mode, "model")
        )
        for mode in expected_existing_fingerprints
    } == expected_existing_fingerprints


def test_base_and_trained_search_configs_differ_only_by_mode_and_model():
    base = benchmark._default_test_run_config("base_search", "base-model")
    trained = benchmark._default_test_run_config("search_rl", "trained-model")

    assert base["model_path"] == "base-model"
    assert trained["model_path"] == "trained-model"
    assert base["mode"] == "base_search"
    assert trained["mode"] == "search_rl"
    for config in (base, trained):
        assert config["greedy"] is True
        assert config["seed"] == 42
        assert config["retriever_topk"] == 3
        assert config["max_turns"] == 2
        assert config["max_start_length"] == 768
        assert config["max_response_length"] == 128
        assert config["max_obs_length"] == 256
        assert config["max_prompt_length"] == 1408

    assert {
        key: value for key, value in base.items()
        if key not in {"mode", "model_path"}
    } == {
        key: value for key, value in trained.items()
        if key not in {"mode", "model_path"}
    }
    assert benchmark.run_config_fingerprint(base) != (
        benchmark.run_config_fingerprint(trained)
    )


def test_base_and_trained_search_wrappers_share_one_agent_evaluator(monkeypatch):
    calls = []

    def fake_evaluate_search_agent(*args):
        calls.append(args)
        return {"mode": args[3], "model_path": args[4]}

    monkeypatch.setattr(
        benchmark, "evaluate_search_agent", fake_evaluate_search_agent
    )
    example = _example()
    tokenizer = FakeTokenizer()
    generator = FakeGenerator([])

    base = benchmark.evaluate_base_search(
        example, tokenizer, generator, "base-model", "http://retriever/retrieve"
    )
    trained = benchmark.evaluate_search_rl(
        example, tokenizer, generator, "trained-model", "http://retriever/retrieve"
    )

    assert base == {"mode": "base_search", "model_path": "base-model"}
    assert trained == {"mode": "search_rl", "model_path": "trained-model"}
    assert calls[0][:3] == calls[1][:3]
    assert calls[0][5:] == calls[1][5:]


@pytest.mark.parametrize(
    ("mode", "expected_model", "expected_evaluator"),
    [
        ("direct", "base-model", "direct"),
        ("static_rag", "base-model", "static_rag"),
        ("base_search", "base-model", "base_search"),
        ("search_rl", "trained-model", "search_rl"),
    ],
)
def test_main_selects_the_expected_checkpoint_and_evaluator(
    tmp_path, monkeypatch, mode, expected_model, expected_evaluator
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        mode=mode,
        eval_data=str(tmp_path / "eval.parquet"),
        eval_manifest=str(manifest_path),
        output_dir=str(tmp_path / "results"),
        base_model="base-model",
        search_model="trained-model",
        retriever_url="http://retriever/retrieve",
        retriever_topk=3,
        static_context_token_budget=512,
        max_start_length=768,
        max_response_length=128,
        max_obs_length=256,
        max_turns=2,
        max_prompt_length=1408,
        gpu_memory_utilization=0.20,
        seed=42,
        overwrite=False,
    )
    example = _example()
    manifest = {"output": {"sha256": "test-eval"}}
    observed = {"evaluators": []}

    class StubVLLMGenerator:
        def __init__(self, model_path, **_kwargs):
            observed["initialized_model"] = model_path
            self.tokenizer = FakeTokenizer()

    def evaluator(name, model_index):
        def evaluate(*call_args):
            observed["evaluators"].append(name)
            observed["evaluator_model"] = call_args[model_index]
            observed["run_config"] = call_args[-1]
            return {"mode": name}

        return evaluate

    def fake_run_with_resume(
        examples, _result_path, selected_mode, evaluate_one, **_kwargs
    ):
        assert selected_mode == mode
        assert examples == [example]
        evaluate_one(example)
        return []

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark, "load_eval_examples", lambda *_args: ([example], manifest)
    )
    monkeypatch.setattr(benchmark, "VLLMGenerator", StubVLLMGenerator)
    monkeypatch.setattr(
        benchmark, "ExistingRetrieverClient", lambda *_args: object()
    )
    monkeypatch.setattr(benchmark, "evaluate_direct", evaluator("direct", 3))
    monkeypatch.setattr(
        benchmark, "evaluate_static_rag", evaluator("static_rag", 4)
    )
    monkeypatch.setattr(
        benchmark, "evaluate_base_search", evaluator("base_search", 3)
    )
    monkeypatch.setattr(
        benchmark, "evaluate_search_rl", evaluator("search_rl", 3)
    )
    monkeypatch.setattr(benchmark, "run_with_resume", fake_run_with_resume)

    benchmark.main()

    assert observed["initialized_model"] == expected_model
    assert observed["evaluator_model"] == expected_model
    assert observed["evaluators"] == [expected_evaluator]
    assert observed["run_config"]["mode"] == mode
    assert observed["run_config"]["model_path"] == expected_model


def test_search_trajectory_decode_preserves_long_token_ids_and_order():
    tokenizer = StrictDecodeTokenizer()
    token_ids = [ord(character) for character in "<answer>Paris</answer>"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trajectory = benchmark._decode_search_trajectory(
            tokenizer, _search_output(torch.tensor(token_ids, dtype=torch.long))
        )

    assert trajectory == "<answer>Paris</answer>"
    assert tokenizer.decode_inputs == [token_ids]
    assert not [warning for warning in caught if warning.category is RuntimeWarning]


def test_search_trajectory_decode_accepts_integer_valued_float_ids():
    tokenizer = StrictDecodeTokenizer()
    token_ids = [ord(character) for character in "<answer>Paris</answer>"]

    with pytest.warns(
        RuntimeWarning,
        match=r"Search-RL trajectory responses: dtype=torch.float32, shape=.*"
        r"exact integer-valued token IDs are being normalized to int64",
    ) as caught:
        trajectory = benchmark._decode_search_trajectory(
            tokenizer, _search_output(torch.tensor(token_ids, dtype=torch.float32))
        )

    assert trajectory == "<answer>Paris</answer>"
    assert tokenizer.decode_inputs == [token_ids]
    assert len(caught) == 1


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([65.0], dtype=torch.float16),
        torch.tensor([65.0], dtype=torch.bfloat16),
        np.array([65.0], dtype=np.float16),
    ],
)
def test_token_id_normalization_rejects_lossy_float_dtypes(token_ids):
    with pytest.raises(ValueError, match="may already have lost integer identity") as error:
        benchmark._normalize_token_ids_for_decode(token_ids, "lossy token IDs")

    error_text = str(error.value)
    assert "lossy token IDs" in error_text
    assert "dtype=" in error_text
    assert "shape=" in error_text
    assert "sample=" in error_text


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([0, 1 << 24], dtype=torch.float32),
        np.array([0, 1 << 24], dtype=np.float32),
    ],
)
def test_float32_token_ids_within_exact_integer_range_succeed(token_ids):
    with pytest.warns(RuntimeWarning, match="normalized to int64"):
        normalized = benchmark._normalize_token_ids_for_decode(
            token_ids, "float32 token IDs"
        )

    assert normalized == [0, 1 << 24]
    assert all(type(token_id) is int for token_id in normalized)


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([float((1 << 24) + 2)], dtype=torch.float32),
        np.array([float((1 << 24) + 2)], dtype=np.float32),
    ],
)
def test_float32_token_ids_outside_exact_integer_range_fail(token_ids):
    with pytest.raises(ValueError, match="exact consecutive-integer range"):
        benchmark._normalize_token_ids_for_decode(token_ids, "float32 token IDs")


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([0, 1 << 53], dtype=torch.float64),
        np.array([0, 1 << 53], dtype=np.float64),
    ],
)
def test_float64_token_ids_within_exact_integer_range_succeed(token_ids):
    with pytest.warns(RuntimeWarning, match="normalized to int64"):
        normalized = benchmark._normalize_token_ids_for_decode(
            token_ids, "float64 token IDs"
        )

    assert normalized == [0, 1 << 53]
    assert all(type(token_id) is int for token_id in normalized)


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([float((1 << 53) + 2)], dtype=torch.float64),
        np.array([float((1 << 53) + 2)], dtype=np.float64),
    ],
)
def test_float64_token_ids_outside_exact_integer_range_fail(token_ids):
    with pytest.raises(ValueError, match="exact consecutive-integer range"):
        benchmark._normalize_token_ids_for_decode(token_ids, "float64 token IDs")


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (torch.tensor([65.5], dtype=torch.float32), "exactly integer-valued"),
        (torch.tensor([float("nan")], dtype=torch.float32), "must be finite"),
        (torch.tensor([float("inf")], dtype=torch.float32), "must be finite"),
        (torch.tensor([-1], dtype=torch.long), "negative token IDs"),
        (torch.tensor([True], dtype=torch.bool), "boolean token IDs"),
    ],
)
def test_search_trajectory_decode_rejects_invalid_token_ids(token_ids, message):
    tokenizer = StrictDecodeTokenizer()

    with pytest.raises(ValueError, match=message) as error:
        benchmark._decode_search_trajectory(tokenizer, _search_output(token_ids))

    error_text = str(error.value)
    assert "Search-RL trajectory responses" in error_text
    assert "dtype=" in error_text
    assert "shape=" in error_text
    assert "sample=" in error_text
    assert tokenizer.decode_inputs == []


def test_search_trajectory_decode_handles_zero_valid_tokens_explicitly():
    tokenizer = StrictDecodeTokenizer()

    trajectory = benchmark._decode_search_trajectory(
        tokenizer,
        _search_output(torch.tensor([65], dtype=torch.long), valid_length=0),
    )

    assert trajectory == ""
    assert tokenizer.decode_inputs == [[]]


def test_search_trajectory_decode_rejects_multidimensional_token_ids():
    tokenizer = StrictDecodeTokenizer()
    output = _search_output(torch.tensor([65], dtype=torch.long))
    output.batch["responses"] = torch.tensor([[[65]]], dtype=torch.long)

    with pytest.raises(ValueError, match="one-dimensional"):
        benchmark._decode_search_trajectory(tokenizer, output)


def test_static_rag_uses_exactly_one_retrieval_and_shared_em_semantics():
    tokenizer = FakeTokenizer()
    generator = FakeGenerator(["<think>known</think><answer>Paris</answer>"])
    retriever = FakeRetriever()

    result = benchmark.evaluate_static_rag(
        _example(), tokenizer, generator, retriever, "base-model",
        max_start_length=10_000, context_token_budget=512,
    )

    assert retriever.queries == ["What is the capital of France?"]
    assert len(generator.calls) == 1
    assert result["retrieval_call_count"] == 1
    assert result["static_context_token_budget"] == 512
    assert result["retrieved_context_tokens_before_truncation"] > 512
    assert result["retrieved_context_tokens_retained"] == 512
    assert result["prediction"] == "Paris"
    assert result["exact_match"] == 1
    benchmark.validate_result_record(result, "static_rag")


@pytest.mark.parametrize(
    ("mode", "evaluator", "model_path"),
    [
        ("base_search", benchmark.evaluate_base_search, "base-model"),
        ("search_rl", benchmark.evaluate_search_rl, "trained-model"),
    ],
)
def test_search_modes_share_agent_loop_and_report_the_same_metrics(
    monkeypatch, mode, evaluator, model_path
):
    tokenizer = FakeTokenizer()
    generator = FakeGenerator([
        "<think>need evidence</think><search>capital of France</search>",
        "<think>use evidence</think><answer>Paris</answer>",
    ])
    retrieval_calls = []

    def fake_batch_search(self, queries):
        retrieval_calls.append(queries)
        passages = [
            {
                "document": {"id": f"doc-{index}", "contents": f"Title {index}\nParis evidence."},
                "score": 1.0 - index / 10,
            }
            for index in range(3)
        ]
        return {"result": [passages for _query in queries]}

    monkeypatch.setattr(
        agent_generation.LLMGenerationManager, "_batch_search", fake_batch_search
    )
    result = evaluator(
        _example(), tokenizer, generator, model_path,
        "http://retriever/retrieve",
    )

    assert len(generator.calls) == 2
    assert retrieval_calls == [["capital of France"]]
    assert result["mode"] == mode
    assert result["model_path"] == model_path
    assert result["prediction"] == "Paris"
    assert result["exact_match"] == 1
    assert result["number_of_actions"] == 2
    assert result["number_of_valid_actions"] == 2
    assert result["number_of_valid_searches"] == 1
    assert result["number_of_successful_retrievals"] == 1
    assert result["finished"] is True
    assert result["search_retrieval_failure_count"] == 0
    assert len(result["retrieved_observation_lengths_before_truncation"]) == 1
    benchmark.validate_result_record(result, mode)


def test_search_rl_full_flow_decodes_integer_valued_float_output(monkeypatch):
    tokenizer = StrictDecodeTokenizer()
    generator = FakeGenerator([
        "<think>need evidence</think><search>capital of France</search>",
        "<think>use evidence</think><answer>Paris</answer>",
    ])
    retrieval_calls = []

    def fake_batch_search(self, queries):
        retrieval_calls.append(queries)
        passages = [
            {
                "document": {
                    "id": f"doc-{index}",
                    "contents": f"Title {index}\nParis evidence.",
                },
                "score": 1.0 - index / 10,
            }
            for index in range(3)
        ]
        return {"result": [passages for _query in queries]}

    original_run_llm_loop = agent_generation.LLMGenerationManager.run_llm_loop
    observed = {}

    def run_llm_loop_with_float_responses(self, gen_batch, initial_input_ids):
        output = original_run_llm_loop(self, gen_batch, initial_input_ids)
        observed["source_dtype"] = output.batch["responses"].dtype
        output.batch["responses"] = output.batch["responses"].to(torch.float32)
        observed["decode_boundary_dtype"] = output.batch["responses"].dtype
        return output

    monkeypatch.setattr(
        agent_generation.LLMGenerationManager, "_batch_search", fake_batch_search
    )
    monkeypatch.setattr(
        agent_generation.LLMGenerationManager,
        "run_llm_loop",
        run_llm_loop_with_float_responses,
    )

    with pytest.warns(
        RuntimeWarning,
        match=r"Search-RL trajectory responses: dtype=torch.float32, shape=.*"
        r"exact integer-valued token IDs are being normalized to int64",
    ) as caught:
        result = benchmark.evaluate_search_rl(
            _example(), tokenizer, generator, "trained-model", "http://retriever/retrieve"
        )

    assert observed == {
        "source_dtype": torch.int64,
        "decode_boundary_dtype": torch.float32,
    }
    assert len(caught) == 1
    assert len(generator.calls) == 2
    assert retrieval_calls == [["capital of France"]]
    assert result["prediction"] == "Paris"
    assert result["exact_match"] == 1
    assert result["number_of_actions"] == 2
    assert result["number_of_valid_searches"] == 1
    assert all(
        type(token_id) is int
        for decode_input in tokenizer.decode_inputs
        for token_id in decode_input
    )
    assert "<search>capital of France</search>" in result["trajectory"]
    assert result["trajectory"].endswith("<answer>Paris</answer>")


def test_result_schema_rejects_static_rag_without_exactly_one_retrieval():
    record = _record("static_rag", "nq:1", "nq", 1, 1.0)
    record["retrieval_call_count"] = 2
    with pytest.raises(ValueError, match="exactly one retrieval"):
        benchmark.validate_result_record(record, "static_rag")


def test_scoring_fails_if_prompt_cropping_removed_the_format_example():
    with pytest.raises(ValueError, match="lost its answer-format example"):
        benchmark.score_completion(
            "Question: capital?", "<answer>Paris</answer>", {"target": ["Paris"]}
        )


def test_latency_quality_search_metrics_and_percentage_point_comparisons():
    direct = [
        _record("direct", "nq:1", "nq", 0, 1.0),
        _record("direct", "hotpotqa:2", "hotpotqa", 0, 3.0),
    ]
    static = [
        _record(
            "static_rag", "nq:1", "nq", 0, 2.0,
            retrieved_context_tokens_before_truncation=700,
            retrieved_context_tokens_retained=512,
        ),
        _record("static_rag", "hotpotqa:2", "hotpotqa", 1, 4.0),
    ]
    base_search = [
        _record(
            "base_search", "nq:1", "nq", 1, 2.0,
            number_of_actions=1,
            number_of_valid_actions=1,
            finished=True,
            retrieval_latency_s=0.0,
        ),
        _record(
            "base_search", "hotpotqa:2", "hotpotqa", 0, 4.0,
            number_of_actions=2,
            number_of_valid_actions=1,
            number_of_valid_searches=1,
            number_of_successful_retrievals=1,
            finished=False,
            retrieval_latency_s=0.25,
            retrieved_observation_lengths_before_truncation=[200],
            retained_observation_lengths_after_truncation=[200],
            observation_excess_tokens=[0],
        ),
    ]
    search = [
        _record(
            "search_rl", "nq:1", "nq", 0, 3.0,
            number_of_actions=2,
            number_of_valid_actions=2,
            number_of_valid_searches=1,
            number_of_successful_retrievals=1,
            observation_truncation_count=1,
            trajectory_had_observation_truncation=True,
            retrieved_observation_lengths_before_truncation=[300],
            retained_observation_lengths_after_truncation=[256],
            observation_excess_tokens=[44],
            retrieval_latency_s=0.5,
        ),
        _record(
            "search_rl", "hotpotqa:2", "hotpotqa", 1, 5.0,
            number_of_actions=1,
            number_of_valid_actions=1,
            finished=False,
        ),
    ]

    result = summary.summarize_all({
        "direct": direct,
        "static_rag": static,
        "base_search": base_search,
        "search_rl": search,
    })

    assert result["direct"]["overall_em"] == 0.0
    assert result["static_rag"]["overall_em"] == 0.5
    assert result["search_rl"]["nq_em"] == 0.0
    assert result["search_rl"]["hotpotqa_em"] == 1.0
    assert result["base_search"]["overall_em"] == 0.5
    assert result["comparisons"] == {
        "base_search_vs_direct_em_pp": 50.0,
        "base_search_vs_static_rag_em_pp": 0.0,
        "search_rl_vs_base_search_em_pp": 0.0,
        "search_rl_vs_direct_em_pp": 50.0,
        "search_rl_vs_static_rag_em_pp": 0.0,
    }
    assert result["direct"]["latency"] == {
        "mean_s": 2.0,
        "p50_s": 2.0,
        "p95_s": pytest.approx(2.9),
    }
    agent = result["search_rl"]["agent"]
    assert agent["finish_ratio"] == 0.5
    assert agent["valid_action_ratio"] == 1.0
    assert agent["valid_search_ratio"] == 0.25
    assert agent["mean_valid_searches_per_trajectory"] == 0.5
    assert agent["mean_successful_retrievals_per_trajectory"] == 0.5
    assert agent["fraction_trajectories_with_retrieval"] == 0.5
    assert agent["mean_number_of_actions"] == 1.5
    assert agent["search_retrieval_failure_count"] == 0
    assert agent["observation_truncation_count"] == 1
    assert agent["fraction_trajectories_with_observation_truncation"] == 0.5
    assert agent["retrieved_observation_tokens_before_truncation_mean"] == 300
    assert agent["retained_observation_tokens_after_truncation_mean"] == 256
    assert agent["excess_tokens_above_max_obs_length_mean"] == 44
    assert agent["excess_tokens_above_max_obs_length_max"] == 44
    deltas = result["agent_behavior_deltas"]
    assert deltas["orientation"] == "search_rl_minus_base_search"
    assert deltas["finish_ratio_delta"] == 0.0
    assert deltas["valid_action_ratio_delta"] == 0.25
    assert deltas["valid_search_ratio_delta"] == 0.0
    assert deltas["mean_valid_searches_per_trajectory_delta"] == 0.0
    assert deltas["mean_successful_retrievals_per_trajectory_delta"] == 0.0
    assert deltas["fraction_trajectories_with_retrieval_delta"] == 0.0
    assert deltas["mean_number_of_actions_delta"] == 0.0
    assert deltas["fraction_trajectories_with_observation_truncation_delta"] == 0.5
    assert deltas["retrieval_latency_mean_s_delta"] == 0.125
    assert deltas["generation_latency_mean_s_delta"] == 0.5
    assert deltas["end_to_end_latency_mean_s_delta"] == 1.0
    failure = result["failure_analysis"]
    assert failure["static_rag_context_truncation"]["plausible_contributor"] is True
    assert failure["base_search_observation_truncation"]["plausible_contributor"] is False
    assert failure["search_rl_observation_truncation"]["plausible_contributor"] is True
    assert "does not establish causality" in (
        failure["search_rl_observation_truncation"]["assessment"]
    )


def test_summary_rejects_nonidentical_mode_uid_sets():
    result_sets = {
        "direct": [_record("direct", "nq:1", "nq", 0, 1.0)],
        "static_rag": [_record("static_rag", "nq:2", "nq", 0, 1.0)],
        "base_search": [_record("base_search", "nq:1", "nq", 0, 1.0)],
        "search_rl": [_record("search_rl", "nq:1", "nq", 0, 1.0)],
    }
    with pytest.raises(ValueError, match="same UID set"):
        summary.summarize_all(result_sets)


def test_summary_rejects_results_not_bound_to_manifest_uids_and_hash():
    result_sets = {
        "direct": [_record("direct", "nq:1", "nq", 0, 1.0)],
        "static_rag": [_record("static_rag", "nq:1", "nq", 0, 1.0)],
        "base_search": [_record("base_search", "nq:1", "nq", 0, 1.0)],
        "search_rl": [_record("search_rl", "nq:1", "nq", 0, 1.0)],
    }
    manifest = {
        "selected_source_rows": [{"uid": "nq:2"}],
        "output": {"sha256": "test-eval"},
    }
    with pytest.raises(ValueError, match="do not match the eval manifest"):
        summary.validate_results_against_manifest(
            result_sets, manifest, "test-manifest"
        )


def test_summary_cli_writes_and_references_paired_statistics_artifacts(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "seed": 42,
        "selected_row_count": 64,
        "non_overlap_audit": {"passed": True},
        "output": {"sha256": "eval-sha"},
    }), encoding="utf-8")
    result_sets = {
        mode: [
            {
                "mode": mode,
                "uid": f"nq:{index}",
                "run_fingerprint": f"{mode}-fingerprint",
                "evaluation_error": None,
            }
            for index in range(64)
        ]
        for mode in benchmark.MODES
    }
    run_configs = {
        mode: {
            "model_path": (
                "trained-model" if mode == "search_rl" else "base-model"
            ),
            "retriever_url": None if mode == "direct" else "retriever",
        }
        for mode in benchmark.MODES
    }
    descriptive = {
        mode: (
            {"agent": {"search_retrieval_failure_count": 0}}
            if mode in benchmark.AGENT_MODES else {}
        )
        for mode in benchmark.MODES
    }
    paired_report = {
        "configuration": {},
        "primary": {
            "comparison": "search_rl_vs_base_search",
            "overall": {"difference_pp": 3.125},
            "interpretation": "Limited paired benchmark interpretation.",
        },
        "quality_claim_ready": True,
        "inference_claim_ready": True,
        "readiness": {
            "warning": None,
            "source_counts": {"nq": 32, "hotpotqa": 32},
        },
    }

    monkeypatch.setattr(
        summary,
        "parse_args",
        lambda: SimpleNamespace(
            results_dir=str(results_dir),
            eval_manifest=str(manifest_path),
            output=None,
        ),
    )
    monkeypatch.setattr(
        summary,
        "read_result_file",
        lambda _path, mode: result_sets[mode],
    )
    monkeypatch.setattr(
        summary,
        "validate_results_against_manifest",
        lambda *_args: run_configs,
    )
    monkeypatch.setattr(summary, "summarize_all", lambda *_args: descriptive)
    monkeypatch.setattr(summary, "join_paired_results", lambda *_args: ["joined"])
    monkeypatch.setattr(
        summary, "analyze_paired_results", lambda *_args: paired_report
    )
    monkeypatch.setattr(
        summary,
        "validate_joined_against_manifest",
        lambda *_args: {"passed": True},
    )
    monkeypatch.setattr(summary, "render_markdown", lambda *_args: "# Paired\n")
    monkeypatch.setattr(
        summary, "sha256_file", lambda path: f"sha256:{Path(path).name}"
    )

    summary.main()

    paired_json = results_dir / "paired_statistics.json"
    paired_markdown = results_dir / "paired_statistics.md"
    summary_payload = json.loads(
        (results_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert paired_json.exists()
    assert paired_markdown.read_text(encoding="utf-8") == "# Paired\n"
    assert summary_payload["paired_statistics"]["primary_comparison"] == (
        "search_rl_vs_base_search"
    )
    assert summary_payload["paired_statistics"]["primary_overall"] == {
        "difference_pp": 3.125
    }
    assert set(summary_payload["artifacts"]).issuperset({
        "paired_statistics_json",
        "paired_statistics_markdown",
    })
    assert summary_payload["artifacts"]["paired_statistics_json"]["path"] == (
        str(paired_json)
    )


def test_search_behavior_metrics_exclude_retrieval_failure_rows():
    successful = _record(
        "search_rl", "nq:1", "nq", 0, 2.0,
        number_of_actions=2,
        number_of_valid_actions=2,
        number_of_valid_searches=1,
        number_of_successful_retrievals=1,
        retrieved_observation_lengths_before_truncation=[100],
        retained_observation_lengths_after_truncation=[100],
        observation_excess_tokens=[0],
    )
    failed = _record(
        "search_rl", "hotpotqa:2", "hotpotqa", 0, 4.0,
        prediction=None,
        number_of_actions=0,
        number_of_valid_actions=0,
        finished=False,
        search_retrieval_failure_count=1,
        evaluation_error="RuntimeError: retriever unavailable",
    )
    benchmark.validate_result_record(failed, "search_rl")

    metrics = summary.search_agent_metrics([successful, failed])

    assert metrics["behavior_trajectory_count"] == 1
    assert metrics["behavior_trajectories_excluded_for_retrieval_failure"] == 1
    assert metrics["mean_number_of_actions"] == 2
    assert metrics["search_retrieval_failure_count"] == 1


@pytest.mark.parametrize(
    "mode", ("direct", "static_rag", "base_search", "search_rl")
)
def test_all_mode_records_validate_and_resume_without_recomputation(mode):
    example = _example("nq:1")
    record = _record(
        mode,
        example.uid,
        example.data_source,
        1,
        1.0,
        question=example.question,
        ground_truth=example.ground_truth["target"],
    )

    benchmark.validate_result_record(record, mode)
    pending, completed = benchmark.resume_plan(
        [example],
        [record],
        expected_run_fingerprint=record["run_fingerprint"],
    )

    assert pending == []
    assert completed == {example.uid: record}


def test_mode_results_resume_without_repeating_completed_examples(tmp_path):
    examples = [_example("nq:1"), _example("nq:2")]
    result_path = tmp_path / "direct.jsonl"
    completed = _record(
        "direct", "nq:1", "nq", 1, 1.0,
        question=examples[0].question,
        ground_truth=examples[0].ground_truth["target"],
    )
    result_path.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    evaluated = []

    def evaluate_one(example):
        evaluated.append(example.uid)
        return _record(
            "direct", example.uid, example.data_source, 0, 2.0,
            question=example.question,
            ground_truth=example.ground_truth["target"],
        )

    records = benchmark.run_with_resume(
        examples, result_path, "direct", evaluate_one
    )

    assert evaluated == ["nq:2"]
    assert [record["uid"] for record in records] == ["nq:1", "nq:2"]
    assert [
        json.loads(line)["uid"]
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ] == ["nq:1", "nq:2"]


def test_resume_rejects_a_stale_model_or_run_configuration():
    example = _example("nq:1")
    existing = _record(
        "direct", "nq:1", "nq", 0, 1.0,
        question=example.question,
        ground_truth=example.ground_truth["target"],
    )
    replacement_config = benchmark._default_test_run_config(
        "direct", "replacement-model"
    )
    with pytest.raises(ValueError, match="run configuration does not match"):
        benchmark.resume_plan(
            [example],
            [existing],
            expected_run_fingerprint=benchmark.run_config_fingerprint(
                replacement_config
            ),
        )


def test_primary_launcher_is_syntax_valid_sequential_and_keeps_observation_cap():
    subprocess.run(["bash", "-n", str(RUN_SCRIPT)], check=True)
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    assert 'MAX_OBS_LENGTH="256"' in source
    assert 'STATIC_CONTEXT_TOKEN_BUDGET="512"' in source
    assert (
        "run_mode direct\n"
        "  run_mode static_rag\n"
        "  run_mode base_search\n"
        "  run_mode search_rl\n"
        "  summarize"
    ) in source
    assert source.count("run_mode base_search") == 1
    assert "all|direct|static_rag|base_search|search_rl|summarize" in source
    assert "PHASE4_SEARCH_MODEL_PATH" in source
    assert "PHASE4_BASE_MODEL_PATH" in source
    assert "phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20" in source
