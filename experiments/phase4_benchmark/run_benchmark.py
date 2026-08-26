#!/usr/bin/env python3
"""Run one deterministic Phase-4 benchmark mode and persist per-example results."""

import argparse
import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

from experiments.phase4_benchmark.prepare_eval_data import sha256_file
from verl.utils.reward_score import qa_em


AGENT_MODES = ("base_search", "search_rl")
MODES = ("direct", "static_rag", "base_search", "search_rl")
SUPPORTED_SOURCES = ("nq", "hotpotqa")
SCHEMA_VERSION = 1
DEFAULT_BASE_MODEL = "/workspace/models/Qwen2.5-3B-Instruct"
DEFAULT_SEARCH_MODEL = (
    "/workspace/Search-R1/verl_checkpoints/"
    "phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20"
)
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"

DIRECT_PROMPT_TEMPLATE = """Answer the given question. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question}\n"""

STATIC_RAG_PROMPT_TEMPLATE = """Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question} Context: {context} \n"""

COMMON_RESULT_FIELDS = {
    "schema_version",
    "uid",
    "data_source",
    "question",
    "ground_truth",
    "mode",
    "model_path",
    "prediction",
    "exact_match",
    "end_to_end_latency_s",
    "generation_latency_s",
    "trajectory",
    "run_config",
    "run_fingerprint",
}
AGENT_RESULT_FIELDS = {
    "number_of_actions",
    "number_of_valid_actions",
    "number_of_valid_searches",
    "number_of_successful_retrievals",
    "finished",
    "search_retrieval_failure_count",
    "retrieval_latency_s",
    "observation_truncation_count",
    "trajectory_had_observation_truncation",
    "retrieved_observation_lengths_before_truncation",
    "retained_observation_lengths_after_truncation",
    "observation_excess_tokens",
}
MODE_RESULT_FIELDS = {
    "direct": set(),
    "static_rag": {
        "retrieval_latency_s",
        "retrieval_call_count",
        "retrieved_documents",
        "static_context_token_budget",
        "retrieved_context_tokens_before_truncation",
        "retrieved_context_tokens_retained",
    },
    "base_search": AGENT_RESULT_FIELDS,
    "search_rl": AGENT_RESULT_FIELDS,
}


@dataclass(frozen=True)
class BenchmarkExample:
    uid: str
    data_source: str
    question: str
    ground_truth: dict
    search_prompt: str


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: list


@dataclass(frozen=True)
class RetrievalOutput:
    context: str
    documents: list
    latency_s: float


def run_config_fingerprint(run_config):
    encoded = json.dumps(
        run_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_config(args, mode, model_path, eval_manifest, eval_manifest_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_contract": "phase4-benchmark-v1",
        "mode": mode,
        "eval_sha256": eval_manifest["output"]["sha256"],
        "eval_manifest_sha256": sha256_file(eval_manifest_path),
        "model_path": str(model_path),
        "seed": args.seed,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        "max_start_length": args.max_start_length,
        "max_response_length": args.max_response_length,
        "max_prompt_length": args.max_prompt_length,
        "retriever_url": args.retriever_url if mode != "direct" else None,
        "retriever_topk": args.retriever_topk if mode != "direct" else 0,
        "static_context_token_budget": (
            args.static_context_token_budget if mode == "static_rag" else None
        ),
        "max_turns": args.max_turns if mode in AGENT_MODES else None,
        "max_obs_length": args.max_obs_length if mode in AGENT_MODES else None,
    }


def _default_test_run_config(mode, model_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_contract": "phase4-test-fixture",
        "mode": mode,
        "eval_sha256": "test-eval",
        "eval_manifest_sha256": "test-manifest",
        "model_path": str(model_path),
        "seed": 42,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.20,
        "attention_backend": "test",
        "max_start_length": 768,
        "max_response_length": 128,
        "max_prompt_length": 1408,
        "retriever_url": "test" if mode != "direct" else None,
        "retriever_topk": 3 if mode != "direct" else 0,
        "static_context_token_budget": 512 if mode == "static_rag" else None,
        "max_turns": 2 if mode in AGENT_MODES else None,
        "max_obs_length": 256 if mode in AGENT_MODES else None,
    }


def _mapping(value, label):
    if isinstance(value, dict):
        return value
    raise ValueError(f"{label} must be a mapping")


def _sequence(value, label):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise ValueError(f"{label} must be a sequence")


def canonical_source(value):
    source = str(value).strip().lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported data_source: {value!r}")
    return source


def _index_text(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid extra_info.index: {value!r}")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"invalid extra_info.index: {value!r}")
    if isinstance(value, float):
        value = int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("extra_info.index is empty")
    return text


def extract_question(search_prompt):
    marker = "Question:"
    if marker not in search_prompt:
        raise ValueError("search prompt does not contain the canonical 'Question:' marker")
    question = search_prompt.split(marker, 1)[1].strip()
    if not question:
        raise ValueError("search prompt contains an empty question")
    return question


def example_from_row(row):
    source = canonical_source(row["data_source"])
    extra_info = _mapping(row["extra_info"], "extra_info")
    uid = f"{source}:{_index_text(extra_info.get('index'))}"
    prompt_messages = _sequence(row["prompt"], "prompt")
    if not prompt_messages:
        raise ValueError(f"{uid} has an empty prompt")
    prompt_message = _mapping(prompt_messages[-1], "prompt message")
    search_prompt = str(prompt_message.get("content", ""))
    question = extract_question(search_prompt)
    reward_model = _mapping(row["reward_model"], "reward_model")
    ground_truth = _mapping(reward_model.get("ground_truth"), "ground_truth")
    targets = ground_truth.get("target")
    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = _sequence(targets, "ground_truth.target")
    if not targets:
        raise ValueError(f"{uid} has no ground-truth answers")
    return BenchmarkExample(
        uid=uid,
        data_source=source,
        question=question,
        ground_truth={"target": [str(target) for target in targets]},
        search_prompt=search_prompt,
    )


def load_eval_examples(eval_path, manifest_path, expected_count=64):
    eval_path = Path(eval_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("seed") != 42:
        raise ValueError("Phase-4 manifest seed must be 42")
    if manifest.get("selected_row_count") != expected_count:
        raise ValueError(
            f"Phase-4 manifest must contain {expected_count} rows; "
            f"found {manifest.get('selected_row_count')!r}"
        )
    if manifest.get("data_source_counts") != {"nq": expected_count // 2, "hotpotqa": expected_count // 2}:
        raise ValueError("Phase-4 manifest is not exactly balanced across NQ and HotpotQA")
    if not manifest.get("non_overlap_audit", {}).get("passed"):
        raise ValueError("Phase-4 manifest overlap audit did not pass")
    if manifest.get("non_overlap_audit", {}).get("uid_overlap"):
        raise ValueError("Phase-4 manifest contains excluded UID overlap")
    if manifest.get("non_overlap_audit", {}).get("source_position_overlap"):
        raise ValueError("Phase-4 manifest contains excluded source-position overlap")
    expected_sha = manifest.get("output", {}).get("sha256")
    actual_sha = sha256_file(eval_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"Phase-4 eval parquet SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    examples = [example_from_row(row) for row in pd.read_parquet(eval_path).to_dict("records")]
    if len(examples) != expected_count:
        raise ValueError(f"expected {expected_count} evaluation rows, found {len(examples)}")
    uids = [example.uid for example in examples]
    if len(uids) != len(set(uids)):
        raise ValueError("Phase-4 evaluation rows contain duplicate UIDs")
    if uids != [entry.get("uid") for entry in manifest.get("selected_source_rows", [])]:
        raise ValueError("Phase-4 eval row order/UIDs do not match the manifest")
    return examples, manifest


def build_direct_prompt(question):
    return DIRECT_PROMPT_TEMPLATE.format(question=question)


def build_static_rag_prompt(question, context):
    return STATIC_RAG_PROMPT_TEMPLATE.format(question=question, context=context)


def render_chat_prompt(tokenizer, user_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def encode_prompt(tokenizer, rendered_prompt, max_tokens):
    token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    return list(token_ids[-max_tokens:])


def decode_actual_prompt(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def score_completion(actual_prompt, completion, ground_truth):
    format_probe = "__phase4_format_probe__"
    if qa_em.extract_solution(
        actual_prompt + f"<answer>{format_probe}</answer>"
    ) != format_probe:
        raise ValueError(
            "the effective prompt lost its answer-format example during token cropping"
        )
    solution = actual_prompt + completion
    prediction = qa_em.extract_solution(solution)
    score = 0 if prediction is None else int(qa_em.em_check(prediction, ground_truth["target"]))
    return prediction, score


def _base_result(example, mode, model_path, prediction, exact_match, end_to_end,
                 generation_latency, trajectory, run_config=None):
    if run_config is None:
        run_config = _default_test_run_config(mode, model_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "uid": example.uid,
        "data_source": example.data_source,
        "question": example.question,
        "ground_truth": example.ground_truth["target"],
        "mode": mode,
        "model_path": str(model_path),
        "prediction": prediction,
        "exact_match": int(exact_match),
        "end_to_end_latency_s": float(end_to_end),
        "generation_latency_s": float(generation_latency),
        "trajectory": trajectory,
        "run_config": run_config,
        "run_fingerprint": run_config_fingerprint(run_config),
    }


def evaluate_direct(example, tokenizer, generator, model_path, max_start_length=768,
                    run_config=None):
    started = time.perf_counter()
    rendered = render_chat_prompt(tokenizer, build_direct_prompt(example.question))
    prompt_ids = encode_prompt(tokenizer, rendered, max_start_length)
    generation_started = time.perf_counter()
    output = generator.generate_batch([prompt_ids])[0]
    generation_latency = time.perf_counter() - generation_started
    actual_prompt = decode_actual_prompt(tokenizer, prompt_ids)
    prediction, score = score_completion(actual_prompt, output.text, example.ground_truth)
    return _base_result(
        example, "direct", model_path, prediction, score,
        time.perf_counter() - started, generation_latency, output.text, run_config,
    )


def truncate_static_context(tokenizer, context, token_budget):
    raw_ids = list(tokenizer.encode(context, add_special_tokens=False))
    retained_ids = raw_ids[:token_budget]
    return tokenizer.decode(retained_ids, skip_special_tokens=True), len(raw_ids), len(retained_ids)


def evaluate_static_rag(example, tokenizer, generator, retriever, model_path,
                        max_start_length=768, context_token_budget=512,
                        run_config=None):
    started = time.perf_counter()
    retrieval = retriever.retrieve_one(example.question)
    context, raw_context_tokens, retained_context_tokens = truncate_static_context(
        tokenizer, retrieval.context, context_token_budget
    )
    rendered = render_chat_prompt(
        tokenizer, build_static_rag_prompt(example.question, context)
    )
    prompt_ids = encode_prompt(tokenizer, rendered, max_start_length)
    generation_started = time.perf_counter()
    output = generator.generate_batch([prompt_ids])[0]
    generation_latency = time.perf_counter() - generation_started
    actual_prompt = decode_actual_prompt(tokenizer, prompt_ids)
    prediction, score = score_completion(actual_prompt, output.text, example.ground_truth)
    result = _base_result(
        example, "static_rag", model_path, prediction, score,
        time.perf_counter() - started, generation_latency, output.text, run_config,
    )
    result.update({
        "retrieval_latency_s": retrieval.latency_s,
        "retrieval_call_count": 1,
        "retrieved_documents": retrieval.documents,
        "static_context_token_budget": context_token_budget,
        "retrieved_context_tokens_before_truncation": raw_context_tokens,
        "retrieved_context_tokens_retained": retained_context_tokens,
    })
    return result


class ExistingRetrieverClient:
    """One-query adapter around the Search-R1 retriever implementation."""

    def __init__(self, search_url, topk):
        from search_r1.llm_agent.generation import LLMGenerationManager

        self.manager = LLMGenerationManager.__new__(LLMGenerationManager)
        self.manager.config = type("RetrieverConfig", (), {
            "search_url": search_url,
            "topk": topk,
        })()
        self.topk = topk

    @staticmethod
    def _document_record(item, rank):
        document = item.get("document", {}) if isinstance(item, dict) else {}
        contents = str(document.get("contents", ""))
        return {
            "rank": rank,
            "id": document.get("id", item.get("id") if isinstance(item, dict) else None),
            "title": contents.split("\n", 1)[0],
            "score": item.get("score") if isinstance(item, dict) else None,
        }

    def retrieve_one(self, question):
        started = time.perf_counter()
        payload = self.manager._batch_search([question])
        latency = time.perf_counter() - started
        results = payload.get("result")
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], list):
            raise RuntimeError("Retriever returned invalid cardinality for one Static-RAG query")
        passages = results[0]
        if len(passages) != self.topk:
            raise RuntimeError(
                f"Static-RAG retriever returned {len(passages)} passages; "
                f"expected top-k {self.topk}"
            )
        context = self.manager._passages2string(passages)
        documents = [
            self._document_record(item, rank)
            for rank, item in enumerate(passages, start=1)
        ]
        return RetrievalOutput(context=context, documents=documents, latency_s=latency)


class VLLMGenerator:
    """Public-vLLM backend shared by all three standalone evaluation modes."""

    def __init__(self, model_path, gpu_memory_utilization, max_model_len, seed,
                 max_response_length):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.engine = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=max_model_len,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_response_length,
        )

    def generate_batch(self, prompt_token_ids):
        requests = self.engine.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        outputs = []
        for request in requests:
            if len(request.outputs) != 1:
                raise RuntimeError("greedy Phase-4 generation returned multiple candidates")
            candidate = request.outputs[0]
            outputs.append(GenerationOutput(
                text=candidate.text,
                token_ids=list(candidate.token_ids),
            ))
        if len(outputs) != len(prompt_token_ids):
            raise RuntimeError("vLLM returned the wrong number of Phase-4 generations")
        return outputs


class VLLMRolloutAdapter:
    """Expose public vLLM through the interface used by the existing agent loop."""

    def __init__(self, generator, pad_token_id):
        self.generator = generator
        self.pad_token_id = pad_token_id
        self.generation_latency_s = 0.0

    def generate_sequences(self, active_batch):
        import torch
        from verl import DataProto

        input_ids = active_batch.batch["input_ids"].detach().cpu()
        attention_mask = active_batch.batch["attention_mask"].detach().cpu()
        prompts = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(input_ids, attention_mask)
        ]
        started = time.perf_counter()
        generated = self.generator.generate_batch(prompts)
        self.generation_latency_s += time.perf_counter() - started
        max_length = max((len(item.token_ids) for item in generated), default=0)
        responses = torch.full(
            (len(generated), max_length), self.pad_token_id, dtype=torch.long
        )
        for index, item in enumerate(generated):
            if item.token_ids:
                responses[index, :len(item.token_ids)] = torch.tensor(
                    item.token_ids, dtype=torch.long
                )
        return DataProto.from_dict({"responses": responses})


def _meta_item(meta_info, key, default):
    value = meta_info.get(key)
    if not isinstance(value, list) or not value:
        return default
    return value[0]


def _token_id_error(label, token_ids, reason):
    value_type = f"{type(token_ids).__module__}.{type(token_ids).__name__}"
    dtype = getattr(token_ids, "dtype", None)
    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = repr(shape)
    try:
        sample_source = token_ids.detach().cpu().tolist()
    except AttributeError:
        try:
            sample_source = token_ids.tolist()
        except AttributeError:
            try:
                sample_source = list(token_ids)
            except TypeError:
                sample_source = token_ids
    sample = sample_source[:8] if isinstance(sample_source, list) else sample_source
    raise ValueError(
        f"{label}: {reason}; type={value_type}, dtype={dtype!s}, "
        f"shape={shape!r}, sample={sample!r}"
    )


def _normalize_token_ids_for_decode(token_ids, label):
    """Validate one token-ID row and return plain Python integers for decoding."""
    import torch

    original = token_ids
    float_exact_limit = None
    float_dtype_label = None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu()
        if token_ids.ndim != 1:
            _token_id_error(label, original, "expected a one-dimensional token-ID row")
        if token_ids.dtype == torch.bool:
            _token_id_error(label, original, "boolean token IDs are not allowed")
        if token_ids.dtype.is_floating_point:
            if token_ids.dtype == torch.float32:
                float_exact_limit = 1 << 24
            elif token_ids.dtype == torch.float64:
                float_exact_limit = 1 << 53
            else:
                _token_id_error(
                    label,
                    original,
                    "low-precision or unsupported floating token IDs may already "
                    "have lost integer identity and cannot be safely normalized; "
                    "only float32 and float64 are accepted",
                )
            float_dtype_label = str(token_ids.dtype)
        values = token_ids.tolist()
    elif isinstance(token_ids, np.ndarray):
        if token_ids.ndim != 1:
            _token_id_error(label, original, "expected a one-dimensional token-ID row")
        if token_ids.dtype == np.dtype(np.bool_):
            _token_id_error(label, original, "boolean token IDs are not allowed")
        try:
            is_floating_dtype = np.issubdtype(token_ids.dtype, np.floating)
        except TypeError:
            is_floating_dtype = False
        is_floating_dtype = is_floating_dtype or "float" in str(token_ids.dtype).lower()
        if is_floating_dtype:
            if token_ids.dtype == np.dtype(np.float32):
                float_exact_limit = 1 << 24
            elif token_ids.dtype == np.dtype(np.float64):
                float_exact_limit = 1 << 53
            else:
                _token_id_error(
                    label,
                    original,
                    "low-precision or unsupported floating token IDs may already "
                    "have lost integer identity and cannot be safely normalized; "
                    "only float32 and float64 are accepted",
                )
            float_dtype_label = str(token_ids.dtype)
        values = token_ids.tolist()
    elif isinstance(token_ids, Sequence) and not isinstance(
        token_ids, (str, bytes, bytearray)
    ):
        values = list(token_ids)
    else:
        _token_id_error(label, original, "expected a tensor, array, or sequence")

    normalized = []
    int64_max = (1 << 63) - 1
    sequence_float_dtypes = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            _token_id_error(label, original, "boolean token IDs are not allowed")
        if isinstance(value, Integral):
            normalized_value = int(value)
        elif isinstance(value, (float, np.floating)):
            value_exact_limit = float_exact_limit
            if value_exact_limit is None:
                if isinstance(value, np.floating):
                    value_dtype = np.asarray(value).dtype
                    if value_dtype == np.dtype(np.float32):
                        value_exact_limit = 1 << 24
                    elif value_dtype == np.dtype(np.float64):
                        value_exact_limit = 1 << 53
                    else:
                        _token_id_error(
                            label,
                            original,
                            "low-precision or unsupported floating token IDs may "
                            "already have lost integer identity and cannot be safely "
                            "normalized; only float32 and float64 are accepted",
                        )
                    sequence_float_dtypes.add(str(value_dtype))
                else:
                    value_exact_limit = 1 << 53
                    sequence_float_dtypes.add("python.float64")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                _token_id_error(label, original, "floating token IDs must be finite")
            if numeric_value < 0:
                _token_id_error(label, original, "negative token IDs are not allowed")
            if not numeric_value.is_integer():
                _token_id_error(
                    label, original, "floating token IDs must be exactly integer-valued"
                )
            if abs(numeric_value) > value_exact_limit:
                _token_id_error(
                    label,
                    original,
                    "floating token IDs exceed the source dtype's exact consecutive-"
                    f"integer range (absolute value must be <= {value_exact_limit})",
                )
            normalized_value = int(numeric_value)
        else:
            _token_id_error(label, original, f"token ID {value!r} is not numeric")
        if normalized_value < 0:
            _token_id_error(label, original, "negative token IDs are not allowed")
        if normalized_value > int64_max:
            _token_id_error(label, original, "token IDs must fit in signed int64")
        normalized.append(normalized_value)

    if float_dtype_label is not None or sequence_float_dtypes:
        dtype_description = float_dtype_label or ",".join(sorted(sequence_float_dtypes))
        shape = getattr(original, "shape", (len(values),))
        shape = tuple(shape)
        warnings.warn(
            f"{label}: dtype={dtype_description}, shape={shape}; exact integer-valued "
            "token IDs are being normalized to int64",
            RuntimeWarning,
            stacklevel=2,
        )
    return normalized


def _decode_search_trajectory(tokenizer, output):
    responses = output.batch["responses"][0]
    prompt_length = output.batch["prompts"].shape[-1]
    response_mask = output.batch["attention_mask"][0, prompt_length:]
    valid_length = int(response_mask.sum().item())
    token_ids = _normalize_token_ids_for_decode(
        responses[:valid_length], "Search-RL trajectory responses"
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def evaluate_search_agent(example, tokenizer, generator, mode, model_path, search_url,
                          topk=3, max_turns=2, max_start_length=768,
                          max_response_length=128, max_obs_length=256,
                          max_prompt_length=1408, run_config=None,
                          configure_manager=None):
    import torch
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from verl import DataProto

    if mode not in AGENT_MODES:
        raise ValueError(f"unsupported Search Agent mode: {mode!r}")
    started = time.perf_counter()
    rendered = render_chat_prompt(tokenizer, example.search_prompt)
    prompt_ids = encode_prompt(tokenizer, rendered, max_start_length)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
    gen_batch = DataProto.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })
    rollout_adapter = VLLMRolloutAdapter(generator, tokenizer.pad_token_id)
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=rollout_adapter,
        config=GenerationConfig(
            max_turns=max_turns,
            max_start_length=max_start_length,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_obs_length=max_obs_length,
            num_gpus=1,
            no_think_rl=False,
            search_url=search_url,
            topk=topk,
        ),
        is_validation=True,
    )

    retrieval_telemetry = {"latency_s": 0.0, "failure_count": 0}
    original_batch_search = manager._batch_search

    def timed_batch_search(self, queries):
        retrieval_started = time.perf_counter()
        try:
            payload = original_batch_search(queries)
            results = payload.get("result")
            if not isinstance(results, list) or len(results) != len(queries):
                raise RuntimeError("Search-RL retriever returned invalid query cardinality")
            if any(not isinstance(passages, list) or len(passages) != topk for passages in results):
                raise RuntimeError(
                    f"Search-RL retriever did not return exactly top-k {topk} passages"
                )
            return payload
        except Exception:
            retrieval_telemetry["failure_count"] += 1
            raise
        finally:
            retrieval_telemetry["latency_s"] += time.perf_counter() - retrieval_started

    manager._batch_search = MethodType(timed_batch_search, manager)
    if configure_manager is not None:
        configure_manager(manager)
    try:
        output = manager.run_llm_loop(gen_batch, input_ids.clone())
    except Exception as error:
        if not retrieval_telemetry["failure_count"]:
            raise
        result = _base_result(
            example, mode, model_path, None, 0,
            time.perf_counter() - started, rollout_adapter.generation_latency_s, "",
            run_config,
        )
        result.update({
            "number_of_actions": 0,
            "number_of_valid_actions": 0,
            "number_of_valid_searches": 0,
            "number_of_successful_retrievals": 0,
            "finished": False,
            "search_retrieval_failure_count": retrieval_telemetry["failure_count"],
            "retrieval_latency_s": retrieval_telemetry["latency_s"],
            "observation_truncation_count": 0,
            "trajectory_had_observation_truncation": False,
            "retrieved_observation_lengths_before_truncation": [],
            "retained_observation_lengths_after_truncation": [],
            "observation_excess_tokens": [],
            "evaluation_error": f"{type(error).__name__}: {error}",
        })
        return result

    trajectory = _decode_search_trajectory(tokenizer, output)
    actual_prompt = decode_actual_prompt(tokenizer, prompt_ids)
    prediction, score = score_completion(actual_prompt, trajectory, example.ground_truth)
    meta = output.meta_info
    truncation_count = int(_meta_item(
        meta, "retrieval_observation_truncation_stats", 0
    ))
    result = _base_result(
        example, mode, model_path, prediction, score,
        time.perf_counter() - started, rollout_adapter.generation_latency_s, trajectory,
        run_config,
    )
    result.update({
        "number_of_actions": int(_meta_item(meta, "turns_stats", 0)),
        "number_of_valid_actions": int(_meta_item(meta, "valid_action_stats", 0)),
        "number_of_valid_searches": int(_meta_item(meta, "valid_search_stats", 0)),
        "number_of_successful_retrievals": int(
            _meta_item(meta, "retrieval_success_stats", 0)
        ),
        "finished": not bool(_meta_item(meta, "active_mask", True)),
        "search_retrieval_failure_count": retrieval_telemetry["failure_count"],
        "retrieval_latency_s": retrieval_telemetry["latency_s"],
        "observation_truncation_count": truncation_count,
        "trajectory_had_observation_truncation": truncation_count > 0,
        "retrieved_observation_lengths_before_truncation": list(_meta_item(
            meta, "retrieval_observation_token_lengths", []
        )),
        "retained_observation_lengths_after_truncation": list(_meta_item(
            meta, "retained_retrieval_observation_token_lengths", []
        )),
        "observation_excess_tokens": list(_meta_item(
            meta, "retrieval_observation_token_excess", []
        )),
    })
    return result


def evaluate_base_search(example, tokenizer, generator, model_path, search_url,
                         topk=3, max_turns=2, max_start_length=768,
                         max_response_length=128, max_obs_length=256,
                         max_prompt_length=1408, run_config=None,
                         configure_manager=None):
    return evaluate_search_agent(
        example, tokenizer, generator, "base_search", model_path, search_url,
        topk, max_turns, max_start_length, max_response_length, max_obs_length,
        max_prompt_length, run_config, configure_manager,
    )


def evaluate_search_rl(example, tokenizer, generator, model_path, search_url,
                       topk=3, max_turns=2, max_start_length=768,
                       max_response_length=128, max_obs_length=256,
                       max_prompt_length=1408, run_config=None,
                       configure_manager=None):
    return evaluate_search_agent(
        example, tokenizer, generator, "search_rl", model_path, search_url,
        topk, max_turns, max_start_length, max_response_length, max_obs_length,
        max_prompt_length, run_config, configure_manager,
    )


def validate_result_record(record, expected_mode=None):
    if not isinstance(record, dict):
        raise ValueError("result row must be a JSON object")
    mode = record.get("mode")
    if mode not in MODES or (expected_mode is not None and mode != expected_mode):
        raise ValueError(f"unexpected result mode: {mode!r}")
    missing = (COMMON_RESULT_FIELDS | MODE_RESULT_FIELDS[mode]).difference(record)
    if missing:
        raise ValueError(f"{mode} result is missing fields: {sorted(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    run_config = record.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("result run_config must be a mapping")
    if record.get("run_fingerprint") != run_config_fingerprint(run_config):
        raise ValueError("result run_fingerprint does not match run_config")
    if run_config.get("mode") != mode:
        raise ValueError("result run_config mode does not match result mode")
    if run_config.get("model_path") != record.get("model_path"):
        raise ValueError("result run_config model does not match result model")
    fixed_config = {
        "seed": 42,
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.20,
        "max_start_length": 768,
        "max_response_length": 128,
        "max_prompt_length": 1408,
    }
    for key, expected in fixed_config.items():
        if run_config.get(key) != expected:
            raise ValueError(f"result run_config has unexpected {key}")
    if not run_config.get("eval_sha256") or not run_config.get("eval_manifest_sha256"):
        raise ValueError("result run_config is not bound to evaluation hashes")
    if mode == "direct" and run_config.get("retriever_topk") != 0:
        raise ValueError("Direct run_config must disable retrieval")
    if mode != "direct" and run_config.get("retriever_topk") != 3:
        raise ValueError("retrieval run_config must use top-k 3")
    if mode == "static_rag" and run_config.get("static_context_token_budget") != 512:
        raise ValueError("Static-RAG run_config must use a 512-token context budget")
    if mode in AGENT_MODES and (
        run_config.get("max_turns") != 2 or run_config.get("max_obs_length") != 256
    ):
        raise ValueError("Search Agent run_config must use two turns and max_obs_length 256")
    if record.get("data_source") not in SUPPORTED_SOURCES:
        raise ValueError("result has unsupported data_source")
    if not str(record.get("uid", "")).startswith(f"{record['data_source']}:"):
        raise ValueError("result UID is not source-aware")
    if not isinstance(record.get("ground_truth"), list) or not record["ground_truth"]:
        raise ValueError("result ground_truth must be a non-empty list")
    if record.get("prediction") is not None and not isinstance(record["prediction"], str):
        raise ValueError("result prediction must be text or null")
    if record.get("exact_match") not in (0, 1):
        raise ValueError("exact_match must be binary")
    for latency_key in ("end_to_end_latency_s", "generation_latency_s"):
        latency = float(record.get(latency_key))
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"{latency_key} must be finite and non-negative")
    if mode == "static_rag":
        if record.get("retrieval_call_count") != 1:
            raise ValueError("every Static-RAG result must have exactly one retrieval call")
        if not isinstance(record.get("retrieved_documents"), list):
            raise ValueError("retrieved_documents must be a list")
        if len(record["retrieved_documents"]) != run_config["retriever_topk"]:
            raise ValueError("Static-RAG result does not contain exactly top-k documents")
        retrieval_latency = float(record.get("retrieval_latency_s"))
        if not math.isfinite(retrieval_latency) or retrieval_latency < 0:
            raise ValueError("retrieval_latency_s must be finite and non-negative")
        budget = record.get("static_context_token_budget")
        raw = record.get("retrieved_context_tokens_before_truncation")
        retained = record.get("retrieved_context_tokens_retained")
        if budget != 512 or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (raw, retained)
        ):
            raise ValueError("Static-RAG context token metrics are invalid")
        if raw < 0 or retained < 0 or retained > raw or retained > budget:
            raise ValueError("Static-RAG context token metrics are inconsistent")
    if mode in AGENT_MODES:
        count_keys = (
            "number_of_actions",
            "number_of_valid_actions",
            "number_of_valid_searches",
            "number_of_successful_retrievals",
            "search_retrieval_failure_count",
            "observation_truncation_count",
        )
        counts = {}
        for key in count_keys:
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            counts[key] = value
        if counts["number_of_valid_actions"] > counts["number_of_actions"]:
            raise ValueError("valid actions cannot exceed actions")
        if counts["number_of_valid_searches"] > counts["number_of_valid_actions"]:
            raise ValueError("valid searches cannot exceed valid actions")
        if (
            counts["number_of_successful_retrievals"]
            > counts["number_of_valid_searches"]
        ):
            raise ValueError("successful retrievals cannot exceed valid searches")
        if not isinstance(record.get("finished"), bool):
            raise ValueError("finished must be boolean")
        if not isinstance(record.get("trajectory_had_observation_truncation"), bool):
            raise ValueError("trajectory truncation flag must be boolean")
        retrieval_latency = float(record.get("retrieval_latency_s"))
        if not math.isfinite(retrieval_latency) or retrieval_latency < 0:
            raise ValueError("retrieval_latency_s must be finite and non-negative")
        event_lists = (
            record.get("retrieved_observation_lengths_before_truncation"),
            record.get("retained_observation_lengths_after_truncation"),
            record.get("observation_excess_tokens"),
        )
        if not all(isinstance(values, list) for values in event_lists):
            raise ValueError("Search Agent observation telemetry must use lists")
        if len({len(values) for values in event_lists}) != 1:
            raise ValueError("Search Agent observation telemetry lists are misaligned")
        raw_lengths, retained_lengths, excess_tokens = event_lists
        if len(raw_lengths) != counts["number_of_successful_retrievals"]:
            raise ValueError("Search Agent observation count does not match retrieval count")
        for raw, retained, excess in zip(raw_lengths, retained_lengths, excess_tokens):
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (raw, retained, excess)
            ):
                raise ValueError("Search Agent observation token metrics must be non-negative integers")
            if retained > raw or excess != max(raw - 256, 0):
                raise ValueError("Search Agent observation token metrics are inconsistent")
        expected_truncations = sum(
            retained < raw for raw, retained in zip(raw_lengths, retained_lengths)
        )
        if counts["observation_truncation_count"] != expected_truncations:
            raise ValueError("Search Agent observation truncation count is inconsistent")
        if record["trajectory_had_observation_truncation"] != (expected_truncations > 0):
            raise ValueError("Search Agent trajectory truncation flag is inconsistent")
        if counts["search_retrieval_failure_count"]:
            if record["exact_match"] != 0 or record["prediction"] is not None:
                raise ValueError("retriever failure rows must not claim a correct prediction")
            if not record.get("evaluation_error"):
                raise ValueError("retriever failure rows must record evaluation_error")
    return record


def read_result_file(path, mode):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from error
        validate_result_record(record, mode)
        if record["uid"] in seen:
            raise ValueError(f"duplicate UID in {path}: {record['uid']}")
        seen.add(record["uid"])
        records.append(record)
    return records


def resume_plan(examples, existing_records, expected_run_fingerprint=None):
    expected_uids = [example.uid for example in examples]
    expected_set = set(expected_uids)
    existing = {record["uid"]: record for record in existing_records}
    unexpected = sorted(set(existing).difference(expected_set))
    if unexpected:
        raise ValueError(f"result file contains UIDs outside the manifest: {unexpected}")
    expected_examples = {example.uid: example for example in examples}
    for uid, record in existing.items():
        example = expected_examples[uid]
        expected_identity = (
            example.data_source,
            example.question,
            example.ground_truth["target"],
        )
        record_identity = (
            record["data_source"],
            record["question"],
            record["ground_truth"],
        )
        if record_identity != expected_identity:
            raise ValueError(f"existing result content does not match eval row {uid}")
        if (
            expected_run_fingerprint is not None
            and record["run_fingerprint"] != expected_run_fingerprint
        ):
            raise ValueError(
                f"existing result run configuration does not match current run for {uid}"
            )
    return [example for example in examples if example.uid not in existing], existing


def _append_result(path, record):
    validate_result_record(record, record["mode"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_results(path, examples, records_by_uid):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(
                records_by_uid[example.uid], ensure_ascii=False, sort_keys=True
            ) + "\n")
    temporary.replace(path)


def run_with_resume(examples, result_path, mode, evaluate_one, overwrite=False,
                    expected_run_fingerprint=None):
    result_path = Path(result_path)
    if overwrite and result_path.exists():
        result_path.unlink()
    pending, records_by_uid = resume_plan(
        examples,
        read_result_file(result_path, mode),
        expected_run_fingerprint=expected_run_fingerprint,
    )
    for index, example in enumerate(pending, start=1):
        record = evaluate_one(example)
        if record["uid"] != example.uid or record["mode"] != mode:
            raise ValueError("evaluator returned a result for the wrong UID or mode")
        _append_result(result_path, record)
        records_by_uid[example.uid] = record
        print(f"[{mode}] persisted {len(records_by_uid)}/{len(examples)}: {example.uid}")
    if len(records_by_uid) != len(examples):
        raise RuntimeError("Phase-4 mode ended without all expected results")
    _rewrite_results(result_path, examples, records_by_uid)
    return [records_by_uid[example.uid] for example in examples]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--search-model", default=DEFAULT_SEARCH_MODEL)
    parser.add_argument("--retriever-url", default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--retriever-topk", type=int, default=3)
    parser.add_argument("--static-context-token-budget", type=int, default=512)
    parser.add_argument("--max-start-length", type=int, default=768)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--max-obs-length", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1408)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != 42:
        raise ValueError("the primary Phase-4 benchmark seed must remain 42")
    if args.retriever_topk != 3:
        raise ValueError("the primary Phase-4 benchmark retriever top-k must remain 3")
    if args.max_obs_length != 256:
        raise ValueError("the primary Search-RL max_obs_length must remain 256")
    if args.max_turns != 2:
        raise ValueError("the primary Search-RL max_turns must remain 2")
    if args.max_prompt_length != 1408:
        raise ValueError("the primary Search-RL max_prompt_length must remain 1408")
    if args.max_start_length != 768:
        raise ValueError("the primary Phase-4 max_start_length must remain 768")
    if args.max_response_length != 128:
        raise ValueError("the primary Phase-4 max_response_length must remain 128")
    if args.static_context_token_budget != 512:
        raise ValueError("the primary Static-RAG context token budget must remain 512")
    if args.gpu_memory_utilization != 0.20:
        raise ValueError("the primary Phase-4 vLLM GPU utilization must remain 0.20")

    examples, manifest = load_eval_examples(args.eval_data, args.eval_manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = output_dir / f"{args.mode}.jsonl"
    model_path = args.search_model if args.mode == "search_rl" else args.base_model
    run_config = build_run_config(
        args, args.mode, model_path, manifest, Path(args.eval_manifest).resolve()
    )
    run_fingerprint = run_config_fingerprint(run_config)
    existing = read_result_file(result_path, args.mode) if not args.overwrite else []
    pending, _ = resume_plan(
        examples, existing, expected_run_fingerprint=run_fingerprint
    )
    if not pending:
        print(f"[{args.mode}] already complete: {result_path}")
        return

    print(json.dumps({
        "mode": args.mode,
        "model": model_path,
        "eval_data": str(Path(args.eval_data).resolve()),
        "eval_sha256": manifest["output"]["sha256"],
        "remaining_examples": len(pending),
        "greedy": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_response_length": args.max_response_length,
        "max_obs_length": args.max_obs_length if args.mode in AGENT_MODES else None,
        "max_turns": args.max_turns if args.mode in AGENT_MODES else None,
        "retriever_topk": args.retriever_topk if args.mode != "direct" else 0,
        "static_context_token_budget": (
            args.static_context_token_budget if args.mode == "static_rag" else None
        ),
        "result_path": str(result_path),
        "run_fingerprint": run_fingerprint,
    }, indent=2, sort_keys=True))

    generator = VLLMGenerator(
        model_path=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        seed=args.seed,
        max_response_length=args.max_response_length,
    )
    tokenizer = generator.tokenizer
    if args.mode == "direct":
        evaluate_one = lambda example: evaluate_direct(
            example, tokenizer, generator, model_path, args.max_start_length,
            run_config,
        )
    elif args.mode == "static_rag":
        retriever = ExistingRetrieverClient(args.retriever_url, args.retriever_topk)
        evaluate_one = lambda example: evaluate_static_rag(
            example, tokenizer, generator, retriever, model_path,
            args.max_start_length, args.static_context_token_budget,
            run_config,
        )
    elif args.mode == "base_search":
        evaluate_one = lambda example: evaluate_base_search(
            example, tokenizer, generator, model_path, args.retriever_url,
            args.retriever_topk, args.max_turns, args.max_start_length,
            args.max_response_length, args.max_obs_length, args.max_prompt_length,
            run_config,
        )
    else:
        evaluate_one = lambda example: evaluate_search_rl(
            example, tokenizer, generator, model_path, args.retriever_url,
            args.retriever_topk, args.max_turns, args.max_start_length,
            args.max_response_length, args.max_obs_length, args.max_prompt_length,
            run_config,
        )
    run_with_resume(
        examples, result_path, args.mode, evaluate_one, overwrite=args.overwrite,
        expected_run_fingerprint=run_fingerprint,
    )
    print(f"[{args.mode}] complete: {result_path}")


if __name__ == "__main__":
    main()
