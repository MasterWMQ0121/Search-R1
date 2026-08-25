from types import MethodType, SimpleNamespace

import torch

from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.tensor_helper import TensorConfig, TensorHelper
from verl import DataProto


class _FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        encoded = []
        for text in texts:
            if text == "long retrieval":
                encoded.append([11, 12, 13, 14, 15, 16])
            elif text:
                encoded.append([21, 22])
            else:
                encoded.append([])

        max_length = max((len(tokens) for tokens in encoded), default=0)
        padded = [tokens + [self.pad_token_id] * (max_length - len(tokens)) for tokens in encoded]
        attention_mask = [
            [1] * len(tokens) + [0] * (max_length - len(tokens))
            for tokens in encoded
        ]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _manager(max_turns=1):
    manager = LLMGenerationManager.__new__(LLMGenerationManager)
    manager.tokenizer = _FakeTokenizer()
    manager.config = SimpleNamespace(
        max_turns=max_turns,
        max_start_length=8,
        max_prompt_length=32,
        max_obs_length=4,
        no_think_rl=False,
    )
    manager.tensor_fn = TensorHelper(TensorConfig(
        pad_token_id=manager.tokenizer.pad_token_id,
        max_prompt_length=manager.config.max_prompt_length,
        max_obs_length=manager.config.max_obs_length,
        max_start_length=manager.config.max_start_length,
    ))
    return manager


def test_observation_stats_preserve_the_existing_prefix_slice(capsys):
    manager = _manager()

    default_result = manager._process_next_obs(["long retrieval", "short"])
    result, before, retained, excess = manager._process_next_obs(
        ["long retrieval", "short"],
        return_token_stats=True,
    )

    expected = torch.tensor([
        [11, 12, 13, 14],
        [21, 22, 0, 0],
    ])
    assert torch.equal(default_result, expected)
    assert torch.equal(result, expected)
    assert before.tolist() == [6, 2]
    assert retained.tolist() == [4, 2]
    assert excess.tolist() == [2, 0]
    assert capsys.readouterr().out.count(
        "[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, 6 & 4"
    ) == 2


def test_agent_meta_info_records_only_successful_search_observations():
    manager = _manager(max_turns=2)
    search_calls = 0

    def generate(self, active_batch):
        batch_size = active_batch.batch["input_ids"].shape[0]
        return DataProto.from_dict({
            "responses": torch.full((batch_size, 1), 31, dtype=torch.long),
        })

    def postprocess(self, responses):
        batch_size = responses.shape[0]
        return responses, ["ignored"] * batch_size

    def execute(self, predictions, pad_token, active_mask=None, do_search=True):
        nonlocal search_calls
        if do_search:
            search_calls += 1
            observation = "long retrieval" if search_calls == 1 else "short"
            return [observation, ""], [0, 1], [1, 0], [1, 0]
        return ["" for _ in predictions], [1 for _ in predictions], [1 for _ in predictions], [0 for _ in predictions]

    manager._generate_with_gpu_padding = MethodType(generate, manager)
    manager._postprocess_responses = MethodType(postprocess, manager)
    manager.execute_predictions = MethodType(execute, manager)

    gen_batch = DataProto.from_dict({
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "position_ids": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    })
    output = manager.run_llm_loop(gen_batch, gen_batch.batch["input_ids"].clone())

    assert output.meta_info["retrieval_observation_token_lengths"] == [[6, 2], []]
    assert output.meta_info["retained_retrieval_observation_token_lengths"] == [[4, 2], []]
    assert output.meta_info["retrieval_observation_token_excess"] == [[2, 0], []]
    assert output.meta_info["retrieval_observation_truncation_stats"] == [1, 0]
    assert output.meta_info["retrieval_success_stats"] == [2, 0]
