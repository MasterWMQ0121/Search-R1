from types import SimpleNamespace

import torch

from search_r1.llm_agent.generation import LLMGenerationManager


def test_only_information_tokens_are_zeroed_in_loss_mask():
    manager = LLMGenerationManager.__new__(LLMGenerationManager)
    manager.tokenizer = SimpleNamespace(pad_token_id=0)

    prompt = torch.tensor([[11, 12]])
    prompt_mask_tokens = prompt.clone()
    search_response = torch.tensor([[21, 22]])
    information = torch.tensor([[31, 32, 33]])
    actual, masked = manager._info_masked_concatenate_with_padding(
        prompt, prompt_mask_tokens, search_response, information, pad_to_left=False
    )
    actual, masked = manager._info_masked_concatenate_with_padding(
        actual, masked, torch.tensor([[41, 42]]), pad_to_left=False
    )

    assert actual.tolist() == [[11, 12, 21, 22, 31, 32, 33, 41, 42]]
    assert (masked != 0).long().tolist() == [[1, 1, 1, 1, 0, 0, 0, 1, 1]]
