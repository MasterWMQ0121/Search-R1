import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_grpo_standardizes_within_each_question_and_respects_mask():
    rewards = torch.zeros((8, 4), dtype=torch.float32)
    rewards[:, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0])
    eos_mask = torch.tensor([[1, 1, 1, 1]] * 8, dtype=torch.float32)
    eos_mask[0, -1] = 0
    indexes = np.array(["q0"] * 4 + ["q1"] * 4, dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(rewards, eos_mask, indexes)

    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, returns)
    assert advantages[0, -1].item() == 0.0
    assert torch.allclose(advantages[:4, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[4:, 0].mean(), torch.tensor(0.0), atol=1e-6)


def test_single_trajectory_group_has_zero_advantage():
    rewards = torch.tensor([[0.0, 1.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, mask, np.array(["only"], dtype=object)
    )
    assert torch.equal(advantages, torch.zeros_like(advantages))
