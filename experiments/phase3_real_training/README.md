# Phase 3: small-scale real-data GRPO training validation

This experiment extends the passed Phase-2 reward-signal gate from one update
to a small, reproducible real-training benchmark. It validates sustained
execution, not convergence, paper reproduction, or a claimed quality gain.

Phase 3 leaves every Phase-2 experiment file unchanged. It reuses the same
binary normalized exact-match reward, Qwen2.5-3B-Instruct actor, reference
policy, vLLM rollout, FSDP offload path, and CPU Wiki-18 retriever.

## Experiment design

- 128 training rows: 64 NQ and 64 HotpotQA when both sources are available.
- 32 validation rows, evaluated greedily before and after training.
- 20 optimizer updates consume the first 80 prepared training rows.
- Four prompts per update, four independently sampled agents per prompt, and
  `rollout.n=1`: `4 x 4 x 1 = 16` trajectories.
- The complete run generates `20 x 16 = 320` rollout trajectories.
- One PPO mini-batch of 16, micro-batches of one, and one PPO epoch. Therefore
  each trainer update corresponds to exactly one AdamW optimizer step.
- Deterministic data order. The 128-row dataset supplies 32 batches, so the
  20-step cap is reachable in one epoch without recycling examples.
- No periodic validation. A single final actor checkpoint is saved at update
  20 for the later Base vs RAG vs Search-RL evaluation.

The selector uses seed `42` and offset `4`. It calls the audited Phase-2 data
preparer rather than copying selection logic, preserves source rows unchanged,
checks known fixture leakage, and writes a SHA256 manifest.

## Prepare data

```bash
cd /workspace/Search-R1
PHASE3_PYTHON=/workspace/Search-R1/.venv-phase1/bin/python \
PHASE3_SOURCE_DATA_DIR=/workspace/searchr1-assets/datasets/nq_hotpotqa_train \
PHASE3_DATA_DIR=/workspace/searchr1-assets/datasets/phase3_real_training \
  bash experiments/phase3_real_training/prepare_data.sh
```

Inspect `manifest.json` before training. It should report 128 training rows,
32 validation rows, seed 42, offset 4, and a passed leakage audit.

## Retriever

Reuse the proven CPU retriever unchanged.

```bash
cd /workspace/Search-R1
PHASE2_REAL_PYTHON=/workspace/Search-R1/.venv-phase1/bin/python \
  bash experiments/phase2_real_data/launch_retriever.sh
```

In a second terminal:

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase2_real_data/verify_retriever.py \
  --url http://127.0.0.1:8000/retrieve \
  --topk 3
```

## Train on one A800

The data-preparation and retriever tools stay in `.venv-phase1`. The GPU RL
workload must instead run in the existing `searchr1` conda environment:

```bash
cd /workspace/Search-R1
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
PHASE3_PYTHON=python3 \
PHASE3_MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
PHASE3_DATA_DIR=/workspace/searchr1-assets/datasets/phase3_real_training \
PHASE3_RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
  bash experiments/phase3_real_training/run_small_training.sh
```

The run preserves the proven BF16, XFormers, tensor-parallel-one, vLLM GPU
utilization `0.20`, token limits `768/128/256/1408`, two-turn, top-k-three,
gradient-checkpointing, remove-padding, state-masking, and FSDP offload
configuration. Actor parameter and optimizer offload are enabled, actor
gradient offload is disabled, and reference parameter offload is enabled.
Learning rate is `1e-6`. Phase 2 deliberately set KL to zero to prove a purely
reward-driven GRPO gradient. Phase 3 restores the stable Search-R1 actor-side
KL loss (`use_kl_loss=true`, `kl_loss_coef=0.001`, `low_var_kl`) across the
20-update run. Entropy coefficient remains zero, and reward-side KL remains
disabled. Reward variance and non-zero advantages establish the reward signal;
with actor-side KL enabled, `actor/grad_norm` alone no longer proves pure reward
attribution.

## Metrics and interpretation

The console log contains one metric record per update:

- Reward: `critic/rewards/mean`, `critic/rewards/std`, min and max.
- GRPO signal: `grpo/groups_with_reward_variance`,
  `grpo/group_reward_variance_fraction`, `grpo/group_reward_std_mean`, and
  `grpo/nonzero_advantage_fraction`.
- Optimization: `actor/grad_norm`, `actor/pg_loss`, `actor/kl_loss`,
  `actor/kl_coef`, `actor/ppo_kl`, `actor/entropy_loss`, `actor/lr`, and
  `training/optimizer_steps`.
- Environment: `env/finish_ratio`, `env/ratio_of_valid_action`,
  `env/number_of_valid_search`, `env/ratio_of_valid_search`,
  `env/number_of_successful_retrievals`, and
  `env/trajectories_with_retrieval`.
- Performance: `timing_s/step`, `timing_s/gen`, `timing_s/ref`,
  `timing_s/adv`, `timing_s/update_actor`, per-token timings, and the existing
  allocated/reserved GPU-memory lifecycle messages.

`env/number_of_valid_search` and `env/ratio_of_valid_search` describe valid
search actions, including a final-pass search for which retrieval is disabled.
The two retrieval metrics count only search-enabled actions returned from the
retriever. HTTP or response-schema errors abort the generation loop, so these
are operational success statistics, not retrieval relevance or answer recall.
The successful-retrieval count is averaged per trajectory, and
`trajectories_with_retrieval` is the fraction with at least one retrieval.
`timing_s/gen` includes both vLLM rollout and retrieval latency.

Validation emits `val/test_score/nq` and `val/test_score/hotpotqa` at step zero
and after the 20th update. It is deterministic, uses one greedy trajectory per
validation prompt, and reads the current actor weights. An unchanged validation
mean is allowed: binary exact match on 32 rows is coarse and parameter updates
need not cross an exact-match decision boundary in 20 steps.

## Final checkpoint

The trainer performs the actor update first, then evaluates its save condition,
then increments `global_steps` and checks the `max_optimizer_steps` early-return
condition. With both values set to 20, the only checkpoint is therefore saved
after the 20th actor update at:

```text
/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20/
```

Step-20 metrics are logged after that save. `global_steps` then increments to
21, the `max_optimizer_steps` branch runs final validation at logger step 21,
and the trainer returns.

The run script verifies that `config.json` and model weights exist there. This
trained policy is the Phase-4 evaluation input; no Phase-4 experiment is added
here. It contains Hugging Face actor weights, configuration, and tokenizer, but
not optimizer or trainer state, so it is an evaluation artifact rather than a
resumable training checkpoint.

```bash
find /workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20 \
  -maxdepth 1 -type f -print
```

Surface the requested curves and failure markers with:

```bash
grep -E 'Initial validation metrics|Final validation metrics|Saving actor checkpoint|training/optimizer_steps|critic/rewards/|grpo/|actor/(grad_norm|pg_loss|kl_loss|kl_coef|ppo_kl|entropy_loss|lr)|env/(finish_ratio|ratio_of_valid_action|number_of_valid_search|ratio_of_valid_search|number_of_successful_retrievals|trajectories_with_retrieval)|timing_s/(step|gen|ref|adv|update_actor|save_checkpoint)|memory.*(allocated|reserved)|allocated.*GB|reserved.*GB|Traceback|OutOfMemoryError|CUDA.*(OOM|out of memory)|:[+-]?(nan|inf)([[:space:]]|$)' \
  /workspace/Search-R1/phase3-qwen2.5-3b-small-real-grpo-training.log
```

The script reports three outcomes separately:

- **Training infrastructure pass:** exactly 20 updates, the final checkpoint,
  both validation passes, and no OOM, traceback, NaN, Inf, or other failed run.
- **Reward-signal pass:** at least two updates with within-group reward variance
  and at least two with non-zero GRPO advantages.
- **Quality signal:** final greedy validation compared with initial validation;
  improvement is desirable but not required at this scale.

An overall Phase-3 pass additionally requires at least two finite positive
actor gradient norms. The reward and gradient counters are independent: with
actor-side KL enabled they demonstrate persistent reward signal plus optimizer
execution, not pure reward attribution on the same updates. Exit status `2`
means infrastructure completed but the reward or optimization evidence was
inconclusive.

## Runtime measurement

Runtime remains pending the real A800 Phase-3 run. Do not extrapolate it from
the old answer-leaky Phase-2 timing: initialization, real retrieval, two 32-row
validation passes, offload cycles, and checkpoint saving all contribute.

## Success boundary and remaining risks

A Phase-3 engineering pass demonstrates optimizer execution with a persistent
reward signal, finite/stable execution, a working environment, and before/after
evaluation. It does not demonstrate monotonic reward, statistically
significant improvement, generalization, or production training stability.

The binary exact-match reward is sparse, individual batches may have no
within-group variance, and the small greedy validation split may not change.
CPU retrieval and micro-batch-one offload favor stability over throughput.
Those are measurements for a later experiment, not reasons to alter this
first Phase-3 gate.

Twenty updates are intentional: the goal is sustained real-data GRPO,
repeated reward signal, numerical stability, checkpoint generation,
retrieval/tool behavior, and timing/memory characterization. The next phase
compares direct Qwen2.5-3B, static RAG, and this trained Search-RL agent, so
50/100/1000-step training is not required before that comparison benchmark.
