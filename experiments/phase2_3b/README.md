# Phase 2: Qwen2.5-3B feasibility gates

These scripts isolate the Phase-2 Qwen2.5-3B checks from the completed Phase-1
launch scripts. They reuse the deliberately answer-leaky Phase-1 smoke fixture,
so their scores validate plumbing only and are not model-quality measurements.

## Rollout/memory gate: passed on A800 80GB

`run_rollout_memory_gate.sh` completed successfully on one NVIDIA A800 80GB
with Qwen2.5-3B-Instruct loaded from a local `PHASE2_MODEL_PATH`, BF16,
PyTorch 2.4.0, vLLM 0.6.3, XFormers, and tensor parallel size one. The full
search -> information -> answer trajectory completed without an OOM or
traceback. Immediately after vLLM rollout construction, allocated GPU memory
was approximately 14.2513 GB and reserved memory was approximately 14.3145 GB.

The initial validation score of `1.0` came from the answer-leaky Phase-1 fixture.
It proves that the validation and retrieval path was connected; it does not
measure generalization or model quality.

With the Phase-1 retriever already serving, run the validation-only gate from
the repository root with:

```bash
PHASE2_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash experiments/phase2_3b/run_rollout_memory_gate.sh
```

## One-optimizer-update GRPO gate

`run_one_update.sh` is the next feasibility gate. It performs one real GRPO
actor update through rollout, reward, grouped advantages, reference log-probs
and KL, backward, and one optimizer step. It uses one training prompt with two
independently generated agent trajectories. Both trajectories retain the same
dataset-index UID, giving GRPO the smallest non-singleton comparison group.
The resulting two trajectories form one PPO mini-batch, split into two
micro-batches for gradient accumulation, so the actor executes one optimizer
step.

The first run keeps the proven conservative settings: vLLM GPU utilization
`0.20`, actor parameter and optimizer offload enabled, actor gradient offload
disabled, and reference parameter offload enabled. It disables validation and
checkpointing for the gate.

With the retriever at `http://127.0.0.1:8000/retrieve`, launch from the
repository root with:

```bash
PHASE2_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct \
  bash experiments/phase2_3b/run_one_update.sh
```

This gate has not yet established training stability or any model-quality
improvement. Passing it will only establish that one complete optimizer update
works end to end on the tested runtime.
