# Phase 2: Qwen2.5-3B rollout/memory gate

This directory is isolated from the completed Phase-1 launch scripts. Its first
gate checks whether `Qwen/Qwen2.5-3B-Instruct` can complete one live Search-R1
validation trajectory on a single 24 GB RTX 4090. It temporarily reuses the
answer-leaky Phase-1 smoke data and retriever, so the result is a GPU/memory and
plumbing feasibility signal—not a model-quality measurement.

`run_rollout_memory_gate.sh` performs validation only: one validation row, batch
size one, one agent, and at most two search turns. It uses BF16 vLLM rollout with
the XFormers backend and tensor parallel size one. The initial conservative
memory settings are:

- actor FSDP parameter offload: enabled
- actor FSDP optimizer offload: enabled
- actor FSDP gradient offload: disabled
- reference FSDP parameter offload: enabled
- vLLM GPU memory utilization: `0.20`

The script prints these settings and the token limits before initializing the
model, then records them with the runtime output in
`phase2-qwen2.5-3b-rollout-memory-gate.log` at the repository root. The existing
Phase-1 retriever must already be serving `http://127.0.0.1:8000/retrieve`.

This preparation step does not run the script or start GPU work. When the gate
is intentionally launched later from the repository root, use:

```bash
bash experiments/phase2_3b/run_rollout_memory_gate.sh
```
