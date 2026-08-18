# Phase 2: real-data GRPO reward-signal gate

This experiment is intentionally separate from the earlier plumbing gates:

- Phase 1 used a deliberately answer-leaky fixture to validate the search state
  machine and reward plumbing.
- The Phase-2 3B smoke gate executed a real optimizer step, but both trajectories
  received reward `1`; its GRPO advantages were zero and its non-zero gradient
  could therefore come from KL regularization.
- This directory uses real NQ/HotpotQA rows, real Wiki-18 retrieval, and four
  trajectories per prompt to test for a reward-driven GRPO gradient.

The full retrieval corpus/index does **not** imply full-dataset training. The
gate trains on four prompts drawn from a deterministic 64-row pool. The CPU
retriever and the single-A800 RL workload are deliberately separated. This is
an engineering diagnostic, not a paper-scale reproduction, and one update does
not establish stability or model-quality improvement.

## Existing measured baseline

The answer-leaky 3B one-update smoke gate took approximately 51.6 seconds:
generation 29.1 seconds and actor update 18.4 seconds. After vLLM weight
offload, CUDA allocated memory was approximately 14.25 GiB. These are prior
smoke measurements, not predictions for this real-data gate.

## External assets and preflight

Expected GPU paths are:

```text
/workspace/searchr1-assets/
├── datasets/
│   ├── nq_hotpotqa_train/{train.parquet,test.parquet}
│   └── phase2_real_data_gate/{train.parquet,test.parquet,manifest.json}
├── models/e5-base-v2/
└── wiki18/{e5_Flat.index,wiki-18.jsonl}
```

The baseline machine has a 178 GiB `/workspace` filesystem with 144 GiB free
and 236 GiB host RAM. `prepare_assets.sh` uses a conservative 96 GiB free-disk
planning threshold and requires 128 GiB available RAM before creating
directories. These are configurable with `PHASE2_REAL_REQUIRED_FREE_GIB` and
`PHASE2_REAL_MIN_RAM_GIB`. The disk value is explicitly an estimate rather
than a claimed artifact size: it allows for the growing temporary index plus
one downloaded shard, followed by the final index plus compressed and
decompressed corpus files. The helper itself never downloads anything.

```bash
cd /workspace/Search-R1
bash experiments/phase2_real_data/prepare_assets.sh preflight
bash experiments/phase2_real_data/prepare_assets.sh init
bash experiments/phase2_real_data/prepare_assets.sh init-index
```

Download and append only one already-built index shard at a time. Each append
is fsynced and checked by byte count. Delete the verified source shard before
downloading the next one, so the two shards and a second full index copy are
never resident together:

```bash
cd /workspace/Search-R1
huggingface-cli download PeterJinGo/wiki-18-e5-index part_aa \
  --repo-type dataset \
  --local-dir /workspace/searchr1-assets/wiki18
bash experiments/phase2_real_data/prepare_assets.sh append-index-part part_aa
rm -- /workspace/searchr1-assets/wiki18/part_aa

huggingface-cli download PeterJinGo/wiki-18-e5-index part_ab \
  --repo-type dataset \
  --local-dir /workspace/searchr1-assets/wiki18
bash experiments/phase2_real_data/prepare_assets.sh append-index-part part_ab
rm -- /workspace/searchr1-assets/wiki18/part_ab

bash experiments/phase2_real_data/prepare_assets.sh finalize-index
```

Download the compressed corpus separately. `prepare-corpus` validates gzip,
decompresses to `wiki-18.jsonl.tmp`, fsyncs and validates the output, then
renames it atomically. It deliberately retains the gzip. Cleanup is a separate,
explicit command:

```bash
cd /workspace/Search-R1
huggingface-cli download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz \
  --repo-type dataset \
  --local-dir /workspace/searchr1-assets/wiki18
bash experiments/phase2_real_data/prepare_assets.sh prepare-corpus

# Optional only after the final corpus has been verified:
bash experiments/phase2_real_data/prepare_assets.sh cleanup-corpus
```

Download the E5 model and already Search-R1-formatted real dataset locally:

```bash
cd /workspace/Search-R1
huggingface-cli download intfloat/e5-base-v2 \
  --local-dir /workspace/searchr1-assets/models/e5-base-v2
huggingface-cli download PeterJinGo/nq_hotpotqa_train \
  --repo-type dataset \
  --local-dir /workspace/searchr1-assets/datasets/nq_hotpotqa_train
bash experiments/phase2_real_data/prepare_assets.sh verify
```

No secrets or access tokens are embedded in these scripts. Configure your
Hugging Face authentication outside the repository if the service requires it.

## Deterministic gate subset

`prepare_gate_data.py` preserves every source column and does not rewrite
prompts, answers, or `extra_info`. With the defaults it selects 32 NQ plus 32
HotpotQA training rows and 8 plus 8 test rows when both sources have enough
examples. If one source is short, the other fills the requested total. It
rejects the known Phase-1 fixture markers before writing output and records
source/output SHA256s, selected source positions and `extra_info` identifiers in
`manifest.json`.

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase2_real_data/prepare_gate_data.py \
  --source-train /workspace/searchr1-assets/datasets/nq_hotpotqa_train/train.parquet \
  --source-test /workspace/searchr1-assets/datasets/nq_hotpotqa_train/test.parquet \
  --output-dir /workspace/searchr1-assets/datasets/phase2_real_data_gate \
  --train-size 64 \
  --test-size 16 \
  --seed 42
```

## CPU retriever

The existing `retrieval_server.py` and official Wiki-18 corpus are already
schema-compatible: the server loads JSONL rows directly, and the Search-R1
generation client consumes each row's non-empty `contents` field. No corpus
adapter is needed. The launch script loads the prebuilt Flat index with CPU
FAISS and runs E5 query encoding on CPU; it never enables GPU FAISS.

Terminal 1:

```bash
cd /workspace/Search-R1
PHASE2_REAL_PYTHON=/workspace/Search-R1/.venv-phase1/bin/python \
  bash experiments/phase2_real_data/launch_retriever.sh
```

Terminal 2:

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase2_real_data/verify_retriever.py \
  --url http://127.0.0.1:8000/retrieve \
  --topk 3
```

Verification checks HTTP 200, the exact top-k count, non-empty `contents`,
finite scores, and reports per-query, mean and p50 latency. It intentionally
does not require a particular document or answer.

## One-update reward-signal gate

The gate uses four prompts, `rollout.n=1`, and four independently sampled
agents per prompt: `4 × 1 × 4 = 16` trajectories. The trainer repeats each row
by `n_agent` before assigning a source-aware prompt UID. One PPO mini-batch of
16, accumulated in micro-batches of one, therefore reaches exactly one
`optimizer.step()` with the current actor implementation. Stochastic sampling
is explicit: temperature `1.0` and top-p `0.95`.

The token cap follows the existing Search-R1 launch formula:

```text
max_prompt_length
  = max_start_length
  + max_response_length × (max_turns - 1)
  + max_obs_length × max_turns
  = 768 + 128 × (2 - 1) + 256 × 2
  = 1408
```

`LLMGenerationManager` applies this value as the rolling prompt cap. The actor
KL path remains enabled so reference log-prob computation is exercised, but
the KL coefficient is `0.0`; entropy coefficient is also `0.0`. In
`DataParallelPPOActor.update_policy`, the differentiable objective is therefore
only `pg_loss`. AdamW's configured weight decay remains an optimizer-side
effect, but it cannot make `actor/grad_norm` positive when the policy-gradient
gradient is zero.

```bash
cd /workspace/Search-R1
PHASE2_MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
PHASE2_REAL_DATA_DIR=/workspace/searchr1-assets/datasets/phase2_real_data_gate \
PHASE2_REAL_RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
  bash experiments/phase2_real_data/run_reward_signal_gate.sh
```

The gate retains the proven BF16/XFormers/tensor-parallel-one configuration,
vLLM GPU utilization `0.20`, actor parameter and optimizer offload enabled,
actor gradient offload disabled, reference parameter offload enabled, state
masking, remove-padding and gradient checkpointing.

## Metrics and PASS criteria

Diagnostics use the same source-aware UID groups and per-sequence outcome
reward as GRPO:

- `grpo/group_count`: number of unique prompt UIDs.
- `grpo/groups_with_reward_variance`: groups whose sample standard deviation is
  greater than zero.
- `grpo/group_reward_variance_fraction`: varying groups divided by group count.
- `grpo/group_reward_std_mean`: mean of per-group sample standard deviations;
  singleton standard deviation is defined as zero to avoid NaN.
- `grpo/reward_std_min` and `grpo/reward_std_max`: extrema across group standard
  deviations.
- `grpo/nonzero_advantage_fraction`: fraction of trajectories having at least
  one non-zero, response-masked advantage token.

A PASS requires one optimizer step, four UID groups, at least one varying group,
positive and negative advantages, a positive finite actor gradient norm, zero
KL and entropy optimization coefficients, and no NaN, Inf, CUDA OOM or
traceback. `actor/pg_loss` itself may be near zero because normalized GRPO
advantages are zero-mean.

```bash
grep -E 'training/optimizer_steps|grpo/|critic/score/|critic/rewards/|critic/advantages/|actor/pg_loss|actor/kl_loss|actor/kl_coef|actor/entropy_loss|actor/grad_norm|timing_s/|memory.*(allocated|reserved)|allocated.*GB|reserved.*GB|(^|[^[:alpha:]])(nan|NaN|inf|Inf)([^[:alpha:]]|$)|CUDA.*(OOM|out of memory)|Traceback' \
  /workspace/Search-R1/phase2-real-data-qwen2.5-3b-reward-signal-gate.log
```

If all four groups have identical outcomes, the script exits with status `2`
and prints `gate inconclusive`; that is not an infrastructure failure. Select
the next deterministic four-row window without changing source code:

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase2_real_data/prepare_gate_data.py \
  --source-train /workspace/searchr1-assets/datasets/nq_hotpotqa_train/train.parquet \
  --source-test /workspace/searchr1-assets/datasets/nq_hotpotqa_train/test.parquet \
  --output-dir /workspace/searchr1-assets/datasets/phase2_real_data_gate_offset4 \
  --train-size 64 --test-size 16 --seed 42 --train-offset 4
PHASE2_MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
PHASE2_REAL_DATA_DIR=/workspace/searchr1-assets/datasets/phase2_real_data_gate_offset4 \
  bash experiments/phase2_real_data/run_reward_signal_gate.sh
```

If needed, continue with offsets `8`, `12`, and so on, or regenerate the pool
with a different recorded seed.

## Later progression (documentation only)

After Gate A (one update, 4 prompts × 4 trajectories), the planned progression
is Gate B at five optimizer steps and Gate C at twenty. A 50–100-step run is
optional only if useful. Training the full roughly 170k-row dataset on one A800
is explicitly out of scope; the goals are correctness, observability, memory
and throughput profiling, and distributed-ready infrastructure.
