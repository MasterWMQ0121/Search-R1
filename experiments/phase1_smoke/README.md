# Phase 1 small-scale smoke run

This directory is a CPU-first reproducibility gate for a later one-GPU GRPO
smoke run. CPU preparation is implemented and verified before any accelerator
work. The corpus is intentionally synthetic, answer-bearing test
infrastructure and **is not a retrieval benchmark**.

The code baseline is Search-R1 commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`
on `phase1-small-scale`. FlashRAG is pinned to dataset revision
`bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`, which is also recorded in
`data/phase1_smoke/manifest.json`.

## CPU preparation

From the repository root, create the isolated preparation environment:

```bash
python3.11 -m venv .venv-phase1
.venv-phase1/bin/python -m pip install -r experiments/phase1_smoke/requirements-retriever.txt
```

Prepare and validate the deterministic data:

```bash
experiments/phase1_smoke/prepare_data.sh
.venv-phase1/bin/python experiments/phase1_smoke/verify_artifacts.py
```

Build the tiny E5/FAISS index on CPU:

```bash
experiments/phase1_smoke/build_index.sh
.venv-phase1/bin/python experiments/phase1_smoke/verify_artifacts.py --require-index
```

Run the CPU unit/contract tests:

```bash
.venv-phase1/bin/python -m pytest -q tests/phase1
```

Launch and exercise the retriever in two terminals:

```bash
experiments/phase1_smoke/launch_retriever.sh
```

```bash
.venv-phase1/bin/python experiments/phase1_smoke/verify_retriever.py
```

The server binds to `127.0.0.1:8000`, runs `intfloat/e5-small-v2` on CPU,
at revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`, uses a FAISS `Flat`
inner-product index, and returns top-2 passages. Override
paths and the port with `PHASE1_DATA_DIR`, `PHASE1_PYTHON`, and
`PHASE1_RETRIEVER_PORT`. The CPU scripts default OpenMP and Apple Accelerate to
one thread to avoid a native Torch/FAISS oversubscription crash observed on
macOS ARM; set `OMP_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` explicitly to tune
another host.

## Artifacts and reproducibility

- `train.parquet`: 128 seeded examples from NQ train.
- `test.parquet`: 64 seeded examples from NQ test, used as validation.
- `corpus.jsonl`: 192 answer-bearing evidence fixtures plus 64 distractors.
- `manifest.json`: source revision, seeds, counts, hashes, and a known-query
  retrieval oracle.
- `index/e5_Flat.index`: CPU-built dense index.

The `data/` directory is ignored by Git. Re-run `prepare_data.sh` to recreate
the artifacts. `PHASE1_NQ_REVISION` may override the source revision for an
intentional data refresh; the default stays pinned for byte-stable recreation.

## GPU command: prepared, not executed

Do **not** run this during CPU preparation:

```bash
bash experiments/phase1_smoke/run.sh
```

The script is configured for `Qwen/Qwen2.5-0.5B-Instruct`, one GPU, GRPO,
four trajectories per each of two base questions (8 trajectories), mini-batch
8, micro-batch 1, exactly 8 actor optimizer updates, two search turns, top-k 2,
96 response tokens, 192 observation tokens, 384 initial prompt tokens, and a
1,152-token rolling prompt ceiling. Tensor parallelism is 1, rollout memory
utilization is 0.4, dtype is BF16, gradient checkpointing is enabled, reference
parameters are offloaded, and actor parameters/gradients/optimizer state are
not offloaded.

Before running it, use a Linux NVIDIA environment with BF16 support, CUDA 12.1
compatible drivers, Python 3.9, PyTorch 2.4.0, vLLM 0.6.3, the repository
installed editable, FlashAttention 2, and a live retriever from
`launch_retriever.sh`. GPU dependency installation and the training command
remain deliberately unexecuted in this phase.

See [the detailed implementation note](../../docs/phase1_small_scale.md) for
the contract tests and compatibility changes.
