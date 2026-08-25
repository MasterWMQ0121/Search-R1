# Phase 4: held-out Direct vs Static RAG vs Search-RL benchmark

Phase 4 is the first product-quality comparison in this project. It evaluates
the same held-out questions with three inference structures:

- **Direct:** base Qwen2.5-3B-Instruct, no retrieval, one generation.
- **Static RAG:** the same base model, exactly one E5/Wiki-18 top-3 retrieval,
  then one generation from a fixed context prompt.
- **Search-RL:** the Phase-3 actor checkpoint and the existing Search-R1 agent
  loop, with dynamic search actions and at most two search-enabled turns.

The earlier phases have narrower meanings. Phase 1 proved small answer-leaky
plumbing, Phase 2 proved a real-data GRPO reward signal, and Phase 3 proved 20
sustained real-data optimizer updates and exported an actor checkpoint. None of
those phases measured a held-out quality advantage. Phase 4 may support a claim
such as “Search-RL improved EM by X.X percentage points versus Static RAG,” but
only after `summary.json` computes X.X from a completed A800 run. This README
does not pre-fill or predict that number.

## Fixed primary benchmark

- 64 questions selected with seed 42: 32 NQ and 32 HotpotQA.
- Base model: `/workspace/models/Qwen2.5-3B-Instruct`.
- Search model:
  `/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20`.
- BF16, one GPU, tensor parallel size 1, XFormers, vLLM 0.20 GPU-memory
  utilization, greedy decoding, and 128 generated tokens per model call.
- Retriever: the existing CPU E5-base-v2/Wiki-18 service at
  `http://127.0.0.1:8000/retrieve`, top-k 3.
- Search-RL: `max_turns=2`, `max_start_length=768`,
  `max_response_length=128`, `max_obs_length=256`, and
  `max_prompt_length=1408`.

The Search-RL observation limit deliberately remains **256**. Phase-3 warnings
such as `549 & 256` are evidence to measure, not permission to alter the trained
configuration. The existing tokenizer call, warning, and literal 256-token
prefix slice are unchanged.

Static RAG has an explicit **512-token retrieved-context budget**. The top-3
passages are concatenated in retriever rank order using the same `Doc N(Title:
...)` format as Search-R1, tokenized with the base-model tokenizer, and the
first 512 context tokens are retained. Raw and retained context lengths are
recorded for every example.

## Held-out preparation and overlap guarantee

`prepare_eval_data.py` reads the original source test parquet and the exact
prepared artifacts used by Phase 2 and Phase 3. Before selecting anything it:

1. verifies the source and prepared parquet SHA256 values against both
   manifests;
2. constructs source-aware UIDs as `lowercase(data_source):extra_info.index`;
3. excludes the Phase-2 training UIDs and the Phase-3 training and validation
   UIDs;
4. excludes Phase-2/3 test source positions using the stronger physical
   identity `(source SHA256, test split, source position)`;
5. rejects missing/duplicate UIDs and known Phase-1 fixture markers; and
6. shuffles the remaining NQ and HotpotQA pools deterministically, takes 32 of
   each, and interleaves them.

Selection fails rather than backfilling an unbalanced source or accepting any
overlap. `manifest.json` binds the selected rows to all input/output hashes and
records both UID and physical-position audit results.

Use the actual successful Phase-2 gate directory. The example below names the
documented offset-4 gate; change only that path if the successful A800 artifact
was stored elsewhere.

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase4_benchmark/prepare_eval_data.py \
  --source-test /workspace/searchr1-assets/datasets/nq_hotpotqa_train/test.parquet \
  --phase2-train /workspace/searchr1-assets/datasets/phase2_real_data_gate_offset4/train.parquet \
  --phase2-manifest /workspace/searchr1-assets/datasets/phase2_real_data_gate_offset4/manifest.json \
  --phase3-train /workspace/searchr1-assets/datasets/phase3_real_training/train.parquet \
  --phase3-test /workspace/searchr1-assets/datasets/phase3_real_training/test.parquet \
  --phase3-manifest /workspace/searchr1-assets/datasets/phase3_real_training/manifest.json \
  --output-dir /workspace/searchr1-assets/datasets/phase4_benchmark \
  --eval-size 64 \
  --seed 42
```

Inspect `manifest.json`. It must show 64 selected rows, `nq: 32`,
`hotpotqa: 32`, `non_overlap_audit.passed: true`, and empty UID and source
position overlap lists before evaluation.

## Evaluation semantics

All modes render a one-message Qwen chat prompt, decode greedily, and use the
existing Search-R1 `qa_em` answer extraction and normalized exact match. The
actual cropped prompt is concatenated with the completion/trajectory before
scoring, matching the training reward path in which the prompt supplies the
format example and the last `<answer>...</answer>` is evaluated.

The Direct prompt is:

```text
Answer the given question. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: {question}
```

The Static-RAG prompt is:

```text
Answer the given question with some potentially useful context. You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: {question} Context: {top-3 context, capped at 512 tokenizer tokens}
```

Search-RL receives the original Search-R1 prompt stored in the parquet and runs
`LLMGenerationManager` with unchanged generation semantics. A thin adapter
exposes public vLLM through the manager's existing rollout interface; it does
not reimplement action parsing, invalid-action feedback, retrieved-document
formatting, state masks, or observation truncation. The only shared-code change
is additive token-length/truncation metadata around the existing slice.

Latency must be interpreted structurally. Model construction is excluded from
example latency, but Direct performs one generation, Static RAG performs one
HTTP retrieval plus one generation, and Search-RL can perform up to three model
calls (two search-enabled turns plus the final generation) and zero to two
completed retrievals. Their end-to-end latency numbers are therefore useful
product measurements, not equivalent single-call kernel benchmarks.

## Retriever and A800 run

Start and verify the existing CPU retriever in one terminal:

```bash
cd /workspace/Search-R1
PHASE2_REAL_PYTHON=/workspace/Search-R1/.venv-phase1/bin/python \
  bash experiments/phase2_real_data/launch_retriever.sh
```

From a second terminal, verify HTTP/schema/top-k behavior before evaluation:

```bash
cd /workspace/Search-R1
/workspace/Search-R1/.venv-phase1/bin/python \
  experiments/phase2_real_data/verify_retriever.py \
  --url http://127.0.0.1:8000/retrieve \
  --topk 3
```

In the GPU environment, run all modes. The shell launcher invokes each mode in
a separate Python process; process exit releases vLLM/CUDA state before the
next checkpoint loads.

```bash
cd /workspace/Search-R1
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
PHASE4_PYTHON=python3 \
PHASE4_DATA_DIR=/workspace/searchr1-assets/datasets/phase4_benchmark \
PHASE4_BASE_MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
PHASE4_SEARCH_MODEL_PATH=/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20 \
PHASE4_RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
PHASE4_RESULTS_DIR=/workspace/Search-R1/phase4_benchmark_results \
  bash experiments/phase4_benchmark/run_benchmark.sh all
```

Modes are resumable and persist after every example. Run one mode without
repeating completed rows:

```bash
bash experiments/phase4_benchmark/run_benchmark.sh direct
bash experiments/phase4_benchmark/run_benchmark.sh static_rag
bash experiments/phase4_benchmark/run_benchmark.sh search_rl
bash experiments/phase4_benchmark/run_benchmark.sh summarize
```

Set `PHASE4_OVERWRITE=true` only when intentionally replacing a mode artifact.

## Artifacts and metrics

The result directory contains `direct.jsonl`, `static_rag.jsonl`,
`search_rl.jsonl`, and `summary.json`. Every JSONL row includes UID, source,
question, targets, model/mode, extracted prediction, binary exact match,
trajectory, generation latency, and end-to-end latency. Static RAG also stores
one retrieval latency, document identifiers/titles/scores, context budget, and
raw/retained context token counts.

Search-RL rows additionally store action/search/retrieval counts, finish state,
retrieval failure count, generation/retrieval timing, and:

- `observation_truncation_count`;
- whether the trajectory had any truncated observation;
- each retrieved observation's token length before truncation;
- each retained token length after the existing slice; and
- each excess-token count above `max_obs_length=256`.

As in the Phase-3 environment metrics, a syntactically valid `<search>` emitted
on the final retrieval-disabled generation counts as a valid search action but
not as a successful retrieval. The two fields are intentionally reported
separately.

Search-RL `generation_latency_s` and `retrieval_latency_s` are cumulative
per-trajectory totals across its dynamic calls. Their summary p50/p95 values
are distributions of those trajectory totals, not distributions of individual
model or HTTP calls.

The summary reports overall/NQ/HotpotQA EM and correct/total, end-to-end mean,
p50 and p95 latency, Static-RAG retrieval latency, the requested Search-Agent
behavior metrics, and observation/context truncation aggregates. Percentage-
point comparisons are calculated only from the three measured files.

Failure analysis reports NQ and HotpotQA incorrect counts that co-occur with
Static-RAG context truncation or Search-RL observation truncation. Co-occurrence
makes truncation a plausible contributor worth investigating; it does **not**
establish causality, and the generated assessment says so explicitly.

Every row also carries a hash-bound run configuration covering the evaluation
parquet/manifest, model, retriever URL/top-k, decoding, token limits, backend,
and prompt contract. Resume rejects stale or mixed configurations. A nonzero
Search-RL retrieval-failure count leaves the rows visible in EM and failure
totals, excludes those incomplete trajectories from agent-behavior ratios, and
marks `benchmark_status.quality_claim_ready=false`; rerun before making a
quality claim.
