# Phase 5: Search-RL observation-context optimization

Phase 5 is a paired inference ablation for the trained Phase-3 Search-RL actor.
It does not train or change a model. It asks whether the held-out Phase-4 result
is limited by the observation-token capacity or by how the existing capacity is
allocated.

The three conditions use the same 64 audited UIDs (32 NQ and 32 HotpotQA),
questions, targets, actor checkpoint, Wiki-18/E5 Retriever, top-k 3, Search-R1
prompt and state machine, two-turn limit, greedy decoding, seed 42, and
normalized exact-match evaluator:

- `raw_256_baseline` is the existing immutable Phase-4 `search_rl.jsonl`. It is
  validated in place and is never rerun, copied, or rewritten. Its measured
  observation-truncation rate was above 95%.
- `raw_512` preserves the current `Doc N(Title: ...)` top-3 rendering and changes
  only the observation cap from 256 to 512 tokens. This is an inference-time
  context-length distribution shift because the actor was trained with a
  256-token observation cap. It also increases rolling-context/KV cost.
- `compressed_256` keeps the trained 256-token budget but uses deterministic,
  query-aware extractive evidence selection before the existing manager-side
  tokenizer/slice boundary. The manager's original 256-token prefix slice
  remains as a safety boundary.

No Phase-5 quality result is asserted here. The A800 artifacts and paired
statistics determine the measured outcome.

## Fixed token budgets

The rolling prompt formula is:

```text
max_prompt_length = max_start_length
                  + max_response_length * (max_turns - 1)
                  + max_obs_length * max_turns
```

With `max_start_length=768`, `max_response_length=128`, and `max_turns=2`, the
raw-512 condition is `768 + 128 * 1 + 512 * 2 = 1920`. Its standalone vLLM
limit is `1920 + 128 = 2048`. The baseline and compressed conditions retain
`max_obs_length=256`, `max_prompt_length=1408`, and `max_model_len=1536`.

All conditions otherwise retain BF16, tensor parallel size 1, XFormers, one
GPU, and vLLM GPU-memory utilization 0.20.

## Deterministic evidence policy

`evidence_compressor.py` receives only the generated search query, the one
structured top-3 response already returned by the Retriever, and its raw
rendering. Its API has no ground-truth argument, performs no retrieval or model
call, and adds no NLP dependency.

It NFKC-normalizes/case-folds Unicode alphanumeric query and sentence terms,
uses a fixed conservative stop-word list, and splits document bodies on stable
newline and `.?!` boundaries. For unique query terms `Q`, candidate sentence
`s` is scored as:

```text
relevance(s) = BM25(s, Q)
             + 0.75 * query_coverage(s)
             + 0.25 * title_query_coverage(s)
             + 0.05 / retrieval_rank
             + 0.05 * normalized_retrieval_score

priority(s) = relevance(s) / sqrt(token_count(s))
```

BM25 uses `k1=1.2`, `b=0.75`, and
`idf(t)=ln(1 + (N-df(t)+0.5)/(df(t)+0.5))`. Rank and Retriever-score priors
apply only when the sentence/title has a lexical query signal. Exact normalized
duplicate sentences are selected once while retaining alias provenance. A
diversity pass first gives each relevant retrieved document one opportunity,
then fills globally by priority. Rendering is grouped by original document rank
and original sentence order.

Every tentative addition re-tokenizes the complete literal observation:

```text
\n\n<information>{compressed content}</information>\n\n
```

against the actual Qwen tokenizer and 256-token cap. If no sentence overlaps
the query, concise leading evidence is chosen deterministically across ranks.
If no complete sentence fits, a token-safe source prefix is used and recorded;
nonempty evidence is not silently replaced by an empty information block.

## Baseline validation and resumability

The baseline validator requires exactly 64 valid Phase-4 Search-RL rows in
evaluation order, the Phase-3 `global_step_20` actor path, greedy seed-42
decoding, top-k 3, two turns, the 256/1408 limits, matching evaluation parquet
and manifest hashes, and zero Retriever failures or evaluation errors. It
computes the baseline SHA256 before and after validation. Every new run config
binds that SHA, baseline fingerprint, evaluation hashes, checkpoint, Retriever,
decoding settings, and token policy. Resume rejects stale or mixed configs.

New results are appended per example and then deterministically ordered. The
writer explicitly rejects an output path equal to the immutable baseline.
`PHASE5_OVERWRITE=true` applies only to Phase-5 condition files.

## A800 execution

Start and verify the existing CPU Retriever as documented in Phase 4. Then use
the GPU environment in a second terminal:

```bash
cd /workspace/Search-R1
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export PHASE5_PYTHON=python3
export PHASE5_DATA_DIR=/workspace/searchr1-assets/datasets/phase4_benchmark
export PHASE5_BASELINE_PATH=/workspace/Search-R1/phase4_benchmark_results/search_rl.jsonl
export PHASE5_SEARCH_MODEL_PATH=/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20
export PHASE5_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export PHASE5_RESULTS_DIR=/workspace/Search-R1/phase5_observation_results

bash experiments/phase5_observation_context/run_context_ablation.sh validate_baseline
bash experiments/phase5_observation_context/run_context_ablation.sh raw_512
bash experiments/phase5_observation_context/run_context_ablation.sh compressed_256
bash experiments/phase5_observation_context/run_context_ablation.sh summarize
```

The two GPU conditions run in separate Python processes so vLLM/CUDA state is
released between them. `all` performs those four operations in order. Verify
the new row counts without touching the baseline:

```bash
wc -l \
  /workspace/Search-R1/phase4_benchmark_results/search_rl.jsonl \
  /workspace/Search-R1/phase5_observation_results/raw_512.jsonl \
  /workspace/Search-R1/phase5_observation_results/compressed_256.jsonl
```

## Artifacts and metrics

Phase 5 writes only:

```text
/workspace/Search-R1/phase5_observation_results/
  raw_512.jsonl
  compressed_256.jsonl
  summary.json
  paired_statistics.json
  paired_statistics.md
```

Each new JSONL row retains Phase-4 prediction, trajectory, exact-match,
end-to-end/generation/retrieval latency, action/search/retrieval counts,
finish/failure state, and manager truncation telemetry. It adds the observation
policy; raw, policy-output, and retained token counts; compression ratio;
post-policy truncation count; documents returned/represented; sentences
considered/selected; compression latency; zero-overlap/partial-prefix flags;
and selected ranks plus auditable sentence identifiers.

For `raw_512`, raw and policy-output observations are identical, compression
ratio is 1.0, and compression latency and sentence-selection counts are zero.
For `compressed_256`, raw top-3 and compressed lengths remain separate from any
remaining manager safety truncation.

The summary reports overall/NQ/HotpotQA EM and correct/total; mean/p50/p95
end-to-end, generation, retrieval, and compression latency; finish, action,
search, and retrieval behavior; and the requested context/token/document,
sentence, truncation, and fallback aggregates. Descriptive deltas are oriented
as compressed-256 minus raw-256, raw-512 minus raw-256, and compressed-256 minus
raw-512.

Paired analysis joins strictly by UID and requires identical question, target,
source, and evaluation hashes. The pre-specified primary comparison is
`compressed_256 - raw_256_baseline`; the two other comparisons are exploratory.
Overall intervals use 10,000 source-stratified paired bootstrap samples; NQ and
HotpotQA intervals resample within source. All use seed 42, 95% percentile
intervals, and exact two-sided McNemar tests.

Engineering completion requires 64 rows per condition, 32/32 source balance,
identical identities/hashes, a validated unchanged baseline, and no retrieval
or evaluation errors. Compression completion additionally requires wrapped
compressed observations at or below 256 tokens, fewer post-policy truncations,
determinism, no answer leakage, and no additional Retriever call. EM improvement
is desirable product signal, not an engineering-pass requirement.

Interpretation remains neutral: a compressed-context gain is evidence only on
this held-out set; a raw-512-only gain favors capacity over this heuristic; a
compressed condition matching longer raw context with fewer tokens indicates
fixed-budget efficiency; and no gain means frequent truncation alone was not
sufficient evidence that more or selected context improves this checkpoint.
