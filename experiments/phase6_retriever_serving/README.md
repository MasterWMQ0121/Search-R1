# Phase 6: Retriever serving optimization

Phase 6 is an inference/serving experiment. It does not train a model, change
the Search-R1 Agent, re-embed Wiki-18, or use the A800 for retrieval. It
separates three questions that must not be conflated:

1. **Exact serving:** where CPU E5 + Flat retrieval time is spent, and how
   FAISS threads, native request batching, and exact process-local caches affect
   latency/throughput.
2. **ANN trade-off:** how a CPU IVF-PQ index reconstructed from the existing
   Flat vectors trades Recall@3, search latency, RSS, and disk.
3. **Product behavior:** whether the selected ANN candidate changes held-out
   `compressed_256` Agent EM, behavior, or end-to-end latency relative to a
   contemporaneous exact replay.

The existing measured Phase-5 result motivates this work, but this README
contains no measured Phase-6 result or quality-preservation claim. Calibration
Recall@3 is a serving selection signal; paired held-out Agent EM remains the
authoritative product check.

## Fixed Agent contract

Both downstream conditions retain the Phase-3 `global_step_20` actor,
Phase-5 query-aware `compressed_256` policy, the same 64 Phase-4 UIDs, prompt,
tokenizer, greedy seed-42 decoding, BF16, `top-k=3`, `max_turns=2`,
`max_obs_length=256`, `max_prompt_length=1408`, and `max_model_len=1536`.
Only the cache-disabled Retriever URL/index configuration differs:

- `flat_exact_replay`: current Flat index on port 8100;
- `ivfpq_selected`: calibration-selected IVF-PQ/nprobe on port 8101.

The optional Phase-4 Retriever hook defaults to `None`; Phase-4/5 callers and
their run fingerprints are unchanged. For Phase 6, the hook installs an
opt-in metrics client before the existing Phase-4 timer. The Phase-5
compression policy remains outside that timer and invokes the wrapped client
exactly once per Agent search. Each runnable condition content-fingerprints the
local Phase-3 Qwen checkpoint before model construction, binds that fingerprint
into resume state, and verifies the checkpoint again at the lifecycle boundary.

## Instrumented CPU server

`optimized_retrieval_server.py` reuses the existing CPU E5 `Encoder`, corpus
loader, and document loader. The default `POST /retrieve` response remains
exactly `{"result": ...}`. A request with `"return_metrics": true` also returns
ordered internal FAISS row IDs and these `time.perf_counter()` stages:

- `request_total_s` — application handler work from request dispatch through
  response-object construction;
- `query_normalization_s` — type/empty checks and exact `str.strip()` only;
- `cache_lookup_s` — result/embedding LRU access and insertion;
- `query_encoding_s` — CPU E5 encoding, including the existing `query: ` prefix,
  mean pooling, normalization, and float32 conversion;
- `faiss_search_s` — native batched `index.search`;
- `document_fetch_s` — corpus rows fetched from FAISS IDs;
- `response_format_s` — Python response-object assembly.

`response_format_s` and `request_total_s` do not claim to include ASGI JSON
serialization or network transfer; client latency measures those boundaries.
`/healthz` exposes content fingerprints, index/corpus integrity, backend,
dimension, metric, threads, nprobe, cache configuration, startup timings, and
stdlib `/proc` RSS when available. `/stats` exposes cumulative request/query/
error counts, stage sums/means, cache counters, uptime, and RSS.

The server is CPU-only and uses one Uvicorn worker. Corpus row count must equal
FAISS `ntotal`; Flat/IVF-PQ class, metric, IDs, scores, and top-k cardinality are
validated before results are returned.

## Exact cache semantics

Both bounded LRU caches are disabled by default, process-local, thread-safe,
cleared on restart, and never persisted. Result keys bind the exact stripped
query, top-k, index/model content fingerprints, and every result-affecting
retrieval setting. Embedding keys bind the exact stripped query and encoder
contract. Queries are never lower-cased or semantically normalized. Empty
queries fail before lookup.

Cold and ANN-quality workloads require both caches disabled. A separate warm
workload enables caches explicitly, primes every calibration query in excluded
requests, and reports result/embedding hit rates. Warm-cache throughput must
not be presented as cold unique-query latency.

## Calibration set

`prepare_calibration_queries.py` reads Phase-3 training rows and deterministically
selects 128 globally unique questions with seed 42, targeting 64 NQ and 64
HotpotQA when available and deterministically filling a short source. It reuses
the Phase-4 source-aware UID/question conventions, audits exact UID overlap
against the Phase-4 held-out manifest, and fails on overlap or insufficient
unique rows.

`queries.jsonl` contains only UID, source, question, question hash, and source
position—never answers, reward targets, or EM labels. Its manifest binds the
source train parquet, held-out eval/manifest, ordered UIDs/hashes, source counts,
selection audit, and output by SHA256.

## IVF-PQ construction and safety

`build_ivfpq_index.py` never re-embeds the corpus. It opens the source Flat
index read-only/mmap where supported (recording a fallback), requires actual
`reconstruct_n` support, and verifies source class, `ntotal`, dimension, metric,
and SHA256. Defaults are `nlist=4096`, `m=96`, `nbits=8`, inner product,
262144 training vectors, 4096-vector sampling blocks, 32768-vector add chunks,
and seed 42; dimension is discovered and must be divisible by `m`.

Training samples are exact-size, disjoint contiguous blocks drawn from broad
deterministic strata. Only those ranges are reconstructed and the ranges are
recorded. Every source vector is then reconstructed in sequential chunks and
added with its explicit original `int64` row ID. This preserves corpus-ID
alignment without an `O(ntotal)` sampling array or full second vector copy.

Before any build, runtime metadata drives conservative free-disk/RAM checks.
The source hash is checked again before publication. The candidate is written
to `.tmp`, fsynced, reread, and checked for trained state, dimension, `ntotal`,
metric, IVF/PQ parameters, and a nonempty search before atomic rename and an
atomic SHA-bound manifest. The source is never overwritten.

Resume means **validated restart from zero**, not add-stage continuation. A
stale temp/state requires `--resume` (matching config fingerprint and source
SHA) or `--overwrite`; mismatched/partial state is refused. A complete output
with a valid manifest is verified and reused.

## Offline definitions and candidate rule

`benchmark_indexes.py` encodes each ordered calibration query exactly once
with the same CPU E5 contract and persists a model/query/hash-bound float32
artifact. The Flat top-3 reference is collected once. IVF-PQ is evaluated for
`nprobe={4,8,16,32,64,128}` and threads `{4,8,16}`; warm-ups are excluded and
each latency sample is one `index.search(query[None, :], 3)` call.

- Recall@1 / top-1 agreement: ANN top-1 equals Flat top-1.
- Recall@3: mean per-query fraction of Flat top-3 IDs present in ANN top-3.
- Full top-3 agreement: fraction with identical unordered top-3 sets.
- QPS: query count divided by the sum of measured pure-search latency.
- Invalid count: rows with missing/out-of-range/duplicate-short ANN IDs.

Selection uses no QA target. Valid configurations with Recall@3 at least
`PHASE6_MIN_RECALL_AT_3` (default 0.95) are ordered by lowest p95, then mean,
then nprobe, then thread count. If none passes, the deterministic
highest-recall Pareto candidate is reported with
`candidate_selection_passed=false`; it is not production-ready and downstream
Agent evaluation requires explicit `--allow-unqualified-candidate`/
`PHASE6_ALLOW_UNQUALIFIED_CANDIDATE=true` authorization.

## HTTP workload definitions

`benchmark_server.py` records native batch sizes `{1,4,8,16,32}`, client and
server latency distributions, QPS/batches-per-second, server stages, cache hit
rates, RSS, and errors. Circular exact-size batches cover every calibration UID;
the 128-row default divides every prescribed batch size exactly. Warm-ups are
excluded.

The report supports append-only sequential targets, allowing the large Flat
index to be loaded once per thread setting rather than four times concurrently.
Completion requires the same Flat/model/corpus identity at threads
`{1,4,8,16}`, complete UIDs/batch sizes, zero errors, identical selected serving
threads and retrieval encode batch size for Flat and IVF-PQ, and live
Flat/IVF/model fingerprints plus nprobe matching `selected_candidate.json`.
Cold and warm sections and ID comparisons remain separate. Final
summarization also requires the HTTP and downstream Agent servers to share the
same corpus fingerprint and retrieval encode batch size.

## A800/Linux workflow

All retrieval and index commands below are CPU workloads. Only the final Agent
runs use the A800 for Qwen inference.

```bash
cd /workspace/Search-R1
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export PHASE6_ROOT=/workspace/Search-R1
export PHASE6_ASSETS=/workspace/searchr1-assets
export PHASE6_RESULTS=/workspace/Search-R1/phase6_retriever_results
export PHASE6_CALIBRATION=/workspace/searchr1-assets/datasets/phase6_retriever_calibration
export PHASE6_FLAT=/workspace/searchr1-assets/wiki18/e5_Flat.index
export PHASE6_IVFPQ=/workspace/searchr1-assets/wiki18/e5_IVFPQ_nlist4096_m96_nbits8.index
export PHASE6_CORPUS=/workspace/searchr1-assets/wiki18/wiki-18.jsonl
export PHASE6_E5=/workspace/searchr1-assets/models/e5-base-v2
export PHASE6_BASELINE=/workspace/Search-R1/phase5_observation_results/compressed_256.jsonl
mkdir -p "${PHASE6_RESULTS}" "${PHASE6_CALIBRATION}"
```

### 1. Freeze/audit the immutable Phase-5 input

```bash
test -s "${PHASE6_BASELINE}"
sha256sum "${PHASE6_BASELINE}" | tee "${PHASE6_RESULTS}/phase5_compressed_256.sha256"
bash experiments/phase6_retriever_serving/run_agent_retriever_ablation.sh validate_baseline
```

### 2. Prepare non-held-out calibration queries

```bash
python -m experiments.phase6_retriever_serving.prepare_calibration_queries \
  --source-train "${PHASE6_ASSETS}/datasets/phase3_real_training/train.parquet" \
  --heldout-eval "${PHASE6_ASSETS}/datasets/phase4_benchmark/eval.parquet" \
  --heldout-manifest "${PHASE6_ASSETS}/datasets/phase4_benchmark/manifest.json" \
  --output-dir "${PHASE6_CALIBRATION}" --sample-size 128 --seed 42
```

### 3. Discover metadata and preflight the ANN build

```bash
python -m experiments.phase6_retriever_serving.build_ivfpq_index \
  --source-index "${PHASE6_FLAT}" --output-index "${PHASE6_IVFPQ}" \
  --nlist 4096 --m 96 --nbits 8 --training-sample-size 262144 \
  --add-chunk-size 32768 --seed 42 --preflight-only \
  | tee "${PHASE6_RESULTS}/ivfpq_preflight.json"
```

Inspect `passed`, discovered `ntotal`/dimension/metric, and the byte estimates
before authorizing the long build.

### 4. Build, then independently verify IVF-PQ

```bash
python -m experiments.phase6_retriever_serving.build_ivfpq_index \
  --source-index "${PHASE6_FLAT}" --output-index "${PHASE6_IVFPQ}" \
  --nlist 4096 --m 96 --nbits 8 --training-sample-size 262144 \
  --add-chunk-size 32768 --seed 42

python -m experiments.phase6_retriever_serving.build_ivfpq_index \
  --source-index "${PHASE6_FLAT}" --output-index "${PHASE6_IVFPQ}" \
  --nlist 4096 --m 96 --nbits 8 --training-sample-size 262144 \
  --add-chunk-size 32768 --seed 42 --verify-only
```

### 5. Benchmark indexes and select calibration-only serving configuration

```bash
python -m experiments.phase6_retriever_serving.benchmark_indexes \
  --queries "${PHASE6_CALIBRATION}/queries.jsonl" \
  --queries-manifest "${PHASE6_CALIBRATION}/manifest.json" \
  --model-path "${PHASE6_E5}" \
  --embedding-artifact "${PHASE6_RESULTS}/calibration_query_embeddings.npz" \
  --flat-index "${PHASE6_FLAT}" --ivfpq-index "${PHASE6_IVFPQ}" \
  --ivfpq-manifest "${PHASE6_IVFPQ%.index}.manifest.json" \
  --output-dir "${PHASE6_RESULTS}" \
  --nprobes 4,8,16,32,64,128 --thread-counts 4,8,16 \
  --min-recall-at-3 "${PHASE6_MIN_RECALL_AT_3:-0.95}"

python -c 'import json; p=json.load(open("/workspace/Search-R1/phase6_retriever_results/selected_candidate.json")); print(json.dumps(p, indent=2))'
```

### 6. Read the selected serving configuration

Read `nprobe` and `faiss_thread_count` from `selected_candidate.json`; do not
guess them. The HTTP workflow below starts and stops its own servers, so first
stop any Retriever processes already using ports 8100 or 8101.

```bash
export PHASE6_PYTHON="${PHASE6_PYTHON:-python}"
export PHASE6_SELECTED="${PHASE6_RESULTS}/selected_candidate.json"
command -v "${PHASE6_PYTHON}" >/dev/null
test -s "${PHASE6_SELECTED}"
export PHASE6_THREADS="$("${PHASE6_PYTHON}" -c 'import json,os; print(json.load(open(os.environ["PHASE6_SELECTED"]))["selected_candidate"]["faiss_thread_count"])')"
export PHASE6_NPROBE="$("${PHASE6_PYTHON}" -c 'import json,os; print(json.load(open(os.environ["PHASE6_SELECTED"]))["selected_candidate"]["nprobe"])')"
printf 'selected threads=%s nprobe=%s\n' "${PHASE6_THREADS}" "${PHASE6_NPROBE}"
```

### 7. Benchmark HTTP serving with explicit restarts

Run this block in one Bash shell. It refuses to overwrite an existing report.
The exact thread sweep loads only one Flat server at a time and stops it before
the next thread setting. Only the selected Flat and IVF-PQ servers coexist for
the paired cold and warm measurements; both are stopped before their
cache-enabled replacements start.

```bash
set -euo pipefail

export PHASE6_SERVER_REPORT="${PHASE6_RESULTS}/server_benchmark.json"
export PHASE6_SERVER_LOGS="${PHASE6_RESULTS}/server_logs"
export PHASE6_SERVER_START_TIMEOUT_S="${PHASE6_SERVER_START_TIMEOUT_S:-1800}"
test ! -e "${PHASE6_SERVER_REPORT}" || {
  printf 'Refusing to overwrite existing report: %s\n' "${PHASE6_SERVER_REPORT}" >&2
  exit 1
}
mkdir -p "${PHASE6_SERVER_LOGS}"

FLAT_PID=""
ANN_PID=""

stop_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

stop_servers() {
  stop_pid "${FLAT_PID:-}"
  stop_pid "${ANN_PID:-}"
  FLAT_PID=""
  ANN_PID=""
}

wait_for_health() {
  local url="$1"
  local pid="$2"
  local label="$3"
  local deadline=$((SECONDS + PHASE6_SERVER_START_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf '%s exited before becoming healthy; inspect %s\n' \
        "${label}" "${PHASE6_SERVER_LOGS}/${label}.log" >&2
      return 1
    fi
    sleep 1
  done
  printf 'Timed out waiting for %s at %s\n' "${label}" "${url}" >&2
  return 1
}

start_flat() {
  local threads="$1"
  local cache_mode="$2"
  local label="$3"
  local -a cache_args=(--cache-disabled)
  if [[ "${cache_mode}" == "warm" ]]; then
    cache_args=(--cache-enabled --result-cache-capacity 256 --embedding-cache-capacity 256)
  fi
  "${PHASE6_PYTHON}" -m experiments.phase6_retriever_serving.optimized_retrieval_server \
    --index-path "${PHASE6_FLAT}" --corpus-path "${PHASE6_CORPUS}" \
    --model-path "${PHASE6_E5}" --index-backend flat \
    --faiss-thread-count "${threads}" --port 8100 "${cache_args[@]}" \
    >"${PHASE6_SERVER_LOGS}/${label}.log" 2>&1 &
  FLAT_PID=$!
  wait_for_health http://127.0.0.1:8100/healthz "${FLAT_PID}" "${label}"
}

start_ann() {
  local cache_mode="$1"
  local label="$2"
  local -a cache_args=(--cache-disabled)
  if [[ "${cache_mode}" == "warm" ]]; then
    cache_args=(--cache-enabled --result-cache-capacity 256 --embedding-cache-capacity 256)
  fi
  "${PHASE6_PYTHON}" -m experiments.phase6_retriever_serving.optimized_retrieval_server \
    --index-path "${PHASE6_IVFPQ}" --corpus-path "${PHASE6_CORPUS}" \
    --model-path "${PHASE6_E5}" --index-backend ivfpq \
    --ivf-nprobe "${PHASE6_NPROBE}" --faiss-thread-count "${PHASE6_THREADS}" \
    --port 8101 "${cache_args[@]}" \
    >"${PHASE6_SERVER_LOGS}/${label}.log" 2>&1 &
  ANN_PID=$!
  wait_for_health http://127.0.0.1:8101/healthz "${ANN_PID}" "${label}"
}

trap stop_servers EXIT

benchmark_args=(
  --queries "${PHASE6_CALIBRATION}/queries.jsonl"
  --calibration-manifest "${PHASE6_CALIBRATION}/manifest.json"
  --selected-candidate "${PHASE6_SELECTED}"
  --output "${PHASE6_SERVER_REPORT}"
)

append_args=()
for threads in 1 4 8 16; do
  start_flat "${threads}" cold "flat_threads_${threads}"
  "${PHASE6_PYTHON}" -m experiments.phase6_retriever_serving.benchmark_server \
    "${benchmark_args[@]}" --workload cold "${append_args[@]}" \
    --target "flat_threads_${threads}|exact_sweep|http://127.0.0.1:8100"
  stop_servers
  append_args=(--append)
done

start_flat "${PHASE6_THREADS}" cold selected_cold_flat
start_ann cold selected_cold_ivfpq
"${PHASE6_PYTHON}" -m experiments.phase6_retriever_serving.benchmark_server \
  "${benchmark_args[@]}" --workload cold --append \
  --target 'flat_exact|flat_exact|http://127.0.0.1:8100' \
  --target 'ivfpq_selected|ivfpq_selected|http://127.0.0.1:8101'
stop_servers

start_flat "${PHASE6_THREADS}" warm selected_warm_flat
start_ann warm selected_warm_ivfpq
"${PHASE6_PYTHON}" -m experiments.phase6_retriever_serving.benchmark_server \
  "${benchmark_args[@]}" --workload warm --append \
  --target 'warm_flat|warm_flat|http://127.0.0.1:8100' \
  --target 'warm_ivfpq|warm_ivfpq|http://127.0.0.1:8101' \
  --require-complete
stop_servers
trap - EXIT
```

### 8–10. Run Agent exact replay, ANN, then summarize

The HTTP workflow leaves both ports stopped. With the A800/Linux exports from
the start of this workflow loaded, start exactly one cache-disabled selected
Flat server in one CPU terminal:

```bash
export PHASE6_SELECTED="${PHASE6_RESULTS}/selected_candidate.json"
export PHASE6_THREADS="$(python -c 'import json,os; print(json.load(open(os.environ["PHASE6_SELECTED"]))["selected_candidate"]["faiss_thread_count"])')"
python -m experiments.phase6_retriever_serving.optimized_retrieval_server \
  --index-path "${PHASE6_FLAT}" --corpus-path "${PHASE6_CORPUS}" \
  --model-path "${PHASE6_E5}" --index-backend flat \
  --faiss-thread-count "${PHASE6_THREADS}" --port 8100 --cache-disabled
```

Start exactly one cache-disabled selected IVF-PQ server in a second CPU
terminal:

```bash
export PHASE6_SELECTED="${PHASE6_RESULTS}/selected_candidate.json"
export PHASE6_THREADS="$(python -c 'import json,os; print(json.load(open(os.environ["PHASE6_SELECTED"]))["selected_candidate"]["faiss_thread_count"])')"
export PHASE6_NPROBE="$(python -c 'import json,os; print(json.load(open(os.environ["PHASE6_SELECTED"]))["selected_candidate"]["nprobe"])')"
python -m experiments.phase6_retriever_serving.optimized_retrieval_server \
  --index-path "${PHASE6_IVFPQ}" --corpus-path "${PHASE6_CORPUS}" \
  --model-path "${PHASE6_E5}" --index-backend ivfpq \
  --ivf-nprobe "${PHASE6_NPROBE}" --faiss-thread-count "${PHASE6_THREADS}" \
  --port 8101 --cache-disabled
```

Verify both health endpoints, then run the Agent conditions serially in the GPU
environment:

```bash
curl -fsS http://127.0.0.1:8100/healthz >/dev/null
curl -fsS http://127.0.0.1:8101/healthz >/dev/null
```

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export PHASE6_PYTHON=python3
export PHASE6_RESULTS_DIR=/workspace/Search-R1/phase6_retriever_results
export PHASE6_SELECTED_CANDIDATE=/workspace/Search-R1/phase6_retriever_results/selected_candidate.json
export PHASE6_PHASE5_BASELINE=/workspace/Search-R1/phase5_observation_results/compressed_256.jsonl
export PHASE6_SEARCH_MODEL_PATH=/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20
export PHASE6_FLAT_RETRIEVER_URL=http://127.0.0.1:8100/retrieve
export PHASE6_IVFPQ_RETRIEVER_URL=http://127.0.0.1:8101/retrieve

bash experiments/phase6_retriever_serving/run_agent_retriever_ablation.sh flat_exact_replay
bash experiments/phase6_retriever_serving/run_agent_retriever_ablation.sh ivfpq_selected
bash experiments/phase6_retriever_serving/run_agent_retriever_ablation.sh summarize

python -c 'import json; p=json.load(open("/workspace/Search-R1/phase6_retriever_results/paired_statistics.json")); print(json.dumps(p["primary"], indent=2))'
sha256sum -c "${PHASE6_RESULTS}/phase5_compressed_256.sha256"
```

## Artifacts

```text
/workspace/searchr1-assets/datasets/phase6_retriever_calibration/
  queries.jsonl
  manifest.json

/workspace/searchr1-assets/wiki18/
  e5_IVFPQ_nlist4096_m96_nbits8.index
  e5_IVFPQ_nlist4096_m96_nbits8.manifest.json

/workspace/Search-R1/phase6_retriever_results/
  phase5_compressed_256.sha256
  ivfpq_preflight.json
  calibration_query_embeddings.npz
  calibration_query_embeddings.manifest.json
  index_benchmark.json
  selected_candidate.json
  server_benchmark.json
  flat_exact_replay.jsonl
  flat_exact_replay.server_audit.json
  ivfpq_selected.jsonl
  ivfpq_selected.server_audit.json
  summary.json
  paired_statistics.json
  paired_statistics.md
```

The Phase-5 baseline remains a read-only input at its original path.

## Runtime-only disk/RAM estimates

No production number is asserted on the Mac. The preflight discovers source
size, dimension `d`, and `ntotal=N`, then reports bytes. Its conservative raw
candidate estimate is:

```text
N * (ceil(m * nbits / 8) + 8-byte ID)
+ nlist * d * 4
+ 2^nbits * d * 4
+ 64 MiB structural allowance
```

The final-index upper bound applies a 1.35 factor and the default disk gate then
applies 1.25. Build RAM includes the larger of source file bytes and `N*d*4`,
the candidate upper bound, three training-sample buffers, two add-chunk buffers,
and 512 MiB workspace, followed by the default 1.20 safety factor. These are
deliberately conservative estimates, not measured peak RSS. Compare them with
the preflight's discovered free disk/RAM and later `/healthz` RSS.

## Remaining risks and interpretation

- FAISS mmap support is build-dependent; a recorded fallback may materialize
  the Flat index and make RAM the limiting resource.
- IVF-PQ training/add is a long restart-from-zero CPU operation. There is no
  claim of durable add-stage resume.
- Calibration Recall@3 can miss downstream effects from ordering, score
  changes, multi-turn query distribution, or errors concentrated on held-out
  questions.
- FAISS thread settings are process-global; benchmark configurations are run
  serially, not concurrently in one process.
- Cache results apply only to repeated exact query strings and should not be
  generalized to cold or shifting Agent traffic.
- Native batch scaling measures the existing multi-query endpoint, not an
  async dynamic microbatcher or concurrent-client scheduler.
- The exact replay can drift from Phase 5 because runtime/model/library state
  may differ; the summary reports exact mismatch UIDs and withholds product
  readiness when drift is observed.
- Paired EM on 64 rows remains statistically limited. The 10,000-sample
  source-stratified bootstrap and exact McNemar result quantify uncertainty but
  do not prove general production equivalence.
