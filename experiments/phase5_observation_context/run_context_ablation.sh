#!/usr/bin/env bash
# Run Phase-5 conditions in separate processes; never rewrite the Phase-4 baseline.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PHASE5_PYTHON:-python3}"
DATA_DIR="${PHASE5_DATA_DIR:-/workspace/searchr1-assets/datasets/phase4_benchmark}"
RESULTS_DIR="${PHASE5_RESULTS_DIR:-${ROOT_DIR}/phase5_observation_results}"
PHASE4_RESULTS_DIR="${PHASE4_RESULTS_DIR:-${ROOT_DIR}/phase4_benchmark_results}"
BASELINE_PATH="${PHASE5_BASELINE_PATH:-${PHASE4_RESULTS_DIR}/search_rl.jsonl}"
SEARCH_MODEL="${PHASE5_SEARCH_MODEL_PATH:-${PHASE4_SEARCH_MODEL_PATH:-/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20}}"
RETRIEVER_URL="${PHASE5_RETRIEVER_URL:-${PHASE4_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}}"
MODE="${1:-${PHASE5_MODE:-all}}"
LOG_PATH="${PHASE5_LOG_PATH:-${RESULTS_DIR}/phase5_observation_context.log}"
OVERWRITE="${PHASE5_OVERWRITE:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

case "${MODE}" in
  all|validate_baseline|raw_512|compressed_256|summarize) ;;
  *) printf 'Unknown Phase-5 mode: %s\n' "${MODE}" >&2; exit 2 ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Missing Phase-5 Python: %s\n' "${PYTHON_BIN}" >&2
  exit 1
}
[[ -s "${DATA_DIR}/eval.parquet" ]] || {
  printf 'Missing Phase-4 eval parquet: %s\n' "${DATA_DIR}/eval.parquet" >&2
  exit 1
}
[[ -s "${DATA_DIR}/manifest.json" ]] || {
  printf 'Missing Phase-4 eval manifest: %s\n' "${DATA_DIR}/manifest.json" >&2
  exit 1
}
[[ -s "${BASELINE_PATH}" ]] || {
  printf 'Missing immutable Phase-4 Search-RL baseline: %s\n' "${BASELINE_PATH}" >&2
  exit 1
}

mkdir -p "${RESULTS_DIR}" "$(dirname "${LOG_PATH}")"
printf '%s\n' \
  "Phase-5 Search-RL observation-context ablation" \
  "  selected mode: ${MODE}" \
  "  immutable raw_256 baseline: ${BASELINE_PATH}" \
  "  eval data / manifest: ${DATA_DIR}/eval.parquet / ${DATA_DIR}/manifest.json" \
  "  trained Search-RL checkpoint: ${SEARCH_MODEL}" \
  "  GPU count / CUDA_VISIBLE_DEVICES: 1 / ${CUDA_VISIBLE_DEVICES}" \
  "  greedy / seed / dtype: true / 42 / bfloat16" \
  "  Retriever URL / top-k: ${RETRIEVER_URL} / 3" \
  "  raw_512 obs / prompt / model length: 512 / 1920 / 2048" \
  "  compressed_256 obs / prompt / model length: 256 / 1408 / 1536" \
  "  results / log: ${RESULTS_DIR} / ${LOG_PATH}" \
  "  overwrite Phase-5 condition rows only: ${OVERWRITE}" | tee -a "${LOG_PATH}"

run_condition() {
  local condition="$1"
  local overwrite_args=()
  if [[ "${OVERWRITE}" == "true" && "${condition}" != "raw_256_baseline" ]]; then
    overwrite_args+=(--overwrite)
  fi
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    -m experiments.phase5_observation_context.run_context_ablation \
    --condition "${condition}" \
    --eval-data "${DATA_DIR}/eval.parquet" \
    --eval-manifest "${DATA_DIR}/manifest.json" \
    --output-dir "${RESULTS_DIR}" \
    --baseline-results "${BASELINE_PATH}" \
    --search-model "${SEARCH_MODEL}" \
    --retriever-url "${RETRIEVER_URL}" \
    --gpu-memory-utilization 0.20 \
    --seed 42 \
    "${overwrite_args[@]}" 2>&1 | tee -a "${LOG_PATH}"
}

summarize() {
  "${PYTHON_BIN}" -m experiments.phase5_observation_context.summarize_results \
    --baseline-results "${BASELINE_PATH}" \
    --results-dir "${RESULTS_DIR}" \
    --eval-manifest "${DATA_DIR}/manifest.json" \
    --search-model "${SEARCH_MODEL}" 2>&1 | tee -a "${LOG_PATH}"
}

cd "${ROOT_DIR}"
if [[ "${MODE}" == "all" ]]; then
  run_condition raw_256_baseline
  run_condition raw_512
  run_condition compressed_256
  summarize
elif [[ "${MODE}" == "validate_baseline" ]]; then
  run_condition raw_256_baseline
elif [[ "${MODE}" == "summarize" ]]; then
  run_condition raw_256_baseline
  summarize
else
  run_condition raw_256_baseline
  run_condition "${MODE}"
fi
