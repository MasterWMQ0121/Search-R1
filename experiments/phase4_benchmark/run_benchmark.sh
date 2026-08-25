#!/usr/bin/env bash
# Run Phase-4 modes in separate processes so only one model occupies the GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PHASE4_PYTHON:-python3}"
DATA_DIR="${PHASE4_DATA_DIR:-/workspace/searchr1-assets/datasets/phase4_benchmark}"
RESULTS_DIR="${PHASE4_RESULTS_DIR:-${ROOT_DIR}/phase4_benchmark_results}"
BASE_MODEL="${PHASE4_BASE_MODEL_PATH:-${PHASE2_MODEL_PATH:-/workspace/models/Qwen2.5-3B-Instruct}}"
SEARCH_MODEL="${PHASE4_SEARCH_MODEL_PATH:-/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20}"
RETRIEVER_URL="${PHASE4_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
MODE="${1:-${PHASE4_MODE:-all}}"
LOG_PATH="${PHASE4_LOG_PATH:-${RESULTS_DIR}/phase4_benchmark.log}"
OVERWRITE="${PHASE4_OVERWRITE:-false}"

SEED="42"
RETRIEVER_TOPK="3"
STATIC_CONTEXT_TOKEN_BUDGET="512"
MAX_START_LENGTH="768"
MAX_RESPONSE_LENGTH="128"
MAX_OBS_LENGTH="256"
MAX_TURNS="2"
MAX_PROMPT_LENGTH="1408"
VLLM_GPU_MEMORY_UTILIZATION="0.20"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

case "${MODE}" in
  all|direct|static_rag|search_rl|summarize) ;;
  *) printf 'Unknown Phase-4 mode: %s\n' "${MODE}" >&2; exit 2 ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Missing Phase-4 Python: %s\n' "${PYTHON_BIN}" >&2
  exit 1
}
[[ -s "${DATA_DIR}/eval.parquet" ]] || {
  printf 'Missing Phase-4 eval parquet: %s\n' "${DATA_DIR}/eval.parquet" >&2
  exit 1
}
[[ -s "${DATA_DIR}/manifest.json" ]] || {
  printf 'Missing Phase-4 manifest: %s\n' "${DATA_DIR}/manifest.json" >&2
  exit 1
}

mkdir -p "${RESULTS_DIR}" "$(dirname "${LOG_PATH}")"
printf '%s\n' \
  "Phase-4 held-out Direct / Static-RAG / Search-RL benchmark" \
  "  selected mode: ${MODE}" \
  "  eval data / manifest: ${DATA_DIR}/eval.parquet / ${DATA_DIR}/manifest.json" \
  "  base model: ${BASE_MODEL}" \
  "  Search-RL checkpoint: ${SEARCH_MODEL}" \
  "  GPU count / CUDA_VISIBLE_DEVICES: 1 / ${CUDA_VISIBLE_DEVICES}" \
  "  dtype / vLLM attention backend: bfloat16 / ${VLLM_ATTENTION_BACKEND}" \
  "  vLLM GPU utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
  "  greedy seed / response tokens: ${SEED} / ${MAX_RESPONSE_LENGTH}" \
  "  retriever URL / top-k: ${RETRIEVER_URL} / ${RETRIEVER_TOPK}" \
  "  Static-RAG retrieved-context token budget: ${STATIC_CONTEXT_TOKEN_BUDGET}" \
  "  Search-RL turns / observation cap / rolling prompt cap: ${MAX_TURNS} / ${MAX_OBS_LENGTH} / ${MAX_PROMPT_LENGTH}" \
  "  results / log: ${RESULTS_DIR} / ${LOG_PATH}" \
  "  overwrite completed rows: ${OVERWRITE}" | tee -a "${LOG_PATH}"

run_mode() {
  local benchmark_mode="$1"
  local overwrite_args=()
  if [[ "${OVERWRITE}" == "true" ]]; then
    overwrite_args+=(--overwrite)
  fi
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m experiments.phase4_benchmark.run_benchmark \
    --mode "${benchmark_mode}" \
    --eval-data "${DATA_DIR}/eval.parquet" \
    --eval-manifest "${DATA_DIR}/manifest.json" \
    --output-dir "${RESULTS_DIR}" \
    --base-model "${BASE_MODEL}" \
    --search-model "${SEARCH_MODEL}" \
    --retriever-url "${RETRIEVER_URL}" \
    --retriever-topk "${RETRIEVER_TOPK}" \
    --static-context-token-budget "${STATIC_CONTEXT_TOKEN_BUDGET}" \
    --max-start-length "${MAX_START_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --max-obs-length "${MAX_OBS_LENGTH}" \
    --max-turns "${MAX_TURNS}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}" \
    "${overwrite_args[@]}" 2>&1 | tee -a "${LOG_PATH}"
}

summarize() {
  "${PYTHON_BIN}" -m experiments.phase4_benchmark.summarize_results \
    --results-dir "${RESULTS_DIR}" \
    --eval-manifest "${DATA_DIR}/manifest.json" 2>&1 | tee -a "${LOG_PATH}"
}

cd "${ROOT_DIR}"
if [[ "${MODE}" == "all" ]]; then
  # Process boundaries provide reliable vLLM/CUDA teardown between checkpoints.
  run_mode direct
  run_mode static_rag
  run_mode search_rl
  summarize
elif [[ "${MODE}" == "summarize" ]]; then
  summarize
else
  run_mode "${MODE}"
fi
