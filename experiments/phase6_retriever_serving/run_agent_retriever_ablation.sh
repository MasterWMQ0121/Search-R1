#!/usr/bin/env bash
# Run or summarize the isolated Phase-6 downstream Search-RL Agent comparison.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PHASE6_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
DATA_DIR="${PHASE6_EVAL_DATA_DIR:-/workspace/searchr1-assets/datasets/phase4_benchmark}"
RESULTS_DIR="${PHASE6_RESULTS_DIR:-/workspace/Search-R1/phase6_retriever_results}"
PHASE5_BASELINE="${PHASE6_PHASE5_BASELINE:-/workspace/Search-R1/phase5_observation_results/compressed_256.jsonl}"
SEARCH_MODEL="${PHASE6_SEARCH_MODEL_PATH:-/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20}"
SELECTED_CANDIDATE="${PHASE6_SELECTED_CANDIDATE:-/workspace/Search-R1/phase6_retriever_results/selected_candidate.json}"
FLAT_URL="${PHASE6_FLAT_RETRIEVER_URL:-http://127.0.0.1:8100/retrieve}"
IVFPQ_URL="${PHASE6_IVFPQ_RETRIEVER_URL:-http://127.0.0.1:8101/retrieve}"
ACTION="${1:-}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Missing executable Phase-6 Python: %s\n' "${PYTHON_BIN}" >&2
  exit 1
}
[[ -s "${DATA_DIR}/eval.parquet" ]] || { printf 'Missing Phase-4 eval parquet: %s\n' "${DATA_DIR}/eval.parquet" >&2; exit 1; }
[[ -s "${DATA_DIR}/manifest.json" ]] || { printf 'Missing Phase-4 eval manifest: %s\n' "${DATA_DIR}/manifest.json" >&2; exit 1; }
[[ -s "${PHASE5_BASELINE}" ]] || { printf 'Missing immutable Phase-5 baseline: %s\n' "${PHASE5_BASELINE}" >&2; exit 1; }

common_args=(
  --eval-data "${DATA_DIR}/eval.parquet"
  --eval-manifest "${DATA_DIR}/manifest.json"
  --output-dir "${RESULTS_DIR}"
  --phase5-baseline "${PHASE5_BASELINE}"
  --search-model "${SEARCH_MODEL}"
)

case "${ACTION}" in
  validate_baseline)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/experiments/phase6_retriever_serving/run_agent_retriever_ablation.py" \
      --condition validate_baseline "${common_args[@]}"
    ;;
  flat_exact_replay|ivfpq_selected)
    [[ -s "${SELECTED_CANDIDATE}" ]] || { printf 'Missing selected ANN candidate: %s\n' "${SELECTED_CANDIDATE}" >&2; exit 1; }
    retriever_url="${FLAT_URL}"
    [[ "${ACTION}" == "ivfpq_selected" ]] && retriever_url="${IVFPQ_URL}"
    extra_args=()
    [[ "${PHASE6_ALLOW_UNQUALIFIED_CANDIDATE:-false}" == "true" ]] && extra_args+=(--allow-unqualified-candidate)
    [[ "${PHASE6_OVERWRITE:-false}" == "true" ]] && extra_args+=(--overwrite)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/experiments/phase6_retriever_serving/run_agent_retriever_ablation.py" \
      --condition "${ACTION}" \
      "${common_args[@]}" \
      --selected-candidate "${SELECTED_CANDIDATE}" \
      --retriever-url "${retriever_url}" \
      "${extra_args[@]}"
    ;;
  summarize)
    [[ -s "${SELECTED_CANDIDATE}" ]] || { printf 'Missing selected ANN candidate: %s\n' "${SELECTED_CANDIDATE}" >&2; exit 1; }
    exec "${PYTHON_BIN}" "${ROOT_DIR}/experiments/phase6_retriever_serving/summarize_results.py" \
      "${common_args[@]}" \
      --selected-candidate "${SELECTED_CANDIDATE}"
    ;;
  *)
    printf 'Usage: %s {validate_baseline|flat_exact_replay|ivfpq_selected|summarize}\n' "$0" >&2
    exit 2
    ;;
esac
