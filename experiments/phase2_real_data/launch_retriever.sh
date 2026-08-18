#!/usr/bin/env bash
# Launch the prebuilt Wiki-18 E5 Flat retriever entirely on CPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INDEX_PATH="${PHASE2_REAL_INDEX_PATH:-/workspace/searchr1-assets/wiki18/e5_Flat.index}"
CORPUS_PATH="${PHASE2_REAL_CORPUS_PATH:-/workspace/searchr1-assets/wiki18/wiki-18.jsonl}"
RETRIEVER_MODEL="${PHASE2_REAL_RETRIEVER_MODEL:-/workspace/searchr1-assets/models/e5-base-v2}"
RETRIEVER_HOST="${PHASE2_REAL_RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${PHASE2_REAL_RETRIEVER_PORT:-8000}"
PYTHON_BIN="${PHASE2_REAL_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
TOPK="3"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

[[ -x "${PYTHON_BIN}" ]] || {
  printf 'Missing executable CPU retriever Python: %s (set PHASE2_REAL_PYTHON to override)\n' "${PYTHON_BIN}" >&2
  exit 1
}
[[ -s "${INDEX_PATH}" ]] || { printf 'Missing or empty Wiki-18 index: %s\n' "${INDEX_PATH}" >&2; exit 1; }
[[ -s "${CORPUS_PATH}" ]] || { printf 'Missing or empty Wiki-18 corpus: %s\n' "${CORPUS_PATH}" >&2; exit 1; }
[[ -d "${RETRIEVER_MODEL}" ]] || { printf 'Missing local E5 model directory: %s\n' "${RETRIEVER_MODEL}" >&2; exit 1; }

printf '%s\n' \
  "Phase-2 real-data CPU retriever" \
  "  index: ${INDEX_PATH}" \
  "  corpus: ${CORPUS_PATH}" \
  "  retriever model: ${RETRIEVER_MODEL}" \
  "  Python: ${PYTHON_BIN}" \
  "  retriever name: e5" \
  "  device: cpu" \
  "  top-k: ${TOPK}" \
  "  listen: http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve" \
  "  CPU threads (OMP/MKL/OpenBLAS/NumExpr): ${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}/${NUMEXPR_NUM_THREADS}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" search_r1/search/retrieval_server.py \
  --index_path "${INDEX_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --topk "${TOPK}" \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --device cpu \
  --host "${RETRIEVER_HOST}" \
  --port "${RETRIEVER_PORT}"
