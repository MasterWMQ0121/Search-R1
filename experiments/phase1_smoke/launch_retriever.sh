#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE1_PYTHON="${PHASE1_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
DATA_DIR="${PHASE1_DATA_DIR:-${ROOT_DIR}/data/phase1_smoke}"
HOST="${PHASE1_RETRIEVER_HOST:-127.0.0.1}"
PORT="${PHASE1_RETRIEVER_PORT:-8000}"
export HF_HOME="${PHASE1_HF_HOME:-${DATA_DIR}/hf-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

exec "${PHASE1_PYTHON}" "${ROOT_DIR}/search_r1/search/retrieval_server.py" \
  --index_path "${DATA_DIR}/index/e5_Flat.index" \
  --corpus_path "${DATA_DIR}/corpus.jsonl" \
  --topk 2 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-small-v2 \
  --retriever_revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --device cpu \
  --host "${HOST}" \
  --port "${PORT}"
