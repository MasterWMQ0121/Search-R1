#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE1_PYTHON="${PHASE1_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
DATA_DIR="${PHASE1_DATA_DIR:-${ROOT_DIR}/data/phase1_smoke}"
export HF_HOME="${PHASE1_HF_HOME:-${DATA_DIR}/hf-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

"${PHASE1_PYTHON}" "${ROOT_DIR}/search_r1/search/index_builder.py" \
  --retrieval_method e5 \
  --model_path intfloat/e5-small-v2 \
  --model_revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --corpus_path "${DATA_DIR}/corpus.jsonl" \
  --save_dir "${DATA_DIR}/index" \
  --max_length 256 \
  --batch_size 32 \
  --pooling_method mean \
  --faiss_type Flat \
  --device cpu
