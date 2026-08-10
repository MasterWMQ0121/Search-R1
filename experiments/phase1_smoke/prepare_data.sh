#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE1_PYTHON="${PHASE1_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
DATA_DIR="${PHASE1_DATA_DIR:-${ROOT_DIR}/data/phase1_smoke}"
SOURCE_REVISION="${PHASE1_NQ_REVISION:-bcafb8dd07d453be3cbeeeb3f78be1841bddf92c}"
export HF_HOME="${PHASE1_HF_HOME:-${DATA_DIR}/hf-cache}"

"${PHASE1_PYTHON}" "${ROOT_DIR}/scripts/data_process/phase1_smoke_nq.py" \
  --output-dir "${DATA_DIR}" \
  --revision "${SOURCE_REVISION}" \
  --seed 42 \
  --train-size 128 \
  --val-size 64
