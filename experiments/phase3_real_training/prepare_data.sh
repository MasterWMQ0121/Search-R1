#!/usr/bin/env bash
# Prepare the deterministic Phase-3 subset by reusing the audited Phase-2 selector.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PHASE3_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"
SOURCE_DATA_DIR="${PHASE3_SOURCE_DATA_DIR:-/workspace/searchr1-assets/datasets/nq_hotpotqa_train}"
OUTPUT_DIR="${PHASE3_DATA_DIR:-/workspace/searchr1-assets/datasets/phase3_real_training}"

TRAIN_SIZE="128"
VAL_SIZE="32"
SEED="42"
TRAIN_OFFSET="4"

[[ -x "${PYTHON_BIN}" ]] || { printf 'Missing executable Phase-3 Python: %s\n' "${PYTHON_BIN}" >&2; exit 1; }
[[ -s "${SOURCE_DATA_DIR}/train.parquet" ]] || { printf 'Missing source train parquet: %s\n' "${SOURCE_DATA_DIR}/train.parquet" >&2; exit 1; }
[[ -s "${SOURCE_DATA_DIR}/test.parquet" ]] || { printf 'Missing source validation parquet: %s\n' "${SOURCE_DATA_DIR}/test.parquet" >&2; exit 1; }

printf '%s\n' \
  "Preparing Phase-3 small real-data subset" \
  "  source: ${SOURCE_DATA_DIR}" \
  "  output: ${OUTPUT_DIR}" \
  "  train / validation rows: ${TRAIN_SIZE} / ${VAL_SIZE}" \
  "  seed / train offset: ${SEED} / ${TRAIN_OFFSET}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/experiments/phase2_real_data/prepare_gate_data.py" \
  --source-train "${SOURCE_DATA_DIR}/train.parquet" \
  --source-test "${SOURCE_DATA_DIR}/test.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --test-size "${VAL_SIZE}" \
  --seed "${SEED}" \
  --train-offset "${TRAIN_OFFSET}"
