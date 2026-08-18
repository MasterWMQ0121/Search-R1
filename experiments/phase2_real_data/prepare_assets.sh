#!/usr/bin/env bash
# Non-downloading helpers for explicit GPU-side Phase-2 real-data asset setup.
set -euo pipefail

ASSET_ROOT="${PHASE2_REAL_ASSET_ROOT:-/workspace/searchr1-assets}"
WIKI_DIR="${ASSET_ROOT}/wiki18"
DATASET_DIR="${ASSET_ROOT}/datasets/nq_hotpotqa_train"
MODEL_DIR="${ASSET_ROOT}/models/e5-base-v2"
REQUIRED_FREE_GIB="${PHASE2_REAL_REQUIRED_FREE_GIB:-96}"
MIN_RAM_GIB="${PHASE2_REAL_MIN_RAM_GIB:-128}"
ASSET_PYTHON="${PHASE2_REAL_ASSET_PYTHON:-python3}"
INDEX_TMP="${WIKI_DIR}/e5_Flat.index.tmp"
INDEX_FINAL="${WIKI_DIR}/e5_Flat.index"
PART_A_SIZE="${WIKI_DIR}/.e5_Flat.part_aa.bytes"
PART_B_SIZE="${WIKI_DIR}/.e5_Flat.part_ab.bytes"
CORPUS_GZ="${WIKI_DIR}/wiki-18.jsonl.gz"
CORPUS_TMP="${WIKI_DIR}/wiki-18.jsonl.tmp"
CORPUS_FINAL="${WIKI_DIR}/wiki-18.jsonl"

available_disk_gib() {
  local probe="${ASSET_ROOT}"
  [[ -e "${probe}" ]] || probe="$(dirname "${ASSET_ROOT}")"
  df -Pk "${probe}" | awk 'NR == 2 {printf "%d\n", $4 / 1024 / 1024}'
}

available_ram_gib() {
  if [[ -r /proc/meminfo ]]; then
    awk '/MemAvailable/ {printf "%d\n", $2 / 1024 / 1024}' /proc/meminfo
  else
    printf '0\n'
  fi
}

file_size_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

fsync_file() {
  "${ASSET_PYTHON}" -c 'import os, sys; handle = open(sys.argv[1], "r+b"); os.fsync(handle.fileno()); handle.close()' "$1"
}

preflight() {
  local disk_gib ram_gib
  disk_gib="$(available_disk_gib)"
  ram_gib="$(available_ram_gib)"
  printf '%s\n' \
    "Asset root: ${ASSET_ROOT}" \
    "Free disk: ${disk_gib} GiB" \
    "Configured free-disk planning threshold: ${REQUIRED_FREE_GIB} GiB" \
    "Available RAM: ${ram_gib} GiB (required: ${MIN_RAM_GIB} GiB)" \
    "Peak-disk strategy: assembled index temporary + one shard at a time; then final index + compressed corpus + decompression temporary." \
    "The threshold is a conservative planning estimate, not a claimed exact artifact size."
  (( disk_gib >= REQUIRED_FREE_GIB )) || {
    printf 'Insufficient free disk; set PHASE2_REAL_REQUIRED_FREE_GIB only after reviewing actual artifact sizes.\n' >&2
    exit 1
  }
  (( ram_gib >= MIN_RAM_GIB )) || { printf 'Insufficient available RAM for the full CPU retriever.\n' >&2; exit 1; }
}

init_dirs() {
  preflight
  mkdir -p "${WIKI_DIR}" "${DATASET_DIR}" "${MODEL_DIR}"
  printf 'Created asset directories under %s\n' "${ASSET_ROOT}"
}

init_index() {
  [[ -d "${WIKI_DIR}" ]] || { printf 'Missing Wiki directory; run init first: %s\n' "${WIKI_DIR}" >&2; exit 1; }
  [[ ! -e "${INDEX_FINAL}" ]] || { printf 'Final index already exists: %s\n' "${INDEX_FINAL}" >&2; exit 1; }
  [[ ! -e "${INDEX_TMP}" ]] || { printf 'Temporary index already exists: %s\n' "${INDEX_TMP}" >&2; exit 1; }
  [[ ! -e "${PART_A_SIZE}" && ! -e "${PART_B_SIZE}" ]] || {
    printf 'Index assembly state already exists under %s\n' "${WIKI_DIR}" >&2
    exit 1
  }
  : > "${INDEX_TMP}"
  fsync_file "${INDEX_TMP}"
  printf 'Initialized empty temporary index: %s\n' "${INDEX_TMP}"
}

append_index_part() {
  local part_name="${1:-}" part_path state_path before_size part_size after_size expected_size
  case "${part_name}" in
    part_aa) state_path="${PART_A_SIZE}" ;;
    part_ab) state_path="${PART_B_SIZE}" ;;
    *) printf 'append-index-part requires part_aa or part_ab.\n' >&2; exit 2 ;;
  esac
  part_path="${WIKI_DIR}/${part_name}"
  [[ -f "${INDEX_TMP}" ]] || { printf 'Missing temporary index; run init-index first.\n' >&2; exit 1; }
  [[ -s "${part_path}" ]] || { printf 'Missing or empty index shard: %s\n' "${part_path}" >&2; exit 1; }
  [[ ! -e "${state_path}" ]] || { printf 'Shard was already appended: %s\n' "${part_name}" >&2; exit 1; }

  before_size="$(file_size_bytes "${INDEX_TMP}")"
  if [[ "${part_name}" == "part_aa" ]]; then
    [[ "${before_size}" == "0" && ! -e "${PART_B_SIZE}" ]] || {
      printf 'part_aa must be the first shard appended to an empty temporary index.\n' >&2
      exit 1
    }
  else
    [[ -s "${PART_A_SIZE}" ]] || { printf 'Append part_aa before part_ab.\n' >&2; exit 1; }
    [[ ! -e "${WIKI_DIR}/part_aa" ]] || {
      printf 'Delete part_aa after its verified append before processing part_ab.\n' >&2
      exit 1
    }
  fi

  part_size="$(file_size_bytes "${part_path}")"
  "${ASSET_PYTHON}" -c '
import os
import shutil
import sys
with open(sys.argv[1], "rb") as source, open(sys.argv[2], "ab", buffering=0) as target:
    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    os.fsync(target.fileno())
' "${part_path}" "${INDEX_TMP}"

  after_size="$(file_size_bytes "${INDEX_TMP}")"
  expected_size="$((before_size + part_size))"
  [[ "${after_size}" == "${expected_size}" ]] || {
    printf 'Index append size mismatch for %s: expected %s bytes, got %s bytes.\n' \
      "${part_name}" "${expected_size}" "${after_size}" >&2
    exit 1
  }
  printf '%s\n' "${part_size}" > "${state_path}"
  printf 'Verified append of %s (%s bytes). Delete %s before downloading the next shard.\n' \
    "${part_name}" "${part_size}" "${part_path}"
}

finalize_index() {
  local expected_size actual_size
  [[ -s "${PART_A_SIZE}" && -s "${PART_B_SIZE}" ]] || { printf 'Both shard append records are required.\n' >&2; exit 1; }
  [[ ! -e "${WIKI_DIR}/part_aa" && ! -e "${WIKI_DIR}/part_ab" ]] || {
    printf 'Delete both verified source shards before finalizing the index.\n' >&2
    exit 1
  }
  expected_size="$(( $(<"${PART_A_SIZE}") + $(<"${PART_B_SIZE}") ))"
  actual_size="$(file_size_bytes "${INDEX_TMP}")"
  [[ "${actual_size}" == "${expected_size}" && "${actual_size}" -gt 0 ]] || {
    printf 'Final temporary index size mismatch: expected %s bytes, got %s bytes.\n' \
      "${expected_size}" "${actual_size}" >&2
    exit 1
  }
  fsync_file "${INDEX_TMP}"
  mv "${INDEX_TMP}" "${INDEX_FINAL}"
  rm "${PART_A_SIZE}" "${PART_B_SIZE}"
  printf 'Finalized index atomically: %s (%s bytes)\n' "${INDEX_FINAL}" "${actual_size}"
}

prepare_corpus() {
  [[ -s "${CORPUS_GZ}" ]] || { printf 'Missing or empty compressed corpus: %s\n' "${CORPUS_GZ}" >&2; exit 1; }
  [[ ! -e "${CORPUS_FINAL}" ]] || { printf 'Final corpus already exists: %s\n' "${CORPUS_FINAL}" >&2; exit 1; }
  [[ ! -e "${CORPUS_TMP}" ]] || { printf 'Temporary corpus already exists: %s\n' "${CORPUS_TMP}" >&2; exit 1; }
  gzip -t "${CORPUS_GZ}"
  gzip -dc "${CORPUS_GZ}" > "${CORPUS_TMP}"
  [[ -s "${CORPUS_TMP}" ]] || { printf 'Decompressed corpus is empty: %s\n' "${CORPUS_TMP}" >&2; exit 1; }
  fsync_file "${CORPUS_TMP}"
  mv "${CORPUS_TMP}" "${CORPUS_FINAL}"
  printf 'Prepared corpus: %s. Compressed source retained: %s\n' "${CORPUS_FINAL}" "${CORPUS_GZ}"
}

cleanup_corpus() {
  [[ -s "${CORPUS_FINAL}" ]] || { printf 'Refusing cleanup because final corpus is missing or empty: %s\n' "${CORPUS_FINAL}" >&2; exit 1; }
  [[ -s "${CORPUS_GZ}" ]] || { printf 'Compressed corpus is already absent: %s\n' "${CORPUS_GZ}" >&2; exit 1; }
  gzip -t "${CORPUS_GZ}"
  rm "${CORPUS_GZ}"
  printf 'Explicitly removed compressed corpus: %s\n' "${CORPUS_GZ}"
}

verify_assets() {
  local path
  for path in \
    "${INDEX_FINAL}" \
    "${CORPUS_FINAL}" \
    "${DATASET_DIR}/train.parquet" \
    "${DATASET_DIR}/test.parquet"; do
    [[ -s "${path}" ]] || { printf 'Missing or empty required file: %s\n' "${path}" >&2; exit 1; }
  done
  [[ -s "${MODEL_DIR}/config.json" ]] || { printf 'Missing or empty E5 model config: %s\n' "${MODEL_DIR}/config.json" >&2; exit 1; }
  printf 'Required real-data assets are present and non-empty.\n'
}

show_instructions() {
  printf '%s\n' \
    "This helper never downloads assets." \
    "Run preflight and init before downloads." \
    "For the index: init-index, download part_aa, append-index-part part_aa, delete part_aa, then repeat for part_ab and finalize-index." \
    "For the corpus: download the gzip, run prepare-corpus, and retain it unless cleanup-corpus is explicitly requested." \
    "See experiments/phase2_real_data/README.md for exact commands."
}

case "${1:-instructions}" in
  preflight) preflight ;;
  init) init_dirs ;;
  init-index) init_index ;;
  append-index-part) append_index_part "${2:-}" ;;
  finalize-index) finalize_index ;;
  prepare-corpus) prepare_corpus ;;
  cleanup-corpus) cleanup_corpus ;;
  verify) verify_assets ;;
  instructions) show_instructions ;;
  *)
    printf 'Usage: %s {preflight|init|init-index|append-index-part part_aa|append-index-part part_ab|finalize-index|prepare-corpus|cleanup-corpus|verify|instructions}\n' "$0" >&2
    exit 2
    ;;
esac
