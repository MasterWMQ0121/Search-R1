#!/usr/bin/env python3
"""Prepare deterministic non-held-out queries for Phase-6 ANN calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import pandas as pd

from experiments.phase4_benchmark.prepare_eval_data import sha256_file


SUPPORTED_SOURCES = ("nq", "hotpotqa")
REQUIRED_COLUMNS = {"prompt", "data_source", "extra_info"}
QUERY_SCHEMA_VERSION = 1


def _json_write_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonl_write_atomic(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_source(value, label: str) -> Optional[str]:
    source = str(value).strip().lower()
    if source in SUPPORTED_SOURCES:
        return source
    return None


def _index_text(value, label: str) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} has an invalid extra_info.index: {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{label} has an invalid extra_info.index: {value!r}")
        value = int(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} has an empty extra_info.index")
    return text


def _mapping(value, label: str) -> Mapping:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{label} must be a mapping")


def _sequence(value, label: str) -> list:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise ValueError(f"{label} must be a sequence")


def _question_from_row(row, label: str) -> str:
    messages = _sequence(row["prompt"], f"{label} prompt")
    if not messages:
        raise ValueError(f"{label} has an empty prompt")
    message = _mapping(messages[-1], f"{label} final prompt message")
    content = str(message.get("content", "")).strip()
    if not content:
        raise ValueError(f"{label} has an empty question prompt")
    marker = "Question:"
    question = content.split(marker, 1)[1].strip() if marker in content else content
    if not question:
        raise ValueError(f"{label} has an empty question")
    return question


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def load_phase4_heldout_uids(eval_path, manifest_path):
    """Read only the audited Phase-4 identities; never inspect answer targets."""

    eval_path = Path(eval_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not eval_path.is_file():
        raise FileNotFoundError(f"Phase-4 eval parquet does not exist: {eval_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase-4 eval manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase-4 eval manifest must contain a JSON object")
    expected_eval_sha = manifest.get("output", {}).get("sha256")
    actual_eval_sha = sha256_file(eval_path)
    if expected_eval_sha != actual_eval_sha:
        raise ValueError(
            "Phase-4 eval parquet SHA256 mismatch: "
            f"expected {expected_eval_sha!r}, got {actual_eval_sha!r}"
        )
    entries = manifest.get("selected_source_rows")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Phase-4 manifest has no selected_source_rows")
    uids = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Phase-4 manifest row {position} is not a mapping")
        uid = entry.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"Phase-4 manifest row {position} has no UID")
        uids.append(uid)
    if len(uids) != len(set(uids)):
        raise ValueError("Phase-4 manifest contains duplicate held-out UIDs")
    if manifest.get("selected_row_count") != len(uids):
        raise ValueError("Phase-4 manifest selected-row count is inconsistent")
    if not manifest.get("non_overlap_audit", {}).get("passed"):
        raise ValueError("Phase-4 held-out non-overlap audit did not pass")
    return set(uids), {
        "eval_path": str(eval_path),
        "eval_sha256": actual_eval_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "uid_count": len(uids),
    }


def select_calibration_queries(frame, sample_size=128, seed=42):
    """Select balanced, unique Phase-3 questions without using answer labels."""

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"source parquet is missing required columns: {sorted(missing)}")

    pools = {source: [] for source in SUPPORTED_SOURCES}
    ignored_source_counts = {}
    seen_uids = {}
    seen_questions = {}
    duplicate_questions = []
    for source_position in range(len(frame)):
        row = frame.iloc[source_position]
        source = _canonical_source(row["data_source"], f"source row {source_position}")
        if source is None:
            ignored = str(row["data_source"]).strip().lower()
            ignored_source_counts[ignored] = ignored_source_counts.get(ignored, 0) + 1
            continue
        label = f"source row {source_position}"
        extra_info = _mapping(row["extra_info"], f"{label} extra_info")
        uid = f"{source}:{_index_text(extra_info.get('index'), label)}"
        if uid in seen_uids:
            raise ValueError(
                f"source parquet has duplicate UID {uid!r} at rows "
                f"{seen_uids[uid]} and {source_position}"
            )
        seen_uids[uid] = source_position
        question = _question_from_row(row, label)
        question_key = question.strip()
        if question_key in seen_questions:
            duplicate_questions.append({
                "question_sha256": _question_sha256(question_key),
                "kept_source_position": seen_questions[question_key],
                "dropped_source_position": source_position,
            })
            continue
        seen_questions[question_key] = source_position
        pools[source].append({
            "schema_version": QUERY_SCHEMA_VERSION,
            "uid": uid,
            "data_source": source,
            "question": question_key,
            "question_sha256": _question_sha256(question_key),
            "source_position": source_position,
        })

    if sum(len(pool) for pool in pools.values()) < sample_size:
        raise ValueError(
            f"requested {sample_size} unique questions but only "
            f"{sum(len(pool) for pool in pools.values())} are available"
        )
    rng = random.Random(seed)
    for source in SUPPORTED_SOURCES:
        rng.shuffle(pools[source])

    targets = {
        "nq": (sample_size + 1) // 2,
        "hotpotqa": sample_size // 2,
    }
    selected = {
        source: pools[source][: min(targets[source], len(pools[source]))]
        for source in SUPPORTED_SOURCES
    }
    selected_uids = {
        record["uid"] for records in selected.values() for record in records
    }
    remaining = [
        record
        for source in SUPPORTED_SOURCES
        for record in pools[source]
        if record["uid"] not in selected_uids
    ]
    rng.shuffle(remaining)
    missing_count = sample_size - sum(len(records) for records in selected.values())
    for record in remaining[:missing_count]:
        selected[record["data_source"]].append(record)

    ordered = []
    by_source = {source: list(selected[source]) for source in SUPPORTED_SOURCES}
    while any(by_source.values()):
        for source in SUPPORTED_SOURCES:
            if by_source[source]:
                ordered.append(by_source[source].pop(0))
    if len(ordered) != sample_size:
        raise RuntimeError("calibration selection did not produce the requested row count")
    if len({record["question"] for record in ordered}) != sample_size:
        raise RuntimeError("calibration selection contains duplicate questions")
    return ordered, {
        "eligible_unique_question_counts": {
            source: len(pools[source]) for source in SUPPORTED_SOURCES
        },
        "ignored_source_counts": dict(sorted(ignored_source_counts.items())),
        "duplicate_questions_dropped": duplicate_questions,
    }


def prepare_calibration_queries(
    source_train,
    heldout_eval,
    heldout_manifest,
    output_dir,
    sample_size=128,
    seed=42,
):
    source_train = Path(source_train).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_train.is_file():
        raise FileNotFoundError(f"Phase-3 train parquet does not exist: {source_train}")

    heldout_uids, heldout_binding = load_phase4_heldout_uids(
        heldout_eval, heldout_manifest
    )
    frame = pd.read_parquet(source_train)
    selected, selection_audit = select_calibration_queries(
        frame, sample_size=sample_size, seed=seed
    )
    selected_uids = [record["uid"] for record in selected]
    overlap = sorted(set(selected_uids).intersection(heldout_uids))
    if overlap:
        raise ValueError(
            "Phase-6 calibration UIDs overlap Phase-4 held-out UIDs: "
            f"{overlap}"
        )

    queries_path = output_dir / "queries.jsonl"
    manifest_path = output_dir / "manifest.json"
    _jsonl_write_atomic(queries_path, selected)
    source_counts = {
        source: sum(record["data_source"] == source for record in selected)
        for source in SUPPORTED_SOURCES
    }
    manifest = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "seed": seed,
        "row_count": len(selected),
        "source_train": {
            "path": str(source_train),
            "sha256": sha256_file(source_train),
            "row_count": len(frame),
        },
        "phase4_heldout": heldout_binding,
        "selected_uids": selected_uids,
        "question_hashes": [record["question_sha256"] for record in selected],
        "source_counts": source_counts,
        "selection_audit": selection_audit,
        "question_extraction_rule": (
            "text after first 'Question:' marker when present; otherwise the "
            "entire stripped final user message"
        ),
        "leakage_overlap_audit": {
            "passed": True,
            "rule": "exact source-aware UID intersection",
            "overlapping_uids": overlap,
            "heldout_uid_count": len(heldout_uids),
        },
        "output": {
            "path": str(queries_path),
            "sha256": sha256_file(queries_path),
        },
        "answer_targets_included": False,
    }
    _json_write_atomic(manifest_path, manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train",
        default="/workspace/searchr1-assets/datasets/phase3_real_training/train.parquet",
    )
    parser.add_argument(
        "--heldout-eval",
        default="/workspace/searchr1-assets/datasets/phase4_benchmark/eval.parquet",
    )
    parser.add_argument(
        "--heldout-manifest",
        default="/workspace/searchr1-assets/datasets/phase4_benchmark/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/searchr1-assets/datasets/phase6_retriever_calibration",
    )
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    manifest = prepare_calibration_queries(
        source_train=args.source_train,
        heldout_eval=args.heldout_eval,
        heldout_manifest=args.heldout_manifest,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
