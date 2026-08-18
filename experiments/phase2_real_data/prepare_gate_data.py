#!/usr/bin/env python3
"""Build a deterministic, balanced real-data subset for the Phase-2 gate."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"prompt", "data_source", "ability", "reward_model", "extra_info"}
SUPPORTED_SOURCES = ("nq", "hotpotqa")
LEAKAGE_MARKERS = (
    "Accepted answer evidence",
    "PHASE-1 FIXTURE EVIDENCE",
    "This state machine applies only to this deliberately answer-leaky Phase-1 fixture",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_schema(frame, source_path):
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")


def _canonical_source(value):
    source = str(value).strip().lower()
    return source if source in SUPPORTED_SOURCES else None


def select_balanced_rows(frame, size, seed, offset=0):
    """Select NQ/HotpotQA positions deterministically and interleave sources."""
    if size <= 0:
        raise ValueError("sample size must be greater than zero")

    positions = {source: [] for source in SUPPORTED_SOURCES}
    for position, value in enumerate(frame["data_source"].tolist()):
        source = _canonical_source(value)
        if source is not None:
            positions[source].append(position)

    eligible_count = sum(len(items) for items in positions.values())
    if eligible_count < size:
        raise ValueError(
            f"requested {size} rows but only {eligible_count} NQ/HotpotQA rows are available"
        )

    rng = random.Random(seed)
    for items in positions.values():
        rng.shuffle(items)

    targets = {"nq": (size + 1) // 2, "hotpotqa": size // 2}
    selected = {
        source: positions[source][:min(targets[source], len(positions[source]))]
        for source in SUPPORTED_SOURCES
    }
    selected_sets = {source: set(items) for source, items in selected.items()}
    remaining = [
        position
        for source in SUPPORTED_SOURCES
        for position in positions[source]
        if position not in selected_sets[source]
    ]
    rng.shuffle(remaining)

    missing_count = size - sum(len(items) for items in selected.values())
    for position in remaining[:missing_count]:
        source = _canonical_source(frame.iloc[position]["data_source"])
        selected[source].append(position)

    ordered_positions = []
    source_lists = {source: list(items) for source, items in selected.items()}
    while any(source_lists.values()):
        for source in SUPPORTED_SOURCES:
            if source_lists[source]:
                ordered_positions.append(source_lists[source].pop(0))

    if offset < 0 or offset >= size:
        raise ValueError(f"offset must be between 0 and {size - 1}")
    ordered_positions = ordered_positions[offset:] + ordered_positions[:offset]
    return frame.iloc[ordered_positions].copy(), ordered_positions


def audit_fixture_leakage(frame, split_name):
    matches = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        serialized = json.dumps(_json_safe(row.to_dict()), ensure_ascii=False, sort_keys=True)
        for marker in LEAKAGE_MARKERS:
            if marker in serialized:
                matches.append({"split": split_name, "row": row_number, "marker": marker})
    if matches:
        raise ValueError(f"Phase-1 fixture leakage markers found: {matches}")
    return {"passed": True, "rows_scanned": len(frame), "rejected_markers": list(LEAKAGE_MARKERS)}


def _selection_manifest(frame, positions, source_frame):
    selected = []
    for output_position, source_position in enumerate(positions):
        row = source_frame.iloc[source_position]
        selected.append({
            "output_position": output_position,
            "source_position": source_position,
            "source_dataframe_index": _json_safe(source_frame.index[source_position]),
            "data_source": _canonical_source(row["data_source"]),
            "extra_info": _json_safe(row["extra_info"]),
        })
    return selected


def prepare_gate_data(source_train, source_test, output_dir, train_size=64, test_size=16,
                      seed=42, train_offset=0):
    source_train = Path(source_train).expanduser().resolve()
    source_test = Path(source_test).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    for path in (source_train, source_test):
        if not path.is_file():
            raise FileNotFoundError(f"source parquet does not exist: {path}")

    train_source = pd.read_parquet(source_train)
    test_source = pd.read_parquet(source_test)
    _validate_schema(train_source, source_train)
    _validate_schema(test_source, source_test)

    train_rows, train_positions = select_balanced_rows(
        train_source, train_size, seed, offset=train_offset
    )
    test_rows, test_positions = select_balanced_rows(test_source, test_size, seed + 1)

    leakage_audit = {
        "train": audit_fixture_leakage(train_rows, "train"),
        "test": audit_fixture_leakage(test_rows, "test"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "train.parquet"
    test_output = output_dir / "test.parquet"
    manifest_output = output_dir / "manifest.json"
    train_rows.to_parquet(train_output, index=False)
    test_rows.to_parquet(test_output, index=False)

    manifest = {
        "seed": seed,
        "train_offset": train_offset,
        "source_paths": {"train": str(source_train), "test": str(source_test)},
        "source_sha256": {
            "train": sha256_file(source_train),
            "test": sha256_file(source_test),
        },
        "selected_row_counts": {"train": len(train_rows), "test": len(test_rows)},
        "data_source_counts": {
            "train": {
                source: int((train_rows["data_source"].str.lower() == source).sum())
                for source in SUPPORTED_SOURCES
            },
            "test": {
                source: int((test_rows["data_source"].str.lower() == source).sum())
                for source in SUPPORTED_SOURCES
            },
        },
        "selected_source_rows": {
            "train": _selection_manifest(train_rows, train_positions, train_source),
            "test": _selection_manifest(test_rows, test_positions, test_source),
        },
        "output_sha256": {
            "train": sha256_file(train_output),
            "test": sha256_file(test_output),
        },
        "leakage_audit": leakage_audit,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-offset",
        type=int,
        default=0,
        help="Rotate the deterministic train pool so a later prompt batch is first.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_gate_data(
        source_train=args.source_train,
        source_test=args.source_test,
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        train_offset=args.train_offset,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
