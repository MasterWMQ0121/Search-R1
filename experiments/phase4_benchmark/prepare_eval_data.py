#!/usr/bin/env python3
"""Prepare a deterministic held-out NQ/HotpotQA benchmark subset."""

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_SOURCES = ("nq", "hotpotqa")
REQUIRED_COLUMNS = {"prompt", "data_source", "ability", "reward_model", "extra_info"}
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
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _expected_manifest_value(manifest, keys, label):
    value = manifest
    try:
        for key in keys:
            value = value[key]
    except (KeyError, TypeError) as error:
        dotted_key = ".".join(keys)
        raise ValueError(f"{label} is missing {dotted_key}") from error
    return value


def _verify_sha256(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _validate_schema(frame, label):
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _canonical_source(value, label):
    source = str(value).strip().lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"{label} has unsupported data_source: {value!r}")
    return source


def _identity_from_row(row, label):
    source = _canonical_source(row["data_source"], label)
    extra_info = row["extra_info"]
    if not isinstance(extra_info, Mapping) or "index" not in extra_info:
        raise ValueError(f"{label} is missing extra_info.index")
    index = extra_info["index"]
    if isinstance(index, np.generic):
        index = index.item()
    if index is None or isinstance(index, bool):
        raise ValueError(f"{label} has invalid extra_info.index: {index!r}")
    if isinstance(index, float):
        if not math.isfinite(index) or not index.is_integer():
            raise ValueError(f"{label} has invalid extra_info.index: {index!r}")
        index = int(index)
    index_text = str(index).strip()
    if not index_text:
        raise ValueError(f"{label} has an empty extra_info.index")
    return source, index, f"{source}:{index_text}"


def _collect_identities(frame, label):
    _validate_schema(frame, label)
    identities = []
    seen = {}
    for position in range(len(frame)):
        row = frame.iloc[position]
        source, index, uid = _identity_from_row(row, f"{label} row {position}")
        if uid in seen:
            raise ValueError(
                f"{label} has duplicate data_source:index identity {uid!r} "
                f"at rows {seen[uid]} and {position}"
            )
        seen[uid] = position
        identities.append({"source": source, "index": index, "uid": uid})
    return identities


def _validate_selected_count(frame, manifest, split, label):
    expected = _expected_manifest_value(manifest, ["selected_row_counts", split], label)
    if int(expected) != len(frame):
        raise ValueError(
            f"{label} selected row count mismatch for {split}: "
            f"manifest has {expected}, parquet has {len(frame)}"
        )


def _manifest_test_positions(manifest, source_test, label):
    entries = _expected_manifest_value(manifest, ["selected_source_rows", "test"], label)
    if not isinstance(entries, list):
        raise ValueError(f"{label} selected_source_rows.test must be a list")

    positions = set()
    for entry_number, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or "source_position" not in entry:
            raise ValueError(f"{label} test selection {entry_number} is missing source_position")
        position = entry["source_position"]
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise ValueError(f"{label} test selection {entry_number} has an invalid source_position")
        position = int(position)
        if position < 0 or position >= len(source_test):
            raise ValueError(
                f"{label} test selection {entry_number} source_position {position} is out of range"
            )
        if position in positions:
            raise ValueError(f"{label} has duplicate test source_position {position}")

        source_row = source_test.iloc[position]
        source, _, uid = _identity_from_row(
            source_row, f"{label} source test row {position}"
        )
        manifest_source = _canonical_source(
            entry.get("data_source"), f"{label} test selection {entry_number}"
        )
        manifest_extra = entry.get("extra_info")
        if not isinstance(manifest_extra, Mapping) or "index" not in manifest_extra:
            raise ValueError(f"{label} test selection {entry_number} is missing extra_info.index")
        _, _, manifest_uid = _identity_from_row(
            {"data_source": manifest_source, "extra_info": manifest_extra},
            f"{label} test selection {entry_number}",
        )
        if source != manifest_source or uid != manifest_uid:
            raise ValueError(
                f"{label} test selection {entry_number} does not match source row {position}"
            )
        positions.add(position)
    return positions


def _audit_fixture_leakage(frame):
    matches = []
    for row_number in range(len(frame)):
        serialized = json.dumps(
            _json_safe(frame.iloc[row_number].to_dict()), ensure_ascii=False, sort_keys=True
        )
        for marker in LEAKAGE_MARKERS:
            if marker in serialized:
                matches.append({"row": row_number, "marker": marker})
    if matches:
        raise ValueError(f"Phase-1 fixture leakage markers found: {matches}")
    return {
        "passed": True,
        "rows_scanned": len(frame),
        "rejected_markers": list(LEAKAGE_MARKERS),
    }


def prepare_eval_data(
    source_test,
    phase2_train,
    phase2_manifest,
    phase3_train,
    phase3_test,
    phase3_manifest,
    output_dir,
    eval_size=64,
    seed=42,
):
    """Write a balanced held-out subset after manifest- and UID-based exclusion."""
    if eval_size <= 0 or eval_size % len(SUPPORTED_SOURCES) != 0:
        raise ValueError("eval_size must be a positive multiple of two")

    paths = {
        "source_test": Path(source_test).expanduser().resolve(),
        "phase2_train": Path(phase2_train).expanduser().resolve(),
        "phase2_manifest": Path(phase2_manifest).expanduser().resolve(),
        "phase3_train": Path(phase3_train).expanduser().resolve(),
        "phase3_test": Path(phase3_test).expanduser().resolve(),
        "phase3_manifest": Path(phase3_manifest).expanduser().resolve(),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    phase2_metadata = _read_json(paths["phase2_manifest"], "Phase-2 manifest")
    phase3_metadata = _read_json(paths["phase3_manifest"], "Phase-3 manifest")

    source_test_sha = sha256_file(paths["source_test"])
    for label, metadata in (
        ("Phase-2 manifest", phase2_metadata),
        ("Phase-3 manifest", phase3_metadata),
    ):
        expected = _expected_manifest_value(metadata, ["source_sha256", "test"], label)
        if source_test_sha != expected:
            raise ValueError(
                f"source test SHA256 mismatch with {label}: expected {expected}, "
                f"got {source_test_sha}"
            )

    phase2_train_sha = _verify_sha256(
        paths["phase2_train"],
        _expected_manifest_value(
            phase2_metadata, ["output_sha256", "train"], "Phase-2 manifest"
        ),
        "Phase-2 train parquet",
    )
    phase3_train_sha = _verify_sha256(
        paths["phase3_train"],
        _expected_manifest_value(
            phase3_metadata, ["output_sha256", "train"], "Phase-3 manifest"
        ),
        "Phase-3 train parquet",
    )
    phase3_test_sha = _verify_sha256(
        paths["phase3_test"],
        _expected_manifest_value(
            phase3_metadata, ["output_sha256", "test"], "Phase-3 manifest"
        ),
        "Phase-3 test parquet",
    )

    source_frame = pd.read_parquet(paths["source_test"])
    phase2_train_frame = pd.read_parquet(paths["phase2_train"])
    phase3_train_frame = pd.read_parquet(paths["phase3_train"])
    phase3_test_frame = pd.read_parquet(paths["phase3_test"])
    _validate_schema(source_frame, "source test parquet")
    _validate_selected_count(
        phase2_train_frame, phase2_metadata, "train", "Phase-2 manifest"
    )
    _validate_selected_count(
        phase3_train_frame, phase3_metadata, "train", "Phase-3 manifest"
    )
    _validate_selected_count(
        phase3_test_frame, phase3_metadata, "test", "Phase-3 manifest"
    )

    source_identities = _collect_identities(source_frame, "source test parquet")
    phase2_identities = _collect_identities(phase2_train_frame, "Phase-2 train parquet")
    phase3_train_identities = _collect_identities(
        phase3_train_frame, "Phase-3 train parquet"
    )
    phase3_test_identities = _collect_identities(phase3_test_frame, "Phase-3 test parquet")

    excluded_uid_sources = {}
    for label, identities in (
        ("phase2_train", phase2_identities),
        ("phase3_train", phase3_train_identities),
        ("phase3_test", phase3_test_identities),
    ):
        for identity in identities:
            excluded_uid_sources.setdefault(identity["uid"], []).append(label)
    excluded_uids = set(excluded_uid_sources)

    phase2_test_positions = _manifest_test_positions(
        phase2_metadata, source_frame, "Phase-2 manifest"
    )
    phase3_test_positions = _manifest_test_positions(
        phase3_metadata, source_frame, "Phase-3 manifest"
    )
    excluded_test_positions = phase2_test_positions | phase3_test_positions

    candidates = {source: [] for source in SUPPORTED_SOURCES}
    for position, identity in enumerate(source_identities):
        if position in excluded_test_positions or identity["uid"] in excluded_uids:
            continue
        candidates[identity["source"]].append(position)

    per_source = eval_size // len(SUPPORTED_SOURCES)
    for source in SUPPORTED_SOURCES:
        if len(candidates[source]) < per_source:
            raise ValueError(
                f"insufficient held-out {source} rows after exclusions: "
                f"need {per_source}, found {len(candidates[source])}"
            )

    rng = random.Random(seed)
    for source in SUPPORTED_SOURCES:
        rng.shuffle(candidates[source])

    selected_by_source = {
        source: candidates[source][:per_source] for source in SUPPORTED_SOURCES
    }
    selected_positions = []
    for index in range(per_source):
        for source in SUPPORTED_SOURCES:
            selected_positions.append(selected_by_source[source][index])

    selected_frame = source_frame.iloc[selected_positions].copy()
    selected_records = []
    selected_uids = set()
    for output_position, source_position in enumerate(selected_positions):
        identity = source_identities[source_position]
        uid = identity["uid"]
        if uid in selected_uids:
            raise ValueError(f"selected rows contain duplicate data_source:index identity {uid!r}")
        selected_uids.add(uid)
        row = source_frame.iloc[source_position]
        selected_records.append({
            "output_position": output_position,
            "source_position": source_position,
            "physical_identity": {
                "source_sha256": source_test_sha,
                "split": "test",
                "source_position": source_position,
            },
            "data_source": identity["source"],
            "index": _json_safe(identity["index"]),
            "uid": uid,
            "extra_info": _json_safe(row["extra_info"]),
        })

    uid_overlap = sorted(selected_uids.intersection(excluded_uids))
    position_overlap = sorted(set(selected_positions).intersection(excluded_test_positions))
    if uid_overlap or position_overlap:
        raise ValueError(
            f"held-out overlap audit failed: uid_overlap={uid_overlap}, "
            f"source_position_overlap={position_overlap}"
        )
    leakage_audit = _audit_fixture_leakage(selected_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output = output_dir / "eval.parquet"
    manifest_output = output_dir / "manifest.json"
    selected_frame.to_parquet(eval_output, index=False)

    manifest = {
        "seed": seed,
        "eval_size": eval_size,
        "per_source_size": per_source,
        "source_test": {
            "path": str(paths["source_test"]),
            "sha256": source_test_sha,
            "split": "test",
        },
        "reference_artifacts": {
            "phase2_train": {
                "path": str(paths["phase2_train"]),
                "sha256": phase2_train_sha,
                "manifest_path": str(paths["phase2_manifest"]),
                "manifest_sha256": sha256_file(paths["phase2_manifest"]),
                "uid_count": len(phase2_identities),
            },
            "phase3_train": {
                "path": str(paths["phase3_train"]),
                "sha256": phase3_train_sha,
                "uid_count": len(phase3_train_identities),
            },
            "phase3_test": {
                "path": str(paths["phase3_test"]),
                "sha256": phase3_test_sha,
                "manifest_path": str(paths["phase3_manifest"]),
                "manifest_sha256": sha256_file(paths["phase3_manifest"]),
                "uid_count": len(phase3_test_identities),
            },
        },
        "excluded": {
            "unique_uid_count": len(excluded_uids),
            "uid_sources": {
                uid: sorted(labels) for uid, labels in sorted(excluded_uid_sources.items())
            },
            "phase2_test_source_positions": sorted(phase2_test_positions),
            "phase3_test_source_positions": sorted(phase3_test_positions),
        },
        "selected_row_count": len(selected_frame),
        "data_source_counts": {
            source: sum(record["data_source"] == source for record in selected_records)
            for source in SUPPORTED_SOURCES
        },
        "selected_source_rows": selected_records,
        "non_overlap_audit": {
            "passed": True,
            "uid_rule": "lowercase(data_source) + ':' + str(extra_info.index)",
            "uid_reference_sets": ["phase2_train", "phase3_train", "phase3_test"],
            "physical_identity_rule": "source_sha256 + split + source_position",
            "uid_overlap": uid_overlap,
            "source_position_overlap": position_overlap,
        },
        "leakage_audit": leakage_audit,
        "output": {
            "path": str(eval_output),
            "sha256": sha256_file(eval_output),
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--phase2-train", required=True)
    parser.add_argument("--phase2-manifest", required=True)
    parser.add_argument("--phase3-train", required=True)
    parser.add_argument("--phase3-test", required=True)
    parser.add_argument("--phase3-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_eval_data(
        source_test=args.source_test,
        phase2_train=args.phase2_train,
        phase2_manifest=args.phase2_manifest,
        phase3_train=args.phase3_train,
        phase3_test=args.phase3_test,
        phase3_manifest=args.phase3_manifest,
        output_dir=args.output_dir,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
