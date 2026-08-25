import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = ROOT / "experiments" / "phase4_benchmark" / "prepare_eval_data.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load_module(PREPARE_PATH, "phase4_prepare_eval_data")


def _row(data_source, index, split="test", marker=None):
    question = marker or f"Held-out {data_source} question {index}?"
    return {
        "prompt": [{"role": "user", "content": question}],
        "data_source": data_source,
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [f"answer-{index}"]}},
        "extra_info": {"split": split, "index": index},
        "preserved_column": f"preserve-{split}-{data_source}-{index}",
    }


def _normalized(value):
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _uid(row):
    return f"{str(row['data_source']).strip().lower()}:{row['extra_info']['index']}"


def _selection_entry(source_frame, source_position, output_position):
    row = source_frame.iloc[source_position]
    return {
        "output_position": output_position,
        "source_position": source_position,
        "source_dataframe_index": source_position,
        "data_source": str(row["data_source"]).lower(),
        "extra_info": _normalized(row["extra_info"]),
    }


def _write_fixture(
    tmp_path,
    source_count=48,
    duplicate_source_uid=False,
    missing_source_index=False,
    leakage_marker=None,
    mixed_source=False,
):
    if mixed_source:
        source_rows = []
        for index in range(source_count):
            source_rows.extend([
                _row("nq", index),
                _row("triviaqa", index),
                _row("hotpotqa", index),
                _row("popqa", index),
            ])
        duplicate_position = 4
        phase2_test_positions = [8, 10]
        phase3_test_positions = [0, 4, 2, 6]
    else:
        source_rows = [_row("nq", index) for index in range(source_count)]
        source_rows += [_row("hotpotqa", index) for index in range(source_count)]
        duplicate_position = 1
        phase2_test_positions = [2, source_count + 2]
        phase3_test_positions = [0, 1, source_count, source_count + 1]
    if duplicate_source_uid:
        source_rows[duplicate_position] = _row("nq", 0)
    if missing_source_index:
        source_rows[duplicate_position]["extra_info"]["index"] = None
    if leakage_marker is not None:
        for row in source_rows:
            row["prompt"][0]["content"] = leakage_marker
    source_frame = pd.DataFrame(source_rows)
    source_test = tmp_path / "source-test.parquet"
    source_frame.to_parquet(source_test, index=False)

    # These train identities intentionally exclude source-test UIDs 3 and 4.
    phase2_train_frame = pd.DataFrame([
        _row("nq", 4, split="train"),
        _row("hotpotqa", 103, split="train"),
    ])
    phase3_train_frame = pd.DataFrame([
        _row("nq", 3, split="train"),
        _row("hotpotqa", 103, split="train"),
    ])

    phase3_test_frame = source_frame.iloc[phase3_test_positions].copy()

    phase2_train = tmp_path / "phase2-train.parquet"
    phase3_train = tmp_path / "phase3-train.parquet"
    phase3_test = tmp_path / "phase3-test.parquet"
    phase2_train_frame.to_parquet(phase2_train, index=False)
    phase3_train_frame.to_parquet(phase3_train, index=False)
    phase3_test_frame.to_parquet(phase3_test, index=False)

    source_sha = PREPARE.sha256_file(source_test)
    phase2_metadata = {
        "source_sha256": {"test": source_sha},
        "output_sha256": {"train": PREPARE.sha256_file(phase2_train)},
        "selected_row_counts": {"train": len(phase2_train_frame)},
        "selected_source_rows": {
            "test": [
                _selection_entry(source_frame, position, output_position)
                for output_position, position in enumerate(phase2_test_positions)
            ]
        },
    }
    phase3_metadata = {
        "source_sha256": {"test": source_sha},
        "output_sha256": {
            "train": PREPARE.sha256_file(phase3_train),
            "test": PREPARE.sha256_file(phase3_test),
        },
        "selected_row_counts": {
            "train": len(phase3_train_frame),
            "test": len(phase3_test_frame),
        },
        "selected_source_rows": {
            "test": [
                _selection_entry(source_frame, position, output_position)
                for output_position, position in enumerate(phase3_test_positions)
            ]
        },
    }
    phase2_manifest = tmp_path / "phase2-manifest.json"
    phase3_manifest = tmp_path / "phase3-manifest.json"
    phase2_manifest.write_text(json.dumps(phase2_metadata), encoding="utf-8")
    phase3_manifest.write_text(json.dumps(phase3_metadata), encoding="utf-8")

    return {
        "source_frame": source_frame,
        "source_test": source_test,
        "phase2_train": phase2_train,
        "phase2_manifest": phase2_manifest,
        "phase3_train": phase3_train,
        "phase3_test": phase3_test,
        "phase3_manifest": phase3_manifest,
        "phase2_test_positions": set(phase2_test_positions),
        "phase3_test_positions": set(phase3_test_positions),
    }


def _prepare(fixture, output_dir, eval_size=8, seed=42):
    return PREPARE.prepare_eval_data(
        source_test=fixture["source_test"],
        phase2_train=fixture["phase2_train"],
        phase2_manifest=fixture["phase2_manifest"],
        phase3_train=fixture["phase3_train"],
        phase3_test=fixture["phase3_test"],
        phase3_manifest=fixture["phase3_manifest"],
        output_dir=output_dir,
        eval_size=eval_size,
        seed=seed,
    )


def test_phase4_selector_is_deterministic_balanced_non_overlapping_and_preserves_rows(tmp_path):
    fixture = _write_fixture(tmp_path)
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"

    manifest_a = _prepare(fixture, output_a)
    manifest_b = _prepare(fixture, output_b)

    assert manifest_a["seed"] == 42
    assert manifest_a["selected_row_count"] == 8
    assert manifest_a["data_source_counts"] == {"nq": 4, "hotpotqa": 4}
    assert manifest_a["output"]["sha256"] == manifest_b["output"]["sha256"]
    assert (output_a / "eval.parquet").read_bytes() == (output_b / "eval.parquet").read_bytes()

    records = manifest_a["selected_source_rows"]
    assert [record["data_source"] for record in records] == ["nq", "hotpotqa"] * 4
    selected_positions = {record["source_position"] for record in records}
    excluded_positions = fixture["phase2_test_positions"] | fixture["phase3_test_positions"]
    assert selected_positions.isdisjoint(excluded_positions)

    phase2_uids = {_uid(row) for row in pd.read_parquet(fixture["phase2_train"]).to_dict("records")}
    phase3_uids = {
        _uid(row)
        for path in (fixture["phase3_train"], fixture["phase3_test"])
        for row in pd.read_parquet(path).to_dict("records")
    }
    selected_uids = {record["uid"] for record in records}
    assert len(selected_uids) == 8
    assert selected_uids.isdisjoint(phase2_uids | phase3_uids)
    assert "nq:3" not in selected_uids
    assert "nq:4" not in selected_uids
    assert manifest_a["non_overlap_audit"]["passed"] is True
    assert manifest_a["non_overlap_audit"]["uid_overlap"] == []
    assert manifest_a["non_overlap_audit"]["source_position_overlap"] == []
    assert manifest_a["excluded"]["uid_sources"]["hotpotqa:103"] == [
        "phase2_train",
        "phase3_train",
    ]

    output_frame = pd.read_parquet(output_a / "eval.parquet")
    assert list(output_frame.columns) == list(fixture["source_frame"].columns)
    for output_position, record in enumerate(records):
        expected = fixture["source_frame"].iloc[record["source_position"]].to_dict()
        actual = output_frame.iloc[output_position].to_dict()
        assert _normalized(actual) == _normalized(expected)
        assert record["physical_identity"] == {
            "source_sha256": PREPARE.sha256_file(fixture["source_test"]),
            "split": "test",
            "source_position": record["source_position"],
        }

    on_disk = json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest_a
    assert on_disk["output"]["sha256"] == PREPARE.sha256_file(output_a / "eval.parquet")


def test_phase4_selector_filters_mixed_source_pool_without_changing_positions(tmp_path):
    fixture = _write_fixture(tmp_path, mixed_source=True)
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"

    manifest_a = _prepare(fixture, output_a)
    manifest_b = _prepare(fixture, output_b)
    output_frame = pd.read_parquet(output_a / "eval.parquet")

    assert manifest_a["data_source_counts"] == {"nq": 4, "hotpotqa": 4}
    assert set(output_frame["data_source"]) == {"nq", "hotpotqa"}
    assert manifest_a["source_filter"] == {
        "target_sources": ["nq", "hotpotqa"],
        "ignored_source_counts": {"popqa": 48, "triviaqa": 48},
        "ignored_row_count": 96,
    }
    assert manifest_a["source_test"]["sha256"] == PREPARE.sha256_file(
        fixture["source_test"]
    )
    assert manifest_a["output"]["sha256"] == manifest_b["output"]["sha256"]
    assert (output_a / "eval.parquet").read_bytes() == (
        output_b / "eval.parquet"
    ).read_bytes()

    selected_positions = {
        record["source_position"] for record in manifest_a["selected_source_rows"]
    }
    excluded_positions = (
        fixture["phase2_test_positions"] | fixture["phase3_test_positions"]
    )
    assert selected_positions.isdisjoint(excluded_positions)
    assert all(position % 4 in (0, 2) for position in selected_positions)
    selected_uids = {record["uid"] for record in manifest_a["selected_source_rows"]}
    assert "nq:3" not in selected_uids
    assert "nq:4" not in selected_uids
    for output_position, record in enumerate(manifest_a["selected_source_rows"]):
        original = fixture["source_frame"].iloc[record["source_position"]].to_dict()
        selected = output_frame.iloc[output_position].to_dict()
        assert _normalized(selected) == _normalized(original)


def test_phase4_selector_ignores_non_target_identity_details(tmp_path):
    fixture = _write_fixture(tmp_path, mixed_source=True)
    source_frame = pd.read_parquet(fixture["source_test"])
    source_frame.at[5, "extra_info"] = source_frame.iloc[1]["extra_info"]
    source_frame.at[3, "extra_info"] = {}
    source_frame.to_parquet(fixture["source_test"], index=False)
    source_sha = PREPARE.sha256_file(fixture["source_test"])
    for manifest_name in ("phase2_manifest", "phase3_manifest"):
        metadata = json.loads(fixture[manifest_name].read_text(encoding="utf-8"))
        metadata["source_sha256"]["test"] = source_sha
        fixture[manifest_name].write_text(json.dumps(metadata), encoding="utf-8")

    manifest = _prepare(fixture, tmp_path / "output")
    assert manifest["source_filter"]["ignored_row_count"] == 96


def test_phase4_selector_verifies_full_mixed_source_sha(tmp_path):
    fixture = _write_fixture(tmp_path, mixed_source=True)
    fixture["source_test"].write_bytes(
        fixture["source_test"].read_bytes() + b"tampered"
    )

    with pytest.raises(ValueError, match="source test SHA256 mismatch"):
        _prepare(fixture, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["source_test", "phase2_train", "phase3_train", "phase3_test"])
def test_phase4_selector_rejects_hash_mismatches_before_writing(artifact, tmp_path):
    fixture = _write_fixture(tmp_path)
    path = fixture[artifact]
    path.write_bytes(path.read_bytes() + b"tampered")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _prepare(fixture, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("failure", ["duplicate", "missing"])
def test_phase4_selector_rejects_duplicate_or_missing_source_identity(failure, tmp_path):
    fixture = _write_fixture(
        tmp_path,
        duplicate_source_uid=failure == "duplicate",
        missing_source_index=failure == "missing",
    )

    expected = "duplicate data_source:index identity" if failure == "duplicate" else "invalid extra_info.index"
    with pytest.raises(ValueError, match=expected):
        _prepare(fixture, tmp_path / "output")


def test_phase4_selector_rejects_duplicate_target_uid_in_mixed_source(tmp_path):
    fixture = _write_fixture(tmp_path, mixed_source=True, duplicate_source_uid=True)
    with pytest.raises(ValueError, match="duplicate data_source:index identity"):
        _prepare(fixture, tmp_path / "output")


@pytest.mark.parametrize(
    ("artifact", "manifest_name", "split"),
    [
        ("phase2_train", "phase2_manifest", "train"),
        ("phase3_train", "phase3_manifest", "train"),
        ("phase3_test", "phase3_manifest", "test"),
    ],
)
def test_phase4_selector_keeps_prepared_artifacts_strict(
    artifact, manifest_name, split, tmp_path
):
    fixture = _write_fixture(tmp_path, mixed_source=True)
    frame = pd.read_parquet(fixture[artifact])
    frame.at[0, "data_source"] = "triviaqa"
    frame.to_parquet(fixture[artifact], index=False)
    metadata = json.loads(fixture[manifest_name].read_text(encoding="utf-8"))
    metadata["output_sha256"][split] = PREPARE.sha256_file(fixture[artifact])
    fixture[manifest_name].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported data_source"):
        _prepare(fixture, tmp_path / "output")


def test_phase4_selector_keeps_manifest_referenced_source_rows_strict(tmp_path):
    fixture = _write_fixture(tmp_path, mixed_source=True)
    metadata = json.loads(fixture["phase2_manifest"].read_text(encoding="utf-8"))
    metadata["selected_source_rows"]["test"][0] = _selection_entry(
        fixture["source_frame"], 1, 0
    )
    fixture["phase2_manifest"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported data_source"):
        _prepare(fixture, tmp_path / "output")


def test_phase4_selector_allows_reference_set_overlap_but_rejects_duplicates_within_a_set(tmp_path):
    fixture = _write_fixture(tmp_path)
    phase3_train = pd.read_parquet(fixture["phase3_train"])
    phase3_train.at[1, "extra_info"] = {"split": "train", "index": 3}
    phase3_train.at[1, "data_source"] = "nq"
    phase3_train.to_parquet(fixture["phase3_train"], index=False)
    metadata = json.loads(fixture["phase3_manifest"].read_text(encoding="utf-8"))
    metadata["output_sha256"]["train"] = PREPARE.sha256_file(fixture["phase3_train"])
    fixture["phase3_manifest"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate data_source:index identity"):
        _prepare(fixture, tmp_path / "output")


def test_phase4_selector_fails_when_either_source_cannot_fill_exact_balance(tmp_path):
    fixture = _write_fixture(tmp_path, source_count=7)
    with pytest.raises(ValueError, match="insufficient held-out nq rows"):
        _prepare(fixture, tmp_path / "output", eval_size=8)


def test_phase4_selector_rejects_phase1_fixture_leakage(tmp_path):
    fixture = _write_fixture(tmp_path, leakage_marker=PREPARE.LEAKAGE_MARKERS[0])
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="Phase-1 fixture leakage"):
        _prepare(fixture, output_dir)
    assert not output_dir.exists()


def test_phase4_cli_defaults_to_the_approved_size_and_seed(monkeypatch):
    result = subprocess.run(
        [sys.executable, str(PREPARE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--source-test" in result.stdout
    assert "--phase2-train" in result.stdout
    assert "--phase3-train" in result.stdout
    monkeypatch.setattr(sys, "argv", [
        str(PREPARE_PATH),
        "--source-test", "source.parquet",
        "--phase2-train", "phase2-train.parquet",
        "--phase2-manifest", "phase2-manifest.json",
        "--phase3-train", "phase3-train.parquet",
        "--phase3-test", "phase3-test.parquet",
        "--phase3-manifest", "phase3-manifest.json",
        "--output-dir", "output",
    ])
    args = PREPARE.parse_args()
    assert args.eval_size == 64
    assert args.seed == 42


def test_uid_index_canonicalization_matches_runtime_rules():
    source, index, uid = PREPARE._identity_from_row(
        {"data_source": "NQ", "extra_info": {"index": 1.0}}, "row"
    )
    assert (source, index, uid) == ("nq", 1, "nq:1")
    with pytest.raises(ValueError, match="invalid extra_info.index"):
        PREPARE._identity_from_row(
            {"data_source": "nq", "extra_info": {"index": 1.5}}, "row"
        )
