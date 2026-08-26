import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.phase6_retriever_serving import prepare_calibration_queries as prepare


def _row(source, index, question=None):
    question = question or f"What is {source} item {index}?"
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": f"Search carefully. Question: {question}"}],
        "extra_info": {"index": index},
        "reward_model": {"ground_truth": {"target": [f"secret-{index}"]}},
    }


def _fixture(tmp_path, per_source=12, heldout_uids=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [_row("nq", index) for index in range(per_source)]
    rows += [_row("hotpotqa", index) for index in range(per_source)]
    source = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    heldout_uids = heldout_uids or ["nq:heldout", "hotpotqa:heldout"]
    heldout_eval = tmp_path / "eval.parquet"
    pd.DataFrame([{"placeholder": index} for index in range(len(heldout_uids))]).to_parquet(
        heldout_eval, index=False
    )
    heldout_manifest = tmp_path / "heldout-manifest.json"
    heldout_manifest.write_text(
        json.dumps({
            "selected_row_count": len(heldout_uids),
            "selected_source_rows": [{"uid": uid} for uid in heldout_uids],
            "non_overlap_audit": {"passed": True},
            "output": {"sha256": prepare.sha256_file(heldout_eval)},
        }),
        encoding="utf-8",
    )
    return source, heldout_eval, heldout_manifest


def _run(tmp_path, output_name="out", per_source=12, size=8, heldout_uids=None):
    source, heldout_eval, heldout_manifest = _fixture(
        tmp_path, per_source=per_source, heldout_uids=heldout_uids
    )
    manifest = prepare.prepare_calibration_queries(
        source,
        heldout_eval,
        heldout_manifest,
        tmp_path / output_name,
        sample_size=size,
        seed=42,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / output_name / "queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return manifest, records, heldout_eval


def test_calibration_is_balanced_deterministic_unique_and_answer_free(tmp_path):
    manifest_a, records_a, heldout_eval = _run(tmp_path, "out-a")
    baseline_bytes = heldout_eval.read_bytes()
    source, heldout_eval_b, heldout_manifest_b = _fixture(tmp_path / "second")
    manifest_b = prepare.prepare_calibration_queries(
        source, heldout_eval_b, heldout_manifest_b, tmp_path / "out-b", 8, 42
    )
    records_b = [
        json.loads(line)
        for line in (tmp_path / "out-b/queries.jsonl").read_text().splitlines()
    ]

    assert records_a == records_b
    assert [record["data_source"] for record in records_a] == ["nq", "hotpotqa"] * 4
    assert len({record["uid"] for record in records_a}) == 8
    assert len({record["question"] for record in records_a}) == 8
    assert manifest_a["source_counts"] == {"nq": 4, "hotpotqa": 4}
    assert manifest_a["leakage_overlap_audit"] == {
        "passed": True,
        "rule": "exact source-aware UID intersection",
        "overlapping_uids": [],
        "heldout_uid_count": 2,
    }
    assert manifest_a["answer_targets_included"] is False
    serialized = json.dumps(records_a)
    assert "secret-" not in serialized
    assert "reward_model" not in serialized
    assert "ground_truth" not in serialized
    assert heldout_eval.read_bytes() == baseline_bytes
    assert manifest_a["output"]["sha256"] == prepare.sha256_file(
        tmp_path / "out-a/queries.jsonl"
    )
    assert manifest_b["question_hashes"] == manifest_a["question_hashes"]


def test_calibration_fails_on_phase4_uid_overlap_before_writing(tmp_path):
    source, heldout_eval, heldout_manifest = _fixture(
        tmp_path, heldout_uids=["nq:0"]
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="overlap Phase-4"):
        prepare.prepare_calibration_queries(
            source, heldout_eval, heldout_manifest, output, sample_size=24, seed=42
        )
    assert not output.exists()


def test_duplicate_questions_are_dropped_and_insufficient_unique_pool_fails(tmp_path):
    frame = pd.DataFrame([
        _row("nq", 0, "same?"),
        _row("nq", 1, "same?"),
        _row("hotpotqa", 0, "different?"),
    ])
    with pytest.raises(ValueError, match="only 2 are available"):
        prepare.select_calibration_queries(frame, sample_size=3, seed=42)


def test_unbalanced_source_pool_uses_deterministic_fill(tmp_path):
    frame = pd.DataFrame(
        [_row("nq", index) for index in range(7)]
        + [_row("hotpotqa", index) for index in range(2)]
    )
    selected, _ = prepare.select_calibration_queries(frame, sample_size=6, seed=42)
    counts = {
        source: sum(record["data_source"] == source for record in selected)
        for source in prepare.SUPPORTED_SOURCES
    }
    assert counts == {"nq": 4, "hotpotqa": 2}


def test_duplicate_source_uid_is_rejected():
    frame = pd.DataFrame([_row("nq", 1), _row("NQ", 1, "another?")])
    with pytest.raises(ValueError, match="duplicate UID"):
        prepare.select_calibration_queries(frame, sample_size=1, seed=42)


def test_plain_question_prompt_is_supported_without_answer_access():
    row = _row("nq", 1)
    row["prompt"] = [{"role": "user", "content": "A plain question?"}]
    selected, _ = prepare.select_calibration_queries(
        pd.DataFrame([row]), sample_size=1, seed=42
    )
    assert selected[0]["question"] == "A plain question?"


def test_heldout_manifest_sha_and_identity_validation(tmp_path):
    _, heldout_eval, heldout_manifest = _fixture(tmp_path)
    heldout_eval.write_bytes(heldout_eval.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.load_phase4_heldout_uids(heldout_eval, heldout_manifest)


def test_default_contract_uses_phase3_and_phase4_paths():
    defaults = prepare.parse_args([])
    assert defaults.sample_size == 128
    assert defaults.seed == 42
    assert defaults.source_train.endswith("phase3_real_training/train.parquet")
    assert defaults.heldout_eval.endswith("phase4_benchmark/eval.parquet")
