#!/usr/bin/env python3
"""Verify deterministic Phase-1 data and index artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase1_smoke"))
    parser.add_argument("--require-index", action="store_true")
    return parser.parse_args()


def check_row(row: dict, split: str) -> str:
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert required <= row.keys(), f"missing columns: {required - row.keys()}"
    assert row["data_source"] == "nq_phase1_smoke"
    assert row["prompt"] and row["prompt"][0]["role"] == "user"
    assert "<search>" in row["prompt"][0]["content"]
    targets = row["reward_model"]["ground_truth"]["target"]
    assert targets and all(isinstance(answer, str) and answer for answer in targets)
    stable_id = row["extra_info"]["index"]
    assert stable_id.startswith(f"nq:{split}:")
    return stable_id


def main() -> None:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "not a valid retrieval benchmark" in manifest["purpose"]

    expected_counts = {"train": 128, "val": 64}
    stable_ids: list[str] = []
    for filename, split in (("train.parquet", "train"), ("test.parquet", "val")):
        path = args.data_dir / filename
        assert sha256(path) == manifest["artifacts"][filename]
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == expected_counts[split]
        stable_ids.extend(check_row(row, split) for row in rows)
    assert len(stable_ids) == len(set(stable_ids))

    corpus_path = args.data_dir / "corpus.jsonl"
    assert sha256(corpus_path) == manifest["artifacts"]["corpus.jsonl"]
    with corpus_path.open(encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    assert len(corpus) == 256
    assert len({doc["id"] for doc in corpus}) == len(corpus)
    evidence = [doc for doc in corpus if doc["fixture_kind"] == "answer-bearing-evidence"]
    distractors = [doc for doc in corpus if doc["fixture_kind"] == "distractor"]
    assert len(evidence) == 192
    assert len(distractors) == 64
    assert manifest["known_query"]["expected_document_id"] in {doc["id"] for doc in evidence}

    index_path = args.data_dir / "index/e5_Flat.index"
    if args.require_index:
        assert index_path.is_file() and index_path.stat().st_size > 0
        import faiss
        index = faiss.read_index(str(index_path))
        assert index.ntotal == len(corpus)
        assert index.d == 384

    print(json.dumps({
        "status": "ok",
        "train_rows": 128,
        "validation_rows": 64,
        "corpus_documents": 256,
        "answer_bearing_documents": 192,
        "distractor_documents": 64,
        "index_present": index_path.is_file(),
        "index_sha256": sha256(index_path) if index_path.is_file() else None,
        "source_revision": manifest["source"]["resolved_revision"],
    }, indent=2))


if __name__ == "__main__":
    main()
