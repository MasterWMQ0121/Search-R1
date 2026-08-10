#!/usr/bin/env python3
"""Exercise the live Phase-1 retriever, including malformed requests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--manifest", type=Path, default=Path("data/phase1_smoke/manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    known = manifest["known_query"]

    response = requests.post(args.url, json={
        "queries": [known["query"]], "topk": 2, "return_scores": True
    }, timeout=120)
    response.raise_for_status()
    body = response.json()
    assert list(body) == ["result"]
    assert len(body["result"]) == 1 and len(body["result"][0]) == 2
    hits = body["result"][0]
    assert all(isinstance(hit["score"], (int, float)) and math.isfinite(hit["score"]) for hit in hits)
    returned_ids = [hit["document"]["id"] for hit in hits]
    assert known["expected_document_id"] in returned_ids, (known["expected_document_id"], returned_ids)

    malformed_statuses = {}
    for label, payload in {
        "empty_queries": {"queries": [], "topk": 2},
        "blank_query": {"queries": ["   "], "topk": 2},
        "non_positive_topk": {"queries": [known["query"]], "topk": 0},
        "missing_queries": {"topk": 2},
    }.items():
        malformed = requests.post(args.url, json=payload, timeout=30)
        assert 400 <= malformed.status_code < 500, (label, malformed.status_code, malformed.text)
        malformed_statuses[label] = malformed.status_code

    print(json.dumps({
        "status": "ok",
        "query": known["query"],
        "expected_document_id": known["expected_document_id"],
        "returned_document_ids": returned_ids,
        "scores": [hit["score"] for hit in hits],
        "malformed_request_statuses": malformed_statuses,
    }, indent=2))


if __name__ == "__main__":
    main()
