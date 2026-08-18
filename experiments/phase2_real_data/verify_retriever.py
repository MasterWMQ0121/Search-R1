#!/usr/bin/env python3
"""Verify the live real Wiki-18 retriever contract and report latency."""

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request


DEFAULT_QUESTIONS = (
    "Who wrote Pride and Prejudice?",
    "What is the capital of Mongolia?",
    "Which river flows through Budapest?",
    "In what year did the Apollo 11 mission land on the Moon?",
)


def validate_response(payload, topk):
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise ValueError("response must contain a list-valued 'result'")
    if len(payload["result"]) != 1:
        raise ValueError("single-query request must return exactly one result list")
    results = payload["result"][0]
    if len(results) != topk:
        raise ValueError(f"expected exactly {topk} results, received {len(results)}")

    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"result {rank} is not an object")
        document = item.get("document")
        if not isinstance(document, dict):
            raise ValueError(f"result {rank} is missing its document object")
        contents = document.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError(f"result {rank} has empty document contents")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"result {rank} has a non-finite score")
    return results


def retrieve_one(url, question, topk, timeout):
    body = json.dumps({"queries": [question], "topk": topk, "return_scores": True}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"retriever returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
    latency = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"retriever returned HTTP {status}")
    validate_response(payload, topk)
    return latency


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--question", action="append", dest="questions")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be greater than zero")
    questions = args.questions or list(DEFAULT_QUESTIONS)
    latencies = []
    for number, question in enumerate(questions, start=1):
        latency = retrieve_one(args.url, question, args.topk, args.timeout)
        latencies.append(latency)
        print(f"query {number}: {latency:.3f}s | {question}")

    print(f"requests: {len(latencies)}")
    print(f"mean latency: {statistics.fmean(latencies):.3f}s")
    print(f"p50 latency: {statistics.median(latencies):.3f}s")
    if len(latencies) >= 20:
        p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94]
        print(f"p95 latency: {p95:.3f}s")


if __name__ == "__main__":
    main()
