#!/usr/bin/env python3
"""Build the deterministic Phase-1 NQ smoke dataset and fixture corpus.

This deliberately downloads only the small NQ JSON files from FlashRAG.  The
generated corpus is synthetic, answer-bearing test infrastructure; it must not
be used for benchmark claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download


SOURCE_REPO = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
SEED = 42
TRAIN_SIZE = 128
VAL_SIZE = 64
DISTRACTOR_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase1_smoke"))
    parser.add_argument("--repo-id", default=SOURCE_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_source_id(example: dict[str, Any]) -> str:
    raw_id = example.get("id")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id).strip()
    return hashlib.sha256(example["question"].encode("utf-8")).hexdigest()[:20]


def answers_for(example: dict[str, Any]) -> list[str]:
    raw_answers = example.get("golden_answers", [])
    if isinstance(raw_answers, str):
        raw_answers = [raw_answers]
    answers = [str(answer).strip() for answer in raw_answers if str(answer).strip()]
    if not answers:
        raise ValueError(f"NQ example has no golden answer: {stable_source_id(example)}")
    return answers


def normalize_question(question: str) -> str:
    question = question.strip()
    return question if question.endswith("?") else f"{question}?"


def make_prompt(question: str) -> str:
    return (
        "PHASE-1 SEARCH-R1 PROTOCOL SMOKE TEST. This deliberately strict instruction validates "
        "plumbing, not model quality. Follow the tagged protocol exactly; do not write plain "
        "prose outside the tags.\n\n"
        "MANDATORY FIRST RESPONSE:\n"
        "Your FIRST assistant response must contain a brief reason inside <think>...</think>, "
        "then end in exactly one non-empty <search>...</search>. The search text must contain "
        "at least one non-whitespace character. Stop immediately after </search>. Do not answer "
        "the question and do not produce any final-answer action before receiving an "
        "<information>...</information> block, even if you already know the answer.\n\n"
        "Use this exact first-response shape:\n"
        "<think>I need to retrieve evidence before answering.</think>\n"
        f"<search>{question}</search>\n\n"
        "AFTER RECEIVING <information>...</information>:\n"
        "PHASE-1 FIXTURE STOP RULE: If any retrieved document contains the literal text "
        "\"Accepted answer evidence:\", retrieval is sufficient regardless of whether you think "
        "more information would be useful. You MUST NOT search again. Copy the answer text "
        "immediately following that prefix and emit exactly:\n"
        "<think>The retrieved fixture contains explicit accepted answer evidence.</think>\n"
        "<answer>ANSWER_TEXT</answer>\n"
        "Replace ANSWER_TEXT with the copied evidence and stop immediately after </answer>. "
        "This stop rule applies only to this deliberately answer-leaky Phase-1 fixture.\n"
        "Every time new information arrives, reason inside <think>...</think> before the next "
        "action. If the retrieved evidence contains the answer, your NEXT assistant response "
        "MUST consist of exactly two tagged blocks and nothing else, using the format above. "
        "If the evidence does not contain the answer, use <think>...</think> followed by exactly "
        "one non-empty <search>...</search> instead. Never write plain prose such as "
        "\"the answer is ...\", \"to answer: ...\", or any other text outside protocol tags.\n\n"
        f"Question: {question}\n"
        "STATE RULE: if the conversation has no information block yet, follow MANDATORY FIRST "
        "RESPONSE and search. Once an information block appears, follow AFTER RECEIVING "
        "INFORMATION instead. At the start there is no information yet, so search and do not answer."
    )


def select_examples(
    examples: list[dict[str, Any]], count: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    if len(examples) < count:
        raise ValueError(f"requested {count} rows from a split containing {len(examples)}")
    indexes = sorted(random.Random(seed).sample(range(len(examples)), count))
    return [(index, examples[index]) for index in indexes]


def make_row(split: str, source_index: int, example: dict[str, Any]) -> dict[str, Any]:
    source_id = stable_source_id(example)
    question = normalize_question(str(example["question"]))
    stable_id = f"nq:{split}:{source_id}"
    return {
        "data_source": "nq_phase1_smoke",
        "prompt": [{"role": "user", "content": make_prompt(question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers_for(example)}},
        "extra_info": {
            "split": split,
            "index": stable_id,
            "source_index": source_index,
            "source_id": source_id,
            "question": question,
        },
    }


def evidence_document(split: str, example: dict[str, Any]) -> dict[str, Any]:
    source_id = stable_source_id(example)
    question = normalize_question(str(example["question"]))
    answers = answers_for(example)
    title = f"PHASE-1 FIXTURE EVIDENCE {split} {source_id}"
    text = f"Question: {question}\nAccepted answer evidence: {'; '.join(answers)}"
    return {
        "id": f"evidence:{split}:{source_id}",
        "title": title,
        "text": text,
        "contents": f'"{title}"\n{text}',
        "fixture_kind": "answer-bearing-evidence",
    }


def distractor_document(index: int, question_example: dict[str, Any], answer_example: dict[str, Any]) -> dict[str, Any]:
    question = normalize_question(str(question_example["question"]))
    unrelated_answers = answers_for(answer_example)
    title = f"PHASE-1 FIXTURE DISTRACTOR {index:04d}"
    text = (
        f"Unrelated question: {question}\n"
        f"Unrelated answer evidence: {'; '.join(unrelated_answers)}"
    )
    return {
        "id": f"distractor:{index:04d}",
        "title": title,
        "text": text,
        "contents": f'"{title}"\n{text}',
        "fixture_kind": "distractor",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_revision = args.revision
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        resolved_revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
    source_paths = {
        "train": Path(hf_hub_download(args.repo_id, "nq/train.jsonl", repo_type="dataset", revision=resolved_revision)),
        "test": Path(hf_hub_download(args.repo_id, "nq/test.jsonl", repo_type="dataset", revision=resolved_revision)),
    }
    source = {split: read_jsonl(path) for split, path in source_paths.items()}

    selected_train = select_examples(source["train"], args.train_size, args.seed)
    selected_val = select_examples(source["test"], args.val_size, args.seed)
    train_rows = [make_row("train", index, example) for index, example in selected_train]
    val_rows = [make_row("val", index, example) for index, example in selected_val]

    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(val_rows).to_parquet(str(test_path))

    selected_all = [("train", example) for _, example in selected_train]
    selected_all.extend(("val", example) for _, example in selected_val)
    corpus = [evidence_document(split, example) for split, example in selected_all]
    for index in range(DISTRACTOR_SIZE):
        question_example = selected_all[index][1]
        answer_example = selected_all[(index + args.val_size + 1) % len(selected_all)][1]
        corpus.append(distractor_document(index, question_example, answer_example))

    corpus_path = args.output_dir / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)

    known_split, known_example = selected_all[0]
    known_source_id = stable_source_id(known_example)
    manifest = {
        "purpose": "Phase-1 smoke-test fixture only; not a valid retrieval benchmark",
        "source": {
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "files": {split: f"nq/{split}.jsonl" for split in source_paths},
        },
        "selection": {
            "seed": args.seed,
            "validation_seed": args.seed,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
        },
        "corpus": {
            "documents": len(corpus),
            "answer_bearing_documents": len(selected_all),
            "distractor_documents": DISTRACTOR_SIZE,
        },
        "known_query": {
            "query": normalize_question(str(known_example["question"])),
            "expected_document_id": f"evidence:{known_split}:{known_source_id}",
            "accepted_answers": answers_for(known_example),
        },
        "artifacts": {
            "train.parquet": sha256(train_path),
            "test.parquet": sha256(test_path),
            "corpus.jsonl": sha256(corpus_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "revision": resolved_revision,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "corpus_documents": len(corpus),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
