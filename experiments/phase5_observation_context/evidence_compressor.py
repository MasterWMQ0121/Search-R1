#!/usr/bin/env python3
"""Deterministic query-aware extractive compression for Phase-5 observations.

The compressor deliberately has no answer/ground-truth input and makes no
retrieval or model calls.  For a sentence ``s`` and the unique normalized query
terms ``Q``, its relevance score is::

    bm25(s, Q) + 0.75 * query_coverage
               + 0.25 * title_query_coverage
               + 0.05 / retrieval_rank
               + 0.05 * normalized_retrieval_score

The priors are applied only when at least one of the three lexical components is
non-zero.  BM25 uses k1=1.2, b=0.75, and
``idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``.  Selection priority is
the relevance score divided by the square root of the Qwen token length of the
sentence.  Every packing decision re-tokenizes the complete wrapped observation
so the manager's wrapper, not just evidence text, is charged to the budget.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real


BM25_K1 = 1.2
BM25_B = 0.75
QUERY_COVERAGE_WEIGHT = 0.75
TITLE_COVERAGE_WEIGHT = 0.25
RANK_PRIOR_WEIGHT = 0.05
RETRIEVAL_SCORE_PRIOR_WEIGHT = 0.05

# Intentionally conservative: WH-words are retained because they can also be
# entity tokens (for example, the band "The Who").  If filtering removes every
# query token, normalization falls back to the unfiltered query.
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "had", "he", "her", "his", "in", "into",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "with",
})

_SENTENCE_CLOSERS = frozenset("\"'\u2019\u201d)]}")


def wrap_observation(content: str) -> str:
    """Apply the manager's exact information wrapper to unwrapped content."""

    if not isinstance(content, str):
        raise TypeError("observation content must be text")
    return f"\n\n<information>{content.strip()}</information>\n\n"


def _retrieval_items(retrieval_result):
    if not isinstance(retrieval_result, Sequence) or isinstance(
        retrieval_result, (str, bytes, bytearray)
    ):
        raise ValueError("retrieval_result must be a sequence of passage mappings")
    return list(retrieval_result)


def _document_contents(item, rank):
    if not isinstance(item, Mapping):
        raise ValueError(f"retrieval result at rank {rank} must be a mapping")
    document = item.get("document")
    if not isinstance(document, Mapping):
        raise ValueError(f"retrieval result at rank {rank} has no document mapping")
    contents = document.get("contents")
    if not isinstance(contents, str):
        raise ValueError(
            f"retrieval document at rank {rank} must have text contents"
        )
    return document, contents


def format_raw_passages(retrieval_result) -> str:
    """Reproduce Search-R1's existing ``_passages2string`` formatting exactly."""

    formatted = ""
    for rank, item in enumerate(_retrieval_items(retrieval_result), start=1):
        _, contents = _document_contents(item, rank)
        title = contents.split("\n")[0]
        body = "\n".join(contents.split("\n")[1:])
        formatted += f"Doc {rank}(Title: {title}) {body}\n"
    return formatted


@dataclass(frozen=True)
class _Document:
    rank: int
    document_id: str
    retrieval_score: float
    normalized_retrieval_score: float
    title: str
    body: str


@dataclass
class _Candidate:
    document: _Document
    sentence_index: int
    char_start: int
    char_end: int
    text: str
    terms: tuple[str, ...]
    canonical_text: str
    token_count: int
    bm25_score: float = 0.0
    query_coverage: float = 0.0
    title_query_coverage: float = 0.0
    relevance_score: float = 0.0
    selection_priority: float = 0.0
    duplicate_sources: list[dict] = field(default_factory=list)

    @property
    def identifier(self):
        return (
            f"document_rank={self.document.rank};"
            f"sentence_index={self.sentence_index}"
        )


def _alphanumeric_terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms = []
    current = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return tuple(terms)


def _content_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in _alphanumeric_terms(text) if term not in STOP_WORDS)


def _query_terms(query: str) -> tuple[str, ...]:
    all_terms = _alphanumeric_terms(query)
    filtered = tuple(term for term in all_terms if term not in STOP_WORDS)
    return filtered or all_terms


def _sentence_spans(text: str):
    """Yield normalized sentences plus stable offsets from line/.!? boundaries."""

    start = 0
    position = 0
    length = len(text)

    def normalized_span(span_start, span_end):
        sentence = " ".join(text[span_start:span_end].split())
        if sentence:
            return span_start, span_end, sentence
        return None

    while position < length:
        character = text[position]
        if character == "\n":
            span = normalized_span(start, position)
            if span is not None:
                yield span
            position += 1
            while position < length and text[position] in "\r\n":
                position += 1
            start = position
            continue

        if character in ".!?":
            boundary = position + 1
            while boundary < length and text[boundary] in _SENTENCE_CLOSERS:
                boundary += 1
            if boundary == length or text[boundary].isspace():
                span = normalized_span(start, boundary)
                if span is not None:
                    yield span
                position = boundary
                while position < length and text[position].isspace():
                    position += 1
                start = position
                continue
        position += 1

    span = normalized_span(start, length)
    if span is not None:
        yield span


class EvidenceCompressor:
    """Select and pack auditable evidence under an exact tokenizer budget."""

    def __init__(self, tokenizer, max_observation_tokens=256):
        if (
            isinstance(max_observation_tokens, bool)
            or not isinstance(max_observation_tokens, int)
            or max_observation_tokens <= 0
        ):
            raise ValueError("max_observation_tokens must be a positive integer")
        self.tokenizer = tokenizer
        self.max_observation_tokens = max_observation_tokens

    def _encode(self, text):
        if hasattr(self.tokenizer, "encode"):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            encoded = self.tokenizer(text, add_special_tokens=False)
            token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("tokenizer returned more than one encoded row")
            token_ids = token_ids[0]
        return list(token_ids)

    def _decode(self, token_ids):
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _wrapped_token_count(self, content):
        return len(self._encode(wrap_observation(content)))

    @staticmethod
    def _parse_documents(retrieval_result):
        parsed = []
        raw = []
        for rank, item in enumerate(_retrieval_items(retrieval_result), start=1):
            document, contents = _document_contents(item, rank)
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError(
                    f"retrieval result at rank {rank} must have a numeric score"
                )
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(
                    f"retrieval result at rank {rank} has a non-finite score"
                )
            title, separator, body = contents.partition("\n")
            if not separator:
                body = ""
            document_id = document.get("id")
            if document_id is None:
                document_id = f"rank:{rank}"
            raw.append((rank, str(document_id), score, title, body))

        scores = [row[2] for row in raw]
        if scores and max(scores) != min(scores):
            low = min(scores)
            scale = max(scores) - low
            normalized_scores = [(score - low) / scale for score in scores]
        else:
            normalized_scores = [0.5 for _ in scores]

        for row, normalized_score in zip(raw, normalized_scores):
            rank, document_id, score, title, body = row
            parsed.append(_Document(
                rank=rank,
                document_id=document_id,
                retrieval_score=score,
                normalized_retrieval_score=normalized_score,
                title=title,
                body=body,
            ))
        return parsed

    def _build_candidates(self, documents):
        candidates = []
        sentences_considered = 0
        for document in documents:
            for sentence_index, (start, end, sentence) in enumerate(
                _sentence_spans(document.body)
            ):
                sentences_considered += 1
                canonical = " ".join(_alphanumeric_terms(sentence))
                if not canonical:
                    continue
                candidates.append(_Candidate(
                    document=document,
                    sentence_index=sentence_index,
                    char_start=start,
                    char_end=end,
                    text=sentence,
                    terms=_content_terms(sentence),
                    canonical_text=canonical,
                    token_count=max(1, len(self._encode(sentence))),
                ))
        return candidates, sentences_considered

    @staticmethod
    def _score_candidates(candidates, query_terms):
        canonical_candidates = {}
        for candidate in candidates:
            canonical_candidates.setdefault(candidate.canonical_text, candidate)
        unique_candidates = list(canonical_candidates.values())
        candidate_count = len(unique_candidates)
        average_length = (
            sum(len(candidate.terms) for candidate in unique_candidates)
            / candidate_count
            if candidate_count else 1.0
        )
        average_length = max(average_length, 1.0)
        document_frequency = Counter()
        for candidate in unique_candidates:
            document_frequency.update(set(candidate.terms))

        query_unique = tuple(sorted(set(query_terms)))
        query_term_set = set(query_unique)
        for candidate in candidates:
            frequencies = Counter(candidate.terms)
            sentence_term_set = set(candidate.terms)
            bm25_score = 0.0
            for term in query_unique:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (candidate_count - df + 0.5) / (df + 0.5)
                )
                length_normalization = BM25_K1 * (
                    1.0 - BM25_B
                    + BM25_B * len(candidate.terms) / average_length
                )
                bm25_score += (
                    inverse_document_frequency
                    * frequency
                    * (BM25_K1 + 1.0)
                    / (frequency + length_normalization)
                )

            denominator = len(query_term_set)
            query_coverage = (
                len(query_term_set.intersection(sentence_term_set)) / denominator
                if denominator else 0.0
            )
            title_terms = set(_content_terms(candidate.document.title))
            title_coverage = (
                len(query_term_set.intersection(title_terms)) / denominator
                if denominator else 0.0
            )
            lexical_signal = bm25_score + query_coverage + title_coverage
            if lexical_signal > 0.0:
                relevance_score = (
                    bm25_score
                    + QUERY_COVERAGE_WEIGHT * query_coverage
                    + TITLE_COVERAGE_WEIGHT * title_coverage
                    + RANK_PRIOR_WEIGHT / candidate.document.rank
                    + RETRIEVAL_SCORE_PRIOR_WEIGHT
                    * candidate.document.normalized_retrieval_score
                )
            else:
                relevance_score = 0.0
            candidate.bm25_score = bm25_score
            candidate.query_coverage = query_coverage
            candidate.title_query_coverage = title_coverage
            candidate.relevance_score = relevance_score
            candidate.selection_priority = (
                relevance_score / math.sqrt(candidate.token_count)
                if relevance_score > 0.0 else 0.0
            )

    @staticmethod
    def _deduplicate(candidates):
        grouped = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.canonical_text].append(candidate)

        deduplicated = []
        for duplicates in grouped.values():
            ordered = sorted(
                duplicates,
                key=lambda item: (
                    -item.selection_priority,
                    -item.relevance_score,
                    item.document.rank,
                    item.sentence_index,
                ),
            )
            retained = ordered[0]
            retained.duplicate_sources = [
                {
                    "document_rank": duplicate.document.rank,
                    "document_id": duplicate.document.document_id,
                    "sentence_index": duplicate.sentence_index,
                }
                for duplicate in ordered[1:]
            ]
            deduplicated.append(retained)
        return deduplicated

    @staticmethod
    def _render(selected):
        grouped = defaultdict(list)
        documents = {}
        for candidate in selected:
            grouped[candidate.document.rank].append(candidate)
            documents[candidate.document.rank] = candidate.document
        lines = []
        for rank in sorted(grouped):
            document = documents[rank]
            ordered = sorted(grouped[rank], key=lambda item: item.sentence_index)
            sentences = " ".join(candidate.text for candidate in ordered)
            lines.append(f"Doc {rank}(Title: {document.title}) {sentences}")
        return "\n".join(lines)

    def _try_add(self, selected, candidate):
        if candidate in selected:
            return False
        proposed = [*selected, candidate]
        if self._wrapped_token_count(self._render(proposed)) <= self.max_observation_tokens:
            selected.append(candidate)
            return True
        return False

    def _select_relevant(self, candidates):
        positive = [candidate for candidate in candidates if candidate.relevance_score > 0]
        selected = []

        # Diversity pass: give every relevant source document one opportunity
        # before globally filling remaining space.  Priority already includes a
        # token-efficiency normalization, so a long rank-1 sentence does not win
        # merely because it appears first.
        by_document = defaultdict(list)
        for candidate in positive:
            by_document[candidate.document.rank].append(candidate)
        seeds = [
            sorted(
                document_candidates,
                key=lambda item: (
                    -item.selection_priority,
                    -item.relevance_score,
                    item.sentence_index,
                ),
            )[0]
            for document_candidates in by_document.values()
        ]
        seeds.sort(key=lambda item: (
            -item.selection_priority,
            -item.relevance_score,
            item.document.rank,
            item.sentence_index,
        ))
        for candidate in seeds:
            self._try_add(selected, candidate)

        remaining = sorted(
            positive,
            key=lambda item: (
                -item.selection_priority,
                -item.relevance_score,
                item.document.rank,
                item.sentence_index,
            ),
        )
        for candidate in remaining:
            self._try_add(selected, candidate)
        return selected

    def _select_zero_overlap(self, candidates):
        selected = []
        by_document = defaultdict(list)
        for candidate in candidates:
            by_document[candidate.document.rank].append(candidate)

        # Concise leading evidence: choose the shorter of each document's first
        # two sentences (ties retain original order), visiting documents by rank.
        leading = []
        for rank in sorted(by_document):
            first_two = sorted(
                by_document[rank], key=lambda item: item.sentence_index
            )[:2]
            if first_two:
                leading.append(min(
                    first_two,
                    key=lambda item: (item.token_count, item.sentence_index),
                ))
        for candidate in leading:
            self._try_add(selected, candidate)

        for candidate in sorted(
            candidates,
            key=lambda item: (item.document.rank, item.sentence_index),
        ):
            self._try_add(selected, candidate)
        return selected

    def _partial_fallback(self, documents):
        for document in documents:
            source = " ".join(
                part.strip() for part in (document.title, document.body) if part.strip()
            )
            if not source:
                continue
            source_ids = self._encode(source)
            maximum = min(len(source_ids), self.max_observation_tokens)
            for retained in range(maximum, 0, -1):
                fragment = self._decode(source_ids[:retained]).strip()
                if not fragment:
                    continue
                content = f"Doc {document.rank}: {fragment}"
                if self._wrapped_token_count(content) <= self.max_observation_tokens:
                    return content, document.rank
        return None, None

    def compress(self, query, retrieval_result, raw_observation=None):
        """Compress one query's structured hits; ``raw_observation`` is unwrapped."""

        if not isinstance(query, str):
            raise TypeError("query must be text")
        documents = self._parse_documents(retrieval_result)
        if raw_observation is None:
            raw_observation = format_raw_passages(retrieval_result)
        elif not isinstance(raw_observation, str):
            raise TypeError("raw_observation must be unwrapped text")
        raw_token_count = self._wrapped_token_count(raw_observation)

        candidates, sentences_considered = self._build_candidates(documents)
        self._score_candidates(candidates, _query_terms(query))
        candidates = self._deduplicate(candidates)
        zero_overlap = not any(
            candidate.relevance_score > 0 for candidate in candidates
        )
        selected = (
            self._select_zero_overlap(candidates)
            if zero_overlap else self._select_relevant(candidates)
        )

        partial_fallback = False
        partial_rank = None
        if selected:
            content = self._render(selected)
        else:
            content, partial_rank = self._partial_fallback(documents)
            if content is not None:
                partial_fallback = True
            else:
                content = "No nonempty evidence returned."
                if self._wrapped_token_count(content) > self.max_observation_tokens:
                    raise ValueError(
                        "observation budget is too small for the information wrapper"
                    )

        policy_token_count = self._wrapped_token_count(content)
        if policy_token_count > self.max_observation_tokens:
            raise AssertionError("compressed observation exceeds its tokenizer budget")

        selected_ordered = sorted(
            selected,
            key=lambda item: (item.document.rank, item.sentence_index),
        )
        represented_ranks = sorted({
            candidate.document.rank for candidate in selected_ordered
        })
        if partial_rank is not None:
            represented_ranks = [partial_rank]
        provenance = [
            {
                "identifier": candidate.identifier,
                "document_rank": candidate.document.rank,
                "document_id": candidate.document.document_id,
                "sentence_index": candidate.sentence_index,
                "char_start": candidate.char_start,
                "char_end": candidate.char_end,
                "retrieval_score": candidate.document.retrieval_score,
                "bm25_score": candidate.bm25_score,
                "query_coverage": candidate.query_coverage,
                "title_query_coverage": candidate.title_query_coverage,
                "relevance_score": candidate.relevance_score,
                "selection_priority": candidate.selection_priority,
                "duplicate_sources": candidate.duplicate_sources,
            }
            for candidate in selected_ordered
        ]
        return {
            "content": content,
            "raw_retrieved_observation_tokens": raw_token_count,
            "policy_output_observation_tokens": policy_token_count,
            "documents_returned": len(documents),
            "documents_represented": len(represented_ranks),
            "sentences_considered": sentences_considered,
            "sentences_selected": len(selected_ordered),
            "zero_overlap_fallback_used": zero_overlap,
            "partial_sentence_fallback_used": partial_fallback,
            "selected_document_ranks": represented_ranks,
            "selected_sentence_identifiers": [
                candidate.identifier for candidate in selected_ordered
            ],
            "selected_sentence_provenance": provenance,
        }


__all__ = ["EvidenceCompressor", "format_raw_passages", "wrap_observation"]
