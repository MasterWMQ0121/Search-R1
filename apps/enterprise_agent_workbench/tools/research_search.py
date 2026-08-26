"""One-call Search-R1 Retriever adapter with Phase-5 evidence compression."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from experiments.phase5_observation_context.evidence_compressor import (
    BM25_B,
    BM25_K1,
    EvidenceCompressor,
    QUERY_COVERAGE_WEIGHT,
    RANK_PRIOR_WEIGHT,
    RETRIEVAL_SCORE_PRIOR_WEIGHT,
    TITLE_COVERAGE_WEIGHT,
)

from ..tokenizer_runtime import TokenizerRuntime, approximate_test_runtime
from .enterprise_kb import SourceRecord


EVIDENCE_COMPRESSOR_POLICY = "deterministic_query_aware_extractive"
EVIDENCE_COMPRESSOR_VERSION = "phase5-extractive-v1"


@lru_cache(maxsize=1)
def evidence_compressor_fingerprint() -> str:
    source = inspect.getsourcefile(EvidenceCompressor)
    if source is None:
        raise RuntimeError("cannot locate the Phase-5 EvidenceCompressor source")
    source_digest = hashlib.sha256()
    with Path(source).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            source_digest.update(chunk)
    policy = {
        "name": EVIDENCE_COMPRESSOR_POLICY,
        "version": EVIDENCE_COMPRESSOR_VERSION,
        "source_sha256": source_digest.hexdigest(),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "query_coverage_weight": QUERY_COVERAGE_WEIGHT,
        "title_coverage_weight": TITLE_COVERAGE_WEIGHT,
        "rank_prior_weight": RANK_PRIOR_WEIGHT,
        "retrieval_score_prior_weight": RETRIEVAL_SCORE_PRIOR_WEIGHT,
        "selection_priority": "relevance/sqrt(sentence_token_count)",
    }
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ResearchSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace text")
        return value


class ResearchDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    rank: int = Field(ge=1)
    title: str
    document_id: str | None = None
    score: float | None = None


class ResearchSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    compressed_evidence: str
    documents: list[ResearchDocument]
    sources: list[SourceRecord]
    retrieval_latency_s: float = Field(ge=0.0)
    compression_latency_s: float = Field(ge=0.0)
    failure_status: str | None = None
    compression_metrics: dict[str, Any] = Field(default_factory=dict)


class _Compressor(Protocol):
    def compress(self, query: str, passages: list[dict[str, Any]]) -> dict[str, Any]: ...


class Phase5EvidenceAdapter:
    """Stable adapter around, not a copy of, the Phase-5 compressor."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        *,
        tokenizer_runtime: TokenizerRuntime | None = None,
        max_observation_tokens: int = 256,
    ):
        if tokenizer_runtime is not None:
            if tokenizer is not None and tokenizer is not tokenizer_runtime.tokenizer:
                raise ValueError("tokenizer and tokenizer_runtime disagree")
            tokenizer = tokenizer_runtime.tokenizer
        if tokenizer is None:
            raise ValueError("an injected tokenizer is required")
        self.tokenizer_runtime = tokenizer_runtime or approximate_test_runtime(tokenizer)
        self.max_observation_tokens = max_observation_tokens
        self._compressor = EvidenceCompressor(
            tokenizer, max_observation_tokens=max_observation_tokens
        )

    def compress(self, query: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
        compressed = self._compressor.compress(query, passages)
        return {
            **compressed,
            "tokenizer_mode": self.tokenizer_runtime.mode,
            "tokenizer_class": self.tokenizer_runtime.tokenizer_class,
            "tokenizer_artifact_fingerprint": (
                self.tokenizer_runtime.artifact_fingerprint
            ),
            "evidence_compressor_policy": EVIDENCE_COMPRESSOR_POLICY,
            "evidence_compressor_version": EVIDENCE_COMPRESSOR_VERSION,
            "evidence_compressor_fingerprint": evidence_compressor_fingerprint(),
            "max_evidence_token_budget": self.max_observation_tokens,
        }


class DeterministicTextTokenizer:
    """Small reversible tokenizer for the dependency-light local prototype.

    It provides the synchronous ``encode``/``decode`` contract required by the
    existing Phase-5 compressor without loading a model. Its counts are lexical
    rather than Qwen-tokenizer exact, so production deployments should inject
    the checkpoint's tokenizer when exact budget parity is required.
    """

    _pieces = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self) -> None:
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._lock = threading.Lock()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pieces = self._pieces.findall(text)
        with self._lock:
            ids = []
            for piece in pieces:
                identifier = self._piece_to_id.get(piece)
                if identifier is None:
                    identifier = len(self._piece_to_id) + 1
                    self._piece_to_id[piece] = identifier
                    self._id_to_piece[identifier] = piece
                ids.append(identifier)
            return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        with self._lock:
            pieces = [self._id_to_piece.get(int(identifier), "") for identifier in token_ids]
        return " ".join(piece for piece in pieces if piece)


def _passage_fields(item: dict[str, Any], rank: int) -> tuple[str, str, str | None, float | None]:
    if not isinstance(item, dict) or not isinstance(item.get("document"), dict):
        raise ValueError(f"retrieval result at rank {rank} has no document mapping")
    document = item["document"]
    contents = document.get("contents")
    if not isinstance(contents, str):
        raise ValueError(f"retrieval result at rank {rank} has no text contents")
    title, _, body = contents.partition("\n")
    document_id = document.get("id")
    document_id = None if document_id is None else str(document_id)
    raw_score = item.get("score")
    score = float(raw_score) if raw_score is not None else None
    return title.strip() or f"Retrieved document {rank}", body or contents, document_id, score


class ResearchSearchTool:
    def __init__(
        self,
        endpoint: str,
        compressor: _Compressor,
        *,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
        snippet_limit: int = 500,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 80 <= snippet_limit <= 800:
            raise ValueError("snippet_limit must be between 80 and 800 characters")
        self.endpoint = endpoint
        self.compressor = compressor
        self.timeout_seconds = timeout_seconds
        self.client = client
        self.snippet_limit = snippet_limit

    async def search(
        self, request: ResearchSearchInput | dict[str, Any]
    ) -> ResearchSearchOutput:
        request = ResearchSearchInput.model_validate(request)
        started = time.perf_counter()
        try:
            payload = {
                "queries": [request.query],
                "topk": request.top_k,
                "return_scores": True,
            }
            if self.client is None:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.endpoint, json=payload, timeout=self.timeout_seconds
                    )
            else:
                response = await self.client.post(
                    self.endpoint, json=payload, timeout=self.timeout_seconds
                )
            response.raise_for_status()
            response_payload = response.json()
            result_groups = response_payload.get("result")
            if not isinstance(result_groups, list) or len(result_groups) != 1:
                raise ValueError("Retriever response must contain one result group")
            passages = result_groups[0]
            if not isinstance(passages, list):
                raise ValueError("Retriever result group must be a list")
        except (httpx.HTTPError, ValueError, TypeError) as error:
            return ResearchSearchOutput(
                query=request.query,
                compressed_evidence="",
                documents=[],
                sources=[],
                retrieval_latency_s=max(0.0, time.perf_counter() - started),
                compression_latency_s=0.0,
                failure_status=f"{type(error).__name__}: {error}",
            )

        retrieval_latency = max(0.0, time.perf_counter() - started)
        compression_started = time.perf_counter()
        try:
            compressed = self.compressor.compress(request.query, passages)
            documents: list[ResearchDocument] = []
            sources: list[SourceRecord] = []
            for rank, item in enumerate(passages, start=1):
                title, body, document_id, score = _passage_fields(item, rank)
                identity = document_id or hashlib.sha256(
                    f"{title}\n{body}".encode("utf-8")
                ).hexdigest()[:16]
                source_id = f"RESEARCH-{identity}"
                documents.append(
                    ResearchDocument(
                        source_id=source_id,
                        rank=rank,
                        title=title,
                        document_id=document_id,
                        score=score,
                    )
                )
                sources.append(
                    SourceRecord(
                        source_id=source_id,
                        source_type="research",
                        title=title,
                        section=None,
                        snippet=" ".join(body.split())[: self.snippet_limit],
                        score=score,
                        metadata={"rank": rank, "document_id": document_id},
                    )
                )
            compression_latency = max(
                0.0, time.perf_counter() - compression_started
            )
            metrics = {
                key: value
                for key, value in compressed.items()
                if key not in {"content", "selected_sentence_provenance"}
            }
            return ResearchSearchOutput(
                query=request.query,
                compressed_evidence=str(compressed.get("content", "")),
                documents=documents,
                sources=sources,
                retrieval_latency_s=retrieval_latency,
                compression_latency_s=compression_latency,
                failure_status=None,
                compression_metrics=metrics,
            )
        except (ValueError, TypeError, KeyError) as error:
            return ResearchSearchOutput(
                query=request.query,
                compressed_evidence="",
                documents=[],
                sources=[],
                retrieval_latency_s=retrieval_latency,
                compression_latency_s=max(
                    0.0, time.perf_counter() - compression_started
                ),
                failure_status=f"{type(error).__name__}: {error}",
            )
