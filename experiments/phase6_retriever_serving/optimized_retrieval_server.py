#!/usr/bin/env python3
"""CPU-only, instrumented Retriever server for the isolated Phase-6 study.

The default ``/retrieve`` response is intentionally identical to the existing
Search-R1 server.  Timing and result-ID diagnostics are added only when a caller
sets ``return_metrics=true``.  The module supports dependency injection so its
serving logic can be tested without loading the production model, corpus, or
index.
"""

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel


TIMING_FIELDS = (
    "request_total_s",
    "query_normalization_s",
    "cache_lookup_s",
    "query_encoding_s",
    "faiss_search_s",
    "document_fetch_s",
    "response_format_s",
)
VALID_BACKENDS = ("flat", "ivfpq")


class InvalidRetrievalRequest(ValueError):
    """An invalid request that should be exposed as HTTP 400."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_asset(path: Path) -> str:
    """Hash one file or a directory tree without following unrelated paths."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return _sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"fingerprinted asset does not exist: {resolved}")
    digest = hashlib.sha256()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"fingerprinted model directory is empty: {resolved}")
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def process_rss_bytes() -> Optional[int]:
    """Return Linux RSS using stdlib only, or ``None`` when unavailable."""

    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, flags=re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def _stable_fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_name(metric_type: int, faiss_module: Any) -> str:
    if metric_type == getattr(faiss_module, "METRIC_INNER_PRODUCT", 0):
        return "inner_product"
    if metric_type == getattr(faiss_module, "METRIC_L2", 1):
        return "l2"
    return f"faiss_metric_{metric_type}"


@dataclass
class ServerConfig:
    index_path: Path
    corpus_path: Path
    model_path: Path
    host: str = "127.0.0.1"
    port: int = 8100
    topk: int = 3
    faiss_thread_count: int = 8
    retrieval_encode_batch_size: int = 32
    index_backend: str = "flat"
    nprobe: Optional[int] = None
    cache_enabled: bool = False
    result_cache_capacity: int = 0
    embedding_cache_capacity: int = 0
    model_name: str = "e5"
    pooling_method: str = "mean"
    query_max_length: int = 256
    index_fingerprint: Optional[str] = None
    model_fingerprint: Optional[str] = None
    corpus_fingerprint: Optional[str] = None

    def validate(self) -> None:
        if self.index_backend not in VALID_BACKENDS:
            raise ValueError(f"unsupported index backend: {self.index_backend!r}")
        for label, value in (
            ("port", self.port),
            ("topk", self.topk),
            ("faiss_thread_count", self.faiss_thread_count),
            ("retrieval_encode_batch_size", self.retrieval_encode_batch_size),
            ("query_max_length", self.query_max_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for label, value in (
            ("result_cache_capacity", self.result_cache_capacity),
            ("embedding_cache_capacity", self.embedding_cache_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.index_backend == "ivfpq":
            if isinstance(self.nprobe, bool) or not isinstance(self.nprobe, int) or self.nprobe <= 0:
                raise ValueError("ivfpq nprobe must be a positive integer")
        elif self.nprobe is not None:
            raise ValueError("nprobe is only valid for the ivfpq backend")


class ThreadSafeLRU:
    """A bounded, process-local LRU with explicit disabled semantics."""

    def __init__(self, capacity: int, enabled: bool):
        if capacity < 0:
            raise ValueError("LRU capacity must be non-negative")
        self.capacity = capacity
        self.enabled = bool(enabled and capacity > 0)
        self._values: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any) -> Tuple[bool, Any]:
        if not self.enabled:
            return False, None
        with self._lock:
            if key not in self._values:
                self.misses += 1
                return False, None
            value = self._values.pop(key)
            self._values[key] = value
            self.hits += 1
            return True, value

    def put(self, key: Any, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            if key in self._values:
                self._values.pop(key)
            self._values[key] = value
            if len(self._values) > self.capacity:
                self._values.popitem(last=False)
                self.evictions += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "capacity": self.capacity,
                "size": len(self._values),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


class ServerStats:
    def __init__(self, wall_clock: Callable[[], float] = time.time):
        self._wall_clock = wall_clock
        self.start_epoch_s = float(wall_clock())
        self.start_time = datetime.fromtimestamp(
            self.start_epoch_s, tz=timezone.utc
        ).isoformat()
        self._lock = threading.RLock()
        self.request_count = 0
        self.query_count = 0
        self.errors = 0
        self.timing_sums = {name: 0.0 for name in TIMING_FIELDS}

    def record(self, metrics: Dict[str, Any], error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.query_count += int(metrics.get("query_count", 0))
            self.errors += int(bool(error))
            for name in TIMING_FIELDS:
                value = float(metrics.get(name, 0.0))
                if math.isfinite(value) and value >= 0:
                    self.timing_sums[name] += value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            denominator = self.request_count or 1
            return {
                "server_start_time": self.start_time,
                "uptime_s": max(0.0, float(self._wall_clock()) - self.start_epoch_s),
                "request_count": self.request_count,
                "query_count": self.query_count,
                "errors": self.errors,
                "timing_sums": dict(self.timing_sums),
                "timing_means": {
                    name: total / denominator
                    for name, total in self.timing_sums.items()
                },
            }


class OptimizedRetrieverEngine:
    """One immutable CPU index/model plus exact process-local caches."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        index: Any = None,
        corpus: Any = None,
        encoder: Any = None,
        faiss_module: Any = None,
        index_loader: Optional[Callable[[str], Any]] = None,
        corpus_loader: Optional[Callable[[str], Any]] = None,
        encoder_factory: Optional[Callable[..., Any]] = None,
        document_loader: Optional[Callable[[Any, Sequence[int]], List[Dict[str, Any]]]] = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        rss_reader: Callable[[], Optional[int]] = process_rss_bytes,
    ):
        config.validate()
        self.config = config
        self._clock = clock
        self._rss_reader = rss_reader
        self._encoder_lock = threading.RLock()
        self._document_lock = threading.RLock()
        self.stats = ServerStats(wall_clock=wall_clock)
        startup_start = clock()
        index_was_injected = index is not None or index_loader is not None
        encoder_was_injected = encoder is not None or encoder_factory is not None
        corpus_was_injected = corpus is not None or corpus_loader is not None

        if faiss_module is None:
            import faiss as faiss_module  # Imported lazily for CPU-safe unit tests.
        self.faiss = faiss_module
        if hasattr(self.faiss, "omp_set_num_threads"):
            self.faiss.omp_set_num_threads(config.faiss_thread_count)

        index_started = clock()
        if index is None:
            if index_loader is None:
                index_loader = self.faiss.read_index
            index = index_loader(str(config.index_path.expanduser().resolve()))
        self.index = index
        index_load_s = clock() - index_started

        corpus_started = clock()
        if corpus is None:
            if corpus_loader is None:
                from search_r1.search.retrieval_server import load_corpus

                corpus_loader = load_corpus
            corpus = corpus_loader(str(config.corpus_path.expanduser().resolve()))
        self.corpus = corpus
        corpus_load_s = clock() - corpus_started

        model_started = clock()
        if encoder is None:
            if encoder_factory is None:
                from search_r1.search.retrieval_server import Encoder

                encoder_factory = Encoder
            encoder = encoder_factory(
                model_name=config.model_name,
                model_path=str(config.model_path.expanduser().resolve()),
                pooling_method=config.pooling_method,
                max_length=config.query_max_length,
                use_fp16=False,
                device="cpu",
                model_revision=None,
            )
        self.encoder = encoder
        model_load_s = clock() - model_started

        if document_loader is None:
            from search_r1.search.retrieval_server import load_docs

            document_loader = load_docs
        self._document_loader = document_loader
        self._validate_assets()
        self._configure_backend()

        self.index_fingerprint = self._resolve_asset_fingerprint(
            config.index_path,
            config.index_fingerprint,
            dependency_was_injected=index_was_injected,
            label="index",
        )
        self.model_fingerprint = self._resolve_asset_fingerprint(
            config.model_path,
            config.model_fingerprint,
            dependency_was_injected=encoder_was_injected,
            label="model",
        )
        self.corpus_fingerprint = self._resolve_asset_fingerprint(
            config.corpus_path,
            config.corpus_fingerprint,
            dependency_was_injected=corpus_was_injected,
            label="corpus",
        )
        self.retrieval_config_fingerprint = _stable_fingerprint({
            "index_fingerprint": self.index_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "index_backend": config.index_backend,
            "metric_type": int(self.index.metric_type),
            "nprobe": config.nprobe,
            "model_name": config.model_name,
            "pooling_method": config.pooling_method,
            "query_max_length": config.query_max_length,
            "retrieval_encode_batch_size": config.retrieval_encode_batch_size,
            "faiss_thread_count": config.faiss_thread_count,
        })
        self.result_cache = ThreadSafeLRU(
            config.result_cache_capacity, config.cache_enabled
        )
        self.embedding_cache = ThreadSafeLRU(
            config.embedding_cache_capacity, config.cache_enabled
        )
        self.startup_load_timings_s = {
            "index_load_s": max(0.0, float(index_load_s)),
            "corpus_load_s": max(0.0, float(corpus_load_s)),
            "model_load_s": max(0.0, float(model_load_s)),
            "startup_total_s": max(0.0, float(clock() - startup_start)),
        }
        self.ready = True

    @staticmethod
    def _resolve_asset_fingerprint(
        path: Path,
        expected: Optional[str],
        *,
        dependency_was_injected: bool,
        label: str,
    ) -> str:
        # Tests inject in-memory dependencies and use explicit synthetic identities.
        # Production dependencies are always content-hashed; an optional CLI value
        # is an expected hash, never an unchecked identity override.
        if dependency_was_injected and expected is not None:
            return expected
        actual = fingerprint_asset(path)
        if expected is not None and expected != actual:
            raise ValueError(
                f"configured {label} fingerprint does not match the asset: "
                f"expected={expected}, actual={actual}"
            )
        return actual

    def _validate_assets(self) -> None:
        for attribute in ("d", "ntotal", "metric_type"):
            if not hasattr(self.index, attribute):
                raise ValueError(f"FAISS index is missing {attribute}")
        if int(self.index.d) <= 0 or int(self.index.ntotal) <= 0:
            raise ValueError("FAISS index must have positive dimension and ntotal")
        if hasattr(self.index, "is_trained") and not bool(self.index.is_trained):
            raise ValueError("FAISS index is not trained")
        if self.config.topk > int(self.index.ntotal):
            raise ValueError("configured topk exceeds index ntotal")
        corpus_rows = len(self.corpus)
        if corpus_rows != int(self.index.ntotal):
            raise ValueError(
                "corpus row count must equal FAISS index ntotal to preserve document IDs"
            )

    def _configure_backend(self) -> None:
        index_type = type(self.index).__name__.lower()
        if self.config.index_backend == "flat":
            if "flat" not in index_type:
                raise ValueError(
                    f"flat backend requires a Flat FAISS index, received {type(self.index).__name__}"
                )
            return
        if "ivfpq" not in index_type:
            raise ValueError(
                f"ivfpq backend requires IndexIVFPQ, received {type(self.index).__name__}"
            )
        if hasattr(self.faiss, "ParameterSpace"):
            self.faiss.ParameterSpace().set_index_parameter(
                self.index, "nprobe", self.config.nprobe
            )
        else:
            self.index.nprobe = self.config.nprobe
        if int(getattr(self.index, "nprobe", -1)) != self.config.nprobe:
            raise ValueError("failed to apply the configured IVF nprobe")

    def _result_cache_key(self, query: str, topk: int) -> Tuple[Any, ...]:
        return (
            query,
            topk,
            self.index_fingerprint,
            self.model_fingerprint,
            self.retrieval_config_fingerprint,
        )

    def _embedding_cache_key(self, query: str) -> Tuple[Any, ...]:
        return (
            query,
            self.model_fingerprint,
            self.config.model_name,
            self.config.pooling_method,
            self.config.query_max_length,
            self.config.retrieval_encode_batch_size,
            "float32",
            "cpu",
        )

    def _base_metrics(self, topk: int) -> Dict[str, Any]:
        metrics = {name: 0.0 for name in TIMING_FIELDS}
        metrics.update({
            "request_id": uuid.uuid4().hex,
            "query_count": 0,
            "topk": topk,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "embedding_cache_hit_count": 0,
            "embedding_cache_miss_count": 0,
            "index_backend": self.config.index_backend,
            "nprobe": self.config.nprobe,
            "faiss_thread_count": self.config.faiss_thread_count,
            "result_ids": [],
        })
        return metrics

    def _normalize_queries(self, queries: Sequence[str]) -> List[str]:
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise InvalidRetrievalRequest("queries must be a list of non-empty strings")
        normalized = []
        for query in queries:
            if not isinstance(query, str):
                raise InvalidRetrievalRequest("queries must be a list of non-empty strings")
            stripped = query.strip()
            if not stripped:
                raise InvalidRetrievalRequest("queries must contain at least one non-empty string")
            normalized.append(stripped)
        if not normalized:
            raise InvalidRetrievalRequest("queries must contain at least one non-empty string")
        return normalized

    def retrieve(
        self,
        queries: Sequence[str],
        topk: Optional[int] = None,
        return_scores: bool = False,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
        request_started = self._clock()
        requested_topk = self.config.topk if topk is None else topk
        metrics = self._base_metrics(requested_topk)
        try:
            normalize_started = self._clock()
            if (
                isinstance(requested_topk, bool)
                or not isinstance(requested_topk, int)
                or requested_topk <= 0
            ):
                raise InvalidRetrievalRequest("topk must be a positive integer")
            if requested_topk > int(self.index.ntotal):
                raise InvalidRetrievalRequest("topk must not exceed index ntotal")
            normalized = self._normalize_queries(queries)
            metrics["query_count"] = len(normalized)
            metrics["query_normalization_s"] = self._clock() - normalize_started

            slots: List[Optional[Tuple[Tuple[int, ...], Tuple[float, ...]]]] = [
                None for _ in normalized
            ]
            missing_positions = []
            cache_started = self._clock()
            for position, query in enumerate(normalized):
                if not self.result_cache.enabled:
                    missing_positions.append(position)
                    continue
                hit, cached = self.result_cache.get(
                    self._result_cache_key(query, requested_topk)
                )
                if hit:
                    metrics["cache_hit_count"] += 1
                    slots[position] = cached
                else:
                    metrics["cache_miss_count"] += 1
                    missing_positions.append(position)
            metrics["cache_lookup_s"] += self._clock() - cache_started

            embeddings: Dict[int, np.ndarray] = {}
            encode_positions = []
            cache_started = self._clock()
            for position in missing_positions:
                query = normalized[position]
                if not self.embedding_cache.enabled:
                    encode_positions.append(position)
                    continue
                hit, cached = self.embedding_cache.get(
                    self._embedding_cache_key(query)
                )
                if hit:
                    metrics["embedding_cache_hit_count"] += 1
                    embeddings[position] = np.asarray(cached, dtype=np.float32).copy()
                else:
                    metrics["embedding_cache_miss_count"] += 1
                    encode_positions.append(position)
            metrics["cache_lookup_s"] += self._clock() - cache_started

            batch_size = self.config.retrieval_encode_batch_size
            for start in range(0, len(encode_positions), batch_size):
                positions = encode_positions[start:start + batch_size]
                batch_queries = [normalized[position] for position in positions]
                encode_started = self._clock()
                with self._encoder_lock:
                    encoded = self.encoder.encode(batch_queries)
                metrics["query_encoding_s"] += self._clock() - encode_started
                encoded = np.asarray(encoded, dtype=np.float32)
                if encoded.ndim != 2 or encoded.shape != (len(positions), int(self.index.d)):
                    raise RuntimeError(
                        "encoder returned an unexpected embedding shape: "
                        f"{encoded.shape}, expected {(len(positions), int(self.index.d))}"
                    )
                if not np.isfinite(encoded).all():
                    raise RuntimeError("encoder returned non-finite embeddings")
                for row, position in enumerate(positions):
                    embedding = np.ascontiguousarray(encoded[row], dtype=np.float32)
                    embeddings[position] = embedding
                    cache_started = self._clock()
                    self.embedding_cache.put(
                        self._embedding_cache_key(normalized[position]), embedding.copy()
                    )
                    metrics["cache_lookup_s"] += self._clock() - cache_started

            if missing_positions:
                matrix = np.ascontiguousarray(
                    np.stack([embeddings[position] for position in missing_positions]),
                    dtype=np.float32,
                )
                search_started = self._clock()
                scores, identifiers = self.index.search(matrix, requested_topk)
                metrics["faiss_search_s"] = self._clock() - search_started
                scores = np.asarray(scores)
                identifiers = np.asarray(identifiers)
                expected_shape = (len(missing_positions), requested_topk)
                if scores.shape != expected_shape or identifiers.shape != expected_shape:
                    raise RuntimeError("FAISS search returned unexpected result cardinality")
                if not np.isfinite(scores).all():
                    raise RuntimeError("FAISS search returned non-finite scores")
                if (identifiers < 0).any() or (identifiers >= int(self.index.ntotal)).any():
                    raise RuntimeError("FAISS search returned an invalid document ID")
                for row, position in enumerate(missing_positions):
                    value = (
                        tuple(int(item) for item in identifiers[row]),
                        tuple(float(item) for item in scores[row]),
                    )
                    slots[position] = value
                    cache_started = self._clock()
                    self.result_cache.put(
                        self._result_cache_key(normalized[position], requested_topk), value
                    )
                    metrics["cache_lookup_s"] += self._clock() - cache_started

            if any(slot is None for slot in slots):
                raise RuntimeError("internal retrieval result assembly failed")
            completed_slots = [slot for slot in slots if slot is not None]
            result_ids = [list(slot[0]) for slot in completed_slots]
            metrics["result_ids"] = result_ids
            flat_ids = [identifier for row in result_ids for identifier in row]
            fetch_started = self._clock()
            with self._document_lock:
                flat_documents = self._document_loader(self.corpus, flat_ids)
            metrics["document_fetch_s"] = self._clock() - fetch_started
            if len(flat_documents) != len(flat_ids):
                raise RuntimeError("corpus loader returned unexpected document cardinality")

            format_started = self._clock()
            response: List[List[Dict[str, Any]]] = []
            offset = 0
            for slot in completed_slots:
                row_documents = flat_documents[offset:offset + requested_topk]
                offset += requested_topk
                if return_scores:
                    response.append([
                        {"document": document, "score": score}
                        for document, score in zip(row_documents, slot[1])
                    ])
                else:
                    response.append(list(row_documents))
            metrics["response_format_s"] = self._clock() - format_started
            metrics["request_total_s"] = self._clock() - request_started
            self._validate_metrics(metrics)
            self.stats.record(metrics)
            return response, metrics
        except Exception:
            metrics["request_total_s"] = max(0.0, self._clock() - request_started)
            self.stats.record(metrics, error=True)
            raise

    @staticmethod
    def _validate_metrics(metrics: Dict[str, Any]) -> None:
        for name in TIMING_FIELDS:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid timing metric {name}: {value!r}")

    def health(self) -> Dict[str, Any]:
        resolved_index_path = self.config.index_path.expanduser().resolve()
        return {
            "ready": self.ready,
            "index_type": type(self.index).__name__,
            "index_backend": self.config.index_backend,
            "index_path": str(resolved_index_path),
            "index_file_size_bytes": (
                resolved_index_path.stat().st_size
                if resolved_index_path.is_file()
                else None
            ),
            "index_ntotal": int(self.index.ntotal),
            "index_dimension": int(self.index.d),
            "metric_type": _metric_name(int(self.index.metric_type), self.faiss),
            "metric_type_code": int(self.index.metric_type),
            "corpus_row_count": len(self.corpus),
            "model_path": str(self.config.model_path.expanduser().resolve()),
            "faiss_thread_count": self.config.faiss_thread_count,
            "nprobe": self.config.nprobe,
            "cache_enabled": self.config.cache_enabled,
            "result_cache_capacity": self.config.result_cache_capacity,
            "embedding_cache_capacity": self.config.embedding_cache_capacity,
            "cache_configuration": {
                "result": self.result_cache.snapshot(),
                "embedding": self.embedding_cache.snapshot(),
            },
            "retrieval_encode_batch_size": self.config.retrieval_encode_batch_size,
            "startup_load_timings_s": dict(self.startup_load_timings_s),
            "process_rss_bytes": self._rss_reader(),
            "index_fingerprint": self.index_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
            "server_start_time": self.stats.start_time,
        }

    def cumulative_stats(self) -> Dict[str, Any]:
        snapshot = self.stats.snapshot()
        snapshot.update({
            "index_backend": self.config.index_backend,
            "nprobe": self.config.nprobe,
            "faiss_thread_count": self.config.faiss_thread_count,
            "result_cache": self.result_cache.snapshot(),
            "embedding_cache": self.embedding_cache.snapshot(),
            "process_rss_bytes": self._rss_reader(),
        })
        return snapshot


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    return_metrics: bool = False


def create_app(engine: OptimizedRetrieverEngine) -> FastAPI:
    app = FastAPI(title="Search-R1 Phase-6 CPU Retriever")

    @app.exception_handler(RequestValidationError)
    async def validation_error_endpoint(
        request: Request, error: RequestValidationError
    ):
        # FastAPI validation happens before the route handler. Count these
        # failures explicitly while delegating to the stock 422 response so the
        # existing endpoint contract is unchanged.
        metrics = engine._base_metrics(engine.config.topk)
        engine.stats.record(metrics, error=True)
        return await request_validation_exception_handler(request, error)

    @app.post("/retrieve")
    def retrieve_endpoint(request: QueryRequest) -> Dict[str, Any]:
        try:
            result, metrics = engine.retrieve(
                request.queries, request.topk, request.return_scores
            )
        except InvalidRetrievalRequest as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        payload: Dict[str, Any] = {"result": result}
        if request.return_metrics:
            payload["metrics"] = metrics
        return payload

    @app.get("/healthz")
    def health_endpoint() -> Dict[str, Any]:
        return engine.health()

    @app.get("/stats")
    def stats_endpoint() -> Dict[str, Any]:
        return engine.cumulative_stats()

    return app


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=os.environ.get(
        "PHASE6_INDEX_PATH", "/workspace/searchr1-assets/wiki18/e5_Flat.index"
    ))
    parser.add_argument("--corpus-path", default=os.environ.get(
        "PHASE6_CORPUS_PATH", "/workspace/searchr1-assets/wiki18/wiki-18.jsonl"
    ))
    parser.add_argument("--model-path", "--retriever-model", dest="model_path", default=os.environ.get(
        "PHASE6_E5_MODEL_PATH", "/workspace/searchr1-assets/models/e5-base-v2"
    ))
    parser.add_argument("--host", default=os.environ.get("PHASE6_RETRIEVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("PHASE6_RETRIEVER_PORT", 8100))
    parser.add_argument("--top-k", type=int, default=_env_int("PHASE6_RETRIEVER_TOPK", 3))
    parser.add_argument("--faiss-thread-count", type=int, default=_env_int("PHASE6_FAISS_THREADS", 8))
    parser.add_argument("--retrieval-encode-batch-size", type=int, default=_env_int(
        "PHASE6_ENCODE_BATCH_SIZE", 32
    ))
    parser.add_argument("--index-backend", choices=VALID_BACKENDS, default=os.environ.get(
        "PHASE6_INDEX_BACKEND", "flat"
    ))
    parser.add_argument("--ivf-nprobe", type=int, default=(
        int(os.environ["PHASE6_IVF_NPROBE"])
        if "PHASE6_IVF_NPROBE" in os.environ else None
    ))
    parser.add_argument("--result-cache-capacity", type=int, default=_env_int(
        "PHASE6_RESULT_CACHE_CAPACITY", 0
    ))
    parser.add_argument("--embedding-cache-capacity", type=int, default=_env_int(
        "PHASE6_EMBEDDING_CACHE_CAPACITY", 0
    ))
    parser.add_argument("--index-fingerprint", default=os.environ.get("PHASE6_INDEX_FINGERPRINT"))
    parser.add_argument("--model-fingerprint", default=os.environ.get("PHASE6_MODEL_FINGERPRINT"))
    parser.add_argument("--corpus-fingerprint", default=os.environ.get("PHASE6_CORPUS_FINGERPRINT"))
    cache_default = _env_bool("PHASE6_CACHE_ENABLED", False)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-enabled", dest="cache_enabled", action="store_true")
    cache_group.add_argument("--cache-disabled", dest="cache_enabled", action="store_false")
    parser.set_defaults(cache_enabled=cache_default)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    return ServerConfig(
        index_path=Path(args.index_path),
        corpus_path=Path(args.corpus_path),
        model_path=Path(args.model_path),
        host=args.host,
        port=args.port,
        topk=args.top_k,
        faiss_thread_count=args.faiss_thread_count,
        retrieval_encode_batch_size=args.retrieval_encode_batch_size,
        index_backend=args.index_backend,
        nprobe=args.ivf_nprobe,
        cache_enabled=args.cache_enabled,
        result_cache_capacity=args.result_cache_capacity,
        embedding_cache_capacity=args.embedding_cache_capacity,
        index_fingerprint=args.index_fingerprint,
        model_fingerprint=args.model_fingerprint,
        corpus_fingerprint=args.corpus_fingerprint,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    engine = OptimizedRetrieverEngine(config)
    health = engine.health()
    print(json.dumps(health, indent=2, sort_keys=True), flush=True)
    import uvicorn

    uvicorn.run(create_app(engine), host=config.host, port=config.port, workers=1)


if __name__ == "__main__":
    main()
