"""Deterministic lexical search over the workbench Markdown fixtures."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DOCUMENT_IDS = {
    "advertising_budget_policy.md": "KB-POLICY-001",
    "campaign_optimization_sop.md": "KB-SOP-001",
    "promotion_compliance_policy.md": "KB-POLICY-002",
}


class SourceRecord(BaseModel):
    """A bounded, API-safe evidence record."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: str
    title: str
    section: str | None = None
    snippet: str = Field(max_length=800)
    score: float | None = None
    document_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnterpriseKBSearchInput(BaseModel):
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


class EnterpriseKBSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    sources: list[SourceRecord]
    result_count: int
    execution_latency_s: float = Field(ge=0.0)


class _Section(BaseModel):
    source_id: str
    document_name: str
    document_title: str
    section_title: str
    text: str


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug or "overview"


def _bounded_snippet(text: str, query_terms: set[str], limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    lowered = compact.casefold()
    positions = [lowered.find(term) for term in query_terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(compact), start + limit)
    start = max(0, end - limit)
    snippet = compact[start:end].strip()
    if start:
        snippet = "…" + snippet[1:]
    if end < len(compact):
        snippet = snippet[:-1] + "…"
    return snippet


class EnterpriseKnowledgeBase:
    """Small deterministic BM25-like index with no embedding dependency."""

    def __init__(self, documents_dir: Path | str, *, snippet_limit: int = 500):
        self.documents_dir = Path(documents_dir).expanduser().resolve()
        if not self.documents_dir.is_dir():
            raise FileNotFoundError(
                f"enterprise document directory does not exist: {self.documents_dir}"
            )
        if not 80 <= snippet_limit <= 800:
            raise ValueError("snippet_limit must be between 80 and 800 characters")
        self.snippet_limit = snippet_limit
        self._sections = self._load_sections()
        if not self._sections:
            raise ValueError("enterprise document directory contains no searchable sections")
        self._section_terms = [Counter(_terms(section.text)) for section in self._sections]
        self._average_length = sum(
            sum(term_counts.values()) for term_counts in self._section_terms
        ) / len(self._sections)
        self._document_frequency = Counter()
        for terms in self._section_terms:
            self._document_frequency.update(terms.keys())

    def _load_sections(self) -> list[_Section]:
        sections: list[_Section] = []
        for path in sorted(self.documents_dir.glob("*.md")):
            base_id = _DOCUMENT_IDS.get(
                path.name, f"KB-DOC-{path.stem.upper().replace('-', '_')}"
            )
            title = path.stem.replace("_", " ").title()
            current_title = "Overview"
            lines: list[str] = []

            def emit() -> None:
                text = "\n".join(lines).strip()
                if text:
                    sections.append(
                        _Section(
                            source_id=f"{base_id}:{_slug(current_title)}",
                            document_name=path.name,
                            document_title=title,
                            section_title=current_title,
                            text=text,
                        )
                    )

            for line in path.read_text(encoding="utf-8").splitlines():
                heading = _HEADING_RE.match(line)
                if heading:
                    if len(heading.group(1)) == 1:
                        title = heading.group(2).strip()
                        continue
                    emit()
                    current_title = heading.group(2).strip()
                    lines = []
                else:
                    lines.append(line)
            emit()
        return sections

    def search(
        self, request: EnterpriseKBSearchInput | dict[str, Any]
    ) -> EnterpriseKBSearchOutput:
        request = EnterpriseKBSearchInput.model_validate(request)
        started = time.perf_counter()
        query_terms = set(_terms(request.query))
        ranked: list[tuple[float, _Section]] = []
        section_count = len(self._sections)
        for section, term_counts in zip(self._sections, self._section_terms):
            length = max(1, sum(term_counts.values()))
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1.0
                    + (section_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(1.0, self._average_length)
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            if score > 0:
                coverage = sum(term in term_counts for term in query_terms) / max(
                    1, len(query_terms)
                )
                ranked.append((score + coverage, section))
        ranked.sort(key=lambda item: (-item[0], item[1].source_id))
        sources = [
            SourceRecord(
                source_id=section.source_id,
                source_type="enterprise_kb",
                title=section.document_title,
                section=section.section_title,
                snippet=_bounded_snippet(
                    section.text, query_terms, self.snippet_limit
                ),
                score=round(score, 8),
                document_path=section.document_name,
            )
            for score, section in ranked[: request.top_k]
        ]
        return EnterpriseKBSearchOutput(
            query=request.query,
            sources=sources,
            result_count=len(sources),
            execution_latency_s=max(0.0, time.perf_counter() - started),
        )
