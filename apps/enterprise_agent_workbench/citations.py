"""Stable source records and deterministic citation validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


_CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})\]")
_EVIDENCE_TERMS = re.compile(
    r"\b(according|policy|data|metric|roi|budget|spend|revenue|campaign|retriev|analysis)\b|\d",
    re.IGNORECASE,
)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    title: str = Field(min_length=1, max_length=300)
    source_type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    snippet: str = Field(min_length=1, max_length=500)
    section: str | None = Field(default=None, max_length=200)
    document_path: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "snippet")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    def public_dict(self) -> dict[str, Any]:
        """Project a source without leaking an internal filesystem path."""

        payload = self.model_dump(exclude={"document_path"})
        payload["snippet"] = payload["snippet"][:500]
        if self.document_path:
            payload["document_name"] = Path(self.document_path).name
        return payload


class CitationValidationResult(BaseModel):
    valid: bool
    cited_ids: list[str]
    unknown_ids: list[str]
    uncited_source_ids: list[str]
    errors: list[str]


class CitationValidationError(ValueError):
    pass


def validate_citations(
    final_answer: str, sources: Sequence[SourceRecord | dict[str, Any]]
) -> CitationValidationResult:
    records = [
        item if isinstance(item, SourceRecord) else SourceRecord.model_validate(item)
        for item in sources
    ]
    ids = [record.source_id for record in records]
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    cited = list(dict.fromkeys(_CITATION_RE.findall(final_answer)))
    known = set(ids)
    unknown = sorted(set(cited) - known)
    errors: list[str] = []
    if duplicates:
        errors.append("duplicate source IDs: " + ", ".join(duplicates))
    if unknown:
        errors.append("unknown citation IDs: " + ", ".join(unknown))
    return CitationValidationResult(
        valid=not errors,
        cited_ids=cited,
        unknown_ids=unknown,
        uncited_source_ids=sorted(known - set(cited)),
        errors=errors,
    )


def require_valid_citations(
    final_answer: str, sources: Sequence[SourceRecord | dict[str, Any]]
) -> CitationValidationResult:
    result = validate_citations(final_answer, sources)
    if not result.valid:
        raise CitationValidationError("; ".join(result.errors))
    return result


def citation_coverage(final_answer: str) -> dict[str, int | float | str]:
    """Approximate coverage over evidence-looking paragraphs/bullets.

    This lexical metric cannot determine whether a citation actually supports a
    claim; it only checks whether sections containing data/policy cues include a
    syntactically valid citation marker.
    """

    sections = [part.strip() for part in re.split(r"\n\s*\n|\n(?=[-*]\s)", final_answer) if part.strip()]
    dependent = [section for section in sections if _EVIDENCE_TERMS.search(section)]
    cited = [section for section in dependent if _CITATION_RE.search(section)]
    coverage = 1.0 if not dependent else len(cited) / len(dependent)
    return {
        "evidence_dependent_sections": len(dependent),
        "cited_evidence_sections": len(cited),
        "coverage": coverage,
        "method": "lexical section approximation; does not establish evidentiary support",
    }
