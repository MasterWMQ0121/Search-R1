from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.enterprise_agent_workbench.citations import (
    CitationValidationError,
    SourceRecord,
    citation_coverage,
    require_valid_citations,
    validate_citations,
)


def _source(source_id="S1", **overrides):
    payload = {
        "source_id": source_id,
        "title": "Advertising budget policy",
        "source_type": "enterprise_kb",
        "snippet": "Budget increases above the threshold require approval.",
        "section": "Approvals",
        "document_path": "/private/enterprise/advertising_budget_policy.md",
    }
    payload.update(overrides)
    return SourceRecord(**payload)


def test_valid_citations_and_safe_source_projection():
    source = _source()
    result = validate_citations("The policy requires approval [S1].", [source])
    assert result.valid
    assert result.cited_ids == ["S1"]
    public = source.public_dict()
    assert "document_path" not in public
    assert public["document_name"] == "advertising_budget_policy.md"


def test_unknown_and_duplicate_citations_fail_validation():
    result = validate_citations("Claim [MISSING].", [_source()])
    assert not result.valid
    assert result.unknown_ids == ["MISSING"]
    with pytest.raises(CitationValidationError, match="unknown citation IDs"):
        require_valid_citations("Claim [MISSING].", [_source()])

    duplicate = validate_citations("Claim [S1].", [_source(), _source()])
    assert not duplicate.valid
    assert "duplicate source IDs" in duplicate.errors[0]


def test_source_id_and_snippet_are_bounded():
    with pytest.raises(ValidationError):
        _source(source_id="bad source id")
    with pytest.raises(ValidationError):
        _source(snippet="x" * 501)


def test_citation_coverage_is_deterministic_and_documents_limit():
    answer = "ROI declined according to campaign data [S1].\n\nBudget policy applies."
    metric = citation_coverage(answer)
    assert metric["evidence_dependent_sections"] == 2
    assert metric["cited_evidence_sections"] == 1
    assert metric["coverage"] == 0.5
    assert "does not establish" in metric["method"]
