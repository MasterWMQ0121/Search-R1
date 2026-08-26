import inspect

import pytest

from experiments.phase5_observation_context.evidence_compressor import (
    EvidenceCompressor,
    format_raw_passages,
    wrap_observation,
)


class CharacterTokenizer:
    """A reversible CPU-only tokenizer that charges one token per character."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


TOKENIZER = CharacterTokenizer()


def _hit(rank, body, *, title=None, score=1.0, document_id=None):
    title = title if title is not None else f"Title {rank}"
    return {
        "document": {
            "id": document_id if document_id is not None else f"doc-{rank}",
            "contents": f"{title}\n{body}",
        },
        "score": score,
    }


def _token_count(content):
    return len(TOKENIZER.encode(wrap_observation(content), add_special_tokens=False))


def _stable_result(result):
    return dict(result)


def test_raw_passage_format_is_byte_identical_to_search_r1_contract():
    passages = [
        _hit(1, "First line.\nSecond line.", title='"Quoted title"', score=0.9),
        _hit(2, "Other evidence.", title="Other", score=0.7),
    ]

    assert format_raw_passages(passages) == (
        'Doc 1(Title: "Quoted title") First line.\nSecond line.\n'
        "Doc 2(Title: Other) Other evidence.\n"
    )


def test_compression_is_deterministic_and_has_no_ground_truth_input():
    compressor = EvidenceCompressor(TOKENIZER, max_observation_tokens=256)
    passages = [
        _hit(1, "Paris is the capital of France. It is in Europe.", score=0.9),
        _hit(2, "France has many cities.", score=0.8),
    ]

    first = compressor.compress("capital France", passages)
    second = compressor.compress("capital France", passages)

    assert _stable_result(first) == _stable_result(second)
    assert set(inspect.signature(EvidenceCompressor.compress).parameters) == {
        "self",
        "query",
        "retrieval_result",
        "raw_observation",
    }
    assert "ground" not in inspect.signature(EvidenceCompressor.compress).__str__()


def test_query_relevant_sentence_outranks_irrelevant_sentence():
    passages = [_hit(
        1,
        "Unrelated material about painting. Paris is the capital of France.",
        title="Geography",
    )]
    compressor = EvidenceCompressor(TOKENIZER, max_observation_tokens=110)

    result = compressor.compress("France capital", passages)

    assert "Paris is the capital of France." in result["content"]
    assert "painting" not in result["content"]
    assert result["sentences_selected"] == 1


def test_equal_score_relevant_sentences_use_rank_prior():
    passages = [
        _hit(1, "alpha red.", title="T", score=0.5),
        _hit(2, "alpha tan.", title="T", score=0.5),
    ]
    one_document_budget = _token_count("Doc 1(Title: T) alpha red.")
    compressor = EvidenceCompressor(
        TOKENIZER, max_observation_tokens=one_document_budget
    )

    result = compressor.compress("alpha", passages)

    assert result["selected_document_ranks"] == [1]
    assert "alpha red." in result["content"]


def test_retrieval_score_prior_can_break_a_relevance_tie():
    passages = [
        _hit(1, "alpha red.", title="T", score=0.0),
        _hit(2, "alpha tan.", title="T", score=10.0),
    ]
    one_document_budget = _token_count("Doc 2(Title: T) alpha tan.")
    compressor = EvidenceCompressor(
        TOKENIZER, max_observation_tokens=one_document_budget
    )

    result = compressor.compress("alpha", passages)

    assert result["selected_document_ranks"] == [2]
    assert "alpha tan." in result["content"]


def test_duplicate_sentences_are_removed_and_alias_provenance_is_retained():
    passages = [
        _hit(1, "Shared alpha evidence.", score=0.9),
        _hit(2, "Shared alpha evidence.", score=0.8),
    ]

    result = EvidenceCompressor(TOKENIZER).compress("alpha", passages)

    assert result["sentences_considered"] == 2
    assert result["sentences_selected"] == 1
    assert result["content"].count("Shared alpha evidence.") == 1
    provenance = result["selected_sentence_provenance"][0]
    assert provenance["document_rank"] == 1
    assert provenance["duplicate_sources"] == [{
        "document_rank": 2,
        "document_id": "doc-2",
        "sentence_index": 0,
    }]


def test_diversity_pass_represents_each_relevant_document_when_they_fit():
    passages = [
        _hit(1, "alpha first.", title="A", score=0.9),
        _hit(2, "alpha second.", title="B", score=0.8),
        _hit(3, "alpha third.", title="C", score=0.7),
    ]

    result = EvidenceCompressor(TOKENIZER).compress("alpha", passages)

    assert result["selected_document_ranks"] == [1, 2, 3]
    assert result["documents_represented"] == 3


def test_complete_wrapper_not_only_content_obeys_exact_budget():
    passages = [_hit(
        1,
        "alpha one. alpha two. alpha three. alpha four. alpha five.",
        title="Evidence",
    )]
    budget = 100
    result = EvidenceCompressor(
        TOKENIZER, max_observation_tokens=budget
    ).compress("alpha", passages)

    assert _token_count(result["content"]) == result[
        "policy_output_observation_tokens"
    ]
    assert _token_count(result["content"]) <= budget
    assert len(result["content"]) < budget


def test_zero_overlap_uses_concise_leading_evidence_and_stays_nonempty():
    passages = [
        _hit(1, "A very long unrelated leading sentence here. Short fact.", title="A"),
        _hit(2, "Another fact.", title="B"),
    ]

    result = EvidenceCompressor(TOKENIZER, 140).compress("zebra", passages)

    assert result["zero_overlap_fallback_used"] is True
    assert result["content"]
    assert "Short fact." in result["content"]
    assert result["policy_output_observation_tokens"] <= 140


def test_no_complete_sentence_fit_uses_token_safe_partial_fallback():
    passages = [_hit(1, "x" * 500, title="Long")]
    budget = 80

    result = EvidenceCompressor(TOKENIZER, budget).compress("x", passages)

    assert result["partial_sentence_fallback_used"] is True
    assert result["sentences_selected"] == 0
    assert result["selected_document_ranks"] == [1]
    assert result["content"].startswith("Doc 1: Long ")
    assert result["policy_output_observation_tokens"] <= budget
    assert result["content"] != ""


def test_empty_documents_produce_an_explicit_budget_safe_marker():
    passages = [_hit(1, "", title="", score=0.0)]

    result = EvidenceCompressor(TOKENIZER, 80).compress("query", passages)

    assert result["content"] == "No nonempty evidence returned."
    assert result["documents_returned"] == 1
    assert result["documents_represented"] == 0
    assert result["sentences_considered"] == 0
    assert result["sentences_selected"] == 0
    assert result["zero_overlap_fallback_used"] is True
    assert result["partial_sentence_fallback_used"] is False


@pytest.mark.parametrize(
    ("passages", "message"),
    [
        ([{"score": 1.0}], "document mapping"),
        ([{"document": {"contents": 123}, "score": 1.0}], "text contents"),
        ([{"document": {"contents": "T\\nB"}}], "numeric score"),
        ([{"document": {"contents": "T\\nB"}, "score": float("nan")}], "non-finite"),
    ],
)
def test_malformed_documents_fail_loudly(passages, message):
    with pytest.raises(ValueError, match=message):
        EvidenceCompressor(TOKENIZER).compress("query", passages)


def test_original_document_sentence_positions_and_offsets_are_auditable():
    body = "Irrelevant first line.\nAlpha evidence second! Tail."
    passages = [_hit(1, body, document_id="wiki:42")]

    result = EvidenceCompressor(TOKENIZER, 120).compress("alpha", passages)

    assert result["selected_sentence_identifiers"] == [
        "document_rank=1;sentence_index=1"
    ]
    provenance = result["selected_sentence_provenance"][0]
    assert provenance["document_id"] == "wiki:42"
    assert provenance["sentence_index"] == 1
    assert body[provenance["char_start"]:provenance["char_end"]] == (
        "Alpha evidence second!"
    )


def test_raw_token_metric_uses_supplied_unwrapped_raw_observation():
    passages = [_hit(1, "alpha evidence.")]
    supplied_raw = "an exact raw observation"

    result = EvidenceCompressor(TOKENIZER).compress(
        "alpha", passages, raw_observation=supplied_raw
    )

    assert result["raw_retrieved_observation_tokens"] == _token_count(supplied_raw)


def test_invalid_budget_and_query_are_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        EvidenceCompressor(TOKENIZER, max_observation_tokens=0)
    with pytest.raises(TypeError, match="query must be text"):
        EvidenceCompressor(TOKENIZER).compress(None, [])
