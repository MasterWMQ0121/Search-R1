import asyncio
import json
import sqlite3
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.phase5_observation_context.evidence_compressor import (
    EvidenceCompressor,
    wrap_observation,
)
from apps.enterprise_agent_workbench.tokenizer_runtime import TokenizerRuntime
from apps.enterprise_agent_workbench.tool_registry import (
    ToolRegistry,
    ToolSpec,
    build_default_registry,
)
from apps.enterprise_agent_workbench.tools.campaign_api import (
    BusinessExecutionContext,
    CampaignAPI,
    IdempotencyConflict,
    UpdateCampaignBudgetInput,
    WriteActionOutput,
    WriteNotAuthorized,
    initialize_demo_database,
)
from apps.enterprise_agent_workbench.tools.enterprise_kb import (
    EnterpriseKnowledgeBase,
)
from apps.enterprise_agent_workbench.tools.merchant_analytics import MerchantAnalytics
from apps.enterprise_agent_workbench.tools.research_search import (
    EVIDENCE_COMPRESSOR_POLICY,
    Phase5EvidenceAdapter,
    ResearchSearchTool,
    evidence_compressor_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "apps" / "enterprise_agent_workbench"
DOCS_DIR = APP_DIR / "fixtures" / "enterprise_docs"
SEED_PATH = APP_DIR / "fixtures" / "merchant_seed.json"


@pytest.fixture()
def demo_database(tmp_path):
    database = tmp_path / "merchant.sqlite3"
    initialize_demo_database(database, SEED_PATH)
    return database


def _context(decision="approve"):
    return BusinessExecutionContext(
        thread_id="thread-1",
        user_id="operator-1",
        role="operator",
        authorization_granted=True,
        approval_decision=decision,
    )


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _OneCallClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return _Response(self.payload)


class _FakeCompressor:
    def __init__(self):
        self.calls = []

    def compress(self, query, passages):
        self.calls.append((query, passages))
        return {
            "content": "Compressed evidence about campaign benchmarks.",
            "documents_returned": len(passages),
            "policy_output_observation_tokens": 12,
        }


class _CharacterTokenizer:
    """Tiny exact-contract tokenizer that makes wrapper accounting transparent."""

    vocab_size = 1_114_112
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(identifier) for identifier in token_ids)


def _research_tool():
    client = _OneCallClient(
        {
            "result": [
                [
                    {
                        "document": {
                            "id": "wiki:42",
                            "contents": "Campaign measurement\nA useful external fact.",
                        },
                        "score": 0.9,
                    }
                ]
            ]
        }
    )
    compressor = _FakeCompressor()
    return ResearchSearchTool(
        "http://retriever/retrieve", compressor, client=client
    ), client, compressor


def test_enterprise_kb_search_is_deterministic_bounded_and_path_safe():
    kb = EnterpriseKnowledgeBase(DOCS_DIR, snippet_limit=160)

    first = kb.search({"query": "maximum campaign budget increase approval", "top_k": 3})
    second = kb.search({"query": "maximum campaign budget increase approval", "top_k": 3})

    assert [source.source_id for source in first.sources] == [
        source.source_id for source in second.sources
    ]
    assert first.sources
    assert first.sources[0].source_id.startswith("KB-POLICY-001:")
    assert all(len(source.snippet) <= 160 for source in first.sources)
    assert all("/" not in (source.document_path or "") for source in first.sources)
    assert str(ROOT) not in json.dumps(first.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_research_search_makes_exactly_one_post_and_returns_sources():
    tool, client, compressor = _research_tool()

    result = await tool.search({"query": "campaign ROI research", "top_k": 3})

    assert len(client.calls) == 1
    assert client.calls[0][1] == {
        "queries": ["campaign ROI research"],
        "topk": 3,
        "return_scores": True,
    }
    assert len(compressor.calls) == 1
    assert result.failure_status is None
    assert result.compressed_evidence.startswith("Compressed evidence")
    assert result.sources[0].source_id == "RESEARCH-wiki:42"
    assert result.retrieval_latency_s >= 0
    assert result.compression_latency_s >= 0


@pytest.mark.asyncio
async def test_research_uses_injected_tokenizer_and_bounds_complete_observation():
    tokenizer = _CharacterTokenizer()
    runtime = TokenizerRuntime(
        tokenizer=tokenizer,
        mode="exact",
        configured_path=".../actor/global_step_20",
        tokenizer_class=type(tokenizer).__name__,
        artifact_fingerprint="a" * 64,
        vocabulary_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    adapter = Phase5EvidenceAdapter(
        tokenizer_runtime=runtime, max_observation_tokens=256
    )
    client = _OneCallClient(
        {
            "result": [
                [
                    {
                        "document": {
                            "id": "wiki:bounded",
                            "contents": (
                                "Campaign measurement\n"
                                + "Campaign ROI evidence improved materially. " * 30
                            ),
                        },
                        "score": 0.95,
                    }
                ]
            ]
        }
    )
    tool = ResearchSearchTool(
        "http://retriever/retrieve", adapter, client=client
    )

    result = await tool.search({"query": "campaign ROI evidence", "top_k": 3})

    assert isinstance(adapter._compressor, EvidenceCompressor)
    assert adapter._compressor.tokenizer is tokenizer
    assert len(client.calls) == 1
    assert client.calls[0][1] == {
        "queries": ["campaign ROI evidence"],
        "topk": 3,
        "return_scores": True,
    }
    assert not {"answer", "ground_truth", "target"}.intersection(
        client.calls[0][1]
    )
    assert result.failure_status is None
    wrapped_tokens = tokenizer.encode(
        wrap_observation(result.compressed_evidence), add_special_tokens=False
    )
    assert len(wrapped_tokens) <= 256
    metrics = result.compression_metrics
    assert metrics["policy_output_observation_tokens"] == len(wrapped_tokens)
    assert metrics["raw_retrieved_observation_tokens"] > 256
    assert metrics["tokenizer_mode"] == "exact"
    assert metrics["tokenizer_class"] == "_CharacterTokenizer"
    assert metrics["tokenizer_artifact_fingerprint"] == "a" * 64
    assert metrics["evidence_compressor_policy"] == EVIDENCE_COMPRESSOR_POLICY
    assert metrics["evidence_compressor_fingerprint"] == (
        evidence_compressor_fingerprint()
    )
    assert metrics["max_evidence_token_budget"] == 256


def test_analytics_uses_allowlisted_operations_and_detects_fixture_roi_decline(
    demo_database,
):
    analytics = MerchantAnalytics(demo_database)
    result = analytics.execute(
        "compare_periods",
        {
            "campaign_id": "C102",
            "previous_start": "2026-08-01",
            "previous_end": "2026-08-07",
            "current_start": "2026-08-08",
            "current_end": "2026-08-14",
        },
    )

    assert result.derived_metrics["roi_fractional_change"] < 0
    assert result.sources[0].source_id == "ANALYTICS:C102:compare_periods"
    with pytest.raises(ValueError, match="unsupported analytics operation"):
        analytics.execute("SELECT * FROM campaigns", {})
    with pytest.raises(ValidationError):
        analytics.execute(
            "campaign_performance_summary",
            {
                "campaign_id": "C102' OR 1=1 --",
                "start_date": "2026-08-01",
                "end_date": "2026-08-14",
            },
        )


def test_business_write_requires_approval_before_any_side_effect(demo_database):
    api = CampaignAPI(demo_database)
    request = {
        "campaign_id": "C102",
        "daily_budget": 1200,
        "idempotency_key": "budget-C102-1200",
    }

    with pytest.raises(WriteNotAuthorized):
        api.execute("update_campaign_budget", request)
    with pytest.raises(WriteNotAuthorized):
        api.execute("update_campaign_budget", request, context=_context("reject"))

    state = api.execute("get_campaign", {"campaign_id": "C102"})
    assert state.records[0]["daily_budget"] == 1000
    with sqlite3.connect(demo_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


def test_business_write_is_transactional_audited_and_idempotent(demo_database):
    api = CampaignAPI(demo_database)
    request = {
        "campaign_id": "C102",
        "daily_budget": 1200,
        "idempotency_key": "budget-C102-1200",
    }

    first = api.execute("update_campaign_budget", request, context=_context())
    replay = api.execute("update_campaign_budget", request, context=_context())

    assert first.before_state["daily_budget"] == 1000
    assert first.after_state["daily_budget"] == 1200
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.audit_event_id == first.audit_event_id
    with sqlite3.connect(demo_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0] == 1

    with pytest.raises(IdempotencyConflict):
        api.execute(
            "update_campaign_budget",
            {**request, "daily_budget": 1100},
            context=_context(),
        )
    assert api.execute("get_campaign", {"campaign_id": "C102"}).records[0][
        "daily_budget"
    ] == 1200


def test_edited_approved_arguments_execute_only_the_edited_value(demo_database):
    api = CampaignAPI(demo_database)
    result = api.execute(
        "update_campaign_budget",
        {
            "campaign_id": "C102",
            "daily_budget": 1200,
            "idempotency_key": "edited-budget-1200",
        },
        context=_context("edit"),
    )
    assert result.after_state["daily_budget"] == 1200


@pytest.mark.asyncio
async def test_registry_metadata_is_safe_and_validates_inputs(demo_database):
    kb = EnterpriseKnowledgeBase(DOCS_DIR)
    research, _, _ = _research_tool()
    registry = build_default_registry(
        kb, research, MerchantAnalytics(demo_database), CampaignAPI(demo_database)
    )

    metadata = registry.safe_metadata()
    names = {item["name"] for item in metadata}
    assert {
        "enterprise_kb_search",
        "compare_periods",
        "get_campaign",
        "update_campaign_budget",
    }.issubset(names)
    assert "handler" not in json.dumps(metadata)
    assert "credentials" not in json.dumps(metadata).lower()
    with pytest.raises(ValidationError):
        await registry.execute(
            "get_campaign", {"campaign_id": "bad campaign id"}
        )
    output = await registry.execute("get_campaign", {"campaign_id": "C102"})
    assert output["records"][0]["campaign_id"] == "C102"


@pytest.mark.asyncio
async def test_side_effect_timeout_reconciles_real_transaction_outcome(demo_database):
    campaign_api = CampaignAPI(demo_database)

    def slow_write(request, *, context):
        time.sleep(0.05)
        return campaign_api.execute(
            "update_campaign_budget", request, context=context
        )

    registry = ToolRegistry(
        [
            ToolSpec(
                name="slow_budget_write",
                description="Test-only delayed transactional write.",
                input_model=UpdateCampaignBudgetInput,
                output_model=WriteActionOutput,
                category="business_write",
                risk_level="high",
                required_roles=frozenset({"operator"}),
                read_only=False,
                side_effecting=True,
                approval_required=True,
                timeout_seconds=0.01,
                idempotent=True,
                source_producing=False,
                enabled=True,
                handler=slow_write,
            )
        ]
    )
    started = time.perf_counter()
    output = await registry.execute(
        "slow_budget_write",
        {
            "campaign_id": "C102",
            "daily_budget": 1200,
            "idempotency_key": "slow-write-0001",
        },
        context=_context(),
    )

    assert time.perf_counter() - started >= 0.04
    assert output["after_state"]["daily_budget"] == 1200
    with sqlite3.connect(demo_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1


def test_initialize_refuses_implicit_overwrite(demo_database):
    with pytest.raises(FileExistsError, match="overwrite=True"):
        initialize_demo_database(demo_database, SEED_PATH)
