from __future__ import annotations

from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from apps.enterprise_agent_workbench.config import (
    ENTERPRISE_DOCS_ROOT,
    MERCHANT_SEED_PATH,
    WorkbenchSettings,
)
from apps.enterprise_agent_workbench.graph import WorkbenchGraph
from apps.enterprise_agent_workbench.memory import PreferenceMemoryStore
from apps.enterprise_agent_workbench.model_client import FakeModelClient
from apps.enterprise_agent_workbench.tool_registry import build_default_registry
from apps.enterprise_agent_workbench.tools.campaign_api import (
    CampaignAPI,
    initialize_demo_database,
)
from apps.enterprise_agent_workbench.tools.enterprise_kb import EnterpriseKnowledgeBase
from apps.enterprise_agent_workbench.tools.merchant_analytics import MerchantAnalytics
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
    Phase5EvidenceAdapter,
    ResearchSearchTool,
)


@pytest.fixture
def graph_factory(tmp_path):
    resources = []

    def build(decisions, answer, *, role="analyst", max_graph_steps=40):
        suffix = len(resources)
        database_path = tmp_path / f"merchant-{suffix}.sqlite"
        initialize_demo_database(database_path, MERCHANT_SEED_PATH)
        memory = PreferenceMemoryStore(tmp_path / f"memory-{suffix}.sqlite")
        resources.append(memory)
        settings = replace(
            WorkbenchSettings.from_env(),
            data_dir=tmp_path / f"runtime-{suffix}",
            max_graph_steps=max_graph_steps,
            max_tool_calls=8,
            max_research_searches=2,
        )
        research = ResearchSearchTool(
            "http://127.0.0.1:9/retrieve",
            Phase5EvidenceAdapter(DeterministicTextTokenizer()),
            timeout_seconds=0.1,
        )
        registry = build_default_registry(
            EnterpriseKnowledgeBase(ENTERPRISE_DOCS_ROOT),
            research,
            MerchantAnalytics(database_path),
            CampaignAPI(database_path),
        )
        model = FakeModelClient(decisions, synthesized_answer=answer)
        graph = WorkbenchGraph(
            model_client=model,
            registry=registry,
            memory_store=memory,
            settings=settings,
        ).build(checkpointer=InMemorySaver())
        return {
            "graph": graph,
            "registry": registry,
            "memory": memory,
            "model": model,
            "settings": settings,
            "database_path": database_path,
            "role": role,
        }

    yield build
    for resource in resources:
        resource.close()
