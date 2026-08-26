from __future__ import annotations

from dataclasses import replace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from apps.enterprise_agent_workbench.api import (
    ThreadCreateRequest,
    build_default_service,
    create_app,
)
from apps.enterprise_agent_workbench.config import WorkbenchSettings
from apps.enterprise_agent_workbench.model_client import FakeModelClient, PlannerDecision
from apps.enterprise_agent_workbench.tools.research_search import (
    DeterministicTextTokenizer,
)


def planner(action, arguments=None, *, completed=False):
    return PlannerDecision(
        objective="Persist this thread safely.",
        next_action=action,
        arguments=arguments or {},
        completed=completed,
        user_visible_reason="Checkpoint persistence test.",
    )


@pytest.mark.asyncio
async def test_default_lifespan_owns_explicit_strict_sqlite_serializer(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WORKBENCH_DATA_DIR", str(tmp_path / "lifespan"))
    monkeypatch.setenv("WORKBENCH_ALLOW_APPROX_TOKENIZER", "true")
    monkeypatch.delenv("WORKBENCH_TOKENIZER_PATH", raising=False)
    app = create_app()

    async with app.router.lifespan_context(app):
        saver = app.state.workbench.graph.checkpointer
        assert isinstance(saver, AsyncSqliteSaver)
        assert saver.serde.pickle_fallback is False
        assert saver.serde._allowed_msgpack_modules is not True

    assert app.state.workbench.settings.checkpoint_path.is_file()


@pytest.mark.asyncio
async def test_sqlite_checkpoint_persists_and_isolates_threads(tmp_path):
    settings = replace(WorkbenchSettings.from_env(), data_dir=tmp_path)
    thread_id = None
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_path)
    ) as saver:
        await saver.setup()
        service = build_default_service(
            settings=settings,
            checkpointer=saver,
            model_client=FakeModelClient(
                [
                    planner("get_campaign", {"campaign_id": "C102"}),
                    planner("finalizer", completed=True),
                ],
                synthesized_answer=(
                    "Campaign C102 persisted [BUSINESS:C102:get_campaign]."
                ),
            ),
            tokenizer=DeterministicTextTokenizer(),
        )
        record = service.thread_directory.create(
            ThreadCreateRequest(
                user_id="user-a", organization_id="org-a", role="analyst"
            )
        )
        thread_id = record.thread_id
        await service.start_run(thread_id, "Show campaign C102.")
        await service.tasks[thread_id]
        snapshot = await service.graph.aget_state(service.config(thread_id))
        assert snapshot.values["completed"] is True
        await service.shutdown()

    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_path)
    ) as saver:
        await saver.setup()
        reopened = build_default_service(
            settings=settings,
            checkpointer=saver,
            model_client=FakeModelClient(),
            tokenizer=DeterministicTextTokenizer(),
        )
        assert reopened.thread_directory.get(thread_id).user_id == "user-a"
        persisted = await reopened.graph.aget_state(reopened.config(thread_id))
        assert persisted.values["final_answer"].startswith("Campaign C102")

        isolated = reopened.thread_directory.create(
            ThreadCreateRequest(
                user_id="user-b", organization_id="org-a", role="viewer"
            )
        )
        empty = await reopened.graph.aget_state(reopened.config(isolated.thread_id))
        assert not empty.values
        await reopened.shutdown()
