import pytest

from apps.enterprise_agent_workbench import ui


def test_approval_arguments_require_a_json_object():
    assert ui.validate_approval_arguments_json('{"daily_budget": 1200}') == {
        "daily_budget": 1200
    }
    with pytest.raises(ValueError, match="JSON object"):
        ui.validate_approval_arguments_json("[1200]")
    with pytest.raises(ValueError, match="invalid JSON"):
        ui.validate_approval_arguments_json("{bad")


def test_trace_renderer_allowlists_safe_fields_and_never_hidden_reasoning():
    rows = ui.trace_render_rows(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "event": "tool_completed",
                "tool": "campaign_read",
                "status": "ok",
                "safe_output_summary": "Campaign C102 read",
                "hidden_chain_of_thought": "must not render",
                "secret_token": "must not render",
            }
        ]
    )
    assert rows[0]["tool_name"] == "campaign_read"
    assert rows[0]["safe_output_summary"] == "Campaign C102 read"
    assert "hidden_chain_of_thought" not in rows[0]
    assert "secret_token" not in rows[0]


def test_source_renderer_bounds_snippets_and_omits_internal_paths():
    rows = ui.source_render_rows(
        [
            {
                "source_id": "S1",
                "title": "Policy",
                "snippet": "x" * 900,
                "document_path": "/private/internal/policy.md",
            }
        ]
    )
    assert rows[0]["source_id"] == "S1"
    assert len(rows[0]["snippet"]) == 600
    assert "document_path" not in rows[0]


def test_planner_renderer_uses_only_user_visible_updates():
    assert ui.planner_update_from_event(
        {
            "event": "planner_decision",
            "user_visible_reason": "Need to inspect campaign metrics.",
            "hidden_reasoning": "private",
        }
    ) == "Need to inspect campaign metrics."
    assert ui.planner_update_from_event(
        {"event": "tool_started", "user_visible_reason": "not planner"}
    ) is None


class _ResumeAPI:
    def __init__(self, *, events, states, stream_error=False):
        self.events = list(events)
        self.states = list(states)
        self.stream_error = stream_error
        self.calls = []

    def resume(self, thread_id, decision, edited_arguments=None, feedback=None):
        self.calls.append(
            ("resume", thread_id, decision, edited_arguments, feedback)
        )
        return {"thread_id": thread_id, "status": "resumed", "run_id": "r1"}

    def stream(self, thread_id):
        self.calls.append(("stream", thread_id))
        if self.stream_error:
            raise RuntimeError("disconnected")
        yield from self.events

    def state(self, thread_id):
        self.calls.append(("state", thread_id))
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def test_resume_follows_sse_to_terminal_before_returning():
    api = _ResumeAPI(
        events=[
            {"event": "tool_completed"},
            {"event": "complete", "event_type": "run_completed"},
        ],
        states=[{"completed": True, "termination_reason": "completed"}],
    )

    outcome = ui.resume_and_follow(
        api,
        "thread-1",
        "approve",
        prior_state={"approval_request": {"action": "update_campaign_budget"}},
        sleep=lambda _: None,
    )

    assert outcome.lifecycle == "terminal"
    assert outcome.state["completed"] is True
    assert [call[0] for call in api.calls] == ["resume", "stream", "state"]
    assert len([call for call in api.calls if call[0] == "resume"]) == 1


def test_resume_follows_sse_to_a_new_pause_without_repeat_resume():
    new_approval = {"action": "create_followup_task", "arguments": {"title": "Review"}}
    api = _ResumeAPI(
        events=[
            {"event": "approval_requested"},
            {"event": "paused", "approval_request": new_approval},
        ],
        states=[{"completed": False, "approval_request": new_approval}],
    )

    outcome = ui.resume_and_follow(
        api,
        "thread-1",
        "edit",
        edited_arguments={"daily_budget": 1200},
        prior_state={"approval_request": {"action": "update_campaign_budget"}},
        sleep=lambda _: None,
    )

    assert outcome.lifecycle == "paused"
    assert outcome.state["approval_request"] == new_approval
    assert len([call for call in api.calls if call[0] == "resume"]) == 1


def test_disconnected_stream_polling_rejects_stale_approval_until_state_changes():
    old_approval = {"action": "update_campaign_budget", "arguments": {"daily_budget": 1200}}
    new_approval = {"action": "create_followup_task", "arguments": {"title": "Review"}}
    api = _ResumeAPI(
        events=[],
        stream_error=True,
        states=[
            {"completed": False, "approval_request": old_approval},
            {"completed": False, "approval_request": None},
            {"completed": False, "approval_request": new_approval},
        ],
    )

    outcome = ui.resume_and_follow(
        api,
        "thread-1",
        "reject",
        prior_state={"approval_request": old_approval},
        poll_attempts=3,
        poll_interval_seconds=0,
        sleep=lambda _: None,
    )

    assert outcome.lifecycle == "paused"
    assert outcome.state["approval_request"] == new_approval
    assert len([call for call in api.calls if call[0] == "state"]) == 3
    assert len([call for call in api.calls if call[0] == "resume"]) == 1


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"event": "approval_requested"}, None),
        ({"event": "paused"}, "paused"),
        ({"event": "complete"}, "terminal"),
        ({"event": "error"}, "terminal"),
    ],
)
def test_only_authoritative_sse_boundaries_are_settled(event, expected):
    assert ui.resumed_event_lifecycle(event) == expected
