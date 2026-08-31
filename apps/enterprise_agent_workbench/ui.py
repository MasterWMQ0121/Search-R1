#!/usr/bin/env python3
"""Streamlit presentation layer for the Enterprise Agent Workbench.

The UI talks only to the public FastAPI contract.  It never imports LangGraph
state objects, model implementations, or business-tool implementation objects.
Keeping this boundary explicit makes the UI safe to run in a separate process
and keeps hidden model reasoning out of rendered state.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


DEFAULT_API_BASE_URL = os.environ.get(
    "WORKBENCH_API_BASE_URL", "http://127.0.0.1:8010"
).rstrip("/")

TRACE_COLUMNS = (
    "timestamp",
    "event",
    "node",
    "tool_name",
    "duration_ms",
    "status",
    "safe_input_summary",
    "safe_output_summary",
    "error_type",
    "retry_number",
)


def validate_approval_arguments_json(value: str) -> dict[str, Any]:
    """Parse edited approval arguments and require one JSON object."""

    if not isinstance(value, str):
        raise ValueError("edited approval arguments must be JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"edited approval arguments are invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("edited approval arguments must decode to a JSON object")
    return parsed


def trace_render_rows(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return bounded, presentation-safe trace rows.

    The backend is responsible for redaction.  The renderer additionally uses
    an allowlist so an accidentally added internal or reasoning field is not
    displayed by the workbench.
    """

    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        row = {column: event.get(column) for column in TRACE_COLUMNS}
        if row["event"] is None:
            row["event"] = event.get("event_type")
        if row["tool_name"] is None:
            row["tool_name"] = event.get("tool")
        rows.append(row)
    return rows


def source_render_rows(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project source records into the fields displayed by Streamlit."""

    rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        snippet = str(source.get("snippet", ""))
        rows.append(
            {
                "source_id": source.get("source_id"),
                "title": source.get("title"),
                "section": source.get("section"),
                "source_type": source.get("source_type"),
                "snippet": snippet[:600],
                "score": source.get("score"),
            }
        )
    return rows


def planner_update_from_event(event: Mapping[str, Any]) -> str | None:
    """Extract only a concise user-visible planner update from one SSE event."""

    if event.get("event", event.get("event_type")) not in {
        "planner_decision",
        "planner_repair",
    }:
        return None
    for key in ("user_visible_reason", "message", "safe_output_summary"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_object(response: Any, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object JSON response")
    return payload


def _thread_id_from_payload(payload: Mapping[str, Any]) -> str:
    value = payload.get("thread_id")
    if value is None and isinstance(payload.get("thread"), Mapping):
        value = payload["thread"].get("thread_id")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("thread creation response has no thread_id")
    return value.strip()


def resumed_event_lifecycle(event: Mapping[str, Any]) -> str | None:
    """Classify only authoritative segment-boundary SSE events.

    A trace event named ``approval_requested`` can precede LangGraph's actual
    interrupt checkpoint, so it is deliberately not considered settled.  The
    FastAPI service emits the distinct ``paused``/``complete``/``error`` SSE
    event names only after the resumed graph segment reaches its boundary.
    """

    event_name = str(event.get("event", ""))
    if event_name == "paused":
        return "paused"
    if event_name in {"complete", "error"}:
        return "terminal"
    return None


def state_lifecycle(state: Mapping[str, Any]) -> str | None:
    """Return the lifecycle represented by a public thread-state response."""

    if bool(state.get("completed")) or state.get("termination_reason"):
        return "terminal"
    if isinstance(state.get("approval_request"), Mapping):
        return "paused"
    return None


@dataclass(frozen=True)
class ResumeFollowResult:
    response: dict[str, Any]
    stream_events: list[dict[str, Any]]
    state: dict[str, Any]
    lifecycle: str


def resume_and_follow(
    api: "WorkbenchAPI",
    thread_id: str,
    decision: str,
    *,
    edited_arguments: Mapping[str, Any] | None = None,
    feedback: str | None = None,
    prior_state: Mapping[str, Any] | None = None,
    poll_attempts: int = 20,
    poll_interval_seconds: float = 0.10,
    sleep: Callable[[float], None] = time.sleep,
) -> ResumeFollowResult:
    """Resume once, then wait for the resumed segment's public boundary.

    The resumed SSE stream is authoritative.  Public state polling is a
    fallback for a disconnected stream.  Polling does not accept the exact
    stale approval request that was visible before the resume until it has
    first observed that request clear or change.
    """

    if poll_attempts < 1:
        raise ValueError("poll_attempts must be positive")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    response = api.resume(
        thread_id,
        decision,
        edited_arguments=edited_arguments,
        feedback=feedback,
    )
    events: list[dict[str, Any]] = []
    stream_lifecycle: str | None = None
    try:
        for event in api.stream(thread_id):
            event = dict(event)
            events.append(event)
            stream_lifecycle = resumed_event_lifecycle(event) or stream_lifecycle
    except Exception:
        # The public state endpoint is the documented recovery path.  The UI
        # does not inspect in-process task or LangGraph objects.
        stream_lifecycle = None

    previous_approval = (
        dict(prior_state["approval_request"])
        if isinstance((prior_state or {}).get("approval_request"), Mapping)
        else None
    )
    observed_clear_or_change = previous_approval is None
    latest: dict[str, Any] = {}
    for attempt in range(poll_attempts):
        latest = api.state(thread_id)
        lifecycle = state_lifecycle(latest)
        approval = latest.get("approval_request")
        if approval is None:
            observed_clear_or_change = True
        elif previous_approval is not None and approval != previous_approval:
            observed_clear_or_change = True

        if lifecycle == "terminal":
            return ResumeFollowResult(response, events, latest, "terminal")
        if lifecycle == "paused" and (
            stream_lifecycle == "paused" or observed_clear_or_change
        ):
            return ResumeFollowResult(response, events, latest, "paused")
        if stream_lifecycle == "terminal":
            # The segment boundary is authoritative even if a remote state
            # replica is fractionally behind the SSE event.
            return ResumeFollowResult(response, events, latest, "terminal")
        if attempt + 1 < poll_attempts:
            sleep(poll_interval_seconds)
    raise RuntimeError(
        "resume was accepted but the graph did not reach a terminal or paused state"
    )


@dataclass
class WorkbenchAPI:
    """Small synchronous client used by Streamlit's rerun execution model."""

    base_url: str = DEFAULT_API_BASE_URL
    timeout_seconds: float = 120.0
    tenant_id: str = "demo-org"
    role: str = "viewer"

    def _client(self):
        import httpx

        return httpx.Client(
            base_url=self.base_url.rstrip("/"), timeout=self.timeout_seconds
        )

    def health(self) -> dict[str, Any]:
        with self._client() as client:
            return _json_object(client.get("/healthz"), "health check")

    def tools(self) -> list[dict[str, Any]]:
        with self._client() as client:
            payload = client.get(
                "/api/tools",
                params={"tenant_id": self.tenant_id, "role": self.role},
            )
            payload.raise_for_status()
            value = payload.json()
        if isinstance(value, dict):
            value = value.get("tools", [])
        if not isinstance(value, list):
            raise RuntimeError("tool registry response must be a list")
        return [dict(item) for item in value if isinstance(item, Mapping)]

    def create_thread(self, user_id: str, organization_id: str, role: str) -> str:
        self.tenant_id = organization_id
        self.role = role
        with self._client() as client:
            payload = _json_object(
                client.post(
                    "/api/threads",
                    json={
                        "user_id": user_id,
                        "tenant_id": organization_id,
                        "role": role,
                    },
                ),
                "thread creation",
            )
        return _thread_id_from_payload(payload)

    def start_run(self, thread_id: str, task: str) -> dict[str, Any]:
        with self._client() as client:
            return _json_object(
                client.post(
                    f"/api/threads/{thread_id}/runs",
                    json={"tenant_id": self.tenant_id, "task": task},
                ),
                "run start",
            )

    def state(self, thread_id: str) -> dict[str, Any]:
        with self._client() as client:
            return _json_object(
                client.get(
                    f"/api/threads/{thread_id}/state",
                    params={"tenant_id": self.tenant_id},
                ),
                "thread state",
            )

    def history(self, thread_id: str) -> Any:
        with self._client() as client:
            response = client.get(
                f"/api/threads/{thread_id}/history",
                params={"tenant_id": self.tenant_id},
            )
            response.raise_for_status()
            return response.json()

    def trace(self, thread_id: str) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get(
                f"/api/threads/{thread_id}/trace",
                params={"tenant_id": self.tenant_id},
            )
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("events", payload.get("trace", []))
        if not isinstance(payload, list):
            raise RuntimeError("trace response must contain a list")
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def resume(
        self,
        thread_id: str,
        decision: str,
        edited_arguments: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "decision": decision,
        }
        if edited_arguments is not None:
            payload["edited_arguments"] = dict(edited_arguments)
        if feedback:
            payload["feedback"] = feedback
        with self._client() as client:
            return _json_object(
                client.post(f"/api/threads/{thread_id}/resume", json=payload),
                "thread resume",
            )

    def stream(self, thread_id: str):
        """Yield decoded SSE data objects for the active thread."""

        with self._client() as client:
            with client.stream(
                "GET",
                f"/api/threads/{thread_id}/stream",
                params={"tenant_id": self.tenant_id},
            ) as response:
                response.raise_for_status()
                event_name: str | None = None
                for line in response.iter_lines():
                    if not line:
                        event_name = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line.partition(":")[2].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line.partition(":")[2].strip()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = {"message": raw}
                    if not isinstance(payload, dict):
                        payload = {"value": payload}
                    payload.setdefault("event", event_name or "message")
                    yield payload

    def memories(self, user_id: str, organization_id: str) -> Any:
        with self._client() as client:
            response = client.get(
                f"/api/users/{user_id}/memories",
                params={"organization_id": organization_id},
            )
            response.raise_for_status()
            return response.json()


def _ensure_session_state(st: Any) -> None:
    defaults = {
        "thread_ids": [],
        "active_thread_id": None,
        "state": {},
        "trace": [],
        "stream_events": [],
        "approval_resume_inflight": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_approval_card(st: Any, api: WorkbenchAPI, thread_id: str, state: Mapping[str, Any]) -> None:
    approval = state.get("approval_request")
    if not isinstance(approval, Mapping):
        st.info("No action is awaiting approval.")
        return

    st.warning(f"Approval required: {approval.get('action_name', approval.get('action'))}")
    st.caption(str(approval.get("reason", "Review the proposed business action.")))
    st.json(
        {
            "risk_level": approval.get("risk_level"),
            "requesting_user": approval.get("requesting_user"),
            "requesting_role": approval.get("requesting_role"),
            "before_state": approval.get("before_state"),
            "arguments": approval.get("arguments", {}),
        }
    )
    arguments_text = st.text_area(
        "Edited arguments (JSON object)",
        value=json.dumps(approval.get("arguments", {}), indent=2, sort_keys=True),
        key=f"approval_arguments_{thread_id}",
    )
    feedback = st.text_input("Reviewer feedback", key=f"approval_feedback_{thread_id}")
    inflight = bool(st.session_state.approval_resume_inflight)

    def submit(
        decision: str, edited_arguments: Mapping[str, Any] | None = None
    ) -> None:
        st.session_state.approval_resume_inflight = True
        try:
            with st.spinner("Resuming the approved graph segment…"):
                outcome = resume_and_follow(
                    api,
                    thread_id,
                    decision,
                    edited_arguments=edited_arguments,
                    feedback=(
                        feedback
                        or ("Rejected by reviewer" if decision == "reject" else None)
                    ),
                    prior_state=state,
                )
            st.session_state.stream_events.extend(outcome.stream_events)
            st.session_state.state = outcome.state
            st.session_state.trace = api.trace(thread_id)
        except Exception as error:
            # Refresh once even after a transport failure: the resume request
            # may already have been accepted, so retaining the old approval
            # card would invite a duplicate request and a misleading 409.
            try:
                st.session_state.state = api.state(thread_id)
                st.session_state.trace = api.trace(thread_id)
            except Exception:
                pass
            st.error(f"Unable to finish approval resume: {error}")
        finally:
            st.session_state.approval_resume_inflight = False
        st.rerun()

    approve, edit, reject = st.columns(3)
    if approve.button("Approve", key=f"approve_{thread_id}", disabled=inflight):
        submit("approve")
    if edit.button(
        "Edit and approve", key=f"edit_{thread_id}", disabled=inflight
    ):
        try:
            edited = validate_approval_arguments_json(arguments_text)
            submit("edit", edited)
        except ValueError as error:
            st.error(str(error))
    if reject.button("Reject", key=f"reject_{thread_id}", disabled=inflight):
        submit("reject")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Enterprise Agent Workbench", layout="wide")
    _ensure_session_state(st)
    api = WorkbenchAPI()

    st.title("Enterprise Agent Workbench")
    st.caption(
        "Merchant and advertising operations prototype. Business writes affect "
        "only the local deterministic demo database."
    )

    with st.sidebar:
        st.header("Identity and thread")
        user_id = st.text_input("User ID", value="demo-operator")
        organization_id = st.text_input("Organization ID", value="demo-org")
        role = st.selectbox("Role", ["viewer", "analyst", "operator", "admin"], index=2)
        api.tenant_id = organization_id
        api.role = role
        thread_options = st.session_state.thread_ids
        selected = st.selectbox(
            "Thread",
            thread_options,
            index=(
                thread_options.index(st.session_state.active_thread_id)
                if st.session_state.active_thread_id in thread_options
                else 0
            ),
            placeholder="Create a thread",
        ) if thread_options else None
        if selected:
            st.session_state.active_thread_id = selected
        if st.button("New thread", use_container_width=True):
            try:
                thread_id = api.create_thread(user_id, organization_id, role)
                st.session_state.thread_ids.append(thread_id)
                st.session_state.active_thread_id = thread_id
                st.session_state.state = {}
                st.session_state.trace = []
                st.rerun()
            except Exception as error:  # Streamlit must render transport failures.
                st.error(f"Unable to create thread: {error}")
        st.divider()
        st.write(f"API: `{api.base_url}`")
        try:
            health = api.health()
            st.success(f"Backend: {health.get('status', 'available')}")
        except Exception as error:
            st.error(f"Backend unavailable: {error}")

    thread_id = st.session_state.active_thread_id
    main_panel, trace_panel = st.columns([2, 1])

    with main_panel:
        st.subheader("Task")
        task = st.text_area(
            "Business request",
            value=(
                "Analyze why Campaign C102's ROI declined during the last seven days, "
                "check the relevant budget policy, recommend corrective actions, and "
                "increase the daily budget to 1200 if the change is compliant."
            ),
            height=130,
        )
        if st.button("Run", type="primary", disabled=thread_id is None):
            try:
                api.start_run(thread_id, task)
                progress = st.status("Executing graph", expanded=True)
                streamed = []
                for event in api.stream(thread_id):
                    streamed.append(event)
                    update = planner_update_from_event(event)
                    if update:
                        progress.write(update)
                    elif event.get("event") in {
                        "node_started",
                        "tool_started",
                        "tool_completed",
                        "approval_requested",
                        "final_answer_created",
                        "run_failed",
                    }:
                        progress.write(
                            event.get("message")
                            or event.get("safe_output_summary")
                            or event.get("event")
                        )
                st.session_state.stream_events = streamed
                progress.update(label="Graph paused or completed", state="complete")
            except Exception as error:
                st.error(f"Run failed: {error}")
        if thread_id:
            try:
                st.session_state.state = api.state(thread_id)
                st.session_state.trace = api.trace(thread_id)
            except Exception as error:
                st.warning(f"Latest state is unavailable: {error}")

        state = st.session_state.state
        st.subheader("Final answer")
        st.write(state.get("final_answer") or "No final answer yet.")
        st.subheader("Sources")
        sources = state.get("sources", [])
        rows = source_render_rows(sources if isinstance(sources, list) else [])
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No sources recorded yet.")
        st.subheader("Tool results")
        tool_results = state.get("tool_results", [])
        st.json(tool_results if tool_results else [])

    with trace_panel:
        st.subheader("Current plan")
        st.json(st.session_state.state.get("plan", []))
        st.subheader("Execution timeline")
        rows = trace_render_rows(st.session_state.trace)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No trace events yet.")
        durations = [
            row["duration_ms"]
            for row in rows
            if isinstance(row.get("duration_ms"), (int, float))
        ]
        st.metric("Recorded latency", f"{sum(durations):.1f} ms")
        errors = [row for row in rows if row.get("error_type")]
        st.metric("Errors", len(errors))
        st.subheader("Approval")
        if thread_id:
            _render_approval_card(st, api, thread_id, st.session_state.state)
        else:
            st.info("Create a thread to submit a task.")

    memory_tab, audit_tab, tools_tab = st.tabs(
        ["Conversation memory", "Audit log", "Tool registry"]
    )
    with memory_tab:
        if thread_id:
            st.write("Thread summary")
            st.write(st.session_state.state.get("conversation_summary") or "Not summarized.")
        try:
            st.write("Explicit long-term preferences")
            st.json(api.memories(user_id, organization_id))
        except Exception as error:
            st.caption(f"Memories unavailable: {error}")
    with audit_tab:
        st.dataframe(trace_render_rows(st.session_state.trace), use_container_width=True)
    with tools_tab:
        try:
            st.dataframe(api.tools(), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Tool registry unavailable: {error}")


if __name__ == "__main__":
    main()
