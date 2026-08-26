from __future__ import annotations

import pytest

from apps.enterprise_agent_workbench.memory import (
    MemoryValidationError,
    PreferenceMemoryStore,
)


def test_preferences_are_explicit_persistent_and_deletable(tmp_path):
    database = tmp_path / "memory.sqlite"
    with PreferenceMemoryStore(database) as store:
        store.set_preference("org-a", "user-1", "preferred_kpi", "ROI")
        store.set_preference(
            "org-a", "user-1", "preferred_reporting_format", "concise"
        )
        assert store.get_preference("org-a", "user-1", "preferred_kpi") == "ROI"
        assert list(store.list_preferences("org-a", "user-1")) == [
            "preferred_kpi",
            "preferred_reporting_format",
        ]
        assert store.delete_preference("org-a", "user-1", "preferred_kpi")
        assert store.get_preference("org-a", "user-1", "preferred_kpi") is None

    with PreferenceMemoryStore(database) as reopened:
        assert reopened.list_preferences("org-a", "user-1") == {
            "preferred_reporting_format": "concise"
        }


def test_preference_namespaces_isolate_users_and_organizations(tmp_path):
    with PreferenceMemoryStore(tmp_path / "memory.sqlite") as store:
        store.set_preference("org-a", "user-1", "preferred_kpi", "ROI")
        store.set_preference("org-a", "user-2", "preferred_kpi", "CPA")
        store.set_preference("org-b", "user-1", "preferred_kpi", "ROAS")

        assert store.list_preferences("org-a", "user-1") == {"preferred_kpi": "ROI"}
        assert store.list_preferences("org-a", "user-2") == {"preferred_kpi": "CPA"}
        assert store.list_preferences("org-b", "user-1") == {"preferred_kpi": "ROAS"}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("api_token", "token"),
        ("password", "password"),
        ("entire_tool_output", "dump"),
        ("preferred_kpi", "Bearer abcdefghijklmnop"),
        ("preferred_kpi", "sk-abcdefghijklmnop"),
        ("preferred_kpi", {"credential": "secret"}),
        ("preferred_kpi", float("nan")),
    ],
)
def test_sensitive_or_unsupported_preferences_are_rejected(tmp_path, key, value):
    with PreferenceMemoryStore(tmp_path / "memory.sqlite") as store:
        with pytest.raises(MemoryValidationError):
            store.set_preference("org-a", "user-1", key, value)
        assert store.list_preferences("org-a", "user-1") == {}


def test_empty_namespace_is_rejected(tmp_path):
    with PreferenceMemoryStore(tmp_path / "memory.sqlite") as store:
        with pytest.raises(MemoryValidationError):
            store.set_preference("", "user-1", "preferred_kpi", "ROI")
