"""Explicit, namespaced long-term preference memory backed by SQLite."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any


ALLOWED_PREFERENCE_KEYS = frozenset(
    {
        "preferred_reporting_format",
        "default_date_range",
        "preferred_kpi",
        "default_merchant_id",
    }
)
_SENSITIVE_FRAGMENTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "private_key",
)
_SECRET_VALUE_PATTERN = re.compile(
    r"\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)


class MemoryValidationError(ValueError):
    pass


def _validate_namespace(organization_id: str, user_id: str) -> None:
    if not organization_id.strip() or not user_id.strip():
        raise MemoryValidationError("organization_id and user_id must be non-empty")


def _contains_sensitive_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS):
                return True
            if _contains_sensitive_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_field(item) for item in value)
    elif isinstance(value, str):
        return bool(_SECRET_VALUE_PATTERN.search(value))
    return False


def validate_preference(key: str, value: Any) -> None:
    normalized = key.strip().lower().replace("-", "_")
    if any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS):
        raise MemoryValidationError("sensitive preference keys are not permitted")
    if normalized not in ALLOWED_PREFERENCE_KEYS:
        raise MemoryValidationError(f"unsupported explicit preference key: {key}")
    if isinstance(value, bool):
        pass
    elif not isinstance(value, (str, int, float)) or value is None:
        raise MemoryValidationError("preference value must be a scalar string or number")
    if isinstance(value, str) and len(value) > 1_000:
        raise MemoryValidationError("preference value exceeds 1000 characters")
    if isinstance(value, float) and not math.isfinite(value):
        raise MemoryValidationError("preference number must be finite")
    if _contains_sensitive_field(value):
        raise MemoryValidationError("sensitive preference values are not permitted")


class PreferenceMemoryStore:
    """Stores only explicitly submitted safe preferences by organization/user."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.database_path), check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._setup()

    def _setup(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    organization_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (organization_id, user_id, preference_key)
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "PreferenceMemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def set_preference(
        self, organization_id: str, user_id: str, key: str, value: Any
    ) -> None:
        _validate_namespace(organization_id, user_id)
        normalized = key.strip().lower().replace("-", "_")
        validate_preference(normalized, value)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO user_preferences (
                    organization_id, user_id, preference_key, value_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (organization_id, user_id, preference_key)
                DO UPDATE SET value_json = excluded.value_json,
                              updated_at = CURRENT_TIMESTAMP
                """,
                (organization_id, user_id, normalized, encoded),
            )

    def get_preference(
        self, organization_id: str, user_id: str, key: str
    ) -> Any | None:
        _validate_namespace(organization_id, user_id)
        normalized = key.strip().lower().replace("-", "_")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT value_json FROM user_preferences
                WHERE organization_id = ? AND user_id = ? AND preference_key = ?
                """,
                (organization_id, user_id, normalized),
            ).fetchone()
        return None if row is None else json.loads(row["value_json"])

    def list_preferences(self, organization_id: str, user_id: str) -> dict[str, Any]:
        _validate_namespace(organization_id, user_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT preference_key, value_json FROM user_preferences
                WHERE organization_id = ? AND user_id = ?
                ORDER BY preference_key
                """,
                (organization_id, user_id),
            ).fetchall()
        return {row["preference_key"]: json.loads(row["value_json"]) for row in rows}

    def delete_preference(self, organization_id: str, user_id: str, key: str) -> bool:
        _validate_namespace(organization_id, user_id)
        normalized = key.strip().lower().replace("-", "_")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM user_preferences
                WHERE organization_id = ? AND user_id = ? AND preference_key = ?
                """,
                (organization_id, user_id, normalized),
            )
        return cursor.rowcount == 1
