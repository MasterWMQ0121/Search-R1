"""Transactional local mock campaign API with mandatory write authorization."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enterprise_kb import SourceRecord


MAX_DAILY_BUDGET = 1500.0
NO_MANAGER_APPROVAL_INCREASE_FRACTION = 0.10


class IdempotencyConflict(RuntimeError):
    pass


class WriteNotAuthorized(PermissionError):
    pass


class CampaignCurrentStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ListCampaignsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "paused", "archived"] | None = None
    limit: int = Field(default=50, ge=1, le=200)


class GetBudgetPolicyStatusInput(CampaignCurrentStateInput):
    proposed_daily_budget: float = Field(gt=0.0, le=1_000_000.0)


class _IdempotentCampaignInput(CampaignCurrentStateInput):
    idempotency_key: str = Field(min_length=8, max_length=128)

    @field_validator("idempotency_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip()
        if not value or any(character.isspace() for character in value):
            raise ValueError("idempotency_key must be non-empty and contain no whitespace")
        return value


class UpdateCampaignBudgetInput(_IdempotentCampaignInput):
    daily_budget: float = Field(gt=0.0, le=1_000_000.0)


class PauseCampaignInput(_IdempotentCampaignInput):
    reason: str = Field(min_length=1, max_length=500)


class ResumeCampaignInput(_IdempotentCampaignInput):
    reason: str = Field(min_length=1, max_length=500)


class CreateFollowupTaskInput(_IdempotentCampaignInput):
    title: str = Field(min_length=1, max_length=200)
    due_date: date | None = None


class BusinessExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    role: Literal["operator", "admin"]
    authorization_granted: bool
    approval_decision: Literal["approve", "edit", "reject"]


class CampaignReadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    records: list[dict[str, Any]]
    sources: list[SourceRecord]
    execution_latency_s: float = Field(ge=0.0)


class ListCampaignsOutput(CampaignReadOutput):
    pass


class WriteActionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    status: Literal["executed"]
    before_state: dict[str, Any] | None
    after_state: dict[str, Any]
    idempotency_key: str
    audit_event_id: str
    replayed: bool
    execution_latency_s: float = Field(ge=0.0)


class _SeedCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: str
    name: str
    status: str
    daily_budget: float
    owner: str
    updated_at: str


class _SeedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    campaign_id: str
    channel: str
    spend: float
    impressions: int
    clicks: int
    conversions: int
    revenue: float


class _SeedOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    campaign_id: str
    order_count: int
    gross_revenue: float
    refund_amount: float


class _MerchantSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaigns: list[_SeedCampaign]
    campaign_daily_metrics: list[_SeedMetric]
    orders: list[_SeedOrder]


_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE campaigns (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'archived')),
    daily_budget REAL NOT NULL CHECK (daily_budget > 0),
    owner TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE campaign_daily_metrics (
    date TEXT NOT NULL,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    channel TEXT NOT NULL,
    spend REAL NOT NULL,
    impressions INTEGER NOT NULL,
    clicks INTEGER NOT NULL,
    conversions INTEGER NOT NULL,
    revenue REAL NOT NULL,
    PRIMARY KEY (date, campaign_id, channel)
);
CREATE TABLE orders (
    date TEXT NOT NULL,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    order_count INTEGER NOT NULL,
    gross_revenue REAL NOT NULL,
    refund_amount REAL NOT NULL,
    PRIMARY KEY (date, campaign_id)
);
CREATE TABLE audit_log (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE followup_tasks (
    task_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    title TEXT NOT NULL,
    due_date TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def initialize_demo_database(
    database_path: Path | str,
    seed_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically initialize a local database from a validated JSON fixture."""

    database_path = Path(database_path).expanduser().resolve()
    seed_path = Path(seed_path).expanduser().resolve()
    if not seed_path.is_file():
        raise FileNotFoundError(f"merchant seed does not exist: {seed_path}")
    if database_path.exists() and not overwrite:
        raise FileExistsError(
            f"database already exists; pass overwrite=True explicitly: {database_path}"
        )
    seed = _MerchantSeed.model_validate_json(seed_path.read_text(encoding="utf-8"))
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_name(database_path.name + ".initializing")
    if temporary_path.exists():
        raise FileExistsError(f"stale initialization file exists: {temporary_path}")
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(_SCHEMA)
        connection.executemany(
            """
            INSERT INTO campaigns
                (campaign_id, name, status, daily_budget, owner, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.campaign_id,
                    item.name,
                    item.status,
                    item.daily_budget,
                    item.owner,
                    item.updated_at,
                )
                for item in seed.campaigns
            ],
        )
        connection.executemany(
            """
            INSERT INTO campaign_daily_metrics
                (date, campaign_id, channel, spend, impressions, clicks,
                 conversions, revenue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.date,
                    item.campaign_id,
                    item.channel,
                    item.spend,
                    item.impressions,
                    item.clicks,
                    item.conversions,
                    item.revenue,
                )
                for item in seed.campaign_daily_metrics
            ],
        )
        connection.executemany(
            """
            INSERT INTO orders
                (date, campaign_id, order_count, gross_revenue, refund_amount)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    item.date,
                    item.campaign_id,
                    item.order_count,
                    item.gross_revenue,
                    item.refund_amount,
                )
                for item in seed.orders
            ],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    except BaseException:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        connection.close()
    if database_path.exists():
        if not overwrite:
            temporary_path.unlink()
            raise FileExistsError(f"database appeared during initialization: {database_path}")
        database_path.unlink()
    os.replace(temporary_path, database_path)
    return database_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dictionary(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class CampaignAPI:
    READ_ACTIONS = frozenset(
        {"get_campaign", "list_campaigns", "get_budget_policy_status"}
    )
    WRITE_ACTIONS = frozenset(
        {
            "update_campaign_budget",
            "pause_campaign",
            "resume_campaign",
            "create_followup_task",
        }
    )

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(f"merchant database does not exist: {self.database_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _campaign(connection: sqlite3.Connection, campaign_id: str) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT campaign_id, name, status, daily_budget, owner, updated_at
            FROM campaigns WHERE campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"campaign does not exist: {campaign_id}")
        return _dictionary(row) or {}

    @staticmethod
    def _source(action: str, record: dict[str, Any]) -> SourceRecord:
        campaign_id = str(record.get("campaign_id", "campaigns"))
        return SourceRecord(
            source_id=f"BUSINESS:{campaign_id}:{action}",
            source_type="business_api",
            title=f"Campaign state: {campaign_id}",
            section=action,
            snippet=(
                f"Campaign {campaign_id}: status={record.get('status')}, "
                f"daily_budget={record.get('daily_budget')}."
            ),
            metadata={"campaign_id": campaign_id, "action": action},
        )

    def execute(
        self,
        action: str,
        request: BaseModel | dict[str, Any],
        *,
        context: BusinessExecutionContext | dict[str, Any] | None = None,
    ) -> BaseModel:
        if action in self.READ_ACTIONS:
            return self._execute_read(action, request)
        if action in self.WRITE_ACTIONS:
            return self._execute_write(action, request, context)
        raise ValueError(f"unsupported campaign action: {action}")

    def _execute_read(
        self, action: str, request: BaseModel | dict[str, Any]
    ) -> CampaignReadOutput:
        input_models: dict[str, type[BaseModel]] = {
            "get_campaign": CampaignCurrentStateInput,
            "list_campaigns": ListCampaignsInput,
            "get_budget_policy_status": GetBudgetPolicyStatusInput,
        }
        request = input_models[action].model_validate(request)
        started = time.perf_counter()
        with self._connect() as connection:
            if action == "get_campaign":
                records = [self._campaign(connection, request.campaign_id)]
            elif action == "list_campaigns":
                if request.status is None:
                    rows = connection.execute(
                        """
                        SELECT campaign_id, name, status, daily_budget, owner, updated_at
                        FROM campaigns ORDER BY campaign_id LIMIT ?
                        """,
                        (request.limit,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT campaign_id, name, status, daily_budget, owner, updated_at
                        FROM campaigns WHERE status = ? ORDER BY campaign_id LIMIT ?
                        """,
                        (request.status, request.limit),
                    ).fetchall()
                records = [_dictionary(row) or {} for row in rows]
            else:
                campaign = self._campaign(connection, request.campaign_id)
                current = float(campaign["daily_budget"])
                proposed = float(request.proposed_daily_budget)
                increase_fraction = (proposed - current) / current
                records = [
                    {
                        **campaign,
                        "proposed_daily_budget": proposed,
                        "increase_fraction": increase_fraction,
                        "within_budget_cap": proposed <= MAX_DAILY_BUDGET,
                        "policy_manager_approval_required": (
                            increase_fraction
                            > NO_MANAGER_APPROVAL_INCREASE_FRACTION
                        ),
                        "workbench_human_approval_required": True,
                        "compliant": proposed <= MAX_DAILY_BUDGET,
                    }
                ]
        sources = [self._source(action, record) for record in records]
        output_model = ListCampaignsOutput if action == "list_campaigns" else CampaignReadOutput
        return output_model(
            action=action,
            records=records,
            sources=sources,
            execution_latency_s=max(0.0, time.perf_counter() - started),
        )

    def _execute_write(
        self,
        action: str,
        request: BaseModel | dict[str, Any],
        context: BusinessExecutionContext | dict[str, Any] | None,
    ) -> WriteActionOutput:
        input_models: dict[str, type[BaseModel]] = {
            "update_campaign_budget": UpdateCampaignBudgetInput,
            "pause_campaign": PauseCampaignInput,
            "resume_campaign": ResumeCampaignInput,
            "create_followup_task": CreateFollowupTaskInput,
        }
        request = input_models[action].model_validate(request)
        if context is None:
            raise WriteNotAuthorized("business writes require an authorization context")
        context = BusinessExecutionContext.model_validate(context)
        if not context.authorization_granted or context.approval_decision not in {
            "approve",
            "edit",
        }:
            raise WriteNotAuthorized(
                "business write requires authorization and an approve/edit decision"
            )
        started = time.perf_counter()
        payload = request.model_dump(mode="json")
        request_hash = hashlib.sha256(
            json.dumps(
                {"action": action, "arguments": payload},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                """
                SELECT action, request_sha256, result_json
                FROM idempotency_keys WHERE idempotency_key = ?
                """,
                (request.idempotency_key,),
            ).fetchone()
            if prior is not None:
                if prior["action"] != action or prior["request_sha256"] != request_hash:
                    raise IdempotencyConflict(
                        "idempotency key was already used for a different request"
                    )
                replay = json.loads(prior["result_json"])
                replay["replayed"] = True
                replay["execution_latency_s"] = max(
                    0.0, time.perf_counter() - started
                )
                return WriteActionOutput.model_validate(replay)

            before, after = self._apply_write(connection, action, request)
            event_id = "AUDIT-" + hashlib.sha256(
                request.idempotency_key.encode("utf-8")
            ).hexdigest()[:24]
            result = WriteActionOutput(
                action=action,
                status="executed",
                before_state=before,
                after_state=after,
                idempotency_key=request.idempotency_key,
                audit_event_id=event_id,
                replayed=False,
                execution_latency_s=max(0.0, time.perf_counter() - started),
            )
            timestamp = _utc_now()
            result_json = json.dumps(result.model_dump(mode="json"), sort_keys=True)
            connection.execute(
                """
                INSERT INTO audit_log
                    (event_id, timestamp, thread_id, user_id, action, payload, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    timestamp,
                    context.thread_id,
                    context.user_id,
                    action,
                    json.dumps(payload, sort_keys=True),
                    result_json,
                ),
            )
            connection.execute(
                """
                INSERT INTO idempotency_keys
                    (idempotency_key, action, request_sha256, result_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request.idempotency_key,
                    action,
                    request_hash,
                    result_json,
                    timestamp,
                ),
            )
            connection.commit()
            return result

    def _apply_write(
        self, connection: sqlite3.Connection, action: str, request: BaseModel
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        before = self._campaign(connection, request.campaign_id)
        timestamp = _utc_now()
        if action == "update_campaign_budget":
            if request.daily_budget > MAX_DAILY_BUDGET:
                raise ValueError(
                    f"daily budget exceeds fixture policy cap of {MAX_DAILY_BUDGET:g}"
                )
            connection.execute(
                "UPDATE campaigns SET daily_budget = ?, updated_at = ? WHERE campaign_id = ?",
                (request.daily_budget, timestamp, request.campaign_id),
            )
            return before, self._campaign(connection, request.campaign_id)
        if action in {"pause_campaign", "resume_campaign"}:
            status = "paused" if action == "pause_campaign" else "active"
            connection.execute(
                "UPDATE campaigns SET status = ?, updated_at = ? WHERE campaign_id = ?",
                (status, timestamp, request.campaign_id),
            )
            return before, self._campaign(connection, request.campaign_id)
        if action == "create_followup_task":
            task_id = "TASK-" + hashlib.sha256(
                request.idempotency_key.encode("utf-8")
            ).hexdigest()[:16]
            after = {
                "task_id": task_id,
                "campaign_id": request.campaign_id,
                "title": request.title,
                "due_date": (
                    request.due_date.isoformat() if request.due_date else None
                ),
                "status": "open",
                "created_at": timestamp,
            }
            connection.execute(
                """
                INSERT INTO followup_tasks
                    (task_id, campaign_id, title, due_date, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                tuple(after.values()),
            )
            return None, after
        raise AssertionError(f"unhandled write action: {action}")
