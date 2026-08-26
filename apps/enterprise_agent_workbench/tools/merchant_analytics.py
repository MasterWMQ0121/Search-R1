"""Structured, parameterized analytics over the deterministic merchant database."""

from __future__ import annotations

import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enterprise_kb import SourceRecord


class _DateRangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "_DateRangeInput":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("analytics date range may not exceed 366 days")
        return self


class CampaignPerformanceSummaryInput(_DateRangeInput):
    pass


class ChannelBreakdownInput(_DateRangeInput):
    pass


class ConversionFunnelInput(_DateRangeInput):
    pass


class ComparePeriodsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    current_start: date
    current_end: date
    previous_start: date
    previous_end: date

    @model_validator(mode="after")
    def validate_ranges(self) -> "ComparePeriodsInput":
        if self.current_end < self.current_start:
            raise ValueError("current_end must not precede current_start")
        if self.previous_end < self.previous_start:
            raise ValueError("previous_end must not precede previous_start")
        return self


class ROIAnomalyDetectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    as_of_date: date | None = None
    current_days: int = Field(default=7, ge=1, le=90)
    reference_days: int = Field(default=7, ge=1, le=90)
    decline_threshold_fraction: float = Field(default=0.10, ge=0.0, le=1.0)


class AnalyticsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    query_identifier: str
    rows: list[dict[str, Any]]
    derived_metrics: dict[str, Any]
    sources: list[SourceRecord]
    execution_latency_s: float = Field(ge=0.0)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _row_dictionary(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class MerchantAnalytics:
    """Allowlisted analytics operations; callers can never provide SQL."""

    OPERATIONS = frozenset(
        {
            "campaign_performance_summary",
            "compare_periods",
            "channel_breakdown",
            "conversion_funnel",
            "roi_anomaly_detection",
            "campaign_current_state",
        }
    )

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(f"merchant database does not exist: {self.database_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=5.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _period_summary(
        connection: sqlite3.Connection,
        campaign_id: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(spend), 0.0) AS spend,
                   COALESCE(SUM(impressions), 0) AS impressions,
                   COALESCE(SUM(clicks), 0) AS clicks,
                   COALESCE(SUM(conversions), 0) AS conversions,
                   COALESCE(SUM(revenue), 0.0) AS revenue
            FROM campaign_daily_metrics
            WHERE campaign_id = ? AND date BETWEEN ? AND ?
            """,
            (campaign_id, start_date.isoformat(), end_date.isoformat()),
        ).fetchone()
        result = _row_dictionary(row)
        result.update(
            {
                "campaign_id": campaign_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "roi": _safe_ratio(float(result["revenue"]), float(result["spend"])),
                "ctr": _safe_ratio(float(result["clicks"]), float(result["impressions"])),
                "conversion_rate": _safe_ratio(
                    float(result["conversions"]), float(result["clicks"])
                ),
            }
        )
        return result

    @staticmethod
    def _source(operation: str, campaign_id: str, snippet: str) -> SourceRecord:
        return SourceRecord(
            source_id=f"ANALYTICS:{campaign_id}:{operation}",
            source_type="merchant_analytics",
            title=f"Campaign {campaign_id} analytics",
            section=operation,
            snippet=snippet[:500],
            metadata={"campaign_id": campaign_id, "operation": operation},
        )

    def execute(self, operation: str, request: BaseModel | dict[str, Any]) -> AnalyticsOutput:
        if operation not in self.OPERATIONS:
            raise ValueError(f"unsupported analytics operation: {operation}")
        started = time.perf_counter()
        if not isinstance(request, BaseModel):
            model_by_operation: dict[str, type[BaseModel]] = {
                "campaign_performance_summary": CampaignPerformanceSummaryInput,
                "compare_periods": ComparePeriodsInput,
                "channel_breakdown": ChannelBreakdownInput,
                "conversion_funnel": ConversionFunnelInput,
                "roi_anomaly_detection": ROIAnomalyDetectionInput,
            }
            if operation == "campaign_current_state":
                from .campaign_api import CampaignCurrentStateInput

                model_by_operation[operation] = CampaignCurrentStateInput
            request = model_by_operation[operation].model_validate(request)
        campaign_id = str(request.campaign_id)
        with self._connect() as connection:
            rows, metrics = self._execute_with_connection(connection, operation, request)
        snippet = (
            f"Structured {operation} result for {campaign_id}; "
            f"{len(rows)} result row(s)."
        )
        return AnalyticsOutput(
            operation=operation,
            query_identifier=f"merchant-analytics-v1:{operation}",
            rows=rows,
            derived_metrics=metrics,
            sources=[self._source(operation, campaign_id, snippet)],
            execution_latency_s=max(0.0, time.perf_counter() - started),
        )

    def _execute_with_connection(
        self, connection: sqlite3.Connection, operation: str, request: BaseModel
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if operation == "campaign_performance_summary":
            summary = self._period_summary(
                connection, request.campaign_id, request.start_date, request.end_date
            )
            return [summary], {key: summary[key] for key in ("roi", "ctr", "conversion_rate")}
        if operation == "compare_periods":
            current = self._period_summary(
                connection, request.campaign_id, request.current_start, request.current_end
            )
            previous = self._period_summary(
                connection, request.campaign_id, request.previous_start, request.previous_end
            )
            current_roi = current["roi"]
            previous_roi = previous["roi"]
            delta = (
                None
                if current_roi is None or previous_roi is None
                else current_roi - previous_roi
            )
            fraction = (
                None
                if delta is None or previous_roi == 0
                else delta / previous_roi
            )
            return [previous, current], {
                "roi_absolute_change": delta,
                "roi_fractional_change": fraction,
                "revenue_change": current["revenue"] - previous["revenue"],
                "spend_change": current["spend"] - previous["spend"],
            }
        if operation == "channel_breakdown":
            records = connection.execute(
                """
                SELECT channel, SUM(spend) AS spend, SUM(impressions) AS impressions,
                       SUM(clicks) AS clicks, SUM(conversions) AS conversions,
                       SUM(revenue) AS revenue
                FROM campaign_daily_metrics
                WHERE campaign_id = ? AND date BETWEEN ? AND ?
                GROUP BY channel ORDER BY channel
                """,
                (
                    request.campaign_id,
                    request.start_date.isoformat(),
                    request.end_date.isoformat(),
                ),
            ).fetchall()
            rows = [_row_dictionary(row) for row in records]
            for row in rows:
                row["roi"] = _safe_ratio(float(row["revenue"]), float(row["spend"]))
            return rows, {"channel_count": len(rows)}
        if operation == "conversion_funnel":
            summary = self._period_summary(
                connection, request.campaign_id, request.start_date, request.end_date
            )
            row = {
                key: summary[key]
                for key in ("impressions", "clicks", "conversions", "revenue")
            }
            return [row], {
                "impression_to_click_rate": summary["ctr"],
                "click_to_conversion_rate": summary["conversion_rate"],
            }
        if operation == "roi_anomaly_detection":
            as_of = request.as_of_date
            if as_of is None:
                raw = connection.execute(
                    "SELECT MAX(date) FROM campaign_daily_metrics WHERE campaign_id = ?",
                    (request.campaign_id,),
                ).fetchone()[0]
                if raw is None:
                    raise ValueError(f"campaign has no metrics: {request.campaign_id}")
                as_of = date.fromisoformat(raw)
            current_start = as_of - timedelta(days=request.current_days - 1)
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=request.reference_days - 1)
            current = self._period_summary(
                connection, request.campaign_id, current_start, as_of
            )
            previous = self._period_summary(
                connection, request.campaign_id, previous_start, previous_end
            )
            decline = (
                None
                if current["roi"] is None or previous["roi"] in (None, 0)
                else (previous["roi"] - current["roi"]) / previous["roi"]
            )
            return [previous, current], {
                "roi_decline_fraction": decline,
                "anomaly_detected": (
                    decline is not None
                    and decline >= request.decline_threshold_fraction
                ),
                "decline_threshold_fraction": request.decline_threshold_fraction,
            }
        if operation == "campaign_current_state":
            row = connection.execute(
                """
                SELECT campaign_id, name, status, daily_budget, owner, updated_at
                FROM campaigns WHERE campaign_id = ?
                """,
                (request.campaign_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"campaign does not exist: {request.campaign_id}")
            return [_row_dictionary(row)], {}
        raise AssertionError(f"unhandled analytics operation: {operation}")
