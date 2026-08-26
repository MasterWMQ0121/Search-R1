"""Explicit, inspectable registry of workbench tools."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from .tools.campaign_api import (
    BusinessExecutionContext,
    CampaignAPI,
    CampaignCurrentStateInput,
    CampaignReadOutput,
    CreateFollowupTaskInput,
    GetBudgetPolicyStatusInput,
    ListCampaignsInput,
    ListCampaignsOutput,
    PauseCampaignInput,
    ResumeCampaignInput,
    UpdateCampaignBudgetInput,
    WriteActionOutput,
)
from .tools.enterprise_kb import (
    EnterpriseKBSearchInput,
    EnterpriseKBSearchOutput,
    EnterpriseKnowledgeBase,
)
from .tools.merchant_analytics import (
    AnalyticsOutput,
    CampaignPerformanceSummaryInput,
    ChannelBreakdownInput,
    ComparePeriodsInput,
    ConversionFunnelInput,
    MerchantAnalytics,
    ROIAnomalyDetectionInput,
)
from .tools.research_search import (
    ResearchSearchInput,
    ResearchSearchOutput,
    ResearchSearchTool,
)


Handler = Callable[..., Any | Awaitable[Any]]
ALL_ROLES = frozenset({"viewer", "analyst", "operator", "admin"})
ANALYST_ROLES = frozenset({"analyst", "operator", "admin"})
WRITE_ROLES = frozenset({"operator", "admin"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    category: str
    risk_level: str
    required_roles: frozenset[str]
    read_only: bool
    side_effecting: bool
    approval_required: bool
    timeout_seconds: float
    idempotent: bool
    source_producing: bool
    enabled: bool
    handler: Handler

    @property
    def input_schema(self) -> type[BaseModel]:
        """Compatibility alias; graph code should prefer ``input_model``."""

        return self.input_model

    @property
    def output_schema(self) -> type[BaseModel]:
        return self.output_model

    def __post_init__(self) -> None:
        if self.read_only == self.side_effecting:
            raise ValueError(
                f"tool {self.name!r} must be exactly one of read-only or side-effecting"
            )
        if self.side_effecting and not self.idempotent:
            raise ValueError(f"side-effecting tool {self.name!r} must be idempotent")
        if self.timeout_seconds <= 0:
            raise ValueError(f"tool {self.name!r} timeout must be positive")

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "category": self.category,
            "risk_level": self.risk_level,
            "required_roles": sorted(self.required_roles),
            "read_only": self.read_only,
            "side_effecting": self.side_effecting,
            "approval_required": self.approval_required,
            "timeout_seconds": self.timeout_seconds,
            "idempotent": self.idempotent,
            "source_producing": self.source_producing,
            "enabled": self.enabled,
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def require(self, name: str) -> ToolSpec:
        return self.get(name)

    def safe_metadata(self) -> list[dict[str, Any]]:
        return [self._specs[name].safe_metadata() for name in sorted(self._specs)]

    def metadata(self) -> list[dict[str, Any]]:
        """Compatibility alias for callers written before ``safe_metadata``."""

        return self.safe_metadata()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: BusinessExecutionContext | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = self.require(name)
        if not spec.enabled:
            raise PermissionError(f"tool {name!r} is disabled")
        validated = spec.input_model.model_validate(arguments)
        positional = (validated,)
        keyword: dict[str, Any] = {}
        if spec.side_effecting:
            if context is None:
                raise PermissionError(
                    f"side-effecting tool {name!r} requires execution authorization"
                )
            keyword["context"] = context
        async def await_handler(awaitable: Awaitable[Any]) -> Any:
            task = asyncio.ensure_future(awaitable)
            if not spec.side_effecting:
                return await asyncio.wait_for(task, timeout=spec.timeout_seconds)

            # A timeout cannot cancel a synchronous write already running in a
            # worker thread.  Shield it and reconcile the real outcome instead
            # of reporting a failure while the transaction may still commit.
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), timeout=spec.timeout_seconds
                )
            except asyncio.TimeoutError:
                return await task
            except asyncio.CancelledError:
                # Process shutdown may cancel the graph task, but it still
                # cannot safely cancel an in-flight transactional worker.
                await asyncio.shield(task)
                raise

        if inspect.iscoroutinefunction(spec.handler):
            outcome = await await_handler(spec.handler(*positional, **keyword))
        else:
            outcome = await await_handler(
                asyncio.to_thread(spec.handler, *positional, **keyword)
            )
            if inspect.isawaitable(outcome):
                outcome = await await_handler(outcome)
        validated_output = spec.output_model.model_validate(outcome)
        return validated_output.model_dump(mode="json")


def _analytics_handler(
    analytics: MerchantAnalytics, operation: str, request: BaseModel
) -> AnalyticsOutput:
    return analytics.execute(operation, request)


def _campaign_read_handler(
    campaign_api: CampaignAPI, action: str, request: BaseModel
) -> BaseModel:
    return campaign_api.execute(action, request)


def _campaign_write_handler(
    campaign_api: CampaignAPI,
    action: str,
    request: BaseModel,
    *,
    context: BusinessExecutionContext | dict[str, Any],
) -> BaseModel:
    return campaign_api.execute(action, request, context=context)


def build_default_registry(
    enterprise_kb: EnterpriseKnowledgeBase,
    research_search: ResearchSearchTool,
    merchant_analytics: MerchantAnalytics,
    campaign_api: CampaignAPI,
) -> ToolRegistry:
    """Build the complete registry with no implicit or discoverable tools."""

    specs = [
        ToolSpec(
            "enterprise_kb_search",
            "Search approved enterprise policy and operating-procedure documents.",
            EnterpriseKBSearchInput,
            EnterpriseKBSearchOutput,
            "knowledge",
            "low",
            ALL_ROLES,
            True,
            False,
            False,
            5.0,
            True,
            True,
            True,
            enterprise_kb.search,
        ),
        ToolSpec(
            "research_search",
            "Retrieve and compress external/open-domain evidence with Search-R1.",
            ResearchSearchInput,
            ResearchSearchOutput,
            "research",
            "low",
            ALL_ROLES,
            True,
            False,
            False,
            15.0,
            True,
            True,
            True,
            research_search.search,
        ),
    ]
    analytics_inputs = {
        "campaign_performance_summary": CampaignPerformanceSummaryInput,
        "compare_periods": ComparePeriodsInput,
        "channel_breakdown": ChannelBreakdownInput,
        "conversion_funnel": ConversionFunnelInput,
        "roi_anomaly_detection": ROIAnomalyDetectionInput,
        "campaign_current_state": CampaignCurrentStateInput,
    }
    for name, input_schema in analytics_inputs.items():
        specs.append(
            ToolSpec(
                name,
                f"Run the structured merchant analytics operation {name}.",
                input_schema,
                AnalyticsOutput,
                "analytics",
                "low",
                ANALYST_ROLES,
                True,
                False,
                False,
                5.0,
                True,
                True,
                True,
                partial(_analytics_handler, merchant_analytics, name),
            )
        )
    read_specs = {
        "get_campaign": (CampaignCurrentStateInput, CampaignReadOutput),
        "list_campaigns": (ListCampaignsInput, ListCampaignsOutput),
        "get_budget_policy_status": (
            GetBudgetPolicyStatusInput,
            CampaignReadOutput,
        ),
    }
    for name, (input_schema, output_schema) in read_specs.items():
        specs.append(
            ToolSpec(
                name,
                f"Run the read-only campaign operation {name}.",
                input_schema,
                output_schema,
                "business_read",
                "low",
                ALL_ROLES,
                True,
                False,
                False,
                5.0,
                True,
                True,
                True,
                partial(_campaign_read_handler, campaign_api, name),
            )
        )
    write_inputs = {
        "update_campaign_budget": UpdateCampaignBudgetInput,
        "pause_campaign": PauseCampaignInput,
        "resume_campaign": ResumeCampaignInput,
        "create_followup_task": CreateFollowupTaskInput,
    }
    for name, input_schema in write_inputs.items():
        specs.append(
            ToolSpec(
                name,
                f"Propose and, after approval, execute campaign action {name}.",
                input_schema,
                WriteActionOutput,
                "business_write",
                "high" if name != "create_followup_task" else "medium",
                WRITE_ROLES,
                False,
                True,
                True,
                5.0,
                True,
                False,
                True,
                partial(_campaign_write_handler, campaign_api, name),
            )
        )
    return ToolRegistry(specs)
