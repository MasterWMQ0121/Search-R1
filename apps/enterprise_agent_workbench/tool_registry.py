"""Explicit, inspectable registry of workbench tools."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

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

_PLANNER_DESCRIPTIONS = {
    "enterprise_kb_search": "Search approved enterprise policies and procedures.",
    "research_search": "Search external evidence with the Search-R1 Retriever.",
    "campaign_performance_summary": "Summarize campaign performance for a date range.",
    "compare_periods": "Compare campaign performance across two date ranges.",
    "channel_breakdown": "Break down campaign performance by channel.",
    "conversion_funnel": "Analyze a campaign conversion funnel for a date range.",
    "roi_anomaly_detection": "Detect recent campaign ROI decline.",
    "campaign_current_state": "Read the current campaign analytics state.",
    "get_campaign": "Read the current campaign record.",
    "list_campaigns": "List campaigns, optionally filtered by status.",
    "get_budget_policy_status": "Check a proposed budget against policy limits.",
    "update_campaign_budget": "Change a campaign daily budget.",
    "pause_campaign": "Pause an active campaign.",
    "resume_campaign": "Resume a paused campaign.",
    "create_followup_task": "Create a follow-up task for a campaign.",
}

_PLANNER_CONSTRAINT_KEYS = (
    "enum",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
)


def _planner_field_contract(field_schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce one JSON Schema field to its useful Planner-facing contract."""

    variants = field_schema.get("anyOf")
    if isinstance(variants, list):
        non_null = [
            variant
            for variant in variants
            if isinstance(variant, dict) and variant.get("type") != "null"
        ]
        if len(non_null) == 1:
            field_schema = non_null[0]

    field_type = field_schema.get("type", "object")
    if isinstance(field_type, list):
        field_type = [item for item in field_type if item != "null"] or ["null"]
        if len(field_type) == 1:
            field_type = field_type[0]

    contract: dict[str, Any] = {"type": field_type}
    for key in _PLANNER_CONSTRAINT_KEYS:
        if key in field_schema:
            contract[key] = field_schema[key]
    return contract


def _concise_argument_error(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False, include_context=False):
        location = ".".join(str(part) for part in item["loc"]) or "arguments"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


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
    planner_description: str | None = None

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

    def planner_metadata(self) -> dict[str, Any]:
        """Return the compact LLM contract; never return control-plane fields."""

        schema = self.input_model.model_json_schema()
        required = set(schema.get("required", []))
        arguments = {}
        for name, field_schema in schema.get("properties", {}).items():
            if self.side_effecting and name == "idempotency_key":
                continue
            arguments[name] = {
                **_planner_field_contract(field_schema),
                "required": name in required,
            }
        return {
            "name": self.name,
            "description": self.planner_description or self.description,
            "category": self.category,
            "risk_level": self.risk_level,
            "approval_required": self.approval_required,
            "side_effecting": self.side_effecting,
            "arguments": arguments,
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

    @staticmethod
    def _validate_role(role: str) -> None:
        if role not in ALL_ROLES:
            raise ValueError(f"unknown Planner role: {role!r}")

    def planner_metadata(self, role: str) -> list[dict[str, Any]]:
        """Return enabled compact tool contracts visible to ``role`` only."""

        self._validate_role(role)
        return [
            spec.planner_metadata()
            for name in sorted(self._specs)
            for spec in [self._specs[name]]
            if spec.enabled and role in spec.required_roles
        ]

    def planner_action_ids(self, role: str) -> tuple[str, ...]:
        """Return canonical enabled tool IDs available to ``role``."""

        return tuple(item["name"] for item in self.planner_metadata(role))

    def validate_planner_arguments(
        self, role: str, action: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate LLM-provided business arguments using the real input model.

        Side-effecting schemas require an idempotency key at execution time, but
        that control-plane value must never be supplied by the Planner. A valid
        placeholder lets Pydantic check the remaining business fields here; the
        graph later injects the run-bound deterministic key before policy checks.
        """

        self._validate_role(role)
        spec = self._specs.get(action)
        if spec is None:
            raise ValueError(
                f"action {action!r} is not available to Planner role {role!r}"
            )
        if not spec.enabled or role not in spec.required_roles:
            raise ValueError(
                f"action {action!r} is not available to Planner role {role!r}"
            )
        if not isinstance(arguments, dict):
            raise ValueError("Planner arguments must be a JSON object")
        if spec.side_effecting and "idempotency_key" in arguments:
            raise ValueError(
                "idempotency_key is control-plane metadata and must not be "
                "provided by the Planner"
            )

        candidate = dict(arguments)
        if spec.side_effecting:
            candidate["idempotency_key"] = "planner-validation"
        try:
            validated = spec.input_model.model_validate(candidate)
        except ValidationError as error:
            raise ValueError(
                f"invalid arguments for {action!r}: {_concise_argument_error(error)}"
            ) from error
        return validated.model_dump(
            mode="json",
            exclude={"idempotency_key"} if spec.side_effecting else None,
        )

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
            planner_description=_PLANNER_DESCRIPTIONS["enterprise_kb_search"],
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
            planner_description=_PLANNER_DESCRIPTIONS["research_search"],
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
                planner_description=_PLANNER_DESCRIPTIONS[name],
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
                planner_description=_PLANNER_DESCRIPTIONS[name],
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
                planner_description=_PLANNER_DESCRIPTIONS[name],
            )
        )
    return ToolRegistry(specs)
