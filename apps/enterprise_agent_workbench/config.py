"""Environment-backed configuration for the isolated workbench application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class WorkbenchSettings:
    """Runtime settings with conservative, local-only defaults."""

    data_dir: Path
    llm_base_url: str
    model_name: str
    retriever_url: str
    retriever_top_k: int
    evidence_token_budget: int
    tokenizer_path: Path | None
    allow_approx_tokenizer: bool
    max_graph_steps: int
    max_tool_calls: int
    max_research_searches: int
    max_planner_repairs: int
    tool_timeout_seconds: float
    summary_message_threshold: int

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def business_db_path(self) -> Path:
        return self.data_dir / "merchant_demo.sqlite"

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "preferences.sqlite"

    @classmethod
    def from_env(cls) -> "WorkbenchSettings":
        data_dir = Path(
            os.getenv("WORKBENCH_DATA_DIR", "~/.search_r1_workbench")
        ).expanduser().resolve()
        tokenizer_path_value = os.getenv("WORKBENCH_TOKENIZER_PATH", "").strip()
        return cls(
            data_dir=data_dir,
            llm_base_url=os.getenv(
                "WORKBENCH_LLM_BASE_URL", "http://127.0.0.1:8001/v1"
            ).rstrip("/"),
            model_name=os.getenv(
                "WORKBENCH_MODEL_NAME",
                "phase3-search-r1",
            ),
            retriever_url=os.getenv(
                "WORKBENCH_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"
            ),
            retriever_top_k=_positive_int("WORKBENCH_RETRIEVER_TOP_K", 3),
            evidence_token_budget=_positive_int(
                "WORKBENCH_EVIDENCE_TOKEN_BUDGET", 256
            ),
            tokenizer_path=(
                Path(tokenizer_path_value).expanduser().resolve()
                if tokenizer_path_value
                else None
            ),
            allow_approx_tokenizer=_boolean(
                "WORKBENCH_ALLOW_APPROX_TOKENIZER", False
            ),
            max_graph_steps=_positive_int("WORKBENCH_MAX_GRAPH_STEPS", 40),
            max_tool_calls=_positive_int("WORKBENCH_MAX_TOOL_CALLS", 8),
            max_research_searches=_positive_int(
                "WORKBENCH_MAX_RESEARCH_SEARCHES", 2
            ),
            max_planner_repairs=_positive_int("WORKBENCH_MAX_PLANNER_REPAIRS", 1),
            tool_timeout_seconds=_positive_float(
                "WORKBENCH_TOOL_TIMEOUT_SECONDS", 20.0
            ),
            summary_message_threshold=_positive_int(
                "WORKBENCH_SUMMARY_MESSAGE_THRESHOLD", 12
            ),
        )

    def ensure_runtime_directory(self) -> None:
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)


APP_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = APP_ROOT / "fixtures"
ENTERPRISE_DOCS_ROOT = FIXTURE_ROOT / "enterprise_docs"
MERCHANT_SEED_PATH = FIXTURE_ROOT / "merchant_seed.json"


__all__ = [
    "APP_ROOT",
    "ENTERPRISE_DOCS_ROOT",
    "FIXTURE_ROOT",
    "MERCHANT_SEED_PATH",
    "WorkbenchSettings",
]
