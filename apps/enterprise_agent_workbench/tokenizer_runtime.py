"""Fail-closed, CPU-only tokenizer loading for evidence compression."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import WorkbenchSettings


TOKENIZER_CONFIG_ARTIFACT = "tokenizer_config.json"
TOKENIZER_VOCAB_ARTIFACTS = frozenset(
    {
        "tokenizer.json",
        "vocab.json",
        "vocab.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
)
TOKENIZER_OPTIONAL_ARTIFACTS = frozenset(
    {
        "config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer_config.json",
    }
)
QWEN_TOKENIZER_CLASSES = frozenset({"Qwen2Tokenizer", "Qwen2TokenizerFast"})


@dataclass(frozen=True)
class TokenizerRuntime:
    tokenizer: Any
    mode: Literal["exact", "approximate_test"]
    configured_path: str
    tokenizer_class: str
    artifact_fingerprint: str
    vocabulary_size: int
    pad_token_id: int | None
    eos_token_id: int | None

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_mode": self.mode,
            "tokenizer_path": self.configured_path,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_artifact_fingerprint": self.artifact_fingerprint,
            "tokenizer_vocabulary_size": self.vocabulary_size,
            "tokenizer_pad_token_id": self.pad_token_id,
            "tokenizer_eos_token_id": self.eos_token_id,
        }


def _sanitized_tokenizer_path(path: Path) -> str:
    parts = path.parts[-3:]
    return ".../" + "/".join(parts)


def _artifact_paths(path: Path) -> list[Path]:
    config = path / TOKENIZER_CONFIG_ARTIFACT
    if not config.is_file():
        raise FileNotFoundError(
            f"tokenizer directory is missing {TOKENIZER_CONFIG_ARTIFACT}"
        )
    vocabulary = sorted(
        candidate
        for name in TOKENIZER_VOCAB_ARTIFACTS
        if (candidate := path / name).is_file()
    )
    if not vocabulary:
        expected = ", ".join(sorted(TOKENIZER_VOCAB_ARTIFACTS))
        raise FileNotFoundError(
            "tokenizer directory has no supported vocabulary artifact; "
            f"expected one of: {expected}"
        )
    selected = set(vocabulary)
    selected.update(
        candidate
        for name in TOKENIZER_OPTIONAL_ARTIFACTS
        if (candidate := path / name).is_file()
    )
    selected.update(path.glob("*.py"))
    return sorted(selected, key=lambda candidate: candidate.name)


def tokenizer_artifact_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for artifact in _artifact_paths(path):
        digest.update(artifact.name.encode("utf-8"))
        digest.update(b"\0")
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _vocabulary_size(tokenizer: Any) -> int:
    try:
        size = len(tokenizer)
    except (TypeError, AttributeError):
        size = getattr(tokenizer, "vocab_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("loaded tokenizer does not expose a positive vocabulary size")
    return size


def _ensure_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError(
                "loaded tokenizer has neither pad_token_id nor eos_token_id"
            )
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = eos_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = int(eos_token_id)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("loaded tokenizer has an invalid pad_token_id")
    return pad_token_id


def _default_loader(path: str, **kwargs: Any) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised in live setup
        raise RuntimeError(
            "transformers is required for exact live tokenizer loading"
        ) from error
    return AutoTokenizer.from_pretrained(path, **kwargs)


def load_exact_tokenizer(
    path: Path | str,
    *,
    loader: Callable[..., Any] | None = None,
) -> TokenizerRuntime:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            "WORKBENCH_TOKENIZER_PATH must be an existing local directory"
        )
    fingerprint = tokenizer_artifact_fingerprint(resolved)
    try:
        tokenizer = (loader or _default_loader)(
            str(resolved),
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError("failed to load the configured local tokenizer") from error
    tokenizer_class = type(tokenizer).__name__
    if tokenizer_class not in QWEN_TOKENIZER_CLASSES:
        expected = ", ".join(sorted(QWEN_TOKENIZER_CLASSES))
        raise ValueError(
            "WORKBENCH_TOKENIZER_PATH must load the checkpoint's Qwen2 "
            f"tokenizer ({expected}); loaded {tokenizer_class}"
        )
    pad_token_id = _ensure_pad_token_id(tokenizer)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_token_id = int(eos_token_id)
    return TokenizerRuntime(
        tokenizer=tokenizer,
        mode="exact",
        configured_path=_sanitized_tokenizer_path(resolved),
        tokenizer_class=tokenizer_class,
        artifact_fingerprint=fingerprint,
        vocabulary_size=_vocabulary_size(tokenizer),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )


def approximate_test_runtime(tokenizer: Any) -> TokenizerRuntime:
    tokenizer_class = type(tokenizer).__name__
    fingerprint = hashlib.sha256(
        f"approximate_test:{tokenizer_class}:v1".encode("utf-8")
    ).hexdigest()
    return TokenizerRuntime(
        tokenizer=tokenizer,
        mode="approximate_test",
        configured_path="development://explicit-approximate-tokenizer",
        tokenizer_class=tokenizer_class,
        artifact_fingerprint=fingerprint,
        vocabulary_size=max(1, int(getattr(tokenizer, "vocab_size", 1))),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )


def load_tokenizer_runtime(
    settings: WorkbenchSettings,
    *,
    loader: Callable[..., Any] | None = None,
) -> TokenizerRuntime:
    if settings.tokenizer_path is not None:
        return load_exact_tokenizer(settings.tokenizer_path, loader=loader)
    if not settings.allow_approx_tokenizer:
        raise RuntimeError(
            "live Workbench startup requires WORKBENCH_TOKENIZER_PATH; "
            "set WORKBENCH_ALLOW_APPROX_TOKENIZER=true only for explicit "
            "deterministic development/tests"
        )
    from .tools.research_search import DeterministicTextTokenizer

    return approximate_test_runtime(DeterministicTextTokenizer())


__all__ = [
    "TokenizerRuntime",
    "approximate_test_runtime",
    "load_exact_tokenizer",
    "load_tokenizer_runtime",
    "tokenizer_artifact_fingerprint",
]
