from __future__ import annotations

import json

import pytest

from apps.enterprise_agent_workbench.config import WorkbenchSettings
from apps.enterprise_agent_workbench import tokenizer_preflight
from apps.enterprise_agent_workbench.tokenizer_runtime import (
    TokenizerRuntime,
    load_exact_tokenizer,
    load_tokenizer_runtime,
    tokenizer_artifact_fingerprint,
)


class Qwen2TokenizerFast:
    """CPU-only tokenizer stand-in used to exercise the local loader contract."""

    vocab_size = 257
    eos_token_id = 2
    eos_token = "</s>"

    def __init__(self) -> None:
        self.pad_token_id = None
        self.pad_token = None

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(identifier) for identifier in token_ids)


def _write_tokenizer_artifacts(path, *, marker="v1"):
    path.mkdir()
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2TokenizerFast", "marker": marker}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text(
        json.dumps({"version": marker}), encoding="utf-8"
    )


def _clear_tokenizer_environment(monkeypatch):
    monkeypatch.delenv("WORKBENCH_TOKENIZER_PATH", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_APPROX_TOKENIZER", raising=False)


def test_live_mode_rejects_missing_tokenizer_path(monkeypatch):
    _clear_tokenizer_environment(monkeypatch)
    settings = WorkbenchSettings.from_env()

    assert settings.tokenizer_path is None
    assert settings.allow_approx_tokenizer is False
    with pytest.raises(RuntimeError, match="WORKBENCH_TOKENIZER_PATH"):
        load_tokenizer_runtime(settings)


def test_live_mode_rejects_nonexistent_tokenizer_path(monkeypatch, tmp_path):
    _clear_tokenizer_environment(monkeypatch)
    missing = tmp_path / "missing-tokenizer"
    monkeypatch.setenv("WORKBENCH_TOKENIZER_PATH", str(missing))
    settings = WorkbenchSettings.from_env()

    with pytest.raises(FileNotFoundError, match="existing local directory"):
        load_tokenizer_runtime(settings, loader=lambda *_args, **_kwargs: None)


def test_live_mode_rejects_missing_tokenizer_artifacts(monkeypatch, tmp_path):
    _clear_tokenizer_environment(monkeypatch)
    empty = tmp_path / "empty-tokenizer"
    empty.mkdir()
    monkeypatch.setenv("WORKBENCH_TOKENIZER_PATH", str(empty))
    settings = WorkbenchSettings.from_env()

    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        load_tokenizer_runtime(settings, loader=lambda *_args, **_kwargs: None)

    config_only = tmp_path / "config-only-tokenizer"
    config_only.mkdir()
    (config_only / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_TOKENIZER_PATH", str(config_only))
    settings = WorkbenchSettings.from_env()
    with pytest.raises(FileNotFoundError, match="vocabulary artifact"):
        load_tokenizer_runtime(settings, loader=lambda *_args, **_kwargs: None)


def test_explicit_invalid_exact_path_never_falls_back_to_approximate(
    monkeypatch, tmp_path
):
    _clear_tokenizer_environment(monkeypatch)
    monkeypatch.setenv("WORKBENCH_TOKENIZER_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("WORKBENCH_ALLOW_APPROX_TOKENIZER", "true")

    with pytest.raises(FileNotFoundError, match="existing local directory"):
        load_tokenizer_runtime(WorkbenchSettings.from_env())


def test_approximate_tokenizer_requires_explicit_development_opt_in(monkeypatch):
    _clear_tokenizer_environment(monkeypatch)
    monkeypatch.setenv("WORKBENCH_ALLOW_APPROX_TOKENIZER", "true")

    runtime = load_tokenizer_runtime(WorkbenchSettings.from_env())

    assert runtime.mode == "approximate_test"
    assert runtime.configured_path.startswith("development://")
    assert len(runtime.artifact_fingerprint) == 64


def test_exact_tokenizer_load_is_local_only_and_sets_safe_pad_id(tmp_path):
    tokenizer_path = tmp_path / "checkpoint" / "global_step_20"
    tokenizer_path.parent.mkdir()
    _write_tokenizer_artifacts(tokenizer_path)
    calls = []
    tokenizer = Qwen2TokenizerFast()

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    runtime = load_exact_tokenizer(tokenizer_path, loader=loader)

    assert calls == [
        (
            str(tokenizer_path.resolve()),
            {"trust_remote_code": True, "local_files_only": True},
        )
    ]
    assert runtime.tokenizer is tokenizer
    assert runtime.mode == "exact"
    assert runtime.tokenizer_class == "Qwen2TokenizerFast"
    assert runtime.vocabulary_size == 257
    assert runtime.pad_token_id == runtime.eos_token_id == 2
    assert runtime.configured_path.endswith("checkpoint/global_step_20")
    assert str(tmp_path) not in runtime.configured_path
    assert runtime.artifact_fingerprint == tokenizer_artifact_fingerprint(
        tokenizer_path
    )


def test_exact_tokenizer_load_failure_is_fail_closed(tmp_path):
    tokenizer_path = tmp_path / "global_step_20"
    _write_tokenizer_artifacts(tokenizer_path)

    def failing_loader(*_args, **_kwargs):
        raise OSError("local tokenizer fixture is unreadable")

    with pytest.raises(RuntimeError, match="failed to load"):
        load_exact_tokenizer(tokenizer_path, loader=failing_loader)


def test_exact_mode_rejects_a_non_qwen_tokenizer(tmp_path):
    tokenizer_path = tmp_path / "wrong-tokenizer"
    _write_tokenizer_artifacts(tokenizer_path)

    class BertTokenizerFast(Qwen2TokenizerFast):
        pass

    with pytest.raises(ValueError, match="must load.*Qwen2"):
        load_exact_tokenizer(
            tokenizer_path, loader=lambda *_args, **_kwargs: BertTokenizerFast()
        )


def test_exact_tokenizer_preflight_reports_required_safe_metadata(monkeypatch):
    runtime = TokenizerRuntime(
        tokenizer=Qwen2TokenizerFast(),
        mode="exact",
        configured_path=".../actor/global_step_20",
        tokenizer_class="Qwen2TokenizerFast",
        artifact_fingerprint="a" * 64,
        vocabulary_size=151_665,
        pad_token_id=151_643,
        eos_token_id=151_643,
    )
    monkeypatch.setattr(
        tokenizer_preflight, "load_tokenizer_runtime", lambda _settings: runtime
    )

    payload = tokenizer_preflight.preflight_payload()

    assert payload["tokenizer_class"] == "Qwen2TokenizerFast"
    assert payload["tokenizer_vocabulary_size"] == 151_665
    assert payload["tokenizer_pad_token_id"] == 151_643
    assert payload["tokenizer_eos_token_id"] == 151_643
    assert payload["tokenizer_artifact_fingerprint"] == "a" * 64
    assert payload["exact_tokenizer_mode"] == "PASS"
    assert payload["model_weights_loaded"] is False


def test_tokenizer_artifact_fingerprint_changes_with_local_artifacts(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_tokenizer_artifacts(first, marker="v1")
    _write_tokenizer_artifacts(second, marker="v2")

    first_fingerprint = tokenizer_artifact_fingerprint(first)
    assert first_fingerprint != tokenizer_artifact_fingerprint(second)

    (first / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}), encoding="utf-8"
    )
    assert tokenizer_artifact_fingerprint(first) != first_fingerprint
