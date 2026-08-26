#!/usr/bin/env python3
"""Validate and report the Workbench evidence-tokenizer contract."""

from __future__ import annotations

import json

from .config import WorkbenchSettings
from .tokenizer_runtime import load_tokenizer_runtime
from .tools.research_search import (
    EVIDENCE_COMPRESSOR_POLICY,
    EVIDENCE_COMPRESSOR_VERSION,
    evidence_compressor_fingerprint,
)


def preflight_payload() -> dict[str, object]:
    settings = WorkbenchSettings.from_env()
    runtime = load_tokenizer_runtime(settings)
    return {
        **runtime.safe_metadata(),
        "evidence_compressor_policy": EVIDENCE_COMPRESSOR_POLICY,
        "evidence_compressor_version": EVIDENCE_COMPRESSOR_VERSION,
        "evidence_compressor_fingerprint": evidence_compressor_fingerprint(),
        "max_evidence_token_budget": settings.evidence_token_budget,
        "exact_tokenizer_mode": (
            "PASS" if runtime.mode == "exact" else "APPROXIMATE_TEST_OPT_IN"
        ),
        "model_weights_loaded": False,
    }


def main() -> None:
    print(json.dumps(preflight_payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
