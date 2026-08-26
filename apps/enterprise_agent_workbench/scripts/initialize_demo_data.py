#!/usr/bin/env python3
"""Initialize deterministic workbench data outside the source tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apps.enterprise_agent_workbench.tools.campaign_api import (
    initialize_demo_database,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEED = APP_DIR / "fixtures" / "merchant_seed.json"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("WORKBENCH_DATA_DIR", "~/.search_r1_workbench")
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATA_ROOT / "merchant_demo.sqlite"),
        help="Output SQLite path (defaults outside the repository).",
    )
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing local demo database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = initialize_demo_database(
        args.database, args.seed, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "status": "initialized",
                "database": str(output),
                "seed": str(Path(args.seed).expanduser().resolve()),
                "real_business_platform_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
