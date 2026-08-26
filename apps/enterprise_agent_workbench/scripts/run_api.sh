#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKBENCH_PYTHON="${WORKBENCH_PYTHON:-python3}"

if ! WORKBENCH_PYTHON_RESOLVED="$(command -v "${WORKBENCH_PYTHON}" 2>/dev/null)"; then
  echo "workbench Python not found: ${WORKBENCH_PYTHON}" >&2
  echo "set WORKBENCH_PYTHON to a Python 3.10+ interpreter in the isolated workbench environment" >&2
  exit 1
fi

python_status=0
python_report="$("${WORKBENCH_PYTHON_RESOLVED}" -c '
import platform
import sys

print(f"Resolved workbench interpreter: {sys.executable}")
print(f"Workbench Python version: {platform.python_version()}")
if sys.version_info < (3, 10):
    raise SystemExit(64)
' 2>&1)" || python_status=$?
printf '%s\n' "${python_report}"
if [[ "${python_status}" -ne 0 ]]; then
  echo "Enterprise Agent Workbench requires Python 3.10 or newer" >&2
  exit "${python_status}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LANGGRAPH_STRICT_MSGPACK="${LANGGRAPH_STRICT_MSGPACK:-true}"
cd "${REPO_ROOT}"
"${WORKBENCH_PYTHON_RESOLVED}" -m apps.enterprise_agent_workbench.tokenizer_preflight
exec "${WORKBENCH_PYTHON_RESOLVED}" -m uvicorn \
  apps.enterprise_agent_workbench.api:app \
  --host "${WORKBENCH_API_HOST:-127.0.0.1}" \
  --port "${WORKBENCH_API_PORT:-8010}"
