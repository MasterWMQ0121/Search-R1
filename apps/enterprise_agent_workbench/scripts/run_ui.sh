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
export WORKBENCH_API_BASE_URL="${WORKBENCH_API_BASE_URL:-http://127.0.0.1:8010}"

cd "${REPO_ROOT}"
exec "${WORKBENCH_PYTHON_RESOLVED}" -m streamlit run \
  apps/enterprise_agent_workbench/ui.py \
  --server.address "${WORKBENCH_UI_HOST:-127.0.0.1}" \
  --server.port "${WORKBENCH_UI_PORT:-8501}"
