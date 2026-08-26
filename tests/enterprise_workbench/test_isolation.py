import ast
import os
import subprocess
from pathlib import Path

import pytest

from apps.enterprise_agent_workbench.evaluation.run_evaluation import (
    validate_output_isolation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "apps" / "enterprise_agent_workbench"


@pytest.mark.parametrize(
    "relative",
    [
        "phase4_benchmark_results/workbench/results.jsonl",
        "phase5_observation_results/workbench/results.jsonl",
    ],
)
def test_workbench_artifacts_cannot_enter_phase4_or_phase5_results(tmp_path, relative):
    with pytest.raises(ValueError, match="must not be written"):
        validate_output_isolation(tmp_path / relative)


def test_normal_workbench_state_path_is_allowed(tmp_path):
    output = tmp_path / ".workbench-state" / "evaluation" / "results.jsonl"
    assert validate_output_isolation(output) == output.resolve()


def test_launch_scripts_use_only_isolated_environment_and_set_pythonpath():
    for name in ("run_api.sh", "run_ui.sh"):
        path = APP_ROOT / "scripts" / name
        text = path.read_text(encoding="utf-8")
        assert 'WORKBENCH_PYTHON="${WORKBENCH_PYTHON:-python3}"' in text
        assert "Resolved workbench interpreter" in text
        assert "Workbench Python version" in text
        assert "sys.version_info < (3, 10)" in text
        assert "export PYTHONPATH=" in text
        assert "LANGGRAPH_STRICT_MSGPACK" in text
        assert "python3.11" not in text
        assert ".venv-phase1" not in text
        assert "conda activate" not in text
        subprocess.run(["bash", "-n", str(path)], check=True)

    api_text = (APP_ROOT / "scripts" / "run_api.sh").read_text(encoding="utf-8")
    assert "apps.enterprise_agent_workbench.tokenizer_preflight" in api_text


def test_documented_runtime_is_python_310_plus_and_offline_exact_tokenizer():
    text = (APP_ROOT / "README.md").read_text(encoding="utf-8")
    assert "python3.11" not in text
    assert 'WORKBENCH_PYTHON="${WORKBENCH_PYTHON:-python3}"' in text
    assert "Python 3.10 and 3.11 are therefore supported" in text
    assert "export HF_HUB_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert "export WORKBENCH_TOKENIZER_PATH=" in text
    assert "apps.enterprise_agent_workbench.tokenizer_preflight" in text
    assert "Shell exports are terminal-local" in text


def _write_fake_python(path: Path, version: str) -> None:
    major, minor, *_ = (int(part) for part in version.split("."))
    supported = major > 3 or (major == 3 and minor >= 10)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        f"  echo \"Resolved workbench interpreter: {path}\"\n"
        f"  echo \"Workbench Python version: {version}\"\n"
        f"  exit {0 if supported else 64}\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize("version", ["3.10.16", "3.11.15"])
@pytest.mark.parametrize("script_name", ["run_api.sh", "run_ui.sh"])
def test_launch_scripts_accept_supported_python_versions(
    tmp_path, version, script_name
):
    fake_python = tmp_path / "workbench-python"
    _write_fake_python(fake_python, version)
    completed = subprocess.run(
        ["bash", str(APP_ROOT / "scripts" / script_name)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "WORKBENCH_PYTHON": str(fake_python)},
    )
    assert completed.returncode == 0, completed.stderr
    assert f"Workbench Python version: {version}" in completed.stdout


@pytest.mark.parametrize("script_name", ["run_api.sh", "run_ui.sh"])
def test_launch_scripts_reject_python_39(tmp_path, script_name):
    fake_python = tmp_path / "workbench-python"
    _write_fake_python(fake_python, "3.9.19")
    completed = subprocess.run(
        ["bash", str(APP_ROOT / "scripts" / script_name)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "WORKBENCH_PYTHON": str(fake_python)},
    )
    assert completed.returncode != 0
    assert "Workbench Python version: 3.9.19" in completed.stdout
    assert "requires Python 3.10 or newer" in completed.stderr


def test_ui_defers_streamlit_import_until_main():
    tree = ast.parse((APP_ROOT / "ui.py").read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "streamlit" for alias in node.names)
        )
        for node in top_level_imports
    )


def test_evaluation_defaults_outside_benchmark_artifact_directories():
    from apps.enterprise_agent_workbench.evaluation import run_evaluation

    assert str(run_evaluation.DEFAULT_OUTPUT).endswith(
        ".search_r1_workbench/evaluation/results.jsonl"
    )
    assert "phase4" not in str(run_evaluation.DEFAULT_OUTPUT).lower()
    assert "phase5" not in str(run_evaluation.DEFAULT_OUTPUT).lower()
