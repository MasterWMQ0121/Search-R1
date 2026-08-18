import ast
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_group_metrics,
    compute_grpo_outcome_advantage,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "phase2_real_data"
PREPARE_PATH = EXPERIMENT_DIR / "prepare_gate_data.py"
LAUNCH_RETRIEVER_PATH = EXPERIMENT_DIR / "launch_retriever.sh"
VERIFY_RETRIEVER_PATH = EXPERIMENT_DIR / "verify_retriever.py"
RUN_GATE_PATH = EXPERIMENT_DIR / "run_reward_signal_gate.sh"
PREPARE_ASSETS_PATH = EXPERIMENT_DIR / "prepare_assets.sh"
TRAINER_PATH = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
PPO_CONFIG_PATH = ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
README_PATH = EXPERIMENT_DIR / "README.md"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load_module(PREPARE_PATH, "phase2_prepare_gate_data")
VERIFY = _load_module(VERIFY_RETRIEVER_PATH, "phase2_verify_retriever")
RUN_GATE = RUN_GATE_PATH.read_text(encoding="utf-8")
LAUNCH_RETRIEVER = LAUNCH_RETRIEVER_PATH.read_text(encoding="utf-8")
PREPARE_ASSETS = PREPARE_ASSETS_PATH.read_text(encoding="utf-8")
TRAINER_SOURCE = TRAINER_PATH.read_text(encoding="utf-8")
PPO_CONFIG = PPO_CONFIG_PATH.read_text(encoding="utf-8")
README = README_PATH.read_text(encoding="utf-8")


def _row(data_source, index, question=None):
    question = question or f"Real {data_source} question {index}?"
    return {
        "prompt": [{"role": "user", "content": question}],
        "data_source": data_source,
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [f"answer-{index}"]}},
        "extra_info": {"split": "train", "index": index},
        "preserved_column": f"preserve-{data_source}-{index}",
    }


def _write_sources(tmp_path, marker=None):
    train_rows = [_row("nq", index) for index in range(12)]
    train_rows += [_row("hotpotqa", index) for index in range(12)]
    if marker is not None:
        train_rows[0] = _row("nq", 0, marker)
    test_rows = [_row("nq", index) for index in range(6)]
    test_rows += [_row("hotpotqa", index) for index in range(6)]
    train_path = tmp_path / "source-train.parquet"
    test_path = tmp_path / "source-test.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(test_rows).to_parquet(test_path, index=False)
    return train_path, test_path


def _normalized(value):
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def test_prepare_gate_data_is_deterministic_balanced_and_preserves_rows(tmp_path):
    train_source, test_source = _write_sources(tmp_path)
    output_a = tmp_path / "gate-a"
    output_b = tmp_path / "gate-b"

    manifest_a = PREPARE.prepare_gate_data(
        train_source, test_source, output_a, train_size=8, test_size=4, seed=42
    )
    manifest_b = PREPARE.prepare_gate_data(
        train_source, test_source, output_b, train_size=8, test_size=4, seed=42
    )

    assert manifest_a["data_source_counts"]["train"] == {"nq": 4, "hotpotqa": 4}
    assert manifest_a["data_source_counts"]["test"] == {"nq": 2, "hotpotqa": 2}
    assert manifest_a["selected_row_counts"] == {"train": 8, "test": 4}
    assert manifest_a["output_sha256"] == manifest_b["output_sha256"]
    assert (output_a / "train.parquet").read_bytes() == (output_b / "train.parquet").read_bytes()
    assert manifest_a["leakage_audit"]["train"]["passed"] is True

    source_frame = pd.read_parquet(train_source)
    output_frame = pd.read_parquet(output_a / "train.parquet")
    assert list(output_frame.columns) == list(source_frame.columns)
    for output_position, selected in enumerate(manifest_a["selected_source_rows"]["train"]):
        source_row = source_frame.iloc[selected["source_position"]].to_dict()
        output_row = output_frame.iloc[output_position].to_dict()
        assert _normalized(output_row) == _normalized(source_row)

    on_disk_manifest = json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest["source_sha256"]["train"] == PREPARE.sha256_file(train_source)
    assert on_disk_manifest["output_sha256"]["test"] == PREPARE.sha256_file(output_a / "test.parquet")


@pytest.mark.parametrize("marker", PREPARE.LEAKAGE_MARKERS)
def test_prepare_gate_data_rejects_known_phase1_leakage(marker, tmp_path):
    train_source, test_source = _write_sources(tmp_path, marker=marker)
    with pytest.raises(ValueError, match="fixture leakage"):
        PREPARE.prepare_gate_data(
            train_source, test_source, tmp_path / "gate", train_size=24, test_size=4, seed=42
        )
    assert not (tmp_path / "gate" / "train.parquet").exists()


def test_prepare_gate_data_offset_rotates_the_same_deterministic_pool(tmp_path):
    train_source, test_source = _write_sources(tmp_path)
    base = PREPARE.prepare_gate_data(
        train_source, test_source, tmp_path / "base", train_size=8, test_size=4, seed=42
    )
    offset = PREPARE.prepare_gate_data(
        train_source, test_source, tmp_path / "offset", train_size=8, test_size=4,
        seed=42, train_offset=4
    )
    base_positions = [item["source_position"] for item in base["selected_source_rows"]["train"]]
    offset_positions = [item["source_position"] for item in offset["selected_source_rows"]["train"]]
    assert offset_positions == base_positions[4:] + base_positions[:4]


def test_retriever_and_asset_scripts_are_valid_and_cpu_only():
    for script in (LAUNCH_RETRIEVER_PATH, PREPARE_ASSETS_PATH):
        subprocess.run(["bash", "-n", str(script)], check=True)

    for expected in (
        "PHASE2_REAL_INDEX_PATH",
        "PHASE2_REAL_CORPUS_PATH",
        "PHASE2_REAL_RETRIEVER_MODEL",
        "PHASE2_REAL_RETRIEVER_HOST",
        "PHASE2_REAL_RETRIEVER_PORT",
        "PHASE2_REAL_PYTHON",
        'PYTHON_BIN="${PHASE2_REAL_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"',
        '--topk "${TOPK}"',
        "--device cpu",
        '[[ -s "${INDEX_PATH}" ]]',
        '[[ -s "${CORPUS_PATH}" ]]',
    ):
        assert expected in LAUNCH_RETRIEVER
    assert 'TOPK="3"' in LAUNCH_RETRIEVER
    assert "--faiss_gpu" not in LAUNCH_RETRIEVER
    assert "never downloads assets" in PREPARE_ASSETS
    assert "PHASE2_REAL_REQUIRED_FREE_GIB" in PREPARE_ASSETS
    assert 'REQUIRED_FREE_GIB="${PHASE2_REAL_REQUIRED_FREE_GIB:-96}"' in PREPARE_ASSETS


def _asset_environment(asset_root):
    return {
        **os.environ,
        "PHASE2_REAL_ASSET_ROOT": str(asset_root),
        "PHASE2_REAL_REQUIRED_FREE_GIB": "0",
        "PHASE2_REAL_MIN_RAM_GIB": "0",
        "PHASE2_REAL_ASSET_PYTHON": sys.executable,
    }


def _run_asset_command(asset_root, *args):
    return subprocess.run(
        ["bash", str(PREPARE_ASSETS_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_asset_environment(asset_root),
    )


def test_asset_index_assembly_keeps_only_one_source_shard_at_a_time(tmp_path):
    asset_root = tmp_path / "assets"
    wiki_dir = asset_root / "wiki18"
    _run_asset_command(asset_root, "init")
    _run_asset_command(asset_root, "init-index")

    part_a = wiki_dir / "part_aa"
    part_a.write_bytes(b"first-shard")
    _run_asset_command(asset_root, "append-index-part", "part_aa")
    assert (wiki_dir / "e5_Flat.index.tmp").read_bytes() == b"first-shard"
    assert part_a.exists()

    part_b = wiki_dir / "part_ab"
    part_b.write_bytes(b"second-shard")
    blocked = subprocess.run(
        ["bash", str(PREPARE_ASSETS_PATH), "append-index-part", "part_ab"],
        capture_output=True,
        text=True,
        env=_asset_environment(asset_root),
    )
    assert blocked.returncode != 0
    assert "Delete part_aa" in blocked.stderr

    part_a.unlink()
    _run_asset_command(asset_root, "append-index-part", "part_ab")
    part_b.unlink()
    _run_asset_command(asset_root, "finalize-index")

    assert not (wiki_dir / "e5_Flat.index.tmp").exists()
    assert (wiki_dir / "e5_Flat.index").read_bytes() == b"first-shardsecond-shard"


def test_asset_preflight_reports_free_disk_and_enforces_configured_threshold(tmp_path):
    environment = _asset_environment(tmp_path / "assets")
    environment["PHASE2_REAL_REQUIRED_FREE_GIB"] = "999999999"
    result = subprocess.run(
        ["bash", str(PREPARE_ASSETS_PATH), "preflight"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    assert "Free disk:" in result.stdout
    assert "conservative planning estimate" in result.stdout
    assert "Insufficient free disk" in result.stderr


def test_corpus_archive_cleanup_is_explicit(tmp_path):
    asset_root = tmp_path / "assets"
    wiki_dir = asset_root / "wiki18"
    _run_asset_command(asset_root, "init")
    compressed = wiki_dir / "wiki-18.jsonl.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(b'{"contents": "Title\\nText"}\n')

    _run_asset_command(asset_root, "prepare-corpus")
    assert compressed.exists()
    assert (wiki_dir / "wiki-18.jsonl").read_bytes() == b'{"contents": "Title\\nText"}\n'

    _run_asset_command(asset_root, "cleanup-corpus")
    assert not compressed.exists()


def test_retriever_fails_clearly_when_configured_python_is_missing(tmp_path):
    missing_python = tmp_path / "missing-python"
    result = subprocess.run(
        ["bash", str(LAUNCH_RETRIEVER_PATH)],
        capture_output=True,
        text=True,
        env={**os.environ, "PHASE2_REAL_PYTHON": str(missing_python)},
    )
    assert result.returncode != 0
    assert "Missing executable CPU retriever Python" in result.stderr


def test_readme_uses_actual_gpu_repository_and_sequential_downloads():
    assert "/workspace/search_rl" not in README
    assert "/workspace/Search-R1" in README
    assert "scripts/download.py" not in README
    assert README.index("wiki-18-e5-index part_aa") < README.index("rm -- /workspace/searchr1-assets/wiki18/part_aa")
    assert README.index("rm -- /workspace/searchr1-assets/wiki18/part_aa") < README.index("wiki-18-e5-index part_ab")
    assert "prepare_assets.sh cleanup-corpus" in README


def test_retriever_verifier_accepts_real_server_schema_and_rejects_bad_results():
    valid_payload = {
        "result": [[
            {"document": {"contents": f"Title {rank}\nText"}, "score": 1.0 / rank}
            for rank in range(1, 4)
        ]]
    }
    assert len(VERIFY.validate_response(valid_payload, topk=3)) == 3

    invalid_payload = {"result": [[{"document": {"contents": ""}, "score": math.nan}]]}
    with pytest.raises(ValueError):
        VERIFY.validate_response(invalid_payload, topk=1)


def _shell_integer(name):
    match = re.search(rf'^{name}="(\d+)"$', RUN_GATE, flags=re.MULTILINE)
    assert match is not None
    return int(match.group(1))


def test_reward_signal_gate_has_consistent_one_step_grpo_batching():
    subprocess.run(["bash", "-n", str(RUN_GATE_PATH)], check=True)
    train_batch_size = _shell_integer("TRAIN_BATCH_SIZE")
    rollout_n = _shell_integer("ROLLOUT_N")
    n_agent = _shell_integer("N_AGENT")
    mini_batch_size = _shell_integer("PPO_MINI_BATCH_SIZE")
    micro_batch_size = _shell_integer("PPO_MICRO_BATCH_SIZE")

    assert (train_batch_size, rollout_n, n_agent) == (4, 1, 4)
    assert train_batch_size * rollout_n * n_agent == mini_batch_size == 16
    assert mini_batch_size % micro_batch_size == 0
    for expected in (
        "+trainer.max_optimizer_steps=1",
        "+trainer.val_only=false",
        "+trainer.val_before_train=false",
        "+trainer.val_after_train=false",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "data.shuffle_train_dataloader=false",
        "data.train_data_num=null",
    ):
        assert expected in RUN_GATE


def test_reward_signal_gate_preserves_runtime_and_isolates_policy_gradient():
    for expected in (
        'BASE_MODEL="${PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"',
        'RETRIEVER_URL="${PHASE2_REAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"',
        'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"',
        'VLLM_GPU_MEMORY_UTILIZATION="0.20"',
        'ROLLOUT_TEMPERATURE="1.0"',
        'ROLLOUT_TOP_P="0.95"',
        'KL_LOSS_COEF="0.0"',
        'ENTROPY_COEF="0.0"',
        'ACTOR_PARAM_OFFLOAD="true"',
        'ACTOR_GRAD_OFFLOAD="false"',
        'ACTOR_OPTIMIZER_OFFLOAD="true"',
        'REFERENCE_PARAM_OFFLOAD="true"',
        "actor_rollout_ref.rollout.dtype=bfloat16",
        'actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"',
        'actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"',
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.actor.state_masking=true",
        "actor_rollout_ref.actor.use_kl_loss=true",
        'actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"',
        'actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}"',
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "retriever.topk=3",
    ):
        assert expected in RUN_GATE

    start = _shell_integer("MAX_START_LENGTH")
    response = _shell_integer("MAX_RESPONSE_LENGTH")
    observation = _shell_integer("MAX_OBS_LENGTH")
    turns = _shell_integer("MAX_TURNS")
    assert start + response * (turns - 1) + observation * turns == 1408
    assert "MAX_START_LENGTH + MAX_RESPONSE_LENGTH * (MAX_TURNS - 1)" in RUN_GATE
    assert re.search(r"^\s+top_p:\s*0\.95\s*$", PPO_CONFIG, flags=re.MULTILINE)


def test_reward_signal_gate_logs_diagnostic_configuration():
    for label in (
        "mode:",
        "model:",
        "data:",
        "retriever URL:",
        "GPU count:",
        "CUDA_VISIBLE_DEVICES:",
        "train batch size:",
        "rollout n:",
        "n_agent:",
        "effective rollout trajectories:",
        "PPO mini / micro batch:",
        "max turns:",
        "retriever top-k:",
        "token limits (start / response / observation / prompt):",
        "KL loss coefficient:",
        "entropy coefficient:",
        "actor FSDP param offload:",
        "actor FSDP gradient offload:",
        "actor FSDP optimizer offload:",
        "reference FSDP param offload:",
        "vLLM GPU memory utilization:",
        "rollout temperature:",
        "rollout top-p:",
        "log:",
    ):
        assert label in RUN_GATE


def test_source_aware_uids_keep_overlapping_dataset_indexes_separate():
    tree = ast.parse(TRAINER_SOURCE)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_search_prompt_uids"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER_PATH), "exec"), namespace)
    build_uids = namespace["_build_search_prompt_uids"]

    uids = build_uids(
        np.array(["nq", "nq", "hotpotqa", "hotpotqa"], dtype=object),
        np.array([0, 0, 0, 0], dtype=object),
    )
    assert uids.tolist() == ["nq:0", "nq:0", "hotpotqa:0", "hotpotqa:0"]
    assert len(set(uids.tolist())) == 2


def test_grpo_group_metrics_measure_within_uid_variance_and_advantages():
    rewards = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
    ])
    mask = torch.ones_like(rewards)
    uids = np.array(["A"] * 4 + ["B"] * 4, dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, uids)
    metrics = compute_grpo_group_metrics(rewards, advantages, mask, uids)

    assert metrics["grpo/group_count"] == 2
    assert metrics["grpo/groups_with_reward_variance"] == 1
    assert metrics["grpo/group_reward_variance_fraction"] == 0.5
    assert math.isfinite(metrics["grpo/group_reward_std_mean"])
    assert metrics["grpo/reward_std_max"] > 0
    assert metrics["grpo/reward_std_min"] == 0
    assert metrics["grpo/nonzero_advantage_fraction"] == 0.5


def test_grpo_group_metrics_handle_singletons_without_nan():
    rewards = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(rewards)
    uids = np.array(["singleton"], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, uids)
    metrics = compute_grpo_group_metrics(rewards, advantages, mask, uids)
    assert metrics["grpo/group_count"] == 1
    assert metrics["grpo/groups_with_reward_variance"] == 0
    assert metrics["grpo/group_reward_std_mean"] == 0
    assert metrics["grpo/nonzero_advantage_fraction"] == 0
    assert all(math.isfinite(value) for value in metrics.values())


def test_existing_phase1_and_phase2_gate_scripts_are_unchanged():
    expected_hashes = {
        ROOT / "experiments" / "phase1_smoke" / "run_one_update.sh":
            "06ca2bfbec4133c9f9c37a7bc7a8f2d95d7d548b1898b1add03fdc5ce9959610",
        ROOT / "experiments" / "phase2_3b" / "run_rollout_memory_gate.sh":
            "790a53e43e124983a22340d28b7b7a0349d129adb5499e0d707fb26d138037ea",
        ROOT / "experiments" / "phase2_3b" / "run_one_update.sh":
            "f669e781f2aa5e339ce54d50a5431330442347f465608715ac75c9d2466d0d3d",
    }
    for path, expected in expected_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
