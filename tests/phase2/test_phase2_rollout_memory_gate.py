from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "experiments" / "phase2_3b" / "run_rollout_memory_gate.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")


def test_phase2_gate_is_isolated_and_reuses_phase1_fixture():
    assert 'BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"' in SCRIPT
    assert 'DATA_DIR="${PHASE1_DATA_DIR:-${ROOT_DIR}/data/phase1_smoke}"' in SCRIPT
    assert 'data.train_files="${DATA_DIR}/train.parquet"' in SCRIPT
    assert 'data.val_files="${DATA_DIR}/test.parquet"' in SCRIPT
    assert "experiments/phase1_smoke" not in SCRIPT


def test_phase2_gate_is_validation_only_with_one_trajectory():
    for expected in (
        "data.val_data_num=1",
        "data.val_batch_size=1",
        'N_AGENT="1"',
        'MAX_TURNS="2"',
        "+trainer.val_only=true",
        "+trainer.val_before_train=true",
    ):
        assert expected in SCRIPT

    assert "trainer.max_optimizer_steps" not in SCRIPT
    assert "mode: validation only; no optimizer updates" in SCRIPT


def test_phase2_gate_has_conservative_4090_memory_configuration():
    for expected in (
        'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"',
        'TENSOR_PARALLEL_SIZE="1"',
        'VLLM_GPU_MEMORY_UTILIZATION="0.20"',
        'ACTOR_PARAM_OFFLOAD="true"',
        'ACTOR_GRAD_OFFLOAD="false"',
        'ACTOR_OPTIMIZER_OFFLOAD="true"',
        'REFERENCE_PARAM_OFFLOAD="true"',
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.state_masking=true",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.dtype=bfloat16",
    ):
        assert expected in SCRIPT


def test_phase2_gate_preserves_token_limits_and_logs_memory_settings():
    for expected in (
        'MAX_START_LENGTH="384"',
        'MAX_RESPONSE_LENGTH="128"',
        'MAX_OBS_LENGTH="192"',
        'MAX_PROMPT_LENGTH="1152"',
        "vLLM GPU memory utilization:",
        "actor FSDP param offload:",
        "actor FSDP gradient offload:",
        "actor FSDP optimizer offload:",
        "reference FSDP param offload:",
        "token limits (start / response / observation / prompt):",
    ):
        assert expected in SCRIPT
