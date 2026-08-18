import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "experiments" / "phase2_3b" / "run_one_update.sh"
PHASE1_SCRIPT_PATH = ROOT / "experiments" / "phase1_smoke" / "run_one_update.sh"
TRAINER_PATH = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
ACTOR_PATH = ROOT / "verl" / "workers" / "actor" / "dp_actor.py"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
TRAINER_SOURCE = TRAINER_PATH.read_text(encoding="utf-8")
ACTOR_SOURCE = ACTOR_PATH.read_text(encoding="utf-8")


def _shell_integer(name):
    match = re.search(rf'^{name}="(\d+)"$', SCRIPT, flags=re.MULTILINE)
    assert match is not None
    return int(match.group(1))


def test_phase2_one_update_script_exists_and_has_valid_bash_syntax():
    assert SCRIPT_PATH.is_file()
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_phase2_one_update_is_training_only_and_stops_after_one_update():
    for expected in (
        "+trainer.val_only=false",
        "+trainer.val_before_train=false",
        "+trainer.val_after_train=false",
        "+trainer.max_optimizer_steps=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "mode: one optimizer update; plumbing validation only",
    ):
        assert expected in SCRIPT

    assert "val_after_train = self.config.trainer.get('val_after_train', True)" in TRAINER_SOURCE
    assert "self.val_reward_fn is not None and val_after_train" in TRAINER_SOURCE


def test_phase2_one_update_preserves_model_retriever_and_runtime_configuration():
    for expected in (
        'BASE_MODEL="${PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"',
        'RETRIEVER_URL="${PHASE2_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"',
        'actor_rollout_ref.model.path="${BASE_MODEL}"',
        'retriever.url="${RETRIEVER_URL}"',
        'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        'TENSOR_PARALLEL_SIZE="1"',
        'VLLM_GPU_MEMORY_UTILIZATION="0.20"',
        'MAX_START_LENGTH="384"',
        'MAX_RESPONSE_LENGTH="128"',
        'MAX_OBS_LENGTH="192"',
        'MAX_PROMPT_LENGTH="1152"',
        'MAX_TURNS="2"',
        'ACTOR_PARAM_OFFLOAD="true"',
        'ACTOR_GRAD_OFFLOAD="false"',
        'ACTOR_OPTIMIZER_OFFLOAD="true"',
        'REFERENCE_PARAM_OFFLOAD="true"',
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.state_masking=true",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "retriever.topk=2",
    ):
        assert expected in SCRIPT


def test_phase2_one_update_grpo_group_and_optimizer_step_are_consistent():
    train_batch_size = _shell_integer("TRAIN_BATCH_SIZE")
    rollout_n = _shell_integer("ROLLOUT_N")
    n_agent = _shell_integer("N_AGENT")
    mini_batch_size = _shell_integer("PPO_MINI_BATCH_SIZE")
    micro_batch_size = _shell_integer("PPO_MICRO_BATCH_SIZE")
    effective_trajectories = train_batch_size * rollout_n * n_agent

    assert train_batch_size == 1
    assert n_agent == 2
    assert n_agent > 1
    assert mini_batch_size == effective_trajectories == 2
    assert mini_batch_size % micro_batch_size == 0
    assert 'batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent' in TRAINER_SOURCE
    assert "batch.non_tensor_batch['uid'] = _build_search_prompt_uids(" in TRAINER_SOURCE
    assert "dataloader = batch.split(self.config.ppo_mini_batch_size)" in ACTOR_SOURCE
    assert "grad_norm = self._optimizer_step()" in ACTOR_SOURCE


def test_phase2_one_update_logs_the_gate_configuration():
    for expected in (
        "GPU count:",
        "train batch size:",
        "n_agent:",
        "effective rollout trajectories per update:",
        "PPO mini / micro batch:",
        "actor FSDP param offload:",
        "actor FSDP optimizer offload:",
        "reference FSDP param offload:",
        "vLLM GPU memory utilization:",
        "token limits (start / response / observation / prompt):",
        "log:",
    ):
        assert expected in SCRIPT


def test_phase1_one_update_script_is_unchanged():
    digest = hashlib.sha256(PHASE1_SCRIPT_PATH.read_bytes()).hexdigest()
    assert digest == "06ca2bfbec4133c9f9c37a7bc7a8f2d95d7d548b1898b1add03fdc5ce9959610"
