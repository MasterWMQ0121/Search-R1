import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "phase3_real_training"
PREPARE_PATH = EXPERIMENT_DIR / "prepare_data.sh"
RUN_PATH = EXPERIMENT_DIR / "run_small_training.sh"
README_PATH = EXPERIMENT_DIR / "README.md"
TRAINER_PATH = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
GENERATION_PATH = ROOT / "search_r1" / "llm_agent" / "generation.py"

PREPARE = PREPARE_PATH.read_text(encoding="utf-8")
RUN = RUN_PATH.read_text(encoding="utf-8")
README = README_PATH.read_text(encoding="utf-8")
TRAINER = TRAINER_PATH.read_text(encoding="utf-8")
GENERATION = GENERATION_PATH.read_text(encoding="utf-8")


def _shell_integer(script, name):
    match = re.search(rf'^{name}="(\d+)"$', script, flags=re.MULTILINE)
    assert match is not None
    return int(match.group(1))


def test_phase3_scripts_exist_and_have_valid_bash_syntax():
    for path in (PREPARE_PATH, RUN_PATH):
        assert path.is_file()
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_phase3_data_preparation_reuses_audited_phase2_selector():
    assert _shell_integer(PREPARE, "TRAIN_SIZE") == 128
    assert _shell_integer(PREPARE, "VAL_SIZE") == 32
    assert _shell_integer(PREPARE, "SEED") == 42
    assert _shell_integer(PREPARE, "TRAIN_OFFSET") == 4
    for expected in (
        "experiments/phase2_real_data/prepare_gate_data.py",
        'PHASE3_SOURCE_DATA_DIR:-/workspace/searchr1-assets/datasets/nq_hotpotqa_train',
        'PHASE3_DATA_DIR:-/workspace/searchr1-assets/datasets/phase3_real_training',
        '--train-size "${TRAIN_SIZE}"',
        '--test-size "${VAL_SIZE}"',
        '--train-offset "${TRAIN_OFFSET}"',
    ):
        assert expected in PREPARE


def test_phase3_update_and_grpo_geometry_is_internally_consistent():
    train_size = _shell_integer(RUN, "TRAIN_SIZE")
    max_steps = _shell_integer(RUN, "MAX_OPTIMIZER_STEPS")
    epochs = _shell_integer(RUN, "TOTAL_EPOCHS")
    train_batch = _shell_integer(RUN, "TRAIN_BATCH_SIZE")
    rollout_n = _shell_integer(RUN, "ROLLOUT_N")
    n_agent = _shell_integer(RUN, "N_AGENT")
    mini_batch = _shell_integer(RUN, "PPO_MINI_BATCH_SIZE")
    micro_batch = _shell_integer(RUN, "PPO_MICRO_BATCH_SIZE")
    ppo_epochs = _shell_integer(RUN, "PPO_EPOCHS")

    assert (train_size, max_steps, epochs) == (128, 20, 1)
    assert _shell_integer(PREPARE, "TRAIN_SIZE") == train_size
    assert _shell_integer(PREPARE, "VAL_SIZE") == _shell_integer(RUN, "VAL_SIZE") == 32
    assert (train_batch, rollout_n, n_agent) == (4, 1, 4)
    trajectories_per_update = train_batch * rollout_n * n_agent
    assert trajectories_per_update == mini_batch == 16
    assert mini_batch % micro_batch == 0
    assert ppo_epochs == 1
    assert max_steps * train_batch == 80
    assert max_steps * trajectories_per_update == 320
    assert max_steps * train_batch <= train_size
    available_batches = (train_size // train_batch) * epochs
    assert max_steps == 20 <= available_batches == 32
    assert 'data.shuffle_train_dataloader=false' in RUN
    assert '+trainer.max_optimizer_steps="${MAX_OPTIMIZER_STEPS}"' in RUN


def test_phase3_runs_training_with_before_and_after_validation_only():
    max_steps = _shell_integer(RUN, "MAX_OPTIMIZER_STEPS")
    save_freq = _shell_integer(RUN, "SAVE_FREQ")
    assert max_steps == save_freq == 20

    for expected in (
        "+trainer.val_only=false",
        "+trainer.val_before_train=true",
        "+trainer.val_after_train=true",
        "trainer.test_freq=-1",
        'trainer.save_freq="${SAVE_FREQ}"',
        "data.train_data_num=null",
        "data.val_data_num=null",
        "validation: greedy, before and after only",
        'global_step_${MAX_OPTIMIZER_STEPS}',
        '${CHECKPOINT_PATH}/config.json',
        '"${CHECKPOINT_PATH}"/model*.safetensors',
        'Saving actor checkpoint to ${CHECKPOINT_PATH}',
        "Initial validation metrics:",
        "Final validation metrics:",
    ):
        assert expected in RUN

    actor_update = TRAINER.index("actor_output = self.actor_rollout_wg.update_actor(batch)")
    checkpoint_save = TRAINER.index("if self.config.trainer.save_freq > 0", actor_update)
    global_step_increment = TRAINER.index("self.global_steps += 1", checkpoint_save)
    early_return = TRAINER.index(
        "if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps",
        global_step_increment,
    )
    assert actor_update < checkpoint_save < global_step_increment < early_return


def test_phase3_preserves_the_proven_phase2_runtime_and_reward():
    for expected in (
        'PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct',
        'PHASE3_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"',
        'TENSOR_PARALLEL_SIZE="1"',
        'VLLM_GPU_MEMORY_UTILIZATION="0.20"',
        'MAX_START_LENGTH="768"',
        'MAX_RESPONSE_LENGTH="128"',
        'MAX_OBS_LENGTH="256"',
        'MAX_PROMPT_LENGTH="$((MAX_START_LENGTH + MAX_RESPONSE_LENGTH * (MAX_TURNS - 1) + MAX_OBS_LENGTH * MAX_TURNS))"',
        'MAX_TURNS="2"',
        'KL_LOSS_COEF="0.001"',
        'ENTROPY_COEF="0.0"',
        'ACTOR_PARAM_OFFLOAD="true"',
        'ACTOR_GRAD_OFFLOAD="false"',
        'ACTOR_OPTIMIZER_OFFLOAD="true"',
        'REFERENCE_PARAM_OFFLOAD="true"',
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.actor.state_masking=true",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.use_kl_loss=true",
        'actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"',
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        'actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}"',
        "algorithm.kl_ctrl.kl_coef=0.0",
        "reward_model.enable=false",
        "retriever.topk=3",
    ):
        assert expected in RUN


def test_phase3_uses_training_conda_python_and_keeps_cpu_tools_isolated():
    assert 'PYTHON_BIN="${PHASE3_PYTHON:-python3}"' in RUN
    assert ".venv-phase1/bin/python" not in RUN
    assert 'command -v "${PYTHON_BIN}"' in RUN
    assert 'training Python: ${PYTHON_RUNTIME}' in RUN
    assert 'PYTHON_BIN="${PHASE3_PYTHON:-${ROOT_DIR}/.venv-phase1/bin/python}"' in PREPARE
    for expected in (
        "source /usr/local/miniconda3/etc/profile.d/conda.sh",
        "conda activate searchr1",
        "PHASE3_PYTHON=python3",
    ):
        assert expected in README


def test_phase3_observability_and_inconclusive_signal_gate_are_present():
    for metric in (
        "critic/rewards/mean",
        "critic/rewards/std",
        "grpo/groups_with_reward_variance",
        "grpo/group_reward_variance_fraction",
        "grpo/nonzero_advantage_fraction",
        "actor/grad_norm",
        "actor/kl_loss",
        "actor/kl_coef",
        "actor/entropy_loss",
        "training/optimizer_steps",
        "env/finish_ratio",
        "env/ratio_of_valid_action",
        "env/number_of_valid_search",
        "env/ratio_of_valid_search",
        "env/number_of_successful_retrievals",
        "env/trajectories_with_retrieval",
        "timing_s/step",
        "timing_s/gen",
        "timing_s/update_actor",
    ):
        assert metric in README

    assert "torch.std(sequence_reward, unbiased=False)" in TRAINER
    assert "where=turns_stats != 0" in TRAINER
    assert "retrieval_success_stats += torch.tensor(is_search" in GENERATION
    assert "meta_info['retrieval_success_stats']" in GENERATION
    assert "VARIANCE_STEPS < 2" in RUN
    assert "ADVANTAGE_STEPS < 2" in RUN
    assert "GRADIENT_STEPS < 2" in RUN
    assert "exit 2" in RUN
    assert "TRAINING INFRA PASS" in RUN
    assert "REWARD-SIGNAL PASS" in RUN
    assert "QUALITY SIGNAL" in RUN


def test_phase3_final_checkpoint_is_the_only_scheduled_checkpoint():
    max_steps = _shell_integer(RUN, "MAX_OPTIMIZER_STEPS")
    save_freq = _shell_integer(RUN, "SAVE_FREQ")
    assert max_steps == save_freq == 20
    scheduled_checkpoints = [
        step for step in range(1, max_steps + 1) if step % save_freq == 0
    ]
    assert scheduled_checkpoints == [20]
    assert RUN.count("trainer.save_freq=") == 1
    assert "actor/global_step_${MAX_OPTIMIZER_STEPS}" in RUN
    assert "actor/global_step_20/" in README
    assert "Phase-4 evaluation input" in README


def test_phase2_real_data_experiment_files_are_unchanged():
    expected_hashes = {
        "README.md": "62f09788e2e5c3a64fb8445289bcd23a54cb91172425ae007616710a4f4a72c0",
        "prepare_gate_data.py": "468e13a5ba94a524946cbdfba173bc256515035655e71c3793e0a13af1c05cf5",
        "prepare_assets.sh": "d97dc836b2d91411463e1e14a83942b76640d359581830afdd049d918eb87490",
        "launch_retriever.sh": "c39b134728edb5defdb70713e9c0111d9518a21a38e07c0c87e5bb660ddb1313",
        "verify_retriever.py": "286cecb73be0befe922046d0b50d6713788f51199a9e6482df383e15ff6752e8",
        "run_reward_signal_gate.sh": "79e77cfb3300ba7f5af426b97338c1896adb126feb86f50c95771649b50ca150",
    }
    phase2_dir = ROOT / "experiments" / "phase2_real_data"
    actual_hashes = {
        name: hashlib.sha256((phase2_dir / name).read_bytes()).hexdigest()
        for name in expected_hashes
    }
    assert actual_hashes == expected_hashes


def test_phase3_readme_does_not_claim_quality_or_gpu_results():
    normalized_readme = " ".join(README.split())
    assert "not convergence, paper reproduction, or a claimed quality gain" in normalized_readme
    assert "does not demonstrate monotonic reward" in normalized_readme
    assert "20 x 16 = 320" in normalized_readme
    assert "20 optimizer updates consume the first 80" in normalized_readme
    assert "Runtime remains pending the real A800 Phase-3 run" in normalized_readme
    for stale_text in (
        "256 training rows",
        "50 optimizer updates",
        "first 200 prepared training rows",
        "50-step cap",
        "global_step_50",
        "50-update run",
        "43 minutes",
    ):
        assert stale_text not in README
    assert "GPU COMMAND PREPARED ONLY" in RUN
