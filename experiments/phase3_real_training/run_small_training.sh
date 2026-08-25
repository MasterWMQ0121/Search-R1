#!/usr/bin/env bash
# GPU COMMAND PREPARED ONLY. Small real-data GRPO learning/stability validation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${PHASE3_DATA_DIR:-/workspace/searchr1-assets/datasets/phase3_real_training}"
PYTHON_BIN="${PHASE3_PYTHON:-python3}"
RETRIEVER_URL="${PHASE3_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
BASE_MODEL="${PHASE3_MODEL_PATH:-${PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}}"
EXPERIMENT_NAME="phase3-qwen2.5-3b-small-real-grpo-training"
LOG_PATH="${PHASE3_LOG_PATH:-${ROOT_DIR}/${EXPERIMENT_NAME}.log}"

TRAIN_SIZE="128"
VAL_SIZE="32"
MAX_OPTIMIZER_STEPS="20"
SAVE_FREQ="20"
TOTAL_EPOCHS="1"
MAX_START_LENGTH="768"
MAX_RESPONSE_LENGTH="128"
MAX_OBS_LENGTH="256"
MAX_TURNS="2"
MAX_PROMPT_LENGTH="$((MAX_START_LENGTH + MAX_RESPONSE_LENGTH * (MAX_TURNS - 1) + MAX_OBS_LENGTH * MAX_TURNS))"
TRAIN_BATCH_SIZE="4"
VAL_BATCH_SIZE="4"
ROLLOUT_N="1"
N_AGENT="4"
EFFECTIVE_ROLLOUT_TRAJECTORIES="$((TRAIN_BATCH_SIZE * ROLLOUT_N * N_AGENT))"
TOTAL_PROMPTS_CONSUMED="$((MAX_OPTIMIZER_STEPS * TRAIN_BATCH_SIZE))"
TOTAL_ROLLOUT_TRAJECTORIES="$((MAX_OPTIMIZER_STEPS * EFFECTIVE_ROLLOUT_TRAJECTORIES))"
PPO_MINI_BATCH_SIZE="16"
PPO_MICRO_BATCH_SIZE="1"
PPO_EPOCHS="1"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="1"
REFERENCE_LOGPROB_MICRO_BATCH_SIZE="1"
TENSOR_PARALLEL_SIZE="1"
VLLM_GPU_MEMORY_UTILIZATION="0.20"
ROLLOUT_TEMPERATURE="1.0"
ROLLOUT_TOP_P="0.95"
KL_LOSS_COEF="0.001"
ENTROPY_COEF="0.0"
ACTOR_PARAM_OFFLOAD="true"
ACTOR_GRAD_OFFLOAD="false"
ACTOR_OPTIMIZER_OFFLOAD="true"
REFERENCE_PARAM_OFFLOAD="true"
CHECKPOINT_PATH="${ROOT_DIR}/verl_checkpoints/${EXPERIMENT_NAME}/actor/global_step_${MAX_OPTIMIZER_STEPS}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { printf 'Missing Phase-3 training Python: %s\n' "${PYTHON_BIN}" >&2; exit 1; }
PYTHON_RUNTIME="$("${PYTHON_BIN}" -c 'import platform, sys; print(f"{sys.executable} (Python {platform.python_version()})")')"
[[ -s "${DATA_DIR}/train.parquet" ]] || { printf 'Missing prepared train parquet: %s\n' "${DATA_DIR}/train.parquet" >&2; exit 1; }
[[ -s "${DATA_DIR}/test.parquet" ]] || { printf 'Missing prepared validation parquet: %s\n' "${DATA_DIR}/test.parquet" >&2; exit 1; }

log_configuration() {
  printf '%s\n' \
    "Phase-3 Qwen2.5-3B small real-data GRPO training validation" \
    "  mode: ${MAX_OPTIMIZER_STEPS} optimizer updates with before/after validation" \
    "  training Python: ${PYTHON_RUNTIME}" \
    "  model: ${BASE_MODEL}" \
    "  train / validation data: ${DATA_DIR}/train.parquet / ${DATA_DIR}/test.parquet" \
    "  expected train / validation rows: ${TRAIN_SIZE} / ${VAL_SIZE}" \
    "  retriever URL: ${RETRIEVER_URL}" \
    "  GPU count: 1" \
    "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" \
    "  dtype: bfloat16" \
    "  vLLM attention backend: ${VLLM_ATTENTION_BACKEND}" \
    "  tensor parallel size: ${TENSOR_PARALLEL_SIZE}" \
    "  vLLM GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
    "  rollout temperature / top-p: ${ROLLOUT_TEMPERATURE} / ${ROLLOUT_TOP_P}" \
    "  optimizer updates / epochs: ${MAX_OPTIMIZER_STEPS} / ${TOTAL_EPOCHS}" \
    "  train / validation batch size: ${TRAIN_BATCH_SIZE} / ${VAL_BATCH_SIZE}" \
    "  rollout n / n_agent: ${ROLLOUT_N} / ${N_AGENT}" \
    "  effective rollout trajectories per update: ${EFFECTIVE_ROLLOUT_TRAJECTORIES}" \
    "  total prompts / rollout trajectories: ${TOTAL_PROMPTS_CONSUMED} / ${TOTAL_ROLLOUT_TRAJECTORIES}" \
    "  PPO mini / micro batch / epochs: ${PPO_MINI_BATCH_SIZE} / ${PPO_MICRO_BATCH_SIZE} / ${PPO_EPOCHS}" \
    "  rollout / reference logprob micro batch: ${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE} / ${REFERENCE_LOGPROB_MICRO_BATCH_SIZE}" \
    "  max turns / retriever top-k: ${MAX_TURNS} / 3" \
    "  token limits (start / response / observation / prompt): ${MAX_START_LENGTH} / ${MAX_RESPONSE_LENGTH} / ${MAX_OBS_LENGTH} / ${MAX_PROMPT_LENGTH}" \
    "  KL loss / entropy coefficient: ${KL_LOSS_COEF} / ${ENTROPY_COEF}" \
    "  actor FSDP param / grad / optimizer offload: ${ACTOR_PARAM_OFFLOAD} / ${ACTOR_GRAD_OFFLOAD} / ${ACTOR_OPTIMIZER_OFFLOAD}" \
    "  reference FSDP param offload: ${REFERENCE_PARAM_OFFLOAD}" \
    "  validation: greedy, before and after only" \
    "  checkpoint save frequency: ${SAVE_FREQ}" \
    "  final checkpoint: ${CHECKPOINT_PATH}" \
    "  log: ${LOG_PATH}"
}

count_positive_metric_steps() {
  local metric_name="$1"
  awk -v key="${metric_name}:" '
    {
      start = index($0, key)
      if (start > 0) {
        value = substr($0, start + length(key))
        split(value, fields, " - ")
        if ((fields[1] + 0) > 0) {
          count += 1
        }
      }
    }
    END { print count + 0 }
  ' "${LOG_PATH}"
}

cd "${ROOT_DIR}"
log_configuration | tee "${LOG_PATH}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=false \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  algorithm.kl_ctrl.kl_coef=0.0 \
  reward_model.enable=false \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}" \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.n_agent="${N_AGENT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${REFERENCE_LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REFERENCE_PARAM_OFFLOAD}" \
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  +trainer.val_after_train=true \
  +trainer.max_optimizer_steps="${MAX_OPTIMIZER_STEPS}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.project_name=Search-R1 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps=null \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${ROOT_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns="${MAX_TURNS}" \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3 \
  2>&1 | tee -a "${LOG_PATH}"

if grep -Eqi 'Traceback|OutOfMemoryError|CUDA.*(OOM|out of memory)|:[+-]?(nan|inf)([[:space:]]|$)' "${LOG_PATH}"; then
  printf 'Phase-3 failed: fatal or non-finite output found in %s\n' "${LOG_PATH}" >&2
  exit 1
fi
if ! grep -Fq "training/optimizer_steps:${MAX_OPTIMIZER_STEPS}.000" "${LOG_PATH}"; then
  printf 'Phase-3 failed: optimizer step %s was not logged.\n' "${MAX_OPTIMIZER_STEPS}" >&2
  exit 1
fi
for validation_marker in 'Initial validation metrics:' 'Final validation metrics:'; do
  if ! grep -Fq "${validation_marker}" "${LOG_PATH}"; then
    printf 'Phase-3 failed: missing %s\n' "${validation_marker}" >&2
    exit 1
  fi
done
if ! grep -Fq "Saving actor checkpoint to ${CHECKPOINT_PATH}" "${LOG_PATH}"; then
  printf 'Phase-3 failed: final checkpoint save was not logged for %s\n' "${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -s "${CHECKPOINT_PATH}/config.json" ]]; then
  printf 'Phase-3 failed: final checkpoint config is missing: %s\n' "${CHECKPOINT_PATH}/config.json" >&2
  exit 1
fi
checkpoint_weights_found="false"
for checkpoint_weights in \
  "${CHECKPOINT_PATH}"/model*.safetensors \
  "${CHECKPOINT_PATH}"/pytorch_model*.bin; do
  if [[ -s "${checkpoint_weights}" ]]; then
    checkpoint_weights_found="true"
    break
  fi
done
if [[ "${checkpoint_weights_found}" != "true" ]]; then
  printf 'Phase-3 failed: final checkpoint weights are missing from %s\n' "${CHECKPOINT_PATH}" >&2
  exit 1
fi

printf 'TRAINING INFRA PASS: %s updates, final checkpoint, validations, and finite execution completed.\n' "${MAX_OPTIMIZER_STEPS}"

VARIANCE_STEPS="$(count_positive_metric_steps 'grpo/groups_with_reward_variance')"
ADVANTAGE_STEPS="$(count_positive_metric_steps 'grpo/nonzero_advantage_fraction')"
GRADIENT_STEPS="$(count_positive_metric_steps 'actor/grad_norm')"
printf '%s\n' \
  "Phase-3 run completed all ${MAX_OPTIMIZER_STEPS} optimizer updates." \
  "  updates with within-group reward variance: ${VARIANCE_STEPS}" \
  "  updates with non-zero GRPO advantages: ${ADVANTAGE_STEPS}" \
  "  updates with positive actor gradient norm: ${GRADIENT_STEPS}"

if (( VARIANCE_STEPS < 2 || ADVANTAGE_STEPS < 2 )); then
  printf '%s\n' \
    "REWARD-SIGNAL INCONCLUSIVE: reward variance and non-zero advantages did not persist across at least two steps." \
    "Inspect ${LOG_PATH} before changing the reward or runtime configuration." >&2
  exit 2
fi

printf 'REWARD-SIGNAL PASS: multiple updates contained reward variance and non-zero GRPO advantages.\n'

if (( GRADIENT_STEPS < 2 )); then
  printf '%s\n' \
    "OPTIMIZATION-SIGNAL INCONCLUSIVE: fewer than two updates had a finite positive actor gradient norm." \
    "Actor-side KL is enabled, so grad_norm is not pure reward attribution." >&2
  exit 2
fi

printf '%s\n' \
  "OPTIMIZATION-SIGNAL PASS: multiple optimizer updates had finite positive actor gradients." \
  "QUALITY SIGNAL: compare initial and final validation EM; improvement is not required for this gate." \
  "Phase-3 engineering gate passed; checkpoint: ${CHECKPOINT_PATH}" \
  "Full curves: ${LOG_PATH}"
