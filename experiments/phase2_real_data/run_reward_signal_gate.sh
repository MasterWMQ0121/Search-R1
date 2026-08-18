#!/usr/bin/env bash
# GPU COMMAND PREPARED ONLY. One real-data reward-signal optimizer gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${PHASE2_REAL_DATA_DIR:-/workspace/searchr1-assets/datasets/phase2_real_data_gate}"
PYTHON_BIN="${PHASE2_REAL_PYTHON:-python3}"
RETRIEVER_URL="${PHASE2_REAL_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
BASE_MODEL="${PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
EXPERIMENT_NAME="phase2-real-data-qwen2.5-3b-reward-signal-gate"
LOG_PATH="${PHASE2_REAL_LOG_PATH:-${ROOT_DIR}/${EXPERIMENT_NAME}.log}"

MAX_START_LENGTH="768"
MAX_RESPONSE_LENGTH="128"
MAX_OBS_LENGTH="256"
MAX_TURNS="2"
MAX_PROMPT_LENGTH="$((MAX_START_LENGTH + MAX_RESPONSE_LENGTH * (MAX_TURNS - 1) + MAX_OBS_LENGTH * MAX_TURNS))"
TRAIN_BATCH_SIZE="4"
ROLLOUT_N="1"
N_AGENT="4"
EFFECTIVE_ROLLOUT_TRAJECTORIES="$((TRAIN_BATCH_SIZE * ROLLOUT_N * N_AGENT))"
PPO_MINI_BATCH_SIZE="16"
PPO_MICRO_BATCH_SIZE="1"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="1"
REFERENCE_LOGPROB_MICRO_BATCH_SIZE="1"
TENSOR_PARALLEL_SIZE="1"
VLLM_GPU_MEMORY_UTILIZATION="0.20"
ROLLOUT_TEMPERATURE="1.0"
ROLLOUT_TOP_P="0.95"
KL_LOSS_COEF="0.0"
ENTROPY_COEF="0.0"
ACTOR_PARAM_OFFLOAD="true"
ACTOR_GRAD_OFFLOAD="false"
ACTOR_OPTIMIZER_OFFLOAD="true"
REFERENCE_PARAM_OFFLOAD="true"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

[[ -s "${DATA_DIR}/train.parquet" ]] || { printf 'Missing prepared train parquet: %s\n' "${DATA_DIR}/train.parquet" >&2; exit 1; }
[[ -s "${DATA_DIR}/test.parquet" ]] || { printf 'Missing prepared test parquet: %s\n' "${DATA_DIR}/test.parquet" >&2; exit 1; }

log_configuration() {
  printf '%s\n' \
    "Phase-2 real-data Qwen2.5-3B GRPO reward-signal gate" \
    "  mode: one optimizer update; real NQ/HotpotQA reward-signal diagnostic" \
    "  model: ${BASE_MODEL}" \
    "  data: ${DATA_DIR}/train.parquet" \
    "  retriever URL: ${RETRIEVER_URL}" \
    "  GPU count: 1" \
    "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" \
    "  dtype: bfloat16" \
    "  vLLM attention backend: ${VLLM_ATTENTION_BACKEND}" \
    "  tensor parallel size: ${TENSOR_PARALLEL_SIZE}" \
    "  vLLM GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
    "  rollout temperature: ${ROLLOUT_TEMPERATURE}" \
    "  rollout top-p: ${ROLLOUT_TOP_P}" \
    "  train batch size: ${TRAIN_BATCH_SIZE}" \
    "  rollout n: ${ROLLOUT_N}" \
    "  n_agent: ${N_AGENT}" \
    "  effective rollout trajectories: ${EFFECTIVE_ROLLOUT_TRAJECTORIES}" \
    "  PPO mini / micro batch: ${PPO_MINI_BATCH_SIZE} / ${PPO_MICRO_BATCH_SIZE}" \
    "  rollout / reference logprob micro batch: ${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE} / ${REFERENCE_LOGPROB_MICRO_BATCH_SIZE}" \
    "  max turns: ${MAX_TURNS}" \
    "  retriever top-k: 3" \
    "  token limits (start / response / observation / prompt): ${MAX_START_LENGTH} / ${MAX_RESPONSE_LENGTH} / ${MAX_OBS_LENGTH} / ${MAX_PROMPT_LENGTH}" \
    "  KL loss coefficient: ${KL_LOSS_COEF}" \
    "  entropy coefficient: ${ENTROPY_COEF}" \
    "  actor FSDP param offload: ${ACTOR_PARAM_OFFLOAD}" \
    "  actor FSDP gradient offload: ${ACTOR_GRAD_OFFLOAD}" \
    "  actor FSDP optimizer offload: ${ACTOR_OPTIMIZER_OFFLOAD}" \
    "  reference FSDP param offload: ${REFERENCE_PARAM_OFFLOAD}" \
    "  log: ${LOG_PATH}"
}

cd "${ROOT_DIR}"
log_configuration | tee "${LOG_PATH}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size=4 \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=false \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  algorithm.kl_ctrl.kl_coef=0.0 \
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
  +trainer.val_before_train=false \
  +trainer.val_after_train=false \
  +trainer.max_optimizer_steps=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=Search-R1 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${ROOT_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns="${MAX_TURNS}" \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3 \
  2>&1 | tee -a "${LOG_PATH}"

if grep -Eq 'grpo/groups_with_reward_variance:0\.000([[:space:]]|$)' "${LOG_PATH}"; then
  printf '%s\n' \
    "gate inconclusive: sampled prompts produced no within-group reward variance" \
    "Retry with the next deterministic four-prompt batch; see experiments/phase2_real_data/README.md."
  exit 2
fi

printf '%s\n' "Gate execution completed with within-group reward variance; verify all documented PASS criteria in ${LOG_PATH}."
