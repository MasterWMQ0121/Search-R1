#!/usr/bin/env bash
# GPU COMMAND PREPARED ONLY. This is a one-update plumbing gate, not a quality experiment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${PHASE1_DATA_DIR:-${ROOT_DIR}/data/phase1_smoke}"
PYTHON_BIN="${PHASE2_PYTHON:-python3}"
RETRIEVER_URL="${PHASE2_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
BASE_MODEL="${PHASE2_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
EXPERIMENT_NAME="phase2-qwen2.5-3b-one-update-grpo-gate"
LOG_PATH="${PHASE2_LOG_PATH:-${ROOT_DIR}/${EXPERIMENT_NAME}.log}"

MAX_START_LENGTH="384"
MAX_RESPONSE_LENGTH="128"
MAX_OBS_LENGTH="192"
MAX_PROMPT_LENGTH="1152"
MAX_TURNS="2"
TRAIN_BATCH_SIZE="1"
ROLLOUT_N="1"
N_AGENT="2"
EFFECTIVE_ROLLOUT_TRAJECTORIES="$((TRAIN_BATCH_SIZE * ROLLOUT_N * N_AGENT))"
PPO_MINI_BATCH_SIZE="2"
PPO_MICRO_BATCH_SIZE="1"
TENSOR_PARALLEL_SIZE="1"
VLLM_GPU_MEMORY_UTILIZATION="0.20"
ACTOR_PARAM_OFFLOAD="true"
ACTOR_GRAD_OFFLOAD="false"
ACTOR_OPTIMIZER_OFFLOAD="true"
REFERENCE_PARAM_OFFLOAD="true"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

log_configuration() {
  printf '%s\n' \
    "Phase-2 Qwen2.5-3B one-optimizer-update GRPO gate" \
    "  mode: one optimizer update; plumbing validation only" \
    "  model: ${BASE_MODEL}" \
    "  data: ${DATA_DIR}/train.parquet (answer-leaky Phase-1 fixture; not model-quality evaluation)" \
    "  retriever: ${RETRIEVER_URL}" \
    "  GPU count: 1" \
    "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" \
    "  dtype: bfloat16" \
    "  vLLM attention backend: ${VLLM_ATTENTION_BACKEND}" \
    "  tensor parallel size: ${TENSOR_PARALLEL_SIZE}" \
    "  vLLM GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
    "  train batch size: ${TRAIN_BATCH_SIZE}" \
    "  rollout n: ${ROLLOUT_N}" \
    "  n_agent: ${N_AGENT}" \
    "  effective rollout trajectories per update: ${EFFECTIVE_ROLLOUT_TRAJECTORIES}" \
    "  PPO mini / micro batch: ${PPO_MINI_BATCH_SIZE} / ${PPO_MICRO_BATCH_SIZE}" \
    "  gradient checkpointing: true" \
    "  state masking: true" \
    "  actor FSDP param offload: ${ACTOR_PARAM_OFFLOAD}" \
    "  actor FSDP gradient offload: ${ACTOR_GRAD_OFFLOAD}" \
    "  actor FSDP optimizer offload: ${ACTOR_OPTIMIZER_OFFLOAD}" \
    "  reference FSDP param offload: ${REFERENCE_PARAM_OFFLOAD}" \
    "  max turns: ${MAX_TURNS}" \
    "  token limits (start / response / observation / prompt): ${MAX_START_LENGTH} / ${MAX_RESPONSE_LENGTH} / ${MAX_OBS_LENGTH} / ${MAX_PROMPT_LENGTH}" \
    "  log: ${LOG_PATH}"
}

cd "${ROOT_DIR}"
log_configuration | tee "${LOG_PATH}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num=1 \
  data.val_data_num=1 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size=1 \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=false \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
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
  retriever.topk=2 \
  2>&1 | tee -a "${LOG_PATH}"
