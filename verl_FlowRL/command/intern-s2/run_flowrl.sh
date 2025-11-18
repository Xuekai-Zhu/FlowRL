#!/bin/bash
set -ex

# =============================================================================
# Environment Variables Setup
# =============================================================================
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=6000
export WORLD_SIZE=$NODE_COUNT
export RANK=$NODE_RANK

export TRITON_CACHE_DIR="/tmp/triton"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1

export NCCL_BLOCKING_WAIT=1  # Enable blocking wait to avoid early timeout
export NCCL_TIMEOUT=2400000  # Timeout: 40 minutes

export VLLM_USE_V1=1

# =============================================================================
# Ray Environment Variables
# =============================================================================
export RAY_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_RANK=${RANK:-0} # 0 for head node, >0 for worker nodes
export RAY_HEAD_PORT=${RAY_HEAD_PORT:-"6390"}
export RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-"10001"}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-"8266"}

# =============================================================================
# Conda Environment Setup
# =============================================================================
cd /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d
source conda.sh
conda activate verl

# =============================================================================
# Project Configuration
# =============================================================================
cd /mnt/shared-storage-user/llmit/user/xuekaizhu/FlowRL/verl_FlowRL

project_name='FlowRL_Scaling'
exp_name='FlowRL-Qwen2.5-7B-Math-v0.4.0'
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="${PWD}/work_dirs/${project_name}/${exp_name}/${timestamp}"
rollout_data_dir="${output_dir}/flowrl_train_results"
validation_data_dir="${output_dir}/flowrl_val_results"
CKPTS_DIR="${output_dir}/ckpts"
export TENSORBOARD_DIR="${output_dir}/tensorboard_log"
export VERL_FILE_LOGGER_PATH="${output_dir}/log"

# =============================================================================
# Model and Data Paths
# =============================================================================
MODEL_PATH="/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170"
TRAIN_FILE="/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.parquet"
TEST_FILE="/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-aime-2024.parquet"

# =============================================================================
# Algorithm Settings
# =============================================================================
adv_estimator=grpo

# KL settings (ref policy needed for FlowRL)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.0

# Sequence lengths
max_prompt_length=$((1024 * 2))
max_response_length=32768

# Clip parameters
clip_ratio_low=0.2
clip_ratio_high=0.28

# =============================================================================
# DAPO Tricks
# =============================================================================
# Overlong reward shaping
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Token-level loss
loss_agg_mode="token-mean"

# Dynamic Sampling
enable_filter_groups=True
filter_groups_max_num_gen_batches=10
filter_groups_metric="acc"

# =============================================================================
# Batch Size Configuration
# =============================================================================
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
train_prompt_mini_bsz=32
n_resp_per_prompt=16

# =============================================================================
# Ray Cluster Configuration
# =============================================================================
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NODE_COUNT:-1}

# Calculate total CPUs based on node count
node_count=${NODE_COUNT:-1}
total_cpus=$((node_count * 128))

# Launch Ray cluster
if [ "$RAY_RANK" -eq 0 ]; then
  ray start --head \
    --node-ip-address="$RAY_MASTER_ADDR" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats \
    --num-cpus=$total_cpus
else
  sleep 10
  ray start --address="$RAY_MASTER_ADDR:$RAY_HEAD_PORT" --block --disable-usage-stats
fi

sleep 10

# =============================================================================
# Sampling Configuration
# =============================================================================
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1
val_temperature=0.7
val_top_k=-1

# =============================================================================
# Performance Configuration
# =============================================================================
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=False
gen_tp=4

# =============================================================================
# Submit Ray Job
# =============================================================================
ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    hydra.run.dir=${output_dir} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.warmup_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.log_val_generations=1 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.resume_mode=auto
