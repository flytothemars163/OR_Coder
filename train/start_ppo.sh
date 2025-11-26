#!/bin/bash
set -euo pipefail

# 切换到项目目录并激活虚拟环境
cd /home/ubuntu/project/OR_coder
source /data/venv/OR_coder/bin/activate

# 配置缓存目录（如有 /data 挂载）
if [ -d "/data" ]; then
  export XDG_CACHE_HOME=/data/.cache
  export HF_HOME=/data/.cache/huggingface
  export HF_HUB_CACHE=/data/.cache/huggingface/hub
  export TRANSFORMERS_CACHE=/data/.cache/transformers
  export TORCH_HOME=/data/.cache/torch
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  mkdir -p "/data/tmp" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
fi

# PPO 训练参数（根据需要调整）
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="data/OR-Instruct-Data-3K/OR-Instruct-Data-3K-Gurobi_sft.jsonl"
OUTPUT_DIR="./model_checkpoints/ppo_qwen_coder_lora_checkpoint_v1"

EPISODES=200
BATCH_SIZE=1
MINI_BATCH_SIZE=1
GRAD_ACCUM=1
PPO_EPOCHS=4
LR=1e-5
CLIPRANGE=0.2
KL_COEFF=0.1

# 代码相似度奖励权重（越相似奖励越高，最终 += weight * similarity）
CODE_SIM_WEIGHT=0.5

MAX_NEW_TOKENS=2048
TEMP=0.85
TOP_P=0.95
EXEC_TIMEOUT=8
REWARD_MODE="default"

# LoRA 配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 可选：加载已训练的 SFT LoRA 适配器（留空则不加载）
SFT_ADAPTER_DIR="./model_checkpoints/sft_qwen_coder_lora_checkpoint_v4/lora_adapter"
if [ -n "$SFT_ADAPTER_DIR" ] && [ -d "$SFT_ADAPTER_DIR" ]; then
  SFT_ARG="--sft_adapter_dir $SFT_ADAPTER_DIR"
else
  SFT_ARG=""
fi

LOG_EVERY=10
EARLY_STOP_PATIENCE=5
EARLY_STOP_DELTA=0.01

nohup python training/train_ppo_lora.py \
  --base_model "$MODEL" \
  --dataset "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --episodes $EPISODES \
  --batch_size $BATCH_SIZE \
  --mini_batch_size $MINI_BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --ppo_epochs $PPO_EPOCHS \
  --learning_rate $LR \
  --cliprange $CLIPRANGE \
  --kl_coeff $KL_COEFF \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMP \
  --top_p $TOP_P \
  --exec_timeout $EXEC_TIMEOUT \
  --reward_mode "$REWARD_MODE" \
  --code_sim_weight $CODE_SIM_WEIGHT \
  --bf16 \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --early_stop_patience $EARLY_STOP_PATIENCE \
  --early_stop_delta $EARLY_STOP_DELTA \
  --log_every $LOG_EVERY \
  $SFT_ARG \
  > ppo_train.log 2>&1 & echo $! > ppo_train.pid