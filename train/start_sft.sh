#!/bin/bash
set -euo pipefail
cd /home/ubuntu/project/OR_Coder
source /data/venv/OR_Coder/bin/activate
if [ -d "/data" ]; then
  export XDG_CACHE_HOME=/data/.cache
  export HF_HOME=/data/.cache/huggingface
  export HF_HUB_CACHE=/data/.cache/huggingface/hub
  export TRANSFORMERS_CACHE=/data/.cache/transformers
  export TORCH_HOME=/data/.cache/torch
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  mkdir -p "/data/tmp" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
fi
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="data/augment_data/IndustryOR/IndustryOR_sft.jsonl"
OUTPUT_DIR="./model_checkpoints/sft_qwen_coder_lora_checkpoint_v4"
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_SEQ_LENGTH=2048
LR=2e-5
EPOCHS=5
nohup python training/train_sft.py \
  --model_name "$MODEL" \
  --dataset "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "/data/.cache/transformers" \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --max_seq_length $MAX_SEQ_LENGTH \
  --learning_rate $LR \
  --num_epochs $EPOCHS \
  > sft_train.log 2>&1 & echo $! > sft_train.pid