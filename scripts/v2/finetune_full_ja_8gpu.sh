#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N sft-full-8gpu
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# v2: Multi-GPU finetune with Accelerate (8x H200) - Full decoder
# Format: システムプロンプト + <|reserved_343|>[audio]<|reserved_342|> + 指示 + 応答

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:    qsub -v MODEL_ID=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/finetune_full_ja_8gpu.sh
#   Resume: qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-Full-step1000 scripts/v2/finetune_full_ja_8gpu.sh

echo "Mode: Full decoder - 8 GPU (v2)"

ARGS="--unfreeze-decoder --max-steps 1000000000 --batch-size 1 --grad-accumulation 16 --warmup-steps 100 --val-check-interval-samples 10000 --lr 5e-5"
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-Instruct-Full-${TIMESTAMP}"

if [ -n "$RESUME_FROM" ]; then
  echo "Resuming from: $RESUME_FROM"
  ARGS="$ARGS --resume-from $RESUME_FROM --model-dir $MODEL_DIR"
elif [ -n "$MODEL_ID" ]; then
  echo "Starting from: $MODEL_ID"
  ARGS="$ARGS --model-id $MODEL_ID --model-dir $MODEL_DIR"
else
  echo "Error: MODEL_ID or RESUME_FROM is required"
  exit 1
fi

echo "Arguments: $ARGS"

uv run accelerate launch --num_processes 8 scripts/v2/finetune_accelerate.py $ARGS
