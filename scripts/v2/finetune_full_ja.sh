#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N sft-full-v2
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# v2: New prompt format with audio markers and system prompt
# Format: システムプロンプト + <|reserved_343|>[audio]<|reserved_342|> + 指示 + 応答

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:    qsub -v MODEL_ID=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/finetune_full_ja.sh
#   Resume: qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-Full-step1000 scripts/v2/finetune_full_ja.sh

echo "Mode: Full decoder (adapter + full decoder ~8B params) (v2)"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    ARGS="--resume-from $RESUME_FROM"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    ARGS="--model-id $MODEL_ID"
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    exit 1
fi

COMMON_ARGS="--unfreeze-decoder --max-steps 1000000000 --batch-size 2 --grad-accumulation 64 --warmup-steps 100 --val-check-interval 1000 --lr 5e-5"
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-Instruct-Full-${TIMESTAMP}"

uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py $COMMON_ARGS $ARGS --model-dir $MODEL_DIR
