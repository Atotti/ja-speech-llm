#!/bin/bash
#PBS -P YOUR_PROJECT_ID
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
#   New:        qsub -v MODEL_ID=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/finetune_full_ja_8gpu.sh
#   Resume:     qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-Full-step1000 scripts/v2/finetune_full_ja_8gpu.sh
#   Weights:    qsub -v MODEL_ID=...,DATASET_WEIGHTS=6_2_9_1_1_1_1 scripts/v2/finetune_full_ja_8gpu.sh
#   Fresh+Kimi: qsub -v ENCODER_ID=Atotti/Kimi-Audio-Whisper-Encoder scripts/v2/finetune_full_ja_8gpu.sh
#   Qwen2-Audio: qsub -v ENCODER_ID=Atotti/qwen2-audio-encoder,ENCODER_TYPE=qwen2-audio,DATASET_WEIGHTS=6_2_9_1_1_1_1 scripts/v2/finetune_full_ja_8gpu.sh

echo "Mode: Full decoder - 8 GPU (v2)"

ARGS="--unfreeze-decoder --use-text-multiturn True --max-steps 1000000000 --batch-size 2 --grad-accumulation 8 --warmup-steps 100 --val-check-interval 1000 --lr 1e-4"

# Parse ENCODER_TYPE environment variable (default: whisper)
ENCODER_TYPE_VALUE=${ENCODER_TYPE:-whisper}
ARGS="$ARGS --encoder-type $ENCODER_TYPE_VALUE"
echo "Encoder type: $ENCODER_TYPE_VALUE"
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-Instruct-Full-${TIMESTAMP}"

# Parse DATASET_WEIGHTS environment variable (underscore-separated, e.g., "6_2_9_1_1_1_1")
if [ -n "$DATASET_WEIGHTS" ]; then
    WEIGHTS_SPACED=$(echo "$DATASET_WEIGHTS" | tr '_' ' ')
    ARGS="$ARGS --dataset-weights $WEIGHTS_SPACED"
    echo "Dataset weights: $WEIGHTS_SPACED"
else
    echo "Dataset weights: default"
fi

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    ARGS="$ARGS --resume-from $RESUME_FROM --model-dir $MODEL_DIR"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    ARGS="$ARGS --model-id $MODEL_ID --model-dir $MODEL_DIR"
elif [ -n "$ENCODER_ID" ]; then
    echo "Creating fresh model with encoder: $ENCODER_ID"
    ARGS="$ARGS --encoder-id $ENCODER_ID --model-dir $MODEL_DIR"
else
    echo "Error: MODEL_ID, RESUME_FROM, or ENCODER_ID is required"
    exit 1
fi

echo "Arguments: $ARGS"

uv run accelerate launch --num_processes 8 scripts/v2/finetune_multi_gpu.py $ARGS
