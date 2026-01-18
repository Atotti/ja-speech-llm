#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N speech-llm-ja-8gpu
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# v2: Multi-GPU training with Accelerate (8x H200)
# Format: システムプロンプト + <|reserved_343|>[audio]<|reserved_342|> + 指示 + 応答

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Arguments for accelerate launch
ARGS="--max-steps 1000000000 --batch-size 4 --grad-accumulation 4 --warmup-steps 10 --val-check-interval 5000 --lr 1e-4"

# Build model-dir and optional resume argument
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-${TIMESTAMP}"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    ARGS="$ARGS --resume-from $RESUME_FROM --model-dir $MODEL_DIR"
else
    echo "Starting new training"
    ARGS="$ARGS --model-dir $MODEL_DIR"
fi

echo "Mode: 8-GPU training with Accelerate"
echo "Arguments: $ARGS"

# Launch with Accelerate (8 GPUs)
uv run accelerate launch --num_processes 8 scripts/v2/train_accelerate.py $ARGS
