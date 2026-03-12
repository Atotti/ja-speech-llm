#!/bin/bash
#PBS -P YOUR_PROJECT_ID
#PBS -q rt_HG
#PBS -N speech-llm-ja-v2
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

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:        qsub scripts/v2/train_ja.sh
#   Resume:     qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/train_ja.sh
#   Kimi-Audio: qsub -v ENCODER_ID=Atotti/Kimi-Audio-Whisper-Encoder scripts/v2/train_ja.sh

# Build resume argument if RESUME_FROM is set
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    RESUME_ARG="resume_from='${RESUME_FROM}',"
else
    echo "Starting new training"
    RESUME_ARG=""
fi

# Build encoder argument if ENCODER_ID is set
if [ -n "$ENCODER_ID" ]; then
    echo "Using encoder: $ENCODER_ID"
    ENCODER_ARG="encoder_id='${ENCODER_ID}',"
else
    echo "Using default encoder: openai/whisper-large-v3"
    ENCODER_ARG=""
fi

uv run python -c "from speech_llm_ja import train; train(${RESUME_ARG} ${ENCODER_ARG} lr=1e-4, max_steps=1000000000, batch_size=32, grad_accumulation=4, warmup_steps=10, val_check_interval=5000, model_dir='models/v2/LlamaForSpeechLM-ja-${TIMESTAMP}')"
