#!/bin/bash
#PBS -P YOUR_PROJECT_ID
#PBS -q rt_HG
#PBS -N sft-adapter-v2
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
#   New:        qsub -v MODEL_ID=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/finetune_adapter_ja.sh
#   Resume:     qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-step1000 scripts/v2/finetune_adapter_ja.sh
#   Weights:    qsub -v MODEL_ID=...,DATASET_WEIGHTS="3_1_4_1_1_1_1" scripts/v2/finetune_adapter_ja.sh
#   LR:         qsub -v MODEL_ID=...,LR=1e-4 scripts/v2/finetune_adapter_ja.sh
#   Fresh+Kimi: qsub -v ENCODER_ID=Atotti/Kimi-Audio-Whisper-Encoder scripts/v2/finetune_adapter_ja.sh

echo "Mode: Adapter only (v2)"

# Parse DATASET_WEIGHTS environment variable (underscore-separated, e.g., "6_2_9_1_1_1_1")
if [ -n "$DATASET_WEIGHTS" ]; then
    WEIGHTS_COMMA=$(echo "$DATASET_WEIGHTS" | tr '_' ',')
    WEIGHTS_ARG="dataset_weights=[${WEIGHTS_COMMA}],"
    echo "Dataset weights: [${WEIGHTS_COMMA}]"
else
    WEIGHTS_ARG=""
    echo "Dataset weights: default"
fi

# Parse LR environment variable (default: 1e-4)
LR_VALUE=${LR:-1e-4}
echo "Learning rate: ${LR_VALUE}"

# Parse ENCODER_ID environment variable (for fresh model creation)
if [ -n "$ENCODER_ID" ]; then
    ENCODER_ARG="encoder_id='${ENCODER_ID}',"
    echo "Encoder: ${ENCODER_ID}"
else
    ENCODER_ARG=""
fi

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    uv run python -c "from speech_llm_ja import finetune; finetune(resume_from='${RESUME_FROM}', ${WEIGHTS_ARG} use_text_multiturn=True, lr=${LR_VALUE}, max_steps=1000000000, batch_size=8, grad_accumulation=16, warmup_steps=100, val_check_interval=1000, model_dir='models/v2/LlamaForSpeechLM-ja-Instruct-${TIMESTAMP}')"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    uv run python -c "from speech_llm_ja import finetune; finetune(model_id='${MODEL_ID}', ${WEIGHTS_ARG} use_text_multiturn=True, lr=${LR_VALUE}, max_steps=1000000000, batch_size=8, grad_accumulation=16, warmup_steps=100, val_check_interval=1000, model_dir='models/v2/LlamaForSpeechLM-ja-Instruct-${TIMESTAMP}')"
elif [ -n "$ENCODER_ID" ]; then
    echo "Creating fresh model with encoder: $ENCODER_ID"
    uv run python -c "from speech_llm_ja import finetune; finetune(${ENCODER_ARG} ${WEIGHTS_ARG} use_text_multiturn=True, lr=${LR_VALUE}, max_steps=1000000000, batch_size=8, grad_accumulation=16, warmup_steps=100, val_check_interval=1000, model_dir='models/v2/LlamaForSpeechLM-ja-Instruct-${TIMESTAMP}')"
else
    echo "Error: MODEL_ID, RESUME_FROM, or ENCODER_ID is required"
    exit 1
fi
