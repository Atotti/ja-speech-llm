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
    uv run python -c "from demo2_ja import finetune; finetune(resume_from='${RESUME_FROM}', unfreeze_decoder=True, lr=5e-5, max_steps=1000000000, batch_size=2, grad_accumulation=64, warmup_steps=100, val_check_interval=1000, model_dir='models/v2/LlamaForSpeechLM-ja-Instruct-Full-${TIMESTAMP}')"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    uv run python -c "from demo2_ja import finetune; finetune(model_id='${MODEL_ID}', unfreeze_decoder=True, lr=5e-5, max_steps=1000000000, batch_size=2, grad_accumulation=64, warmup_steps=100, val_check_interval=1000, model_dir='models/v2/LlamaForSpeechLM-ja-Instruct-Full-${TIMESTAMP}')"
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    exit 1
fi
