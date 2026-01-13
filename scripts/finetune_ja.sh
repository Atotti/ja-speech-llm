#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N speech-llm-ja-sft
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=48:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5


# HuggingFace timeout settings (avoid network timeout during validation)
export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New finetune: qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step20000 scripts/finetune_ja.sh
#   Resume:       qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-step1000 scripts/finetune_ja.sh

if [ -n "$RESUME_FROM" ]; then
    # Resume mode
    echo "Resuming from: $RESUME_FROM"
    uv run python -c "from demo2_ja import finetune; finetune(resume_from='${RESUME_FROM}', max_steps=100000, batch_size=8, grad_accumulation=8, warmup_steps=100, val_check_interval=1000, model_dir='models/LlamaForSpeechLM-ja-Instruct-${TIMESTAMP}')"
elif [ -n "$MODEL_ID" ]; then
    # New finetune mode
    echo "Starting new finetune from: $MODEL_ID"
    uv run python -c "from demo2_ja import finetune; finetune(model_id='${MODEL_ID}', max_steps=100000, batch_size=8, grad_accumulation=8, warmup_steps=100, val_check_interval=1000, model_dir='models/LlamaForSpeechLM-ja-Instruct-${TIMESTAMP}')"
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    echo "New finetune: qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step20000 scripts/finetune_ja.sh"
    echo "Resume:       qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-step1000 scripts/finetune_ja.sh"
    exit 1
fi
