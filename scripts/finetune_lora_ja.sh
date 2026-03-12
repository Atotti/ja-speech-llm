#!/bin/bash
#PBS -P YOUR_PROJECT_ID
#PBS -q rt_HG
#PBS -N sft-lora
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:    qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step20000 scripts/finetune_lora_ja.sh
#   Resume: qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-step1000 scripts/finetune_lora_ja.sh
#   (LoRA checkpoints are auto-detected on resume)

echo "Mode: LoRA (adapter + LoRA)"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    uv run python -c "from speech_llm_ja import finetune; finetune(resume_from='${RESUME_FROM}', use_lora=True, max_steps=100000, batch_size=4, grad_accumulation=32, warmup_steps=100, val_check_interval=1000, model_dir='models/LlamaForSpeechLM-ja-Instruct-LoRA-${TIMESTAMP}')"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    uv run python -c "from speech_llm_ja import finetune; finetune(model_id='${MODEL_ID}', use_lora=True, max_steps=100000, batch_size=4, grad_accumulation=32, warmup_steps=100, val_check_interval=1000, model_dir='models/LlamaForSpeechLM-ja-Instruct-LoRA-${TIMESTAMP}')"
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    exit 1
fi
