#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N dpo
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
#   New:    qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-step5000 scripts/dpo_ja.sh
#   Resume: qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-DPO-step1000 scripts/dpo_ja.sh

echo "Mode: DPO (adapter only)"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    uv run python -c "from demo2_ja import dpo; dpo(model_id=None, resume_from='${RESUME_FROM}', max_steps=10000, batch_size=2, grad_accumulation=64, warmup_steps=100, val_check_interval=500, beta=0.1, lr=1e-5, model_dir='models/LlamaForSpeechLM-ja-DPO-${TIMESTAMP}')"
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    uv run python -c "from demo2_ja import dpo; dpo(model_id='${MODEL_ID}', max_steps=10000, batch_size=2, grad_accumulation=64, warmup_steps=100, val_check_interval=500, beta=0.1, lr=1e-5, model_dir='models/LlamaForSpeechLM-ja-DPO-${TIMESTAMP}')"
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    exit 1
fi
