#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N speech-llm-ja-full
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:    qsub scripts/train_full_ja.sh
#   Resume: qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Full-step20000 scripts/train_full_ja.sh

echo "Mode: Full decoder (adapter + full decoder ~8B params)"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    RESUME_ARG="resume_from='${RESUME_FROM}',"
else
    echo "Starting new training"
    RESUME_ARG=""
fi

uv run python -c "from demo2_ja import train; train(${RESUME_ARG} unfreeze_decoder=True, lr=1e-4, max_steps=10000000, batch_size=16, grad_accumulation=8, warmup_steps=100, val_check_interval=5000, model_dir='models/LlamaForSpeechLM-ja-Full-${TIMESTAMP}')"
