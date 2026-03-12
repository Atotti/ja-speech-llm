#!/bin/bash
#PBS -P YOUR_PROJECT_ID
#PBS -N dpo-full-8gpu
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=72:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# Usage (予約ノード):
#   qsub -q R1512750 -v RTYPE=rt_HF,MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-step5000 scripts/dpo_full_8gpu_ja.sh
#   qsub -q R1512750 -v RTYPE=rt_HF,RESUME_FROM=models/LlamaForSpeechLM-ja-DPO-Full-step500 scripts/dpo_full_8gpu_ja.sh
#
# Usage (通常キュー):
#   qsub -q rt_HF -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-step5000 scripts/dpo_full_8gpu_ja.sh

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120
export NCCL_DEBUG=INFO

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
NUM_GPUS=8

# 43k samples / 128 (effective batch) ≈ 336 steps/epoch
# epoch=30 → ~10,000 steps total

echo "Mode: DPO 8GPU Full Decoder (adapter + full decoder)"
echo "GPUs: $NUM_GPUS"

if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
    uv run accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        scripts/run_dpo.py \
        --resume_from="${RESUME_FROM}" \
        --epoch=30 \
        --batch_size=1 \
        --grad_accumulation=16 \
        --warmup_steps=100 \
        --val_check_interval=500 \
        --beta=0.1 \
        --lr=5e-6 \
        --model_dir="models/LlamaForSpeechLM-ja-DPO-Full-8gpu-${TIMESTAMP}" \
        --unfreeze_decoder \
        --use_accelerate
elif [ -n "$MODEL_ID" ]; then
    echo "Starting from: $MODEL_ID"
    uv run accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        scripts/run_dpo.py \
        --model_id="${MODEL_ID}" \
        --epoch=30 \
        --batch_size=1 \
        --grad_accumulation=16 \
        --warmup_steps=100 \
        --val_check_interval=500 \
        --beta=0.1 \
        --lr=5e-6 \
        --model_dir="models/LlamaForSpeechLM-ja-DPO-Full-8gpu-${TIMESTAMP}" \
        --unfreeze_decoder \
        --use_accelerate
else
    echo "Error: MODEL_ID or RESUME_FROM is required"
    exit 1
fi
