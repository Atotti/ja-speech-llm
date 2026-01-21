#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N sft-lora-v2
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# v2: New prompt format with audio markers and system prompt
# Format: システムプロンプト + <|reserved_343|>[audio]<|reserved_342|> + 指示 + 応答

cd $PBS_O_WORKDIR

# ログ用の設定
export PYTHONUNBUFFERED=1
JOBID=${PBS_JOBID%%.*}
if [ -n "$JOBID" ]; then
  export WANDB_NAME="finetune-lora-1gpu-${JOBID}"
fi
mkdir -p ./logs
LOGFILE=./logs/finetune-lora-1gpu-$JOBID.out
ERRFILE=./logs/finetune-lora-1gpu-$JOBID.err
exec >$LOGFILE 2>$ERRFILE
# ログ用の設定終わり

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage:
#   New:    qsub -v MODEL_ID=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/finetune_lora_ja.sh
#   Resume: qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-LoRA-step1000 scripts/v2/finetune_lora_ja.sh
#   (LoRA checkpoints are auto-detected on resume)

echo "Mode: LoRA (adapter + LoRA) (v2)"

if [ -n "$RESUME_FROM" ]; then
  echo "Resuming from: $RESUME_FROM"
  ARGS="--resume-from $RESUME_FROM"
elif [ -n "$MODEL_ID" ]; then
  echo "Starting from: $MODEL_ID"
  ARGS="--model-id $MODEL_ID"
else
  echo "Error: MODEL_ID or RESUME_FROM is required"
  exit 1
fi

COMMON_ARGS="--use-lora --max-steps 1000000000 --batch-size 4 --grad-accumulation 32 --warmup-steps 100 --val-check-interval-samples 100000 --lr 1e-4"
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-Instruct-LoRA-${TIMESTAMP}"

uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py $COMMON_ARGS $ARGS --model-dir $MODEL_DIR
