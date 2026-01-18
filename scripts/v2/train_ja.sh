#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N speech-llm-ja-v2
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=12:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# v2: New prompt format with audio markers and system prompt
# Format: システムプロンプト + <|reserved_343|>[audio]<|reserved_342|> + 指示 + 応答

cd $PBS_O_WORKDIR

# ログ用の設定
export PYTHONUNBUFFERED=1
JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/train-1gpu-$JOBID.out
ERRFILE=./logs/train-1gpu-$JOBID.err
exec >$LOGFILE 2>$ERRFILE
# ログ用の設定終わり

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Build resume argument if RESUME_FROM is set
# Usage: qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-step20000 scripts/v2/train_ja.sh
if [ -n "$RESUME_FROM" ]; then
  echo "Resuming from: $RESUME_FROM"
  RESUME_ARG="--resume-from $RESUME_FROM"
else
  echo "Starting new training"
  RESUME_ARG=""
fi

COMMON_ARGS="--decoder-id /groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4 --max-steps 20000 --batch-size 16 --grad-accumulation 1 --warmup-steps 100 --val-check-interval-samples 100000 --lr 1e-4"
MODEL_DIR="models/v2/LlamaForSpeechLM-ja-${TIMESTAMP}"

uv run accelerate launch --num_processes 1 scripts/v2/train_accelerate.py $COMMON_ARGS $RESUME_ARG --model-dir $MODEL_DIR
