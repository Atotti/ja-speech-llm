#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N speech-llm-ja
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=48:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

uv run python -c "from demo2_ja import train; train(max_steps=1000000, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=5000, model_dir='models/LlamaForSpeechLM-ja-${TIMESTAMP}')"
