#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N sft-full-afwhisper
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# AFWhisper + LLM-JP-4-8B + Full decoder + MultiTaskSFT

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Default dataset weights: 6_2_9_1_1_1_1
WEIGHTS_ARG="dataset_weights=[6,2,9,1,1,1,1],"

echo "=== AFWhisper + LLM-JP-4-8B Full Decoder SFT ==="
echo "Encoder: Atotti/AFWhisper (encoder_type=afwhisper)"
echo "Decoder: LLM-JP-4-8B-instruct4"
echo "Mode: Full decoder (~8B params)"
echo "Batch: 2 x 64 = 128"
echo "LR: 1e-4"
echo "Walltime: 168:00:00"

uv run python -c "
from demo2_ja import finetune
finetune(
    encoder_id='Atotti/AFWhisper',
    encoder_type='afwhisper',
    ${WEIGHTS_ARG}
    unfreeze_decoder=True,
    use_text_multiturn=True,
    lr=1e-4,
    max_steps=1000000000,
    batch_size=2,
    grad_accumulation=64,
    warmup_steps=100,
    val_check_interval=1000,
    model_dir='models/v2/LlamaForSpeechLM-AFWhisper-Full-${TIMESTAMP}'
)
"
