#!/bin/bash
#PBS -P YOUR_PROJECT_ID
#PBS -q rt_HG
#PBS -N train-afwhisper-adapter
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=168:00:00
#PBS -o logs/
#PBS -e logs/
#PBS -m n

# AFWhisper + LLM-JP-4-8B + Adapter only
# Datasets: reazon_sft(9), fsd50k_cc0(1), fsd50k_ccby(1), librispeech(1)

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn/9.5

export HF_HUB_DOWNLOAD_TIMEOUT=120

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=== AFWhisper Adapter Training ==="
echo "Encoder: Atotti/AFWhisper (encoder_type=afwhisper)"
echo "Decoder: LLM-JP-4-8B-instruct4 (frozen)"
echo "Mode: Adapter only (~10M params)"
echo "Datasets: reazon_sft(9), fsd50k_cc0(1), fsd50k_ccby(1), librispeech(1)"
echo "Batch: 2 x 64 = 128"
echo "LR: 1e-3"

uv run python -c "
from speech_llm_ja import finetune
finetune(
    encoder_id='Atotti/AFWhisper',
    encoder_type='afwhisper',
    use_spoken_magpie=False,
    use_spoken_multiturn=False,
    use_reazon_sft=True,
    use_fsd50k_cc0=True,
    use_fsd50k_ccby=True,
    use_librispeech=True,
    use_text_multiturn=False,
    dataset_weights=[9, 1, 1, 1],
    unfreeze_decoder=False,
    lr=1e-3,
    max_steps=1000000000,
    batch_size=2,
    grad_accumulation=64,
    warmup_steps=100,
    val_check_interval=1000,
    model_dir='models/v2/LlamaForSpeechLM-AFWhisper-Adapter-${TIMESTAMP}'
)
"
