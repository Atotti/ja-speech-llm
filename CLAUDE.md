# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tutorial repository for the 2025 Audio-Visual Speech and Language Processing Symposium (音学シンポジウム 2025) on "Introduction to Multimodal Large Language Models". The project demonstrates how to build speech-aware LLMs by connecting frozen audio encoders to frozen text decoders via trainable adapters.

## Setup

```bash
pip install -r requirements.txt
sh scripts/download_clotho.sh  # Download Clotho audio captioning dataset
```

## Key Commands

```bash
# Run demo notebooks in Colab or local Jupyter
jupyter notebook demo1.ipynb  # Phi-4-Multimodal speech translation
jupyter notebook demo2.ipynb  # Llama + Whisper adapter training

# Run demo2 as Python script (for full training pipeline)
python demo2.py

# Japanese version: LLM-jp + Whisper-large-v3 + ReazonSpeech
python -c "from demo2_ja import train; train(max_steps=1000)"  # Pretrain (ASR)
python -c "from demo2_ja import finetune; finetune(model_id='...', max_steps=1000)"  # SFT
python -c "from demo2_ja import validate; validate(...)"  # Evaluation
```

## Architecture

### English Version (demo2.py)

**LlamaForSpeechLM** (`demo2.py:68`): Main model combining:
- Whisper encoder (frozen) - audio feature extraction
- Llama 3.2 1B decoder (frozen) - text generation
- Adapter module (trainable) - bridges audio to text representations

**Adapter** (`demo2.py:27`): 2-layer MLP with average pooling that projects Whisper hidden states to Llama embedding space. Only trainable component (~minimal parameters).

**Training Pipeline**:
1. `train()` (`demo2.py:421`) - Pretrain on LibriSpeech ASR + Clotho audio captioning
2. `generate_data()` (`demo2.py:526`) - Synthesize speech from Alpaca text using VITS TTS
3. `finetune()` (`demo2.py:565`) - Instruction tuning on synthetic spoken Alpaca
4. `eval()` - Evaluate on test benchmarks

### Japanese Version (demo2_ja.py)

**LlamaForSpeechLM** (`demo2_ja.py:65`): Japanese Speech LLM:
- Whisper-large-v3 encoder (frozen, 1280 dim)
- LLM-jp-4 8B decoder (frozen/LoRA/full, 4096 dim)
- Adapter module (trainable, 1280→8192→4096)

**Training Pipeline**:
1. `train()` (`demo2_ja.py:859`) - Pretrain on ReazonSpeech (Japanese ASR)
2. `finetune()` (`demo2_ja.py:1013`) - SFT on spoken-magpie-ja (instruction following)
3. `validate()` (`demo2_ja.py:597`) - Evaluate on ReazonSpeech test

**Finetune Modes** (mutually exclusive):
| Mode | Parameter | Trainable Params |
|------|-----------|------------------|
| Adapter only | (default) | ~10M |
| LoRA | `use_lora=True` | ~18M |
| Full decoder | `unfreeze_decoder=True` | ~8B |

**Chat Template** (LLM-jp format):
```
### 指示:
音声を書き起こしてください。

### 応答:
{transcript}<|eos|>
```

## Models & Datasets

### English (demo2.py)
- **Encoder**: `openai/whisper-small.en`
- **Decoder**: `meta-llama/Llama-3.2-1B-Instruct`
- **TTS for data synthesis**: `kakao-enterprise/vits-vctk`
- **Pretrained model**: `ryota-komatsu/Llama-for-SpeechLM-Instruct` (HuggingFace)
- **Datasets**: LibriSpeech (torchaudio), Clotho, spoken-alpaca

### Japanese (demo2_ja.py)
- **Encoder**: `openai/whisper-large-v3`
- **Decoder**: `/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4` (LLM-jp-4 8B)
- **Pretrain Dataset**: `japanese-asr/whisper_transcriptions.reazonspeech.all` (HuggingFace, streaming)
- **SFT Dataset**: `Atotti/spoken-magpie-ja` (HuggingFace, streaming)

## Job Scripts (ABCI)

```bash
# Pretrain (ASR)
qsub -v MODEL_DIR=models/LlamaForSpeechLM-ja scripts/train_ja.sh

# Finetune - Adapter only
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step45000 scripts/finetune_adapter_ja.sh

# Finetune - LoRA
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step45000 scripts/finetune_lora_ja.sh

# Finetune - Full decoder
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step45000 scripts/finetune_full_ja.sh

# Resume (all scripts support RESUME_FROM)
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-step1000 scripts/finetune_adapter_ja.sh
```

## Training Notes

- **Mixed Precision**: BFloat16 (no GradScaler needed)
- **LoRA dtype**: Explicitly converted to bfloat16 after PEFT initialization
- **Recommended LR**:
  - Adapter/LoRA: `1e-3` ~ `1e-4`
  - Full decoder: `1e-5`

## Hardware Requirements

- **English (demo2.py)**: NVIDIA RTX A6000 48GB VRAM (or equivalent), CUDA 12.1
- **Japanese (demo2_ja.py)**: NVIDIA H200 80GB VRAM (LLM-jp 8B + Whisper-large-v3)
