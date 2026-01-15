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

# Or import from package directly
python -c "from speech_llm_ja import train; train(max_steps=1000)"
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

### Japanese Version (src/speech_llm_ja/)

The Japanese implementation is organized as a package under `src/speech_llm_ja/`. The file `demo2_ja.py` serves as a thin wrapper for backward compatibility.

**Package Structure**:
```
src/speech_llm_ja/
├── __init__.py      # Public API exports
├── model.py         # Adapter, LlamaForSpeechLMConfig, LlamaForSpeechLM
├── datasets.py      # All dataset classes (10 classes)
├── train.py         # train(), _train(), get_lr_schedule(), _save_checkpoint()
├── finetune.py      # finetune()
└── validate.py      # validate(), validate_finetune()
```

**LlamaForSpeechLM** (`src/speech_llm_ja/model.py:61`): Japanese Speech LLM:
- Whisper-large-v3 encoder (frozen, 1280 dim)
- LLM-jp-4 8B decoder (frozen/LoRA/full, 4096 dim)
- Adapter module (trainable, 1280→8192→4096)

**Training Pipeline**:
1. `train()` (`src/speech_llm_ja/train.py:176`) - Pretrain on ReazonSpeech (Japanese ASR)
2. `finetune()` (`src/speech_llm_ja/finetune.py:22`) - SFT on multiple datasets
3. `validate()` (`src/speech_llm_ja/validate.py:14`) - Evaluate on ReazonSpeech test

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

### Japanese (src/speech_llm_ja/)
- **Encoder**: `openai/whisper-large-v3`
- **Decoder**: `/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4` (LLM-jp-4 8B)

**Pretrain Datasets**:
- `japanese-asr/whisper_transcriptions.reazonspeech.all` - Japanese ASR (streaming)
- `Atotti/clotho-ja` - Audio captioning in Japanese

**SFT Datasets** (configurable in `finetune()`):
| Dataset | Parameter | Description |
|---------|-----------|-------------|
| `Atotti/spoken-magpie-ja` | `use_spoken_magpie=True` | Audio instruction following (default) |
| `Atotti/spoken-multiturn-sft` | `use_spoken_multiturn=True` | Multi-turn conversations |
| ReazonSpeech (SFT format) | `use_reazon_sft=True` | Japanese ASR (forgetting prevention) |
| `Atotti/fsd50k-cc0-Qwen3-Omni-captioned` | `use_fsd50k_cc0=True` | Audio captioning |
| `Atotti/fsd50k-ccby-Qwen3-Omni-captioned` | `use_fsd50k_ccby=True` | Audio captioning |
| `openslr/librispeech_asr` | `use_librispeech=True` | English ASR |

**Dataset Classes** (`src/speech_llm_ja/datasets.py`):
- `ReazonSpeech` - Japanese ASR (streaming, split_0~7 train, split_8 test)
- `ClothoJA` - Clotho audio captioning in Japanese
- `SpokenMagpie` - Audio instruction following
- `SpokenMultiturnSFT` - Multi-turn spoken conversations
- `FSD50KCaptioned` - FSD50K with Qwen3-Omni captions
- `LibriSpeechASR` - English ASR
- `ReazonSpeechSFT` - ReazonSpeech in SFT format
- `TextMultiturn` - Text-only multi-turn (for capability preservation)
- `InterleavedDataset` - Interleave multiple datasets with configurable weights

## Job Scripts (ABCI)

```bash
# Pretrain (ASR)
qsub scripts/train_ja.sh

# Pretrain with full decoder
qsub scripts/train_full_ja.sh

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
- **Japanese (src/speech_llm_ja/)**: NVIDIA H200 140GB VRAM (LLM-jp 8B + Whisper-large-v3)
