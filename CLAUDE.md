# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese Speech LLM: Whisper-large-v3 エンコーダと LLM-jp デコーダを学習可能なアダプタで接続し、日本語音声を理解できるLLMを構築する。

## Setup

```bash
uv sync
```

## Key Commands

```bash
# Import from package
python -c "from speech_llm_ja import train; train(max_steps=1000)"

# Gradio demo
python gradio_demo.py
```

## Architecture

**Package Structure**:
```
src/speech_llm_ja/
├── __init__.py      # Public API exports
├── model.py         # Adapter, LlamaForSpeechLMConfig, LlamaForSpeechLM
├── datasets.py      # All dataset classes (11 classes including SpokenDPO)
├── train.py         # train(), _train(), get_lr_schedule(), _save_checkpoint()
├── finetune.py      # finetune()
├── dpo.py           # dpo(), dpo_loss(), get_batch_logps()
└── validate.py      # validate(), validate_finetune()
```

**LlamaForSpeechLM** (`src/speech_llm_ja/model.py:61`):
- Whisper-large-v3 encoder (frozen, 1280 dim)
- LLM-jp decoder (frozen/LoRA/full, 4096 dim)
- Adapter module (trainable, 1280→8192→4096)

**Training Pipeline**:
1. `train()` (`src/speech_llm_ja/train.py`) - Pretrain on ReazonSpeech (Japanese ASR)
2. `finetune()` (`src/speech_llm_ja/finetune.py`) - SFT on multiple datasets
3. `dpo()` (`src/speech_llm_ja/dpo.py`) - DPO on spoken preference data
4. `validate()` (`src/speech_llm_ja/validate.py`) - Evaluate on ReazonSpeech test

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

- **Encoder**: `openai/whisper-large-v3`
- **Decoder**: LLM-jp (set via `decoder_id` parameter, default is placeholder `"your-decoder-model-path"`)

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
- `SpokenMagpie` - Audio instruction following
- `SpokenMultiturnSFT` - Multi-turn spoken conversations
- `FSD50KCaptioned` - FSD50K with Qwen3-Omni captions
- `LibriSpeechASR` - English ASR
- `ReazonSpeechSFT` - ReazonSpeech in SFT format
- `TextMultiturn` - Text-only multi-turn (for capability preservation)
- `InterleavedDataset` - Interleave multiple datasets with configurable weights
- `SpokenDPO` - Spoken preference data for DPO (`Atotti/spoken-dpo-49k`)

## Job Scripts (PBS)

```bash
# Pretrain (ASR)
qsub scripts/train_ja.sh

# Finetune - Adapter only
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-step45000 scripts/finetune_adapter_ja.sh

# DPO
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-step5000 scripts/dpo_ja.sh

# Resume (all scripts support RESUME_FROM)
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-step1000 scripts/finetune_adapter_ja.sh
```

## Training Notes

- **Mixed Precision**: BFloat16 (no GradScaler needed)
- **LoRA dtype**: Explicitly converted to bfloat16 after PEFT initialization
- **Recommended LR**:
  - Pretrain/SFT Adapter/LoRA: `1e-3` ~ `1e-4`
  - Pretrain/SFT Full decoder: `1e-5`
  - DPO: `1e-5` ~ `1e-6` (lower than SFT)
- **DPO Parameters**:
  - `beta`: Temperature parameter (default: 0.1)
  - Reference model: Frozen copy of initial policy model

## Hardware Requirements

- NVIDIA H200 140GB VRAM (LLM-jp 8B + Whisper-large-v3) or equivalent
