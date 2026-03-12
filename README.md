# Japanese Speech LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)

日本語音声を理解できる大規模言語モデルの学習パイプライン。
frozen Whisper-large-v3 エンコーダと frozen LLM-jp デコーダを、学習可能なアダプタで接続します。

## Architecture

```
Audio Input
    │
    ▼
┌──────────────────────┐
│  Whisper-large-v3    │  (frozen, 1280 dim)
│  Encoder             │
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  Adapter (MLP)       │  (trainable, 1280 → 8192 → 4096)
│  + Average Pooling   │
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  LLM-jp 8B           │  (frozen / LoRA / full)
│  Decoder             │
└──────────────────────┘
    │
    ▼
Text Output
```

## Features

- **ASR事前学習**: ReazonSpeech (Japanese ASR) + Clotho-ja (Audio Captioning)
- **マルチタスクSFT**: 音声指示追従、マルチターン対話、音声キャプション、ASR
- **DPO**: Spoken preference data によるアラインメント
- **Gradio WebUI**: 音声入力/テキスト入力に対応したデモUI
- **複数の学習モード**: Adapter only / LoRA / Full decoder

## Setup

```bash
pip install -r requirements.txt
# or
uv sync
```

## Quick Start

### Gradio Demo

```bash
uv run python gradio_demo.py
```

### Training Pipeline

```python
from speech_llm_ja import train, finetune, dpo, validate

# 1. Pretrain (ASR)
train(decoder_id="your-decoder-model-path", max_steps=45000)

# 2. SFT (Multi-task)
finetune(model_id="models/LlamaForSpeechLM-ja-step45000", max_steps=5000)

# 3. DPO
dpo(model_id="models/LlamaForSpeechLM-ja-Instruct-step5000", max_steps=1000)

# 4. Evaluate
validate(model_id="models/LlamaForSpeechLM-ja-DPO-step1000")
```

### SFT Datasets

| Dataset | Parameter | Description |
|---------|-----------|-------------|
| `Atotti/spoken-magpie-ja` | `use_spoken_magpie=True` | 音声指示追従 (default) |
| `Atotti/spoken-multiturn-sft` | `use_spoken_multiturn=True` | マルチターン対話 |
| ReazonSpeech (SFT format) | `use_reazon_sft=True` | 日本語ASR (忘却防止) |
| `Atotti/fsd50k-cc0-Qwen3-Omni-captioned` | `use_fsd50k_cc0=True` | 音声キャプション |
| `openslr/librispeech_asr` | `use_librispeech=True` | 英語ASR |

### Finetune Modes

| Mode | Parameter | Trainable Params |
|------|-----------|------------------|
| Adapter only | (default) | ~10M |
| LoRA | `use_lora=True` | ~18M |
| Full decoder | `unfreeze_decoder=True` | ~8B |

## Project Structure

```
src/speech_llm_ja/
├── __init__.py      # Public API exports
├── model.py         # Adapter, LlamaForSpeechLMConfig, LlamaForSpeechLM
├── datasets.py      # Dataset classes (ReazonSpeech, SpokenMagpie, etc.)
├── train.py         # Pretrain pipeline
├── finetune.py      # SFT pipeline
├── dpo.py           # DPO pipeline
└── validate.py      # Evaluation
```

## Acknowledgements

本プロジェクトは [ryota-komatsu/slp2025](https://github.com/ryota-komatsu/slp2025) をベースに開発されました。
オリジナルのコードは音学シンポジウム 2025 チュートリアル「マルチモーダル大規模言語モデル入門」のために Ryota Komatsu 氏によって作成されたものです。

## License

MIT License (see [LICENSE](LICENSE))
