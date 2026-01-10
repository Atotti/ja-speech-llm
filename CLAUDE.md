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
```

## Architecture

The core implementation is in `demo2.py` with these main components:

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

## Models & Datasets

- **Encoder**: `openai/whisper-small.en`
- **Decoder**: `meta-llama/Llama-3.2-1B-Instruct`
- **TTS for data synthesis**: `kakao-enterprise/vits-vctk`
- **Pretrained model**: `ryota-komatsu/Llama-for-SpeechLM-Instruct` (HuggingFace)
- **Datasets**: LibriSpeech (torchaudio), Clotho, spoken-alpaca

## Hardware Requirements

- NVIDIA RTX A6000 48GB VRAM (or equivalent)
- CUDA 12.1
