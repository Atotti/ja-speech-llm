"""
Demo 2: Japanese Speech LLM (LLM-jp + Whisper-large-v3)

This file is a thin wrapper for backward compatibility.
The actual implementation is in src/speech_llm_ja/.

Usage:
    # Pretrain (ASR)
    python -c "from demo2_ja import train; train(max_steps=1000)"

    # Finetune (SFT)
    python -c "from demo2_ja import finetune; finetune(model_id='...', max_steps=1000)"

    # Validation
    python -c "from demo2_ja import validate; validate(...)"
"""

import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Re-export everything from the package for backward compatibility
from speech_llm_ja import (
    # Model
    Adapter,
    LlamaForSpeechLM,
    LlamaForSpeechLMConfig,
    AUDIO_START_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    # Datasets
    ReazonSpeech,
    AutoMultiTurn,
    SpokenMagpie,
    SpokenMultiturnSFT,
    FSD50KCaptioned,
    LibriSpeechASR,
    ReazonSpeechSFT,
    TextMultiturn,
    InterleavedDataset,
    SpokenDPO,
    IF_INSTRUCTION,
    # Training
    train,
    _train,
    get_lr_schedule,
    _save_checkpoint,
    finetune,
    dpo,
    dpo_loss,
    get_batch_logps,
    # Validation
    validate,
    validate_finetune,
)

__all__ = [
    # Model
    "Adapter",
    "LlamaForSpeechLM",
    "LlamaForSpeechLMConfig",
    "AUDIO_START_TOKEN_ID",
    "AUDIO_END_TOKEN_ID",
    # Datasets
    "ReazonSpeech",
    "AutoMultiTurn",
    "SpokenMagpie",
    "SpokenMultiturnSFT",
    "FSD50KCaptioned",
    "LibriSpeechASR",
    "ReazonSpeechSFT",
    "TextMultiturn",
    "InterleavedDataset",
    "SpokenDPO",
    "IF_INSTRUCTION",
    # Training
    "train",
    "_train",
    "get_lr_schedule",
    "_save_checkpoint",
    "finetune",
    "dpo",
    "dpo_loss",
    "get_batch_logps",
    # Validation
    "validate",
    "validate_finetune",
]
