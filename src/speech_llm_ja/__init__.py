"""Speech LLM for Japanese - Package for training and inference."""

from .model import (
    Adapter,
    LlamaForSpeechLM,
    LlamaForSpeechLMConfig,
    AUDIO_TOKEN_ID,
)
from .datasets import (
    ReazonSpeech,
    AutoMultiTurn,
    SpokenMagpie,
    SpokenMultiturnSFT,
    FSD50KCaptioned,
    LibriSpeechASR,
    ReazonSpeechSFT,
    TextMultiturn,
    InterleavedDataset,
    IF_INSTRUCTION,
)
from .processor import SpeechLlamaProcessor, SpeechLlamaProcessorConfig
from .train import train, _train, get_lr_schedule, _save_checkpoint
from .finetune import finetune
from .validate import validate, validate_finetune

__all__ = [
    # Model
    "Adapter",
    "LlamaForSpeechLM",
    "LlamaForSpeechLMConfig",
    "AUDIO_TOKEN_ID",
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
    "IF_INSTRUCTION",
    # Processor
    "SpeechLlamaProcessor",
    "SpeechLlamaProcessorConfig",
    # Training
    "train",
    "_train",
    "get_lr_schedule",
    "_save_checkpoint",
    "finetune",
    # Validation
    "validate",
    "validate_finetune",
]
