"""Speech LLM for Japanese - Package for training and inference."""

import importlib

# Inference-only imports (no heavy dependencies)
from .model import Adapter, LlamaForSpeechLM, LlamaForSpeechLMConfig, AUDIO_START_TOKEN_ID, AUDIO_END_TOKEN_ID

# Mapping: attribute name -> submodule name (for lazy loading)
_LAZY_IMPORTS = {
    # datasets
    "ReazonSpeech": "datasets", "AutoMultiTurn": "datasets", "SpokenMagpie": "datasets",
    "SpokenMultiturnSFT": "datasets", "FSD50KCaptioned": "datasets", "LibriSpeechASR": "datasets",
    "ReazonSpeechSFT": "datasets", "TextMultiturn": "datasets", "InterleavedDataset": "datasets",
    "SpokenDPO": "datasets", "IF_INSTRUCTION": "datasets",
    # train
    "train": "train", "_train": "train", "get_lr_schedule": "train", "_save_checkpoint": "train",
    # finetune
    "finetune": "finetune",
    # dpo
    "dpo": "dpo", "dpo_loss": "dpo", "get_batch_logps": "dpo",
    # validate
    "validate": "validate", "validate_finetune": "validate",
}


def __getattr__(name):
    """Lazy import for training/dataset modules (require wandb, accelerate, etc.)."""
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache to avoid repeated lazy loading
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Model (always available)
    "Adapter",
    "LlamaForSpeechLM",
    "LlamaForSpeechLMConfig",
    "AUDIO_START_TOKEN_ID",
    "AUDIO_END_TOKEN_ID",
    # Datasets (lazy)
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
    # Training (lazy)
    "train",
    "_train",
    "get_lr_schedule",
    "_save_checkpoint",
    "finetune",
    "dpo",
    "dpo_loss",
    "get_batch_logps",
    # Validation (lazy)
    "validate",
    "validate_finetune",
]
