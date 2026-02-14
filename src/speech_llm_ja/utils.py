"""Shared utilities for speech LLM."""

import numpy as np
from transformers import AutoProcessor, AutoFeatureExtractor


# AFWhisper/Qwen2-Audio constants
AFWHISPER_SAMPLE_RATE = 16000
AFWHISPER_MAX_DURATION = 30  # seconds
AFWHISPER_MAX_SAMPLES = AFWHISPER_SAMPLE_RATE * AFWHISPER_MAX_DURATION  # 480000


def get_encoder_processor(encoder_id: str, encoder_type: str):
    """Get the appropriate processor for the encoder type."""
    if encoder_type in ("afwhisper", "qwen2-audio"):
        # Qwen2AudioEncoder-based models use Qwen2-Audio feature extractor
        return AutoFeatureExtractor.from_pretrained("Qwen/Qwen2-Audio-7B")
    else:
        # Whisper uses AutoProcessor
        return AutoProcessor.from_pretrained(encoder_id)


def pad_or_trim_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim audio to target length (for AFWhisper 30s fixed input)."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio
