"""Smoke test for SpeechLlamaProcessor."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speech_llm_ja.processor import SpeechLlamaProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeechLlamaProcessor smoke test")
    parser.add_argument(
        "--encoder-id",
        default="openai/whisper-large-v3",
        help="Whisper encoder model ID",
    )
    parser.add_argument(
        "--decoder-id",
        default="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4",
        help="LLM decoder model ID",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Audio length (s)")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()

    processor = SpeechLlamaProcessor.from_pretrained(
        encoder_id=args.encoder_id, decoder_id=args.decoder_id
    )

    audio = torch.zeros(int(args.sampling_rate * args.duration), dtype=torch.float32)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "音声を書き起こしてください。"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "テストです。"}],
        },
    ]

    batch = processor([messages], audios=[audio], return_labels=True)
    for key, value in batch.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
