#!/usr/bin/env python
"""Push selected checkpoints to Hugging Face Hub."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.speech_llm_ja import LlamaForSpeechLM

# Models to upload
MODELS = [
    {
        "path": "models/v2/LlamaForSpeechLM-ja-Instruct-Full-20260121-201515-step30000",
        "repo_id": "Atotti/LlamaForSpeechLM-ja-Instruct-Full-step30000",
    },
    {
        "path": "models/LlamaForSpeechLM-ja-DPO-Full-8gpu-20260129-170227-step1000",
        "repo_id": "Atotti/LlamaForSpeechLM-ja-DPO-Full-step1000",
    },
]


def main():
    for info in MODELS:
        print(f"\n{'='*60}")
        print(f"Uploading: {info['path']}")
        print(f"To: {info['repo_id']}")
        print('='*60)

        model = LlamaForSpeechLM.from_pretrained(info["path"])

        print("Model dtype:")
        print(f"  Adapter: {next(model.adapter.parameters()).dtype}")
        print(f"  Encoder: {next(model.encoder.parameters()).dtype}")
        print(f"  Decoder: {next(model.decoder.parameters()).dtype}")

        model.push_to_hub(info["repo_id"], safe_serialization=True)
        print(f"Uploaded: {info['repo_id']}")


if __name__ == "__main__":
    main()
