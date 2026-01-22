#!/usr/bin/env python
"""Quick inference test using a pushed Hub repo."""

import argparse
from typing import Any, Dict

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.audio_utils import load_audio


def _move_to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype.is_floating_point:
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--audio-url",
        default="https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
    )
    parser.add_argument(
        "--text",
        default="What can you tell me about this audio?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = Accelerator().device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        args.repo_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    audio = load_audio(args.audio_url, sampling_rate=16000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": args.text},
            ],
        }
    ]
    model_inputs = processor(
        messages,
        audios=[audio],
        add_generation_prompt=True,
        return_labels=False,
    )
    model_inputs = _move_to_device(model_inputs, device, dtype)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs, max_new_tokens=args.max_new_tokens
        )

    prompt_len = int(model_inputs["decoder_attention_mask"][0].sum().item())
    new_tokens = outputs[0, prompt_len:]
    reply = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\nGenerated response:")
    print("=" * 80)
    print(reply)
    print("=" * 80)


if __name__ == "__main__":
    main()
