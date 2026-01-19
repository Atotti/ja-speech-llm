#!/usr/bin/env python
"""Finetune script for Accelerate (single or multi-GPU).

Usage:
    accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py \
        --model-id models/v2/LlamaForSpeechLM-ja-step50000

    accelerate launch --num_processes 8 scripts/v2/finetune_accelerate.py \
        --model-id models/v2/LlamaForSpeechLM-ja-step50000 --use-lora
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from speech_llm_ja import finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune Speech LLM with Accelerate")
    parser.add_argument("--model-id", default="models/v2/LlamaForSpeechLM-ja")
    parser.add_argument("--model-dir", default="models/v2/LlamaForSpeechLM-ja-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accumulation", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-check-interval-samples", type=int, default=10000)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    # Decoder training options
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--unfreeze-decoder", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    args = parser.parse_args()

    finetune(
        model_id=args.model_id,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epoch=args.epoch,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        grad_accumulation=args.grad_accumulation,
        max_steps=args.max_steps,
        val_check_interval_samples=args.val_check_interval_samples,
        resume_from=args.resume_from,
        start_step=args.start_step,
        use_lora=args.use_lora,
        unfreeze_decoder=args.unfreeze_decoder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
