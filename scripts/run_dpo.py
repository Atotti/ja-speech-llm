#!/usr/bin/env python
"""Entry point for DPO training with accelerate."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speech_llm_ja import dpo


def main():
    parser = argparse.ArgumentParser(description="DPO training for Speech LLM")
    parser.add_argument("--model_id", type=str, default=None, help="Pretrained model path")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--model_dir", type=str, default="models/LlamaForSpeechLM-ja-DPO", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--grad_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--val_check_interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--unfreeze_decoder", action="store_true", help="Unfreeze decoder for full fine-tuning")
    parser.add_argument("--use_accelerate", action="store_true", help="Use Accelerate for multi-GPU")

    args = parser.parse_args()

    if args.model_id is None and args.resume_from is None:
        parser.error("Either --model_id or --resume_from is required")

    dpo(
        model_id=args.model_id,
        resume_from=args.resume_from,
        model_dir=args.model_dir,
        max_steps=args.max_steps,
        epoch=args.epoch,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        lr=args.lr,
        beta=args.beta,
        warmup_steps=args.warmup_steps,
        val_check_interval=args.val_check_interval,
        use_lora=args.use_lora,
        unfreeze_decoder=args.unfreeze_decoder,
        use_accelerate=args.use_accelerate,
    )


if __name__ == "__main__":
    main()
