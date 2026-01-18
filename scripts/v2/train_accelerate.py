#!/usr/bin/env python
"""Train script for Accelerate (single or multi-GPU).

Usage:
    accelerate launch --num_processes 1 scripts/v2/train_accelerate.py

    accelerate launch --num_processes 8 scripts/v2/train_accelerate.py \
        --max-steps 100000 --batch-size 4 --model-dir models/v2/LlamaForSpeechLM-ja
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from speech_llm_ja import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Speech LLM with Accelerate")
    parser.add_argument("--encoder-id", default="openai/whisper-large-v3")
    parser.add_argument("--decoder-id", default="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accumulation", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-check-interval", type=int, default=5000)
    parser.add_argument("--model-dir", default="models/v2/LlamaForSpeechLM-ja")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--unfreeze-decoder", action="store_true")
    parser.add_argument("--asr-weight", type=int, default=1)
    parser.add_argument("--aac-weight", type=int, default=0)

    args = parser.parse_args()

    train(
        encoder_id=args.encoder_id,
        decoder_id=args.decoder_id,
        batch_size=args.batch_size,
        lr=args.lr,
        epoch=args.epoch,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        grad_accumulation=args.grad_accumulation,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        model_dir=args.model_dir,
        resume_from=args.resume_from,
        start_step=args.start_step,
        unfreeze_decoder=args.unfreeze_decoder,
        asr_weight=args.asr_weight,
        aac_weight=args.aac_weight,
    )


if __name__ == "__main__":
    main()
