#!/usr/bin/env python
"""Standalone finetune script for multi-GPU with Accelerate.

Usage:
    # 8 GPU training
    accelerate launch --num_processes 8 scripts/v2/finetune_multi_gpu.py \
        --model-id models/v2/LlamaForSpeechLM-ja-step50000

    # With LoRA
    accelerate launch --num_processes 8 scripts/v2/finetune_multi_gpu.py \
        --model-id models/v2/LlamaForSpeechLM-ja-step50000 --use-lora
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from speech_llm_ja import finetune


def main():
    parser = argparse.ArgumentParser(description="Finetune Speech LLM with multi-GPU support")
    parser.add_argument("--model-id", default=None, help="Pretrained checkpoint path (None to create fresh model)")
    parser.add_argument("--model-dir", default="models/v2/LlamaForSpeechLM-ja-Instruct")
    parser.add_argument("--encoder-id", default="openai/whisper-large-v3", help="Encoder model (for fresh model creation)")
    parser.add_argument("--decoder-id", default="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4", help="Decoder model (for fresh model creation)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accumulation", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    # Decoder training options
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--unfreeze-decoder", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    # Dataset options
    parser.add_argument("--dataset-weights", type=int, nargs="+", default=None,
                        help="Weights for datasets: [magpie, multiturn, reazon, fsd50k_cc0, fsd50k_ccby, librispeech, text_multiturn]")
    parser.add_argument("--use-text-multiturn", type=bool, default=True,
                        help="Enable text-only multiturn dataset for capability preservation")

    args = parser.parse_args()

    finetune(
        model_id=args.model_id,
        model_dir=args.model_dir,
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
        resume_from=args.resume_from,
        start_step=args.start_step,
        use_lora=args.use_lora,
        unfreeze_decoder=args.unfreeze_decoder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dataset_weights=args.dataset_weights,
        use_text_multiturn=args.use_text_multiturn,
        use_accelerate=True,  # Enable multi-GPU
    )


if __name__ == "__main__":
    main()
