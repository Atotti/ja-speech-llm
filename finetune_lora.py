"""
LoRA fine-tuning script for LlamaForSpeechLM.

This script trains both:
- Adapter (audio projection)
- Decoder LoRA (for text capability preservation)

Datasets:
- Audio: Atotti/spoken-magpie-ja
- Text: kanhatakeyama/AutoMultiTurnByCalm3-22B
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
from datasets import load_dataset, DownloadConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from demo2_ja import (
    LlamaForSpeechLM,
    AutoMultiTurn,
    SpokenMagpie,
    get_lr_schedule,
)


def finetune_lora(
    model_id: str = "models/LlamaForSpeechLM-ja",
    audio_dataset_id: str = "Atotti/spoken-magpie-ja",
    text_dataset_id: str = "kanhatakeyama/AutoMultiTurnByCalm3-22B",
    model_dir: str = "models/LlamaForSpeechLM-ja-Instruct-LoRA",
    batch_size: int = 4,
    lr: float = 1e-4,
    epoch: int = 3,
    warmup_steps: int = 100,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 32,
    max_steps: Optional[int] = None,
    audio_text_ratio: Tuple[int, int] = (1, 1),
    # LoRA config
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    wandb_project: str = "speech-llm-ja-sft-lora",
):
    """
    Finetune with LoRA on decoder + adapter training.

    This enables:
    - Audio instruction following (via adapter)
    - Text capability preservation (via decoder LoRA)

    Args:
        audio_text_ratio: Ratio of audio to text batches (e.g., (1, 1) = alternate)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA (default: q_proj, v_proj)
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "model_id": model_id,
            "audio_dataset_id": audio_dataset_id,
            "text_dataset_id": text_dataset_id,
            "batch_size": batch_size,
            "lr": lr,
            "audio_text_ratio": audio_text_ratio,
            "max_steps": max_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_target_modules": lora_target_modules,
        },
    )

    # Load model
    model = LlamaForSpeechLM.from_pretrained(model_id).cuda()

    # Apply LoRA to decoder
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.print_trainable_parameters()

    # Adapter is already trainable, encoder is frozen
    # Now decoder LoRA layers are also trainable

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    # Audio dataset (spoken-magpie-ja) - streaming with lazy filtering
    audio_dataset = SpokenMagpie(
        dataset_id=audio_dataset_id,
        max_duration=30.0,
        max_response_length=2048,
    )

    # Text dataset (AutoMultiTurn)
    text_dataset = AutoMultiTurn(dataset_id=text_dataset_id)

    # Audio collate function
    audio_prompt = """以下は、タスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}

### 応答:
{}<|eos|>"""

    def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # SpokenMagpie yields: {"instruction": str, "response": str, "audio": torch.Tensor}
        encoder_inputs = encoder_processor(
            [item["audio"].numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [audio_prompt.format(item["instruction"], item["response"]) for item in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        return {
            "input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "input_features": encoder_inputs.input_features,
            "encoder_attention_mask": encoder_inputs.attention_mask,
        }

    # Text collate function (no audio)
    text_prompt = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}

### 応答:
{}<|eos|>"""

    def text_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        decoder_inputs = decoder_processor(
            [text_prompt.format(item["instruction"], item["response"]) for item in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        return {
            "input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            # No input_features for text-only
        }

    # Note: IterableDataset doesn't support shuffle=True
    audio_loader = DataLoader(
        audio_dataset, batch_size, collate_fn=audio_collate_fn
    )
    text_loader = DataLoader(
        text_dataset, batch_size, collate_fn=text_collate_fn
    )

    # Optimizer: both adapter and LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=init_grad_scale)

    total_steps = max_steps or (len(audio_loader) * epoch)
    lr_scheduler = get_lr_schedule(optimizer, total_steps, warmup_steps, lr, lr * 0.1)

    step = 0
    audio_ratio, text_ratio = audio_text_ratio

    for ep in range(1, epoch + 1):
        model.train()
        model.encoder.eval()  # Encoder always eval (frozen)

        audio_iter = iter(audio_loader)
        text_iter = iter(text_loader)

        pbar = tqdm(desc=f"epoch {ep}")
        batch_idx = 0

        while True:
            # Alternate based on ratio
            position_in_cycle = batch_idx % (audio_ratio + text_ratio)
            if position_in_cycle < audio_ratio:
                try:
                    batch = next(audio_iter)
                    batch_type = "audio"
                except StopIteration:
                    break
            else:
                try:
                    batch = next(text_iter)
                    batch_type = "text"
                except StopIteration:
                    # Text dataset exhausted, reinitialize
                    text_iter = iter(text_loader)
                    batch = next(text_iter)
                    batch_type = "text"

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch)
                loss = loss / grad_accumulation
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    clip_grad_norm,
                )

                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

                current_lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1

                wandb.log({
                    "train/loss": loss.item() * grad_accumulation,
                    "train/lr": current_lr,
                    "train/scale": scale,
                    "train/grad_norm": grad_norm.item(),
                    "train/batch_type": 0 if batch_type == "audio" else 1,
                }, step=step)

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item() * grad_accumulation:.4f}",
                    "type": batch_type,
                })

                if max_steps is not None and step >= max_steps:
                    break

            batch_idx += 1

        pbar.close()

        # Save checkpoint
        checkpoint_dir = f"{model_dir}-step{step}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        model.adapter.save_pretrained(checkpoint_dir)

        # Save LoRA weights
        model.decoder.save_pretrained(f"{checkpoint_dir}/decoder_lora")

        # Save full model config
        model.config.save_pretrained(checkpoint_dir)

        print(f"Checkpoint saved: {checkpoint_dir}")

        if max_steps is not None and step >= max_steps:
            break

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="models/LlamaForSpeechLM-ja")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--audio-ratio", type=int, default=1)
    parser.add_argument("--text-ratio", type=int, default=1)
    args = parser.parse_args()

    finetune_lora(
        model_id=args.model_id,
        batch_size=args.batch_size,
        lr=args.lr,
        epoch=args.epoch,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        audio_text_ratio=(args.audio_ratio, args.text_ratio),
    )
