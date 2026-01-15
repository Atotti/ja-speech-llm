"""Training functions for speech LLM."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from .model import LlamaForSpeechLM, LlamaForSpeechLMConfig
from .datasets import ReazonSpeech, ClothoJA, InterleavedDataset
from .validate import validate


def get_lr_schedule(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * current_step / warmup_steps) / base_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


def _save_checkpoint(model: LlamaForSpeechLM, checkpoint_dir: str, use_lora: bool = False):
    """Save model checkpoint, handling LoRA if present."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if use_lora:
        # Save adapter, LoRA, and config separately
        torch.save(model.adapter.state_dict(), f"{checkpoint_dir}/adapter.pt")
        model.decoder.save_pretrained(f"{checkpoint_dir}/lora")
        model.config.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved (LoRA): {checkpoint_dir}")
    else:
        model.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}")


def _train(
    model: LlamaForSpeechLM,
    encoder_processor,
    decoder_processor,
    loader: torch.utils.data.DataLoader,
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 1,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    # validation
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 1,
    data_dir="data",
    model_dir="models/LlamaForSpeechLM-ja",
    max_steps: int = None,
    val_check_interval: int = None,
    start_step: int = 0,
    validate_fn=None,  # Custom validation function (default: validate)
    use_lora: bool = False,  # Whether to save LoRA checkpoints
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # learning rate scheduler
    # max_steps is additional steps from start_step
    if max_steps is not None:
        total_steps = max_steps
        target_step = start_step + max_steps
    else:
        total_steps = len(loader) // grad_accumulation * epoch
        target_step = start_step + total_steps

    lr_scheduler = get_lr_schedule(
        optimizer,
        total_steps,
        warmup_steps,
        lr,
        lr * 0.1,
    )

    # Note: GradScaler is not needed for bfloat16 (same exponent range as float32)
    step = start_step

    for epoch in range(1, epoch + 1):
        model.train()
        model.encoder.eval()
        model.decoder.eval()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch)
                loss = loss / grad_accumulation
            loss.backward()

            if (batch_idx + 1) % grad_accumulation == 0:
                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # update
                optimizer.step()
                optimizer.zero_grad()

                # update learning rate
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1

                # wandb log
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                }, step=step)

                # validation at interval
                if val_check_interval is not None and step % val_check_interval == 0:
                    _validate_fn = validate_fn or validate
                    _validate_fn(
                        model,
                        encoder_processor,
                        decoder_processor,
                        step,
                        batch_size,
                        max_length,
                        do_sample,
                        num_beams,
                        data_dir,
                    )
                    # save checkpoint with step number
                    checkpoint_dir = f"{model_dir}-step{step}"
                    _save_checkpoint(model, checkpoint_dir, use_lora)

                    model.train()
                    model.encoder.eval()
                    model.decoder.eval()

                # max_steps check (target_step = start_step + max_steps)
                if max_steps is not None and step >= target_step:
                    break

        # validation at epoch end (if val_check_interval is not set)
        if val_check_interval is None:
            _validate_fn = validate_fn or validate
            _validate_fn(
                model,
                encoder_processor,
                decoder_processor,
                step,
                batch_size,
                max_length,
                do_sample,
                num_beams,
                data_dir,
            )

        # save checkpoint with step number at epoch end
        checkpoint_dir = f"{model_dir}-step{step}"
        _save_checkpoint(model, checkpoint_dir, use_lora)

        if max_steps is not None and step >= target_step:
            break


def train(
    encoder_id="openai/whisper-large-v3",
    decoder_id="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4",
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 5,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    # validation
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 1,
    data_dir="data",
    model_dir="models/LlamaForSpeechLM-ja",
    max_steps: int = None,
    val_check_interval: int = None,
    wandb_project: str = "speech-llm-ja",
    resume_from: str = None,
    start_step: int = 0,
    unfreeze_decoder: bool = False,
):
    """
    Train adapter on ASR (ReazonSpeech) + AAC (ClothoJA).

    Args:
        resume_from: Path to checkpoint to resume from (e.g., "models/LlamaForSpeechLM-ja-step20000")
        start_step: Step number to resume from. max_steps is added to this.
        unfreeze_decoder: If True, unfreeze decoder for full training (~8B params).
    """
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "encoder_id": encoder_id,
            "decoder_id": decoder_id,
            "batch_size": batch_size,
            "lr": lr,
            "epoch": epoch,
            "warmup_steps": warmup_steps,
            "grad_accumulation": grad_accumulation,
            "max_steps": max_steps,
            "val_check_interval": val_check_interval,
            "resume_from": resume_from,
            "start_step": start_step,
            "unfreeze_decoder": unfreeze_decoder,
        },
    )

    if resume_from is not None:
        # Resume from checkpoint
        print(f"Resuming from checkpoint: {resume_from}")
        model = LlamaForSpeechLM.from_pretrained(resume_from).cuda()
        # Use encoder/decoder IDs from checkpoint
        encoder_id = model.config.encoder_id
        decoder_id = model.config.decoder_id
        # Auto-extract start_step from checkpoint path (e.g., "...-step20000" -> 20000)
        match = re.search(r"step(\d+)", resume_from)
        if match and start_step == 0:
            start_step = int(match.group(1))
            print(f"Auto-detected start_step: {start_step}")
    else:
        # Create new model
        model = LlamaForSpeechLM(LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id)).cuda()

    # Unfreeze decoder for full training
    if unfreeze_decoder:
        model.decoder.requires_grad_(True)
        trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"Decoder unfrozen: {trainable_params:,} / {total_params:,} params trainable")

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    # Interleave ReazonSpeech (ASR) and ClothoJA (AAC)
    # Skip first 100 samples of ClothoJA (used for validation)
    # Ratio: ASR 10 : AAC 1
    asr_dataset = ReazonSpeech(split="train")
    aac_dataset = ClothoJA(split="train", skip_samples=100, max_duration=30.0)
    dataset = InterleavedDataset([asr_dataset, aac_dataset], weights=[1, 0])  # ASRとAACの比率を設定

    def get_collate_fn(encoder_processor, decoder_processor):
        asr_prompt = """### 指示:
音声を書き起こしてください。

### 応答:
{}<|eos|>"""

        aac_prompt = """### 指示:
音声の内容を説明してください。

### 応答:
{}<|eos|>"""

        def collate_fn(
            batch: List[Tuple[torch.Tensor, int, str, int, int, int] | Tuple[torch.Tensor, int, str, List[str]]],
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of tuples.
                    ASR: (waveform, sample rate, transcript, speaker ID, chapter ID, utterance ID) - 6 elements
                    AAC: (waveform, sample rate, caption, captions) - 4 elements
            """

            encoder_inputs = encoder_processor(
                [item[0].squeeze(0).numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            # Use different prompts based on task (6-tuple=ASR, 4-tuple=AAC)
            decoder_inputs = decoder_processor(
                [
                    asr_prompt.format(item[2]) if len(item) == 6
                    else aac_prompt.format(item[2])
                    for item in batch
                ],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            return {
                "input_features": encoder_inputs.input_features,
                "input_ids": decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": decoder_inputs.attention_mask,
            }

        return collate_fn

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
    )

    _train(
        model,
        encoder_processor,
        decoder_processor,
        loader,
        batch_size,
        lr,
        epoch,
        warmup_steps,
        init_grad_scale,
        clip_grad_norm,
        grad_accumulation,
        max_length,
        do_sample,
        num_beams,
        data_dir,
        model_dir,
        max_steps,
        val_check_interval,
        start_step,
    )

    wandb.finish()
