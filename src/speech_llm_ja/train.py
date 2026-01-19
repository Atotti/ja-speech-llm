"""Training functions for speech LLM."""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoProcessor, AutoTokenizer

from .model import LlamaForSpeechLM, LlamaForSpeechLMConfig
from .datasets import ReazonSpeechSFT, FSD50KCaptioned, InterleavedDataset
from .validate import validate


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    warmup_steps: int = 10
    init_grad_scale: float = 1e32
    clip_grad_norm: float = 1.0
    grad_accumulation: int = 128


@dataclass
class TrainingDataConfig:
    batch_size: int = 4
    epoch: int = 1
    data_dir: str = "data"
    model_dir: str = "models/LlamaForSpeechLM-ja"
    max_steps: int = None
    val_check_interval_samples: int = None
    start_step: int = 0


@dataclass
class ValidationConfig:
    max_length: int = 1024
    do_sample: bool = False
    num_beams: int = 1


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


def _save_checkpoint(
    model: LlamaForSpeechLM,
    checkpoint_dir: str,
    use_lora: bool = False,
    accelerator: Accelerator = None,
):
    """Save model checkpoint, handling LoRA if present."""
    if accelerator is None:
        raise ValueError("accelerator is required")
    # Only save on main process
    if not accelerator.is_main_process:
        return

    # Unwrap model for saving
    unwrapped_model = accelerator.unwrap_model(model)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if use_lora:
        # Save adapter, LoRA, and config separately
        torch.save(unwrapped_model.adapter.state_dict(), f"{checkpoint_dir}/adapter.pt")
        unwrapped_model.decoder.save_pretrained(f"{checkpoint_dir}/lora")
        unwrapped_model.config.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved (LoRA): {checkpoint_dir}")
    else:
        unwrapped_model.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}")


def _train(
    model: LlamaForSpeechLM,
    encoder_processor,
    decoder_processor,
    loader: torch.utils.data.DataLoader,
    optimizer_config: OptimizerConfig,
    training_data_config: TrainingDataConfig,
    validation_config: ValidationConfig,
    validate_fn=None,  # Custom validation function (default: validate)
    use_lora: bool = False,  # Whether to save LoRA checkpoints
    accelerator: Accelerator = None,  # Accelerator for single/multi-GPU
):
    if accelerator is None:
        raise ValueError("accelerator is required")
    is_main_process = accelerator.is_main_process

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.lr)
    val_check_interval_samples = training_data_config.val_check_interval_samples
    if val_check_interval_samples is not None:
        consumed_at_start = (
            training_data_config.start_step
            * training_data_config.batch_size
            * optimizer_config.grad_accumulation
            * accelerator.num_processes
        )
        next_val_samples = (
            (consumed_at_start // val_check_interval_samples) + 1
        ) * val_check_interval_samples
    else:
        next_val_samples = None

    # learning rate scheduler
    # max_steps is additional steps from start_step
    if training_data_config.max_steps is not None:
        total_steps = training_data_config.max_steps
        target_step = training_data_config.start_step + training_data_config.max_steps
    else:
        total_steps = (
            len(loader)
            // optimizer_config.grad_accumulation
            * training_data_config.epoch
        )
        target_step = training_data_config.start_step + total_steps

    lr_scheduler = get_lr_schedule(
        optimizer,
        total_steps,
        optimizer_config.warmup_steps,
        optimizer_config.lr,
        optimizer_config.lr * 0.1,
    )

    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler
    )

    # Note: GradScaler is not needed for bfloat16 (same exponent range as float32)
    step = training_data_config.start_step
    last_train_metrics = None

    # configなどを記録しておく〜これこそが実験管理〜
    if is_main_process:
        config_payload = {
            "optimizer": asdict(optimizer_config),
            "training_data": asdict(training_data_config),
            "validation": asdict(validation_config),
        }
        print(
            f"Training config: optimizer: {optimizer_config}, training_data: {training_data_config}, validation: {validation_config}",
            flush=True,
        )
        wandb.config.update(config_payload, allow_val_change=True)

    for epoch in range(1, training_data_config.epoch + 1):
        model.train()
        # Keep encoder/decoder in eval mode (frozen)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.encoder.eval()
        unwrapped.decoder.eval()

        accum_loss = None
        for batch_idx, batch in enumerate(loader):
            with accelerator.autocast():
                raw_loss = model(**batch)
                # loss = 1 / grad_accum * \Sigma loss
                # とすることでeffective batch size内の平均Lossになるようにする
                loss = raw_loss / optimizer_config.grad_accumulation

            if accum_loss is None:
                accum_loss = raw_loss.detach().float()
            else:
                accum_loss += raw_loss.detach().float()

            accelerator.backward(loss)

            if (batch_idx + 1) % optimizer_config.grad_accumulation == 0:
                # gradient clipping
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), optimizer_config.clip_grad_norm
                )

                # update
                optimizer.step()
                optimizer.zero_grad()

                # update learning rate
                current_lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1
                consumed_samples = (
                    step
                    * training_data_config.batch_size
                    * optimizer_config.grad_accumulation
                    * accelerator.num_processes
                )

                avg_loss = accum_loss / optimizer_config.grad_accumulation
                accum_loss = None
                last_train_metrics = {
                    "consumed_samples": consumed_samples,
                    "global_step": step,
                    "train_loss": avg_loss.item(),
                }

                # wandb log (main process only)
                if is_main_process:
                    print(
                        f"step={step}|train_loss={avg_loss.item():.6f}|consumed_samples={consumed_samples}|lr={current_lr:.6e}",
                        flush=True,
                    )
                    wandb.log(
                        {
                            "train/loss": avg_loss.item(),
                            "train/consumed_samples": consumed_samples,
                            "train/global_step": step,
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm.item()
                            if hasattr(grad_norm, "item")
                            else grad_norm,
                        },
                        step=step,
                    )

                # validation at interval (main process only)
                if (
                    val_check_interval_samples is not None
                    and consumed_samples >= next_val_samples
                ):
                    if is_main_process:
                        # Get unwrapped model for validation
                        eval_model = accelerator.unwrap_model(model)
                        _validate_fn = validate_fn or validate
                        _validate_fn(
                            eval_model,
                            encoder_processor,
                            decoder_processor,
                            step,
                            training_data_config.batch_size,
                            validation_config.max_length,
                            validation_config.do_sample,
                            validation_config.num_beams,
                            training_data_config.data_dir,
                            train_metrics=last_train_metrics,
                        )
                        # save checkpoint with step number
                        checkpoint_dir = f"{training_data_config.model_dir}-step{step}"
                        _save_checkpoint(model, checkpoint_dir, use_lora, accelerator)
                        next_val_samples += val_check_interval_samples

                    # Sync all processes after validation
                    accelerator.wait_for_everyone()

                    model.train()
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.encoder.eval()
                    unwrapped.decoder.eval()

                # max_steps check (target_step = start_step + max_steps)
                if training_data_config.max_steps is not None and step >= target_step:
                    break

        # validation at epoch end (if val_check_interval_samples is not set)
        if val_check_interval_samples is None and is_main_process:
            eval_model = accelerator.unwrap_model(model)
            _validate_fn = validate_fn or validate
            _validate_fn(
                eval_model,
                encoder_processor,
                decoder_processor,
                step,
                training_data_config.batch_size,
                validation_config.max_length,
                validation_config.do_sample,
                validation_config.num_beams,
                training_data_config.data_dir,
                train_metrics=last_train_metrics,
            )

        # save checkpoint with step number at epoch end
        if is_main_process:
            checkpoint_dir = f"{training_data_config.model_dir}-step{step}"
            _save_checkpoint(model, checkpoint_dir, use_lora, accelerator)

        accelerator.wait_for_everyone()

        if training_data_config.max_steps is not None and step >= target_step:
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
    val_check_interval_samples: int = None,
    wandb_project: str = "speech-llm-ja-harui",
    resume_from: str = None,
    start_step: int = 0,
    unfreeze_decoder: bool = False,
    # Dataset weights
    asr_weight: int = 1,
    aac_weight: int = 0,  # Default: ASR only. Set to 1 to enable AAC (FSD50K)
):
    """
    Train adapter on ASR (ReazonSpeech) + AAC (FSD50K).

    Args:
        resume_from: Path to checkpoint to resume from (e.g., "models/LlamaForSpeechLM-ja-step20000")
        start_step: Step number to resume from. max_steps is added to this.
        unfreeze_decoder: If True, unfreeze decoder for full training (~8B params).
        asr_weight: Weight for ASR dataset (ReazonSpeech). Default: 1.
        aac_weight: Weight for AAC dataset (FSD50K). Default: 0 (disabled).
    """
    # Initialize Accelerator (single GPU or multi-GPU)
    # dispatch_batches=False: each process fetches its own batch independently
    # (required for variable-length audio sequences with different padding sizes)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        mixed_precision="bf16", dataloader_config=dataloader_config
    )
    is_main_process = accelerator.is_main_process

    # Initialize wandb (main process only)
    if is_main_process:
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
                "val_check_interval_samples": val_check_interval_samples,
                "resume_from": resume_from,
                "start_step": start_step,
                "unfreeze_decoder": unfreeze_decoder,
                "asr_weight": asr_weight,
                "aac_weight": aac_weight,
                "num_processes": accelerator.num_processes,
            },
        )

    if resume_from is not None:
        # Resume from checkpoint
        if is_main_process:
            print(f"Resuming from checkpoint: {resume_from}")
        # Don't move to cuda - Accelerate handles device placement
        model = LlamaForSpeechLM.from_pretrained(resume_from)
        # Use encoder/decoder IDs from checkpoint
        encoder_id = model.config.encoder_id
        decoder_id = model.config.decoder_id
        # Auto-extract start_step from checkpoint path (e.g., "...-step20000" -> 20000)
        match = re.search(r"step(\d+)", resume_from)
        if match and start_step == 0:
            start_step = int(match.group(1))
            if is_main_process:
                print(f"Auto-d/etected start_step: {start_step}")
    else:
        # Create new model
        model = LlamaForSpeechLM(
            LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id)
        )

    # Unfreeze decoder for full training
    if unfreeze_decoder:
        model.decoder.requires_grad_(True)
        trainable_params = sum(
            p.numel() for p in model.decoder.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.decoder.parameters())
        if is_main_process:
            print(
                f"Decoder unfrozen: {trainable_params:,} / {total_params:,} params trainable"
            )

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(decoder_id)
    decoder_processor.pad_token = (
        decoder_processor.pad_token or decoder_processor.eos_token
    )

    # Build dataset: ASR (ReazonSpeech) + AAC (FSD50K)
    datasets = []
    weights = []
    dataset_names = []

    if asr_weight > 0:
        datasets.append(ReazonSpeechSFT(split="train"))
        weights.append(asr_weight)
        dataset_names.append("asr")

    if aac_weight > 0:
        datasets.append(
            FSD50KCaptioned(dataset_id="Atotti/fsd50k-cc0-Qwen3-Omni-captioned")
        )
        weights.append(aac_weight)
        dataset_names.append("aac")

    if not datasets:
        raise ValueError("At least one of asr_weight or aac_weight must be > 0")

    if is_main_process:
        print(f"Train datasets: {list(zip(dataset_names, weights))}")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = InterleavedDataset(datasets, weights)

    # Prompt template for pretrain
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    # collate: 照合する
    # ref: https://docs.pytorch.org/docs/stable/data.html#working-with-collate-fn
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dicts with keys: instruction, response, audio
        """
        # Don't move to cuda here - Accelerate handles device placement
        encoder_inputs = encoder_processor(
            [item["audio"].numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )

        decoder_inputs = decoder_processor(
            [prompt.format(item["instruction"], item["response"]) for item in batch],
            padding=True,
            return_tensors="pt",
        )

        result = {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_inputs.attention_mask,
            "decoder_attention_mask": decoder_inputs.attention_mask,
        }

        return result

    loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn)

    optimizer_config = OptimizerConfig(
        lr=lr,
        warmup_steps=warmup_steps,
        init_grad_scale=init_grad_scale,
        clip_grad_norm=clip_grad_norm,
        grad_accumulation=grad_accumulation,
    )
    training_data_config = TrainingDataConfig(
        batch_size=batch_size,
        epoch=epoch,
        data_dir=data_dir,
        model_dir=model_dir,
        max_steps=max_steps,
        val_check_interval_samples=val_check_interval_samples,
        start_step=start_step,
    )
    validation_config = ValidationConfig(
        max_length=max_length,
        do_sample=do_sample,
        num_beams=num_beams,
    )

    _train(
        model,
        encoder_processor,
        decoder_processor,
        loader,
        optimizer_config,
        training_data_config,
        validation_config,
        accelerator=accelerator,
    )

    if is_main_process:
        wandb.finish()
