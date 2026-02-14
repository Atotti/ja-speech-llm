"""DPO (Direct Preference Optimization) training for speech LLM."""

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DataLoaderConfiguration
from tqdm import tqdm
from transformers import AutoTokenizer

from .model import LlamaForSpeechLM, LlamaForSpeechLMConfig
from .datasets import SpokenDPO
from .train import get_lr_schedule, _save_checkpoint
from .utils import (
    get_encoder_processor,
    pad_or_trim_audio,
    AFWHISPER_SAMPLE_RATE,
    AFWHISPER_MAX_SAMPLES,
)


def get_batch_logps(
    model: LlamaForSpeechLM,
    input_ids: torch.LongTensor,
    decoder_attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
    input_features: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    """Compute log probabilities for the given batch.

    Args:
        model: Speech LLM model
        input_ids: Token ids (batch_size, seq_length)
        decoder_attention_mask: Attention mask (batch_size, seq_length)
        labels: Labels with -100 for ignored positions (batch_size, seq_length)
        input_features: Audio features (batch_size, feature_size, feature_length)
        encoder_attention_mask: Audio attention mask

    Returns:
        Log probabilities summed over sequence (batch_size,)
    """
    # Get embeddings with audio inserted
    inputs_embeds, attention_mask, audio_lengths = model.embed(
        input_ids, decoder_attention_mask, input_features, encoder_attention_mask
    )

    # Forward through decoder
    outputs = model.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Adjust labels to match embedded sequence length (insert -100 for audio tokens)
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Build adjusted labels with audio positions masked
    audio_len = audio_lengths[0] if audio_lengths else 0
    if audio_len > 0:
        # Find audio start marker position
        from .model import AUDIO_START_TOKEN_ID
        audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[1]
        if len(audio_start_positions) == batch_size:
            insert_pos = audio_start_positions[0].item() + 1
            labels_before = labels[:, :insert_pos]
            labels_after = labels[:, insert_pos:]
            audio_labels = torch.full((batch_size, audio_len), -100, dtype=labels.dtype, device=device)
            labels = torch.cat((labels_before, audio_labels, labels_after), dim=1)
        else:
            # Fallback: prepend -100 for audio
            labels = F.pad(labels, (audio_len, 0), value=-100)

    # Compute per-token log probabilities
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for the target tokens
    # shift_labels: [batch, seq-1], log_probs: [batch, seq-1, vocab]
    loss_mask = shift_labels != -100
    shift_labels_clamped = shift_labels.clone()
    shift_labels_clamped[~loss_mask] = 0  # Replace -100 with 0 for gather

    per_token_logps = log_probs.gather(dim=-1, index=shift_labels_clamped.unsqueeze(-1)).squeeze(-1)
    per_token_logps = per_token_logps * loss_mask.float()  # Zero out ignored positions

    # Sum log probs over sequence
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute DPO loss.

    Args:
        policy_chosen_logps: Policy model log probs for chosen (batch_size,)
        policy_rejected_logps: Policy model log probs for rejected (batch_size,)
        ref_chosen_logps: Reference model log probs for chosen (batch_size,)
        ref_rejected_logps: Reference model log probs for rejected (batch_size,)
        beta: Temperature parameter

    Returns:
        loss: DPO loss (scalar)
        chosen_rewards: Implicit rewards for chosen (batch_size,)
        rejected_rewards: Implicit rewards for rejected (batch_size,)
    """
    # Compute log ratios
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Implicit rewards (for logging)
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return loss, chosen_rewards, rejected_rewards


def dpo(
    model_id: str,  # Pretrained/finetuned checkpoint path (required for DPO)
    model_dir: str = "models/LlamaForSpeechLM-ja-DPO",
    batch_size: int = 2,
    lr: float = 1e-5,
    beta: float = 0.1,
    epoch: int = 1,
    warmup_steps: int = 100,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 64,
    max_steps: Optional[int] = None,
    val_check_interval: Optional[int] = None,
    wandb_project: str = "speech-llm-ja-dpo",
    max_duration: float = 30.0,
    max_response_length: int = 2048,
    resume_from: Optional[str] = None,
    start_step: int = 0,
    # Training mode options (mutually exclusive)
    use_lora: bool = False,
    unfreeze_decoder: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # Multi-GPU
    use_accelerate: bool = False,
):
    """
    Train speech LLM with DPO on spoken preference data.

    DPO requires a pretrained/finetuned model as starting point.
    The reference model is created as a frozen copy of the initial model.

    Training modes (mutually exclusive):
        - Default: Adapter only (encoder/decoder frozen)
        - use_lora=True: Adapter + LoRA
        - unfreeze_decoder=True: Adapter + Full decoder

    Args:
        model_id: Path to pretrained/finetuned model checkpoint (required)
        model_dir: Directory to save DPO checkpoints
        batch_size: Batch size per GPU
        lr: Learning rate (recommended: 1e-5 ~ 1e-6 for DPO)
        beta: DPO temperature parameter (default: 0.1)
        epoch: Number of epochs
        warmup_steps: Warmup steps for learning rate scheduler
        clip_grad_norm: Gradient clipping norm
        grad_accumulation: Gradient accumulation steps
        max_steps: Maximum training steps (overrides epoch if set)
        val_check_interval: Validation interval in steps
        wandb_project: Wandb project name
        max_duration: Maximum audio duration in seconds
        max_response_length: Maximum response length
        resume_from: Resume from DPO checkpoint
        start_step: Starting step number
        use_lora: Use LoRA for decoder training
        unfreeze_decoder: Unfreeze decoder for full fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: LoRA target modules
        use_accelerate: Use HuggingFace Accelerate for multi-GPU
    """
    if use_lora and unfreeze_decoder:
        raise ValueError("use_lora and unfreeze_decoder are mutually exclusive")
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # Initialize Accelerator for multi-GPU
    accelerator = None
    if use_accelerate:
        dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
        accelerator = Accelerator(mixed_precision="bf16", dataloader_config=dataloader_config)
        is_main_process = accelerator.is_main_process
    else:
        is_main_process = True

    # Auto-extract start_step from resume_from path
    if resume_from is not None and start_step == 0:
        match = re.search(r"step(\d+)", resume_from)
        if match:
            start_step = int(match.group(1))
            if is_main_process:
                print(f"Auto-detected start_step: {start_step}")

    # Check if resuming from LoRA checkpoint
    lora_checkpoint_path = None
    if resume_from is not None:
        lora_path = Path(resume_from) / "lora"
        if lora_path.exists():
            lora_checkpoint_path = lora_path
            use_lora = True
            if is_main_process:
                print(f"Detected LoRA checkpoint, enabling LoRA mode")

    # Initialize wandb
    if is_main_process:
        wandb.init(
            project=wandb_project,
            config={
                "model_id": model_id,
                "batch_size": batch_size,
                "lr": lr,
                "beta": beta,
                "max_steps": max_steps,
                "resume_from": resume_from,
                "start_step": start_step,
                "use_lora": use_lora,
                "unfreeze_decoder": unfreeze_decoder,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "use_accelerate": use_accelerate,
                "num_processes": accelerator.num_processes if accelerator else 1,
            },
        )

    # Load policy model
    if resume_from is not None and lora_checkpoint_path is not None:
        # Resume from LoRA checkpoint
        if is_main_process:
            print(f"Resuming from LoRA checkpoint: {resume_from}")
        from peft import PeftModel

        config = LlamaForSpeechLMConfig.from_pretrained(resume_from)
        model = LlamaForSpeechLM(config)
        if not use_accelerate:
            model = model.cuda()
        adapter_path = Path(resume_from) / "adapter.pt"
        model.adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
        model.decoder = PeftModel.from_pretrained(model.decoder, str(lora_checkpoint_path))
        if is_main_process:
            print(f"Loaded adapter from {adapter_path}")
            print(f"Loaded LoRA from {lora_checkpoint_path}")
    elif resume_from is not None:
        # Resume from non-LoRA checkpoint
        if is_main_process:
            print(f"Resuming from checkpoint: {resume_from}")
        model = LlamaForSpeechLM.from_pretrained(resume_from)
        if not use_accelerate:
            model = model.cuda()
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            model.decoder = get_peft_model(model.decoder, lora_config)
            if is_main_process:
                print("Applied fresh LoRA to resumed model")
    else:
        # Load from model_id (required for DPO)
        if is_main_process:
            print(f"Loading model from: {model_id}")
        model = LlamaForSpeechLM.from_pretrained(model_id)
        if not use_accelerate:
            model = model.cuda()
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            model.decoder = get_peft_model(model.decoder, lora_config)
            for name, param in model.decoder.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)
            if is_main_process:
                print("Applied LoRA to decoder (bfloat16)")

    if use_lora and is_main_process:
        model.decoder.print_trainable_parameters()

    # Unfreeze decoder for full fine-tuning
    if unfreeze_decoder:
        model.decoder.requires_grad_(True)
        trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.decoder.parameters())
        if is_main_process:
            print(f"Decoder unfrozen: {trainable_params:,} / {total_params:,} params trainable")

    # Create reference model (frozen copy)
    if is_main_process:
        print("Creating reference model (frozen copy)...")
    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()

    # Get processors
    model_encoder_type = getattr(model.config, 'encoder_type', 'whisper')
    encoder_processor = get_encoder_processor(model.config.encoder_id, model_encoder_type)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    if is_main_process:
        print(f"Encoder type: {model_encoder_type}, ID: {model.config.encoder_id}")

    # Build dataset
    dataset = SpokenDPO(max_duration=max_duration, max_response_length=max_response_length)

    # Prompt template for DPO
    prompt_template = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DPO batch."""
        # Process audio
        if model_encoder_type in ("afwhisper", "qwen2-audio"):
            audios = []
            original_lengths = []
            for item in batch:
                audio = item["audio"].numpy()
                original_lengths.append(len(audio))
                audio = pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES)
                audios.append(audio)

            encoder_inputs = encoder_processor(
                audios,
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=AFWHISPER_SAMPLE_RATE,
            )

            mel_length = encoder_inputs.input_features.shape[-1]
            original_mel_lengths = [min(int(l / 160), mel_length) for l in original_lengths]
            encoder_attention_mask = torch.zeros(len(batch), mel_length, dtype=torch.long)
            for i, mel_len in enumerate(original_mel_lengths):
                encoder_attention_mask[i, :mel_len] = 1
        else:
            encoder_inputs = encoder_processor(
                [item["audio"].numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            encoder_attention_mask = encoder_inputs.attention_mask

        # Tokenize chosen responses
        chosen_texts = [prompt_template.format(item["instruction"], item["chosen"]) for item in batch]
        chosen_inputs = decoder_processor(chosen_texts, padding=True, return_tensors="pt")

        # Tokenize rejected responses
        rejected_texts = [prompt_template.format(item["instruction"], item["rejected"]) for item in batch]
        rejected_inputs = decoder_processor(rejected_texts, padding=True, return_tensors="pt")

        # Create labels (mask prompt, only compute loss on response)
        # Find "### 応答:\n" position to create labels
        def create_labels(input_ids: torch.LongTensor) -> torch.LongTensor:
            """Create labels by masking prompt portion."""
            labels = input_ids.clone()
            # Mask everything before and including "### 応答:\n"
            # We need to find the response start position
            response_marker = decoder_processor.encode("### 応答:\n", add_special_tokens=False)
            marker_len = len(response_marker)

            for i in range(labels.shape[0]):
                seq = input_ids[i].tolist()
                # Find the marker position
                marker_pos = -1
                for j in range(len(seq) - marker_len + 1):
                    if seq[j:j + marker_len] == response_marker:
                        marker_pos = j + marker_len
                        break

                if marker_pos > 0:
                    labels[i, :marker_pos] = -100
                else:
                    # Fallback: mask first 50% as prompt
                    labels[i, :labels.shape[1] // 2] = -100

            return labels

        chosen_labels = create_labels(chosen_inputs.input_ids)
        rejected_labels = create_labels(rejected_inputs.input_ids)

        result = {
            "input_features": encoder_inputs.input_features,
            "encoder_attention_mask": encoder_attention_mask,
            "chosen_input_ids": chosen_inputs.input_ids,
            "chosen_attention_mask": chosen_inputs.attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_inputs.input_ids,
            "rejected_attention_mask": rejected_inputs.attention_mask,
            "rejected_labels": rejected_labels,
        }

        if not use_accelerate:
            result = {k: v.cuda() for k, v in result.items()}

        return result

    loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler
    if max_steps is not None:
        total_steps = max_steps
        target_step = start_step + max_steps
    else:
        total_steps = len(loader) // grad_accumulation * epoch
        target_step = start_step + total_steps

    lr_scheduler = get_lr_schedule(optimizer, total_steps, warmup_steps, lr, lr * 0.1)

    # Prepare with Accelerate if distributed
    if use_accelerate:
        model, optimizer, loader, lr_scheduler = accelerator.prepare(
            model, optimizer, loader, lr_scheduler
        )
        # Move ref_model to same device
        ref_model = ref_model.to(accelerator.device)

    step = start_step

    for epoch_num in range(1, epoch + 1):
        model.train()
        # Keep encoder in eval mode (frozen)
        unwrapped = accelerator.unwrap_model(model) if use_accelerate else model
        unwrapped.encoder.eval()

        loader_iter = tqdm(loader, desc=f"epoch {epoch_num}", disable=not is_main_process)

        for batch_idx, batch in enumerate(loader_iter):
            with accelerator.autocast() if use_accelerate else torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Get the actual model (unwrap for method access, but keep gradient flow)
                policy_model = accelerator.unwrap_model(model) if use_accelerate else model

                # Compute policy log probs for chosen
                policy_chosen_logps = get_batch_logps(
                    policy_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                    batch["input_features"],
                    batch["encoder_attention_mask"],
                )

                # Compute policy log probs for rejected
                policy_rejected_logps = get_batch_logps(
                    policy_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                    batch["input_features"],
                    batch["encoder_attention_mask"],
                )

                # Compute reference log probs (no grad)
                with torch.no_grad():
                    ref_chosen_logps = get_batch_logps(
                        ref_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                        batch["input_features"],
                        batch["encoder_attention_mask"],
                    )
                    ref_rejected_logps = get_batch_logps(
                        ref_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                        batch["input_features"],
                        batch["encoder_attention_mask"],
                    )

                # Compute DPO loss
                loss, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=beta,
                )
                loss = loss / grad_accumulation

            if use_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accumulation == 0:
                # Gradient clipping
                if use_accelerate:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Update
                optimizer.step()
                optimizer.zero_grad()

                # Update learning rate
                current_lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1

                # Wandb logging
                if is_main_process:
                    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                    reward_margin = (chosen_rewards - rejected_rewards).mean().item()

                    wandb.log({
                        "train/loss": loss.item() * grad_accumulation,
                        "train/lr": current_lr,
                        "train/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        "train/rewards/chosen": chosen_rewards.mean().item(),
                        "train/rewards/rejected": rejected_rewards.mean().item(),
                        "train/rewards/accuracy": reward_accuracy,
                        "train/rewards/margin": reward_margin,
                        "train/logps/policy_chosen": policy_chosen_logps.mean().item(),
                        "train/logps/policy_rejected": policy_rejected_logps.mean().item(),
                    }, step=step)

                # Checkpoint at interval
                if val_check_interval is not None and step % val_check_interval == 0:
                    if is_main_process:
                        checkpoint_dir = f"{model_dir}-step{step}"
                        _save_checkpoint(model, checkpoint_dir, use_lora, accelerator)

                    if use_accelerate:
                        accelerator.wait_for_everyone()

                    model.train()
                    unwrapped = accelerator.unwrap_model(model) if use_accelerate else model
                    unwrapped.encoder.eval()

                # Max steps check
                if max_steps is not None and step >= target_step:
                    break

        # Save checkpoint at epoch end
        if is_main_process:
            checkpoint_dir = f"{model_dir}-step{step}"
            _save_checkpoint(model, checkpoint_dir, use_lora, accelerator)

        if use_accelerate:
            accelerator.wait_for_everyone()

        if max_steps is not None and step >= target_step:
            break

    if is_main_process:
        wandb.finish()
