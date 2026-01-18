"""Finetune functions for speech LLM."""

import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoProcessor, AutoTokenizer

from .model import LlamaForSpeechLM, LlamaForSpeechLMConfig
from .datasets import (
    SpokenMagpie,
    SpokenMultiturnSFT,
    ReazonSpeechSFT,
    FSD50KCaptioned,
    LibriSpeechASR,
    InterleavedDataset,
)
from .train import _train
from .validate import validate_finetune


def finetune(
    model_id="models/LlamaForSpeechLM-ja",
    model_dir="models/LlamaForSpeechLM-ja-Instruct",
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 5,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    max_steps: int = None,
    val_check_interval: int = None,
    wandb_project: str = "speech-llm-ja-sft",
    max_duration: float = 30.0,
    max_response_length: int = 2048,
    resume_from: str = None,
    start_step: int = 0,
    # Decoder training options (mutually exclusive)
    use_lora: bool = False,
    unfreeze_decoder: bool = False,
    # LoRA options (only used if use_lora=True)
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # Dataset selection and weights
    use_spoken_magpie: bool = True,
    use_spoken_multiturn: bool = True,
    use_reazon_sft: bool = True,
    use_fsd50k_cc0: bool = True,
    use_fsd50k_ccby: bool = True,
    use_librispeech: bool = True,
    dataset_weights: List[int] = None,  # Default: [2, 1, 4, 1, 1, 1] when all enabled
):
    """
    Finetune adapter on multiple audio instruction datasets.

    Training modes:
        - Default: Adapter only (encoder/decoder frozen)
        - use_lora=True: Adapter + LoRA (efficient decoder tuning)
        - unfreeze_decoder=True: Adapter + Full decoder (full fine-tuning)

    Available datasets:
        - spoken_magpie: Atotti/spoken-magpie-ja (audio instruction following)
        - spoken_multiturn: Atotti/spoken-multiturn-sft (multi-turn conversations)
        - reazon_sft: ReazonSpeech ASR (Japanese ASR, forgetting prevention)
        - fsd50k_cc0: Atotti/fsd50k-cc0-Qwen3-Omni-captioned (audio captioning)
        - fsd50k_ccby: Atotti/fsd50k-ccby-Qwen3-Omni-captioned (audio captioning)
        - librispeech: openslr/librispeech_asr (English ASR)

    Args:
        model_id: Pretrained model to finetune from (used when resume_from is None)
        resume_from: Path to finetune checkpoint to resume from
        start_step: Step number to resume from. Auto-detected from resume_from path if 0.
        use_lora: Enable LoRA for decoder fine-tuning (trains adapter + LoRA)
        unfreeze_decoder: Unfreeze decoder for full fine-tuning (trains adapter + full decoder)
        use_spoken_magpie: Enable spoken-magpie-ja dataset
        use_spoken_multiturn: Enable spoken-multiturn-sft dataset
        use_reazon_sft: Enable ReazonSpeech ASR dataset
        use_fsd50k_cc0: Enable FSD50K CC0 audio captioning dataset
        use_fsd50k_ccby: Enable FSD50K CCBY audio captioning dataset
        use_librispeech: Enable LibriSpeech English ASR dataset
        dataset_weights: Weights for enabled datasets (default: equal weights)
    """
    if use_lora and unfreeze_decoder:
        raise ValueError("use_lora and unfreeze_decoder are mutually exclusive")

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # Initialize Accelerator (single GPU or multi-GPU)
    # dispatch_batches=False: each process fetches its own batch independently
    # (required for variable-length audio sequences with different padding sizes)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(mixed_precision="bf16", dataloader_config=dataloader_config)
    is_main_process = accelerator.is_main_process

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
            use_lora = True  # Force LoRA mode when resuming from LoRA checkpoint
            if is_main_process:
                print(f"Detected LoRA checkpoint, enabling LoRA mode")

    # Initialize wandb (main process only)
    if is_main_process:
        wandb.init(
            project=wandb_project,
            config={
                "model_id": model_id,
                "batch_size": batch_size,
                "lr": lr,
                "max_steps": max_steps,
                "resume_from": resume_from,
                "start_step": start_step,
                "use_lora": use_lora,
                "unfreeze_decoder": unfreeze_decoder,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "use_spoken_magpie": use_spoken_magpie,
                "use_spoken_multiturn": use_spoken_multiturn,
                "use_reazon_sft": use_reazon_sft,
                "use_fsd50k_cc0": use_fsd50k_cc0,
                "use_fsd50k_ccby": use_fsd50k_ccby,
                "use_librispeech": use_librispeech,
                "dataset_weights": dataset_weights,
                "num_processes": accelerator.num_processes,
            },
        )

    # Load model (Accelerate handles device placement)
    if resume_from is not None and lora_checkpoint_path is not None:
        # Resume from LoRA checkpoint: load config + adapter + LoRA separately
        if is_main_process:
            print(f"Resuming from LoRA checkpoint: {resume_from}")
        from peft import PeftModel

        config = LlamaForSpeechLMConfig.from_pretrained(resume_from)
        model = LlamaForSpeechLM(config)
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
        if use_lora:
            # Apply fresh LoRA to resumed model
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
        # Load from pretrained model_id
        if is_main_process:
            print(f"Loading model from: {model_id}")
        model = LlamaForSpeechLM.from_pretrained(model_id)
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
            # PEFT creates LoRA in float32 by default, convert to bfloat16 to match decoder
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

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    # Build dataset from enabled sources
    datasets = []
    dataset_names = []

    if use_spoken_magpie:
        datasets.append(SpokenMagpie(max_duration=max_duration, max_response_length=max_response_length))
        dataset_names.append("spoken_magpie")

    if use_spoken_multiturn:
        datasets.append(SpokenMultiturnSFT(max_duration=max_duration, max_response_length=max_response_length))
        dataset_names.append("spoken_multiturn")

    if use_reazon_sft:
        datasets.append(ReazonSpeechSFT(max_duration=max_duration))
        dataset_names.append("reazon_sft")

    if use_fsd50k_cc0:
        datasets.append(FSD50KCaptioned(dataset_id="Atotti/fsd50k-cc0-Qwen3-Omni-captioned", max_duration=max_duration, max_response_length=max_response_length))
        dataset_names.append("fsd50k_cc0")

    if use_fsd50k_ccby:
        datasets.append(FSD50KCaptioned(dataset_id="Atotti/fsd50k-ccby-Qwen3-Omni-captioned", max_duration=max_duration, max_response_length=max_response_length))
        dataset_names.append("fsd50k_ccby")

    if use_librispeech:
        datasets.append(LibriSpeechASR(max_duration=max_duration))
        dataset_names.append("librispeech")

    if not datasets:
        raise ValueError("At least one dataset must be enabled")

    # Use weights or default weights
    # Default: [3, 1, 4, 1, 1, 1] for [magpie, multiturn, reazon, fsd50k_cc0, fsd50k_ccby, librispeech]
    default_weights = [3, 1, 4, 1, 1, 1]
    if dataset_weights:
        weights = dataset_weights
    elif len(datasets) == 6:
        weights = default_weights
    else:
        weights = [1] * len(datasets)
    if len(weights) != len(datasets):
        raise ValueError(f"dataset_weights length ({len(weights)}) must match enabled datasets ({len(datasets)})")

    if is_main_process:
        print(f"Enabled datasets: {list(zip(dataset_names, weights))}")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = InterleavedDataset(datasets, weights)

    # Prompt template for finetune
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # All datasets yield: {"instruction": str, "response": str, "audio": torch.Tensor}
        # Audio is already filtered and resampled to 16kHz in each dataset's __iter__
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
            "input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "input_features": encoder_inputs.input_features,
            "encoder_attention_mask": encoder_inputs.attention_mask,
        }

        return result

    # Note: IterableDataset doesn't support shuffle=True, data order depends on dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size, collate_fn=collate_fn
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
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        model_dir=model_dir,
        validate_fn=validate_finetune,
        start_step=start_step,
        use_lora=use_lora,
        accelerator=accelerator,
    )

    if is_main_process:
        wandb.finish()
