"""Finetune functions for speech LLM."""

import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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
    TextMultiturn,
    InterleavedDataset,
)
from .train import _train
from .utils import (
    get_encoder_processor,
    pad_or_trim_audio,
    AFWHISPER_SAMPLE_RATE,
    AFWHISPER_MAX_SAMPLES,
)
from .validate import validate_finetune


def finetune(
    model_id: str = None,  # Pretrained checkpoint path (None to create fresh model)
    model_dir="models/LlamaForSpeechLM-ja-Instruct",
    # For fresh model creation (used when model_id is None)
    encoder_id: str = "openai/whisper-large-v3",
    decoder_id: str = "your-decoder-model-path",
    encoder_type: str = "whisper",  # "whisper", "afwhisper", or "qwen2-audio"
    batch_size: int = 4,
    lr: float = 1e-4,
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
    use_text_multiturn: bool = True,  # Text-only dataset for capability preservation
    dataset_weights: List[int] = None,  # Default: [3, 1, 4, 1, 1, 1, 0] when all enabled
    # Multi-GPU
    use_accelerate: bool = False,
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
        - text_multiturn: kanhatakeyama/ramdom-to-fixed-multiturn-Calm3 (text-only, capability preservation)

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
        use_accelerate: If True, use HuggingFace Accelerate for multi-GPU training.
    """
    if use_lora and unfreeze_decoder:
        raise ValueError("use_lora and unfreeze_decoder are mutually exclusive")

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # Initialize Accelerator for multi-GPU
    accelerator = None
    if use_accelerate:
        # dispatch_batches=False: each process fetches its own batch independently
        # (required for variable-length audio sequences with different padding sizes)
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
            use_lora = True  # Force LoRA mode when resuming from LoRA checkpoint
            if is_main_process:
                print(f"Detected LoRA checkpoint, enabling LoRA mode")

    # Initialize wandb (main process only)
    if is_main_process:
        wandb.init(
            project=wandb_project,
            config={
                "model_id": model_id,
                "encoder_id": encoder_id,
                "decoder_id": decoder_id,
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
                "use_accelerate": use_accelerate,
                "num_processes": accelerator.num_processes if accelerator else 1,
            },
        )

    # Load model (don't move to cuda when using Accelerate)
    if resume_from is not None and lora_checkpoint_path is not None:
        # Resume from LoRA checkpoint: load config + adapter + LoRA separately
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
    elif model_id is not None:
        # Load from pretrained model_id
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
            # PEFT creates LoRA in float32 by default, convert to bfloat16 to match decoder
            for name, param in model.decoder.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)
            if is_main_process:
                print("Applied LoRA to decoder (bfloat16)")
    else:
        # Create fresh model from encoder_id and decoder_id (skip pretrain)
        if is_main_process:
            print(f"Creating fresh model: encoder={encoder_id} ({encoder_type}), decoder={decoder_id}")
        config = LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id, encoder_type=encoder_type)
        model = LlamaForSpeechLM(config)
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

    # Get encoder_type from model config
    model_encoder_type = getattr(model.config, 'encoder_type', 'whisper')
    encoder_processor = get_encoder_processor(model.config.encoder_id, model_encoder_type)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    if is_main_process:
        print(f"Encoder type: {model_encoder_type}, ID: {model.config.encoder_id}")

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

    if use_text_multiturn:
        datasets.append(TextMultiturn(max_response_length=max_response_length))
        dataset_names.append("text_multiturn")

    if not datasets:
        raise ValueError("At least one dataset must be enabled")

    # Use weights or default weights
    # Default: [6, 2, 9, 1, 1, 1, 1] for [magpie, multiturn, reazon, fsd50k_cc0, fsd50k_ccby, librispeech, text_multiturn]
    default_weights = [6, 2, 9, 1, 1, 1, 1]
    if dataset_weights:
        weights = dataset_weights
    elif len(datasets) == 7:
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

    # Prompt templates for finetune
    # Single-turn audio: audio is inserted between <|reserved_343|> and <|reserved_342|>
    prompt_audio = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    # Text-only samples: no audio markers
    prompt_text = """あなたは親切なAIアシスタントです。

### 指示:
{}

### 応答:
{}<|eos|>"""

    # System prompt for multi-turn
    system_prompt = "あなたは音声を理解できるAIアシスタントです。\n\n"

    def build_multiturn_prompt(turns: List[Dict]) -> str:
        """Build prompt for multi-turn conversation."""
        prompt = system_prompt
        for i, turn in enumerate(turns):
            if turn.get("audio") is not None:
                prompt += "<|reserved_343|><|reserved_342|>"
            prompt += f"### 指示:\n{turn['instruction']}\n\n"
            prompt += f"### 応答:\n{turn['response']}"
            if i < len(turns) - 1:
                prompt += "\n\n"
            else:
                prompt += "<|eos|>"
        return prompt

    def process_audio_for_encoder(audio: torch.Tensor) -> torch.Tensor:
        """Process a single audio tensor for the encoder."""
        if model_encoder_type in ("afwhisper", "qwen2-audio"):
            audio_np = pad_or_trim_audio(audio.numpy(), AFWHISPER_MAX_SAMPLES)
            features = encoder_processor(
                audio_np,
                return_tensors="pt",
                sampling_rate=AFWHISPER_SAMPLE_RATE,
            )
        else:
            features = encoder_processor(
                audio.numpy(),
                return_tensors="pt",
                sampling_rate=16000,
            )
        return features.input_features.squeeze(0)  # [feature_size, feature_length]

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function supporting both single-turn and multi-turn samples.

        Single-turn format: {"instruction": str, "response": str, "audio": Tensor or None}
        Multi-turn format: {"turns": [{"instruction": str, "response": str, "audio": Tensor or None}, ...]}
        """
        # Check if any sample is multi-turn
        has_multiturn = any("turns" in item for item in batch)

        if has_multiturn:
            # Multi-turn mode: convert all samples to multi-turn format
            all_prompts = []
            all_audios = []  # List[List[Tensor]]

            for item in batch:
                if "turns" in item:
                    # Already multi-turn
                    turns = item["turns"]
                else:
                    # Convert single-turn to multi-turn format
                    turns = [{
                        "audio": item.get("audio"),
                        "instruction": item["instruction"],
                        "response": item["response"],
                    }]

                # Build prompt
                prompt = build_multiturn_prompt(turns)
                all_prompts.append(prompt)

                # Collect audios for this sample
                sample_audios = []
                for turn in turns:
                    if turn.get("audio") is not None:
                        audio_features = process_audio_for_encoder(turn["audio"])
                        sample_audios.append(audio_features)
                all_audios.append(sample_audios)

            # Tokenize prompts
            decoder_inputs = decoder_processor(
                all_prompts,
                padding=True,
                return_tensors="pt",
            )

            result = {
                "input_ids": decoder_inputs.input_ids,
                "decoder_attention_mask": decoder_inputs.attention_mask,
                "audios": all_audios,  # List[List[Tensor]]
            }

        else:
            # Legacy single-turn mode (no multi-turn samples in batch)
            # Separate audio and text-only samples
            audio_samples = [item for item in batch if item.get("audio") is not None]
            text_samples = [item for item in batch if item.get("audio") is None]

            # Process audio samples
            if audio_samples:
                if model_encoder_type in ("afwhisper", "qwen2-audio"):
                    # Qwen2AudioEncoder-based: pad/trim to fixed 30s length
                    audios = []
                    original_lengths = []
                    for item in audio_samples:
                        audio = item["audio"].numpy()
                        original_lengths.append(len(audio))
                        audio = pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES)
                        audios.append(audio)

                    audio_encoder_inputs = encoder_processor(
                        audios,
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=AFWHISPER_SAMPLE_RATE,
                    )

                    # Create attention mask based on original audio lengths
                    mel_length = audio_encoder_inputs.input_features.shape[-1]
                    original_mel_lengths = [min(int(l / 160), mel_length) for l in original_lengths]
                    audio_encoder_attention_mask = torch.zeros(len(audio_samples), mel_length, dtype=torch.long)
                    for i, mel_len in enumerate(original_mel_lengths):
                        audio_encoder_attention_mask[i, :mel_len] = 1
                else:
                    # Whisper: variable length processing
                    audio_encoder_inputs = encoder_processor(
                        [item["audio"].numpy() for item in audio_samples],
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=16000,
                    )
                    audio_encoder_attention_mask = audio_encoder_inputs.attention_mask

                audio_decoder_inputs = decoder_processor(
                    [prompt_audio.format(item["instruction"], item["response"]) for item in audio_samples],
                    padding=True,
                    return_tensors="pt",
                )

            # Process text-only samples (use dummy silent audio for encoder)
            if text_samples:
                if model_encoder_type in ("afwhisper", "qwen2-audio"):
                    # Qwen2AudioEncoder-based: use 30s dummy audio
                    dummy_audio = np.zeros(AFWHISPER_MAX_SAMPLES, dtype=np.float32)
                    text_encoder_inputs = encoder_processor(
                        [dummy_audio for _ in text_samples],
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=AFWHISPER_SAMPLE_RATE,
                    )
                    # Set attention mask to zeros (no valid audio)
                    mel_length = text_encoder_inputs.input_features.shape[-1]
                    text_encoder_attention_mask = torch.zeros(len(text_samples), mel_length, dtype=torch.long)
                else:
                    # Whisper: use short dummy audio
                    dummy_audio = np.zeros(1600, dtype=np.float32)
                    text_encoder_inputs = encoder_processor(
                        [dummy_audio for _ in text_samples],
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=16000,
                    )
                    text_encoder_attention_mask = text_encoder_inputs.attention_mask

                text_decoder_inputs = decoder_processor(
                    [prompt_text.format(item["instruction"], item["response"]) for item in text_samples],
                    padding=True,
                    return_tensors="pt",
                )

            # Combine results (audio first, then text)
            if audio_samples and text_samples:
                # Need to pad to same length before concatenating
                max_input_len = max(audio_decoder_inputs.input_ids.shape[1], text_decoder_inputs.input_ids.shape[1])

                # Pad audio decoder inputs
                audio_pad_len = max_input_len - audio_decoder_inputs.input_ids.shape[1]
                if audio_pad_len > 0:
                    audio_decoder_inputs.input_ids = torch.nn.functional.pad(
                        audio_decoder_inputs.input_ids, (0, audio_pad_len), value=decoder_processor.pad_token_id
                    )
                    audio_decoder_inputs.attention_mask = torch.nn.functional.pad(
                        audio_decoder_inputs.attention_mask, (0, audio_pad_len), value=0
                    )

                # Pad text decoder inputs
                text_pad_len = max_input_len - text_decoder_inputs.input_ids.shape[1]
                if text_pad_len > 0:
                    text_decoder_inputs.input_ids = torch.nn.functional.pad(
                        text_decoder_inputs.input_ids, (0, text_pad_len), value=decoder_processor.pad_token_id
                    )
                    text_decoder_inputs.attention_mask = torch.nn.functional.pad(
                        text_decoder_inputs.attention_mask, (0, text_pad_len), value=0
                    )

                result = {
                    "input_ids": torch.cat([audio_decoder_inputs.input_ids, text_decoder_inputs.input_ids], dim=0),
                    "decoder_attention_mask": torch.cat([audio_decoder_inputs.attention_mask, text_decoder_inputs.attention_mask], dim=0),
                    "input_features": torch.cat([audio_encoder_inputs.input_features, text_encoder_inputs.input_features], dim=0),
                    "encoder_attention_mask": torch.cat([audio_encoder_attention_mask, text_encoder_attention_mask], dim=0),
                }
            elif audio_samples:
                result = {
                    "input_ids": audio_decoder_inputs.input_ids,
                    "decoder_attention_mask": audio_decoder_inputs.attention_mask,
                    "input_features": audio_encoder_inputs.input_features,
                    "encoder_attention_mask": audio_encoder_attention_mask,
                }
            else:  # text_samples only
                result = {
                    "input_ids": text_decoder_inputs.input_ids,
                    "decoder_attention_mask": text_decoder_inputs.attention_mask,
                    "input_features": text_encoder_inputs.input_features,
                    "encoder_attention_mask": text_encoder_attention_mask,
                }

        # Move to cuda only for single-GPU (non-Accelerate) mode
        if not use_accelerate:
            if "audios" in result:
                # Multi-turn: move audios to cuda
                result["input_ids"] = result["input_ids"].cuda()
                result["decoder_attention_mask"] = result["decoder_attention_mask"].cuda()
                result["audios"] = [[a.cuda() for a in sample_audios] for sample_audios in result["audios"]]
            else:
                result = {k: v.cuda() for k, v in result.items()}

        return result

    # Note: IterableDataset doesn't support shuffle=True, data order depends on dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size, collate_fn=collate_fn
    )

    # Create validate_fn with dataset config
    validate_fn = partial(
        validate_finetune,
        use_spoken_magpie=use_spoken_magpie,
        use_spoken_multiturn=use_spoken_multiturn,
        use_reazon_sft=use_reazon_sft,
        use_fsd50k_cc0=use_fsd50k_cc0,
        use_fsd50k_ccby=use_fsd50k_ccby,
        use_librispeech=use_librispeech,
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
        validate_fn=validate_fn,
        start_step=start_step,
        use_lora=use_lora,
        accelerator=accelerator,
    )

    if is_main_process:
        wandb.finish()
