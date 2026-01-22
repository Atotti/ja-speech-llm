"""Validation functions for speech LLM."""

from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
import torchaudio
import wandb
from datasets import load_dataset, DownloadConfig

from .datasets import ReazonSpeech, FSD50KCaptioned
from .utils import pad_or_trim_audio, AFWHISPER_SAMPLE_RATE, AFWHISPER_MAX_SAMPLES


def validate(
    model,
    encoder_processor,
    decoder_processor,
    step: int,
    batch_size: int = 4,
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 1,
    data_dir="data",
    aac_val_samples: int = 100,
):
    """Validation for pretrain: compute CER (ASR) and BLEU (AAC)."""

    model.eval()

    # Get encoder_type from model config
    encoder_type = getattr(model.config, 'encoder_type', 'whisper')

    # ASR validation (ReazonSpeech test set)
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    asr_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
音声を書き起こしてください。

### 応答:
"""

    def asr_collate_fn(batch):
        audios = [item[0].squeeze(0).numpy() for item in batch]

        if encoder_type in ("afwhisper", "qwen2-audio"):
            # Qwen2AudioEncoder-based: pad/trim to fixed 30s length
            original_lengths = [len(audio) for audio in audios]
            audios = [pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES) for audio in audios]

            encoder_inputs = encoder_processor(
                audios,
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=AFWHISPER_SAMPLE_RATE,
            ).to("cuda")

            # Create attention mask based on original audio lengths
            mel_length = encoder_inputs.input_features.shape[-1]
            original_mel_lengths = [min(int(l / 160), mel_length) for l in original_lengths]
            encoder_attention_mask = torch.zeros(len(batch), mel_length, dtype=torch.long, device="cuda")
            for i, mel_len in enumerate(original_mel_lengths):
                encoder_attention_mask[i, :mel_len] = 1
        else:
            # Whisper: variable length processing
            encoder_inputs = encoder_processor(
                audios,
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            ).to("cuda")
            encoder_attention_mask = encoder_inputs.attention_mask

        decoder_inputs = decoder_processor(
            [asr_prompt for _ in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        refs = [item[2] for item in batch]  # transcript

        return {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "refs": refs,
        }

    asr_dataset = ReazonSpeech(split="test", max_duration=30.0)
    asr_loader = torch.utils.data.DataLoader(asr_dataset, batch_size, collate_fn=asr_collate_fn)

    asr_hyps = []
    asr_refs = []
    for batch in asr_loader:
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            input_features=batch["input_features"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        asr_hyps += decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)
        asr_refs += batch["refs"]

        if len(asr_hyps) >= 100:
            break

    asr_hyps = asr_hyps[:100]
    asr_refs = asr_refs[:100]

    # AAC validation (FSD50K)
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    aac_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
音声を説明してください。

### 応答:
"""

    def aac_collate_fn(batch):
        audios = [item["audio"].numpy() for item in batch]

        if encoder_type in ("afwhisper", "qwen2-audio"):
            # Qwen2AudioEncoder-based: pad/trim to fixed 30s length
            original_lengths = [len(audio) for audio in audios]
            audios = [pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES) for audio in audios]

            encoder_inputs = encoder_processor(
                audios,
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=AFWHISPER_SAMPLE_RATE,
            ).to("cuda")

            # Create attention mask based on original audio lengths
            mel_length = encoder_inputs.input_features.shape[-1]
            original_mel_lengths = [min(int(l / 160), mel_length) for l in original_lengths]
            encoder_attention_mask = torch.zeros(len(batch), mel_length, dtype=torch.long, device="cuda")
            for i, mel_len in enumerate(original_mel_lengths):
                encoder_attention_mask[i, :mel_len] = 1
        else:
            # Whisper: variable length processing
            encoder_inputs = encoder_processor(
                audios,
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            ).to("cuda")
            encoder_attention_mask = encoder_inputs.attention_mask

        decoder_inputs = decoder_processor(
            [aac_prompt for _ in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        refs = [item["response"] for item in batch]  # caption

        return {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "refs": refs,
        }

    aac_dataset = FSD50KCaptioned(
        dataset_id="Atotti/fsd50k-ccby-Qwen3-Omni-captioned",
        max_duration=30.0,
    )
    aac_loader = torch.utils.data.DataLoader(aac_dataset, batch_size, collate_fn=aac_collate_fn)

    aac_hyps = []
    aac_refs = []
    for batch in aac_loader:
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            input_features=batch["input_features"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        aac_hyps += decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)
        aac_refs += batch["refs"]

        if len(aac_hyps) >= aac_val_samples:
            break

    aac_hyps = aac_hyps[:aac_val_samples]
    aac_refs = aac_refs[:aac_val_samples]

    # Metrics
    cer_evaluator = evaluate.load("cer")
    bleu_evaluator = evaluate.load("bleu")

    cer = cer_evaluator.compute(predictions=asr_hyps, references=asr_refs) * 100
    bleu = bleu_evaluator.compute(predictions=aac_hyps, references=aac_refs)["bleu"]

    # wandb log
    wandb.log({"dev/cer": cer, "dev/bleu": bleu}, step=step)

    # Log sample predictions as table
    asr_table = wandb.Table(columns=["Reference", "Prediction"])
    for ref, hyp in zip(asr_refs[:10], asr_hyps[:10]):
        asr_table.add_data(ref, hyp)
    wandb.log({"dev/asr_samples": asr_table}, step=step)

    aac_table = wandb.Table(columns=["Reference", "Prediction"])
    for ref, hyp in zip(aac_refs[:10], aac_hyps[:10]):
        aac_table.add_data(ref, hyp)
    wandb.log({"dev/aac_samples": aac_table}, step=step)


def validate_finetune(
    model,
    encoder_processor,
    decoder_processor,
    step: int,
    batch_size: int = 4,
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 1,
    data_dir="data",  # unused, kept for compatibility
    val_samples: int = 50,
    gen_samples: int = 10,
    # Dataset selection (same as finetune)
    use_spoken_magpie: bool = True,
    use_spoken_multiturn: bool = True,
    use_reazon_sft: bool = True,
    use_fsd50k_cc0: bool = True,
    use_fsd50k_ccby: bool = True,
    use_librispeech: bool = True,
):
    """Validation for finetune: compute loss and generate samples on all enabled datasets."""
    from .datasets import (
        IF_INSTRUCTION,
        SpokenMagpie,
        SpokenMultiturnSFT,
        ReazonSpeechSFT,
        FSD50KCaptioned,
        LibriSpeechASR,
    )

    model.eval()

    # Get encoder_type from model config
    encoder_type = getattr(model.config, 'encoder_type', 'whisper')

    # Prompt for generation (without response) - single turn
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    gen_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
"""

    # Prompt for loss computation (with response) - single turn
    loss_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    # System prompt for multi-turn
    system_prompt = "あなたは音声を理解できるAIアシスタントです。\n\n"

    def build_multiturn_loss_prompt(turns: List[Dict]) -> str:
        """Build prompt for multi-turn loss computation."""
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

    def build_multiturn_gen_prompt(turns: List[Dict]) -> str:
        """Build prompt for multi-turn generation (without last response)."""
        prompt = system_prompt
        for i, turn in enumerate(turns):
            if turn.get("audio") is not None:
                prompt += "<|reserved_343|><|reserved_342|>"
            prompt += f"### 指示:\n{turn['instruction']}\n\n"
            if i < len(turns) - 1:
                # Previous turns: include response
                prompt += f"### 応答:\n{turn['response']}\n\n"
            else:
                # Last turn: prompt for generation (no response)
                prompt += "### 応答:\n"
        return prompt

    def process_audio_for_validation(audio: torch.Tensor) -> torch.Tensor:
        """Process a single audio tensor for validation."""
        audio_np = audio.numpy()
        if encoder_type in ("afwhisper", "qwen2-audio"):
            audio_np = pad_or_trim_audio(audio_np, AFWHISPER_MAX_SAMPLES)
            features = encoder_processor(
                audio_np,
                return_tensors="pt",
                sampling_rate=AFWHISPER_SAMPLE_RATE,
            )
        else:
            features = encoder_processor(
                audio_np,
                return_tensors="pt",
                sampling_rate=16000,
            )
        return features.input_features.squeeze(0).to("cuda")  # [feature_size, feature_length]

    # Define dataset configs
    dataset_configs = []

    if use_spoken_magpie:
        dataset_configs.append({
            "name": "spoken_magpie",
            "dataset": SpokenMagpie(max_duration=30.0),
            "instruction_key": "instruction",  # IF_INSTRUCTION is already set in dataset
        })

    if use_spoken_multiturn:
        dataset_configs.append({
            "name": "spoken_multiturn",
            "dataset": SpokenMultiturnSFT(max_duration=30.0),
            "instruction_key": "instruction",
        })

    if use_reazon_sft:
        dataset_configs.append({
            "name": "reazon_sft",
            "dataset": ReazonSpeechSFT(split="test", max_duration=30.0),
            "instruction_key": "instruction",
        })

    if use_fsd50k_cc0:
        dataset_configs.append({
            "name": "fsd50k_cc0",
            "dataset": FSD50KCaptioned(dataset_id="Atotti/fsd50k-cc0-Qwen3-Omni-captioned", max_duration=30.0),
            "instruction_key": "instruction",
        })

    if use_fsd50k_ccby:
        dataset_configs.append({
            "name": "fsd50k_ccby",
            "dataset": FSD50KCaptioned(dataset_id="Atotti/fsd50k-ccby-Qwen3-Omni-captioned", max_duration=30.0),
            "instruction_key": "instruction",
        })

    if use_librispeech:
        dataset_configs.append({
            "name": "librispeech",
            "dataset": LibriSpeechASR(split="test.clean", max_duration=30.0),
            "instruction_key": "instruction",
        })

    if not dataset_configs:
        print("[validate_finetune] No datasets enabled, skipping validation")
        return

    # Evaluate each dataset
    all_losses = {}

    for config in dataset_configs:
        dataset_name = config["name"]
        dataset = config["dataset"]

        print(f"[validate_finetune] Evaluating {dataset_name}...")

        # Load samples (handle both single-turn and multi-turn formats)
        samples = []
        is_multiturn_dataset = False

        for i, item in enumerate(dataset):
            if i >= val_samples:
                break
            try:
                if "turns" in item:
                    # Multi-turn format
                    is_multiturn_dataset = True
                    samples.append({"turns": item["turns"]})
                else:
                    # Single-turn format
                    samples.append({
                        "instruction": item["instruction"],
                        "response": item["response"],
                        "audio": item["audio"],
                    })
            except Exception as e:
                print(f"[validate_finetune/{dataset_name} error] {type(e).__name__}: {e}")
                continue

        if not samples:
            print(f"[validate_finetune/{dataset_name}] No samples loaded, skipping")
            continue

        # Compute loss
        total_loss = 0.0
        num_batches = 0

        if is_multiturn_dataset:
            # Multi-turn evaluation: process one sample at a time
            for sample in samples:
                turns = sample["turns"]

                # Build prompt and collect audios
                prompt = build_multiturn_loss_prompt(turns)
                sample_audios = []
                for turn in turns:
                    if turn.get("audio") is not None:
                        audio_features = process_audio_for_validation(turn["audio"])
                        sample_audios.append(audio_features)

                decoder_inputs = decoder_processor(
                    [prompt],
                    return_tensors="pt",
                ).to("cuda")

                # Use new audios parameter for multi-turn
                audios_batch = [sample_audios]  # List[List[Tensor]]

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        input_ids=decoder_inputs.input_ids,
                        decoder_attention_mask=decoder_inputs.attention_mask,
                        audios=audios_batch,
                    )
                total_loss += loss.item()
                num_batches += 1
        else:
            # Single-turn evaluation: batch processing
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                audios = [item["audio"].numpy() for item in batch]

                if encoder_type in ("afwhisper", "qwen2-audio"):
                    # Qwen2AudioEncoder-based: pad/trim to fixed 30s length
                    original_lengths = [len(audio) for audio in audios]
                    audios = [pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES) for audio in audios]

                    encoder_inputs = encoder_processor(
                        audios,
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=AFWHISPER_SAMPLE_RATE,
                    ).to("cuda")

                    # Create attention mask based on original audio lengths
                    mel_length = encoder_inputs.input_features.shape[-1]
                    original_mel_lengths = [min(int(l / 160), mel_length) for l in original_lengths]
                    encoder_attention_mask = torch.zeros(len(batch), mel_length, dtype=torch.long, device="cuda")
                    for j, mel_len in enumerate(original_mel_lengths):
                        encoder_attention_mask[j, :mel_len] = 1
                else:
                    # Whisper: variable length processing
                    encoder_inputs = encoder_processor(
                        audios,
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=16000,
                    ).to("cuda")
                    encoder_attention_mask = encoder_inputs.attention_mask

                decoder_inputs = decoder_processor(
                    [loss_prompt.format(item["instruction"], item["response"]) for item in batch],
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        input_ids=decoder_inputs.input_ids,
                        decoder_attention_mask=decoder_inputs.attention_mask,
                        input_features=encoder_inputs.input_features,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        all_losses[dataset_name] = avg_loss
        wandb.log({f"dev/loss/{dataset_name}": avg_loss}, step=step)

        # Generate samples
        sample_hyps = []
        sample_refs = []
        sample_instructions = []

        for item in samples[:gen_samples]:
            if is_multiturn_dataset:
                # Multi-turn generation
                turns = item["turns"]
                if not turns:
                    continue

                # Build generation prompt and collect audios
                prompt = build_multiturn_gen_prompt(turns)
                sample_audios = []
                for turn in turns:
                    if turn.get("audio") is not None:
                        audio_features = process_audio_for_validation(turn["audio"])
                        sample_audios.append(audio_features)

                decoder_inputs = decoder_processor(
                    [prompt],
                    return_tensors="pt",
                ).to("cuda")

                # Use audios parameter for multi-turn
                audios_batch = [sample_audios]  # List[List[Tensor]]

                # Use max_new_tokens for multi-turn (inputs_embeds mode)
                # max_length doesn't work well with inputs_embeds in Transformers
                generated_ids = model.generate(
                    input_ids=decoder_inputs.input_ids,
                    decoder_attention_mask=decoder_inputs.attention_mask,
                    audios=audios_batch,
                    max_new_tokens=512,  # Generate up to 512 new tokens
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=decoder_processor.eos_token_id,
                )
                hyp = decoder_processor.decode(generated_ids[0], skip_special_tokens=True)
                sample_hyps.append(hyp)
                # Reference is the last turn's response
                sample_refs.append(turns[-1]["response"])
                # For multi-turn, show turn count and last instruction
                sample_instructions.append(f"[{len(turns)}ターン] {turns[-1]['instruction']}")
            else:
                # Single-turn generation
                audio = item["audio"].numpy()

                if encoder_type in ("afwhisper", "qwen2-audio"):
                    # Qwen2AudioEncoder-based: pad/trim to fixed 30s length
                    original_length = len(audio)
                    audio = pad_or_trim_audio(audio, AFWHISPER_MAX_SAMPLES)

                    encoder_inputs = encoder_processor(
                        [audio],
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=AFWHISPER_SAMPLE_RATE,
                    ).to("cuda")

                    # Create attention mask based on original audio length
                    mel_length = encoder_inputs.input_features.shape[-1]
                    original_mel_length = min(int(original_length / 160), mel_length)
                    encoder_attention_mask = torch.zeros(1, mel_length, dtype=torch.long, device="cuda")
                    encoder_attention_mask[0, :original_mel_length] = 1
                else:
                    # Whisper: variable length processing
                    encoder_inputs = encoder_processor(
                        [audio],
                        return_tensors="pt",
                        return_attention_mask=True,
                        sampling_rate=16000,
                    ).to("cuda")
                    encoder_attention_mask = encoder_inputs.attention_mask

                decoder_inputs = decoder_processor(
                    [gen_prompt.format(item["instruction"])],
                    return_tensors="pt",
                ).to("cuda")

                generated_ids = model.generate(
                    input_ids=decoder_inputs.input_ids,
                    decoder_attention_mask=decoder_inputs.attention_mask,
                    input_features=encoder_inputs.input_features,
                    encoder_attention_mask=encoder_attention_mask,
                    max_length=max_length,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=decoder_processor.eos_token_id,
                )
                hyp = decoder_processor.decode(generated_ids[0], skip_special_tokens=True)
                sample_hyps.append(hyp)
                sample_refs.append(item["response"])
                sample_instructions.append(item["instruction"])

        # Log samples to wandb table
        table = wandb.Table(columns=["Instruction", "Reference", "Prediction"])
        for inst, ref, hyp in zip(sample_instructions, sample_refs, sample_hyps):
            table.add_data(
                inst[:100] + "..." if len(inst) > 100 else inst,
                ref[:200] + "..." if len(ref) > 200 else ref,
                hyp[:200] + "..." if len(hyp) > 200 else hyp,
            )
        wandb.log({f"dev/samples/{dataset_name}": table}, step=step)

        print(f"[validate_finetune/{dataset_name}] loss={avg_loss:.4f}")

    # Log overall average loss
    if all_losses:
        overall_avg_loss = sum(all_losses.values()) / len(all_losses)
        wandb.log({"dev/loss": overall_avg_loss}, step=step)
        print(f"[validate_finetune] step={step}, overall_loss={overall_avg_loss:.4f}")
