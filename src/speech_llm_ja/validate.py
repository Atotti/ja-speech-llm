"""Validation functions for speech LLM."""

from typing import Any, Dict, List

import evaluate
import torch
import torchaudio
import wandb
from datasets import load_dataset, DownloadConfig

from .datasets import ReazonSpeech, FSD50KCaptioned


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

    # ASR validation (ReazonSpeech test set)
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    asr_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
音声を書き起こしてください。

### 応答:
"""

    def asr_collate_fn(batch):
        encoder_inputs = encoder_processor(
            [item[0].squeeze(0).numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [asr_prompt for _ in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        refs = [item[2] for item in batch]  # transcript

        return {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_inputs.attention_mask,
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
        encoder_inputs = encoder_processor(
            [item["audio"].numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [aac_prompt for _ in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        refs = [item["response"] for item in batch]  # caption

        return {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_inputs.attention_mask,
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
    dataset_id: str = "Atotti/spoken-magpie-ja",
):
    """Validation for finetune: compute loss and generate samples on spoken-magpie-ja."""
    import numpy as np

    from .datasets import IF_INSTRUCTION

    model.eval()

    # Prompt for generation (without response)
    # Audio is inserted between <|reserved_343|> and <|reserved_342|>
    gen_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
"""

    # Prompt for loss computation (with response)
    loss_prompt = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{}

### 応答:
{}<|eos|>"""

    # Load validation samples (use streaming to avoid caching issues)
    dl_config = DownloadConfig(resume_download=True, max_retries=20)
    val_dataset = load_dataset(dataset_id, split="train", streaming=True, download_config=dl_config)

    samples = []
    for i, item in enumerate(val_dataset):
        if i >= val_samples:
            break
        try:
            audio_data = item["instruction_audio"]
            wav = audio_data["array"]
            sr = audio_data["sampling_rate"]

            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)

            audio = torch.from_numpy(wav).float()
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)

            samples.append({
                "instruction": item["instruction"],
                "response": item["response"],
                "audio": audio,
            })
        except Exception as e:
            print(f"[validate_finetune error] {type(e).__name__}: {e}")
            continue

    if not samples:
        print("[validate_finetune] No samples loaded, skipping validation")
        return

    # Compute loss
    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]

        encoder_inputs = encoder_processor(
            [item["audio"].numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [loss_prompt.format(IF_INSTRUCTION, item["response"]) for item in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                input_ids=decoder_inputs.input_ids,
                decoder_attention_mask=decoder_inputs.attention_mask,
                input_features=encoder_inputs.input_features,
                encoder_attention_mask=encoder_inputs.attention_mask,
            )
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    wandb.log({"dev/loss": avg_loss}, step=step)

    # Generate samples (first 10)
    gen_samples = samples[:10]
    hyps = []
    refs = []
    instructions = []

    for item in gen_samples:
        encoder_inputs = encoder_processor(
            [item["audio"].numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [gen_prompt.format(IF_INSTRUCTION)],
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(
            input_ids=decoder_inputs.input_ids,
            decoder_attention_mask=decoder_inputs.attention_mask,
            input_features=encoder_inputs.input_features,
            encoder_attention_mask=encoder_inputs.attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=decoder_processor.eos_token_id,
        )
        hyp = decoder_processor.decode(generated_ids[0], skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(item["response"])
        instructions.append(item["instruction"])

    # Log to wandb
    table = wandb.Table(columns=["Instruction", "Reference", "Prediction"])
    for inst, ref, hyp in zip(instructions, refs, hyps):
        table.add_data(inst[:100] + "..." if len(inst) > 100 else inst, ref[:200] + "..." if len(ref) > 200 else ref, hyp[:200] + "..." if len(hyp) > 200 else hyp)
    wandb.log({"dev/finetune_samples": table}, step=step)

    print(f"[validate_finetune] step={step}, loss={avg_loss:.4f}")
