"""Validation functions for speech LLM."""

from typing import Any, Dict, List, Tuple

import evaluate
import torch
import torchaudio
import wandb
from datasets import load_dataset, DownloadConfig

from .datasets import ReazonSpeech, ClothoJA


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

    def get_collate_fn(encoder_processor, decoder_processor, task: str = "asr"):
        if task == "asr":
            prompt = """### 指示:
音声を書き起こしてください。

### 応答:
"""
        else:  # aac
            prompt = """### 指示:
音声の内容を説明してください。

### 応答:
"""

        def collate_fn(
            batch: List[Tuple[torch.Tensor, int, str, int, int, int] | Tuple[torch.Tensor, int, str, List[str]]],
        ) -> Dict[str, torch.Tensor]:
            encoder_inputs = encoder_processor(
                [item[0].squeeze(0).numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            decoder_inputs = decoder_processor(
                [prompt for item in batch],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # ASR: item[2] is transcript, AAC: item[3] is captions list
            refs = [item[2] if len(item) == 6 else item[3] for item in batch]

            return {
                "input_features": encoder_inputs.input_features,
                "input_ids": decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": decoder_inputs.attention_mask,
                "refs": refs,
            }

        return collate_fn

    def _validate(model, encoder_processor, decoder_processor, loader, num_samples=100):
        hyps = []
        refs = []

        for batch in loader:
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                input_features=batch["input_features"],
                encoder_attention_mask=batch["encoder_attention_mask"],
                max_length=max_length,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            hyps += decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)
            refs += batch["refs"]

            if len(hyps) >= num_samples:
                break

        return hyps[:num_samples], refs[:num_samples]

    model.eval()

    # ASR validation (ReazonSpeech test set)
    asr_dataset = ReazonSpeech(split="test", max_duration=30.0)
    asr_loader = torch.utils.data.DataLoader(
        asr_dataset, batch_size, collate_fn=get_collate_fn(encoder_processor, decoder_processor, task="asr")
    )
    asr_hyps, asr_refs = _validate(model, encoder_processor, decoder_processor, asr_loader)

    # AAC validation (ClothoJA first N samples from train)
    aac_dataset = ClothoJA(max_samples=aac_val_samples)
    aac_loader = torch.utils.data.DataLoader(
        aac_dataset, batch_size, collate_fn=get_collate_fn(encoder_processor, decoder_processor, task="aac")
    )
    aac_hyps, aac_refs = _validate(model, encoder_processor, decoder_processor, aac_loader, num_samples=aac_val_samples)

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
        ref_str = ref[0] if isinstance(ref, list) else ref
        aac_table.add_data(ref_str, hyp)
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

    model.eval()

    # Prompt for generation (without response)
    gen_prompt = """以下は、タスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}

### 応答:
"""

    # Prompt for loss computation (with response)
    loss_prompt = """以下は、タスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。

### 指示:
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
            [loss_prompt.format(item["instruction"], item["response"]) for item in batch],
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
            [gen_prompt.format(item["instruction"])],
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
