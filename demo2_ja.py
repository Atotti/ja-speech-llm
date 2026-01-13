import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset, interleave_datasets, DownloadConfig
from torch import nn
import wandb
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    WhisperForConditionalGeneration,
)


class Adapter(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        kernel_size: int,
        bias: bool,
    ):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size)
        self.linear1 = nn.Linear(encoder_hidden_size, 2 * decoder_hidden_size, bias=bias)
        self.linear2 = nn.Linear(2 * decoder_hidden_size, decoder_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.pool(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LlamaForSpeechLMConfig(PretrainedConfig):
    model_type = "llama_for_speech_lm"

    def __init__(
        self,
        encoder_id: str = "openai/whisper-large-v3",
        decoder_id: str = "/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4",
        adapter_kernel_size: int = 4,
        adapter_linear_bias: bool = False,
        **kwargs,
    ):
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_linear_bias = adapter_linear_bias
        super().__init__(**kwargs)


class LlamaForSpeechLM(PreTrainedModel):
    config_class = LlamaForSpeechLMConfig
    # Note: LLM-jp does NOT tie embed_tokens and lm_head weights.
    # Do NOT add _tied_weights_keys here - it would break model loading.

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__(config)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(config.encoder_id).model.encoder
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_id, torch_dtype=torch.bfloat16)
        self.adapter = Adapter(
            self.encoder.config.d_model,
            self.decoder.config.hidden_size,
            config.adapter_kernel_size,
            config.adapter_linear_bias,
        )

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def get_input_embeddings(self):
        return self.decoder.model.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.decoder.lm_head = new_embeddings

    def tie_weights(self):
        # Override to prevent automatic weight tying.
        # LLM-jp has separate embed_tokens and lm_head weights.
        pass

    def embed(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.decoder.model.embed_tokens(input_ids)

        if input_features is not None:
            # Audio + text
            encoder_outputs = self.encoder(input_features)
            encoder_hidden_states = encoder_outputs[0]

            lengths = self.encoder._get_feat_extract_output_lengths(encoder_attention_mask.sum(dim=1, keepdim=True))
            lengths = lengths // self.config.adapter_kernel_size
            max_len = lengths.max()

            encoder_hidden_states = self.adapter(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states[:, :max_len]

            inputs_embeds = torch.cat((encoder_hidden_states, inputs_embeds), dim=1)

            attention_mask = torch.cat(
                (
                    (
                        torch.arange(encoder_hidden_states.shape[1], device=decoder_attention_mask.device).unsqueeze(0)
                        < lengths
                    ).long(),
                    decoder_attention_mask,
                ),
                dim=1,
            )
        else:
            # Text-only
            attention_mask = decoder_attention_mask

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token ids.
            decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                1: non-mask, 0: mask
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_length)`, optional):
                Log mel spectrogram. None for text-only.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_length)`, optional):
                1: non-mask, 0: mask. None for text-only.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional):
                Labels for language modeling. If None, auto-generated from input_ids.
        """
        inputs_embeds, attention_mask = self.embed(
            input_ids, decoder_attention_mask, input_features, encoder_attention_mask
        )

        if labels is None:
            # Auto-generate labels: mask audio positions with -100
            labels = F.pad(input_ids, (inputs_embeds.shape[1] - input_ids.shape[1], 0), value=-100)

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return decoder_outputs.loss

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds, attention_mask = self.embed(
            input_ids, decoder_attention_mask, input_features, encoder_attention_mask
        )

        generated_ids = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return generated_ids


class ReazonSpeech(torch.utils.data.IterableDataset):
    """ReazonSpeech ASR dataset (streaming)"""

    def __init__(self, split: str = "train", max_duration: float = 30.0):
        """
        Args:
            split: train (split_0~7) | test (split_8)
            max_duration: Maximum audio duration in seconds
        """
        from datasets import Audio

        dataset_id = "japanese-asr/whisper_transcriptions.reazonspeech.all"
        dl_config = DownloadConfig(max_retries=20)
        if split == "test":
            # split_8 を test として使用
            configs = ["split_8"]
        else:
            # split_0~7 を train として使用
            configs = [f"split_{i}" for i in range(8)]

        datasets_list = []
        for cfg in configs:
            ds = load_dataset(dataset_id, cfg, split="train", streaming=True, download_config=dl_config)
            # decode=False にして、iterator 内で例外が起きても iterator が死なないようにする
            ds = ds.cast_column("audio", Audio(decode=False))
            datasets_list.append(ds)

        self.dataset = interleave_datasets(datasets_list, stopping_strategy="all_exhausted")
        self.max_duration = max_duration

    def __iter__(self):
        import io
        import soundfile as sf

        for item in self.dataset:
            try:
                a = item["audio"]  # decode=False なので bytes/path が取れる
                # streaming モードでは bytes を使う（path はファイル名のみでローカルにない）
                if a.get("bytes"):
                    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                else:
                    wav, sr = sf.read(a["path"], dtype="float32")

                # mono化
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)

                # duration filter
                if len(wav) / sr > self.max_duration:
                    continue

                audio = torch.from_numpy(wav).float()
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)

                yield audio.unsqueeze(0), 16000, item["transcription"], None, None, None

            except Exception as e:
                # FLAC decode エラー等が出てもこのサンプルだけスキップして続行
                print(f"[decode error] {type(e).__name__}: {e}")
                continue


class ClothoJA(torch.utils.data.IterableDataset):
    """Clotho-JA dataset from HuggingFace (streaming)."""

    def __init__(
        self,
        dataset_id: str = "Atotti/clotho-ja",
        split: str = "train",
        max_duration: float = 30.0,
        max_samples: int = None,
        skip_samples: int = 0,
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.skip_samples = skip_samples

    def __iter__(self):
        count = 0
        skipped = 0
        for item in self.dataset:
            # Skip first N samples (for train/val split)
            if skipped < self.skip_samples:
                skipped += 1
                continue

            try:
                audio_data = item["audio"]
                wav = audio_data["array"]
                sr = audio_data["sampling_rate"]

                # duration filter
                if len(wav) / sr > self.max_duration:
                    continue

                audio = torch.from_numpy(wav).float()
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)

                caption_ja = item["text_ja"]

                # Return format: (audio, sr, caption, captions_list)
                # Using 4-tuple to distinguish from ASR's 6-tuple
                yield audio.unsqueeze(0), 16000, caption_ja, [caption_ja]

                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    break

            except Exception as e:
                print(f"[ClothoJA decode error] {type(e).__name__}: {e}")
                continue


class AutoMultiTurn(torch.utils.data.IterableDataset):
    """Text-only multi-turn conversation dataset for maintaining text capability during SFT."""

    def __init__(
        self,
        dataset_id: str = "kanhatakeyama/AutoMultiTurnByCalm3-22B",
        split: str = "train",
        max_samples: Optional[int] = None,
        use_multi_turn: bool = False,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split
            max_samples: Maximum number of samples to yield
            use_multi_turn: If True, yield both (q1,a1) and (q2,a2) as separate samples
        """
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_samples = max_samples
        self.use_multi_turn = use_multi_turn

    def __iter__(self):
        count = 0
        for item in self.dataset:
            try:
                # First turn
                q1, a1 = item["q1"], item["a1"]
                if q1 and a1:
                    yield {"instruction": q1, "response": a1, "is_text_only": True}
                    count += 1
                    if self.max_samples is not None and count >= self.max_samples:
                        break

                # Optional second turn
                if self.use_multi_turn:
                    q2, a2 = item.get("q2"), item.get("a2")
                    if q2 and a2:
                        yield {"instruction": q2, "response": a2, "is_text_only": True}
                        count += 1
                        if self.max_samples is not None and count >= self.max_samples:
                            break

            except Exception as e:
                print(f"[AutoMultiTurn error] {type(e).__name__}: {e}")
                continue


class SpokenMagpie(torch.utils.data.IterableDataset):
    """Spoken Magpie-JA dataset for audio instruction following (streaming)."""

    def __init__(
        self,
        dataset_id: str = "Atotti/spoken-magpie-ja",
        split: str = "train",
        max_duration: float = 30.0,
        max_response_length: int = 2048,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split
            max_duration: Maximum audio duration in seconds
            max_response_length: Maximum response text length
        """
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_response_length = max_response_length

    def __iter__(self):
        import numpy as np

        for item in self.dataset:
            try:
                # response length filter (before audio decode)
                if len(item["response"]) > self.max_response_length:
                    continue

                audio_data = item["instruction_audio"]
                wav = audio_data["array"]  # list in streaming mode
                sr = audio_data["sampling_rate"]

                # streaming mode returns list, not np.ndarray
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)

                audio = torch.from_numpy(wav).float()
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)

                yield {
                    "instruction": item["instruction"],
                    "response": item["response"],
                    "audio": audio,
                }

            except Exception as e:
                print(f"[SpokenMagpie error] {type(e).__name__}: {e}")
                continue


class InterleavedDataset(torch.utils.data.IterableDataset):
    """Interleave multiple PyTorch IterableDatasets with configurable ratio."""

    def __init__(self, datasets: List[torch.utils.data.IterableDataset], weights: List[int] = None):
        """
        Args:
            datasets: List of IterableDatasets
            weights: List of integers for sampling ratio (e.g., [10, 1] = 10:1 ratio)
        """
        self.datasets = datasets
        self.weights = weights or [1] * len(datasets)

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        exhausted = [False] * len(iterators)

        while not all(exhausted):
            for i, it in enumerate(iterators):
                if exhausted[i]:
                    continue
                # Yield weight[i] samples from dataset i
                for _ in range(self.weights[i]):
                    try:
                        yield next(it)
                    except StopIteration:
                        exhausted[i] = True
                        break


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

    scaler = torch.amp.GradScaler("cuda", init_scale=init_grad_scale)

    step = start_step

    for epoch in range(1, epoch + 1):
        model.train()
        model.encoder.eval()
        model.decoder.eval()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch)
                loss = loss / grad_accumulation
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulation == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # update
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

                # update learning rate
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1

                # wandb log
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/scale": scale,
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
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    print(f"Checkpoint saved: {checkpoint_dir}")

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
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}")

        if max_steps is not None and step >= target_step:
            break


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
):
    """
    Train adapter on ASR (ReazonSpeech) + AAC (ClothoJA).

    Args:
        resume_from: Path to checkpoint to resume from (e.g., "models/LlamaForSpeechLM-ja-step20000")
        start_step: Step number to resume from. max_steps is added to this.
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

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    # Interleave ReazonSpeech (ASR) and ClothoJA (AAC)
    # Skip first 100 samples of ClothoJA (used for validation)
    # Ratio: ASR 10 : AAC 1
    asr_dataset = ReazonSpeech(split="train")
    aac_dataset = ClothoJA(split="train", skip_samples=100, max_duration=30.0)
    dataset = InterleavedDataset([asr_dataset, aac_dataset], weights=[1, 0]) # ASRとAACの比率を設定

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


def finetune(
    model_id="models/LlamaForSpeechLM-ja",
    dataset_id="Atotti/spoken-magpie-ja",
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
):
    """
    Finetune adapter on spoken-magpie-ja dataset (audio instructions).

    Note: Only adapter is trained. Encoder and decoder are frozen.
    For text capability preservation with LoRA, see finetune_lora.py.

    Args:
        model_id: Pretrained model to finetune from (used when resume_from is None)
        resume_from: Path to finetune checkpoint to resume from (e.g., "models/LlamaForSpeechLM-ja-Instruct-step1000")
        start_step: Step number to resume from. Auto-detected from resume_from path if 0.
    """
    # Determine which model to load
    load_path = resume_from if resume_from is not None else model_id

    # Auto-extract start_step from resume_from path
    if resume_from is not None and start_step == 0:
        match = re.search(r"step(\d+)", resume_from)
        if match:
            start_step = int(match.group(1))
            print(f"Auto-detected start_step: {start_step}")

    wandb.init(
        project=wandb_project,
        config={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "batch_size": batch_size,
            "lr": lr,
            "max_steps": max_steps,
            "resume_from": resume_from,
            "start_step": start_step,
        },
    )

    print(f"Loading model from: {load_path}")
    model = LlamaForSpeechLM.from_pretrained(load_path).cuda()

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    # Use SpokenMagpie for streaming + lazy filtering (no upfront audio loading)
    dataset = SpokenMagpie(
        dataset_id=dataset_id,
        max_duration=max_duration,
        max_response_length=max_response_length,
    )

    # Prompt: distinct from text-only to help model recognize audio input
    prompt = """以下は、タスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}

### 応答:
{}<|eos|>"""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # SpokenMagpie yields: {"instruction": str, "response": str, "audio": torch.Tensor}
        # Audio is already filtered and resampled to 16kHz in SpokenMagpie.__iter__
        encoder_inputs = encoder_processor(
            [item["audio"].numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [prompt.format(item["instruction"], item["response"]) for item in batch],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        return {
            "input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "input_features": encoder_inputs.input_features,
            "encoder_attention_mask": encoder_inputs.attention_mask,
        }

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
    )

    wandb.finish()
