import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset, interleave_datasets, DownloadConfig
from torch import nn
from torch.utils.data import ConcatDataset
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
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
    ):
        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs[0]

        lengths = self.encoder._get_feat_extract_output_lengths(encoder_attention_mask.sum(dim=1, keepdim=True))
        lengths = lengths // self.config.adapter_kernel_size
        max_len = lengths.max()

        encoder_hidden_states = self.adapter(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states[:, :max_len]

        inputs_embeds = self.decoder.model.embed_tokens(input_ids)
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
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
    ):
        """
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_length)`):
                Log mel spectrogram.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token ids.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_length)`):
                1: non-mask
                0: mask
            decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                1: non-mask
                0: mask
        """
        inputs_embeds, attention_mask = self.embed(
            input_features, input_ids, encoder_attention_mask, decoder_attention_mask
        )

        labels = F.pad(input_ids, (inputs_embeds.shape[1] - input_ids.shape[1], 0), value=-100)

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return decoder_outputs.loss

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        **kwargs,
    ):
        inputs_embeds, attention_mask = self.embed(
            input_features, input_ids, encoder_attention_mask, decoder_attention_mask
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


class Clotho(torch.utils.data.Dataset):
    def __init__(self, root="data", split: str = "development", caption_idx: int = 1):
        """
        Args:
            split: development | validation | evaluation
        """
        self.audio_dir = os.path.join(root, "clotho", split)
        caption_path = os.path.join(root, f"clotho/clotho_captions_{split}.csv")

        self.captions = pd.read_csv(caption_path, encoding="ISO-8859-1")
        self.caption_idx = caption_idx

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, n: int) -> Tuple[torch.FloatTensor, int, str, List[str]]:
        """
        Returns:
            audio: 15 to 30 seconds duration
            caption: 8 to 20 words length
        """
        item = self.captions.iloc[n]  # file_name,caption_1,caption_2,caption_3,caption_4,caption_5

        audio_path = os.path.join(self.audio_dir, item["file_name"])
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 16000)

        caption = item[f"caption_{self.caption_idx}"].strip('"')
        captions = [item[f"caption_{caption_idx}"].strip('"') for caption_idx in range(1, 6)]

        return audio, 16000, caption, captions


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
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # learning rate scheduler
    if max_steps is not None:
        total_steps = max_steps
    else:
        total_steps = len(loader) // grad_accumulation * epoch

    lr_scheduler = get_lr_schedule(
        optimizer,
        total_steps,
        warmup_steps,
        lr,
        lr * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=init_grad_scale)

    step = 0

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
                    validate(
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

                # max_steps check
                if max_steps is not None and step >= max_steps:
                    break

        # validation at epoch end (if val_check_interval is not set)
        if val_check_interval is None:
            validate(
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

        if max_steps is not None and step >= max_steps:
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
):
    def get_collate_fn(encoder_processor, decoder_processor):
        prompt = """### 指示:
音声を書き起こしてください。

### 応答:
"""

        def collate_fn(
            batch: List[Tuple[torch.Tensor, int, str, int, int, int] | Tuple[torch.Tensor, int, str, List[str]]],
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of tuples.
                    ASR: (waveform, sample rate, transcript, speaker ID, chapter ID, utterance ID)
                    AAC: (waveform, sample rate, caption, captions)
            """

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

            refs = [item[2] for item in batch]

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
                batch["input_features"],
                batch["input_ids"],
                batch["encoder_attention_mask"],
                batch["decoder_attention_mask"],
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

    asr_dataset = ReazonSpeech(split="test", max_duration=30.0)
    asr_loader = torch.utils.data.DataLoader(
        asr_dataset, batch_size, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
    )

    asr_hyps, asr_refs = _validate(model, encoder_processor, decoder_processor, asr_loader)

    cer_evaluator = evaluate.load("cer")

    cer = cer_evaluator.compute(predictions=asr_hyps, references=asr_refs) * 100

    # wandb log
    wandb.log({"dev/cer": cer}, step=step)

    # Log sample predictions as table
    table = wandb.Table(columns=["Reference", "Prediction"])
    for ref, hyp in zip(asr_refs[:10], asr_hyps[:10]):
        table.add_data(ref, hyp)
    wandb.log({"dev/samples": table}, step=step)


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
):
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
        },
    )

    model = LlamaForSpeechLM(LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id)).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    dataset = ReazonSpeech(split="train")

    def get_collate_fn(encoder_processor, decoder_processor):
        prompt = """### 指示:
音声を書き起こしてください。

### 応答:
{}<|eos|>"""

        def collate_fn(
            batch: List[Tuple[torch.Tensor, int, str, int, int, int] | Tuple[torch.Tensor, int, str, List[str]]],
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of tuples.
                    ASR: (waveform, sample rate, transcript, speaker ID, chapter ID, utterance ID)
                    AAC: (waveform, sample rate, caption, captions)
            """

            encoder_inputs = encoder_processor(
                [item[0].squeeze(0).numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            decoder_inputs = decoder_processor(
                [prompt.format(item[2]) for item in batch],
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
    )

    wandb.finish()


def generate_data(
    model_id="models/LlamaForSpeechLM-ja",
    tts_id="kakao-enterprise/vits-vctk",
):
    model = LlamaForSpeechLM.from_pretrained(model_id).cuda()

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    tts_model = AutoModel.from_pretrained(tts_id).cuda()
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_id)

    def filter_by_input(example):
        pattern = "[A-Za-z,.'!? ]+"
        noinput_pattern = r"no\s*input\s*(required)?\.?"
        return (
            example["input"] != ""
            and example["input"] != "Mon cheval est blanc"
            and example["input"] != "The bakery that I visited yesterday had freshly made croissants."
            and example["input"] != "Croissants are French pastries. The sky is blue."
            and not re.match(noinput_pattern, example["input"], re.IGNORECASE)
            and re.fullmatch(pattern, example["input"]) is not None
        )

    @torch.inference_mode()
    def add_audio(example):
        inputs = tts_tokenizer(example["input"], return_tensors="pt").to("cuda")
        output = tts_model(**inputs).waveform
        output = torchaudio.functional.resample(output, tts_model.config.sampling_rate, 16000)
        output = output.squeeze(0).cpu().numpy()
        return {"audio": {"array": output, "sampling_rate": 16000}}

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.filter(filter_by_input)
    dataset = dataset.map(add_audio)
    dataset.push_to_hub("spoken-alpaca")


def finetune(
    model_id="models/LlamaForSpeechLM-ja",
    dataset_id="ryota-komatsu/spoken-alpaca",
    data_dir="data",
    model_dir="models/LlamaForSpeechLM-ja-Instruct",
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
):
    model = LlamaForSpeechLM.from_pretrained(model_id).cuda()

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    def is_train_example(example):
        return (
            len(example["audio"]["array"]) < 16000 * 30
            and len(example["instruction"]) < 102
            and len(example["output"]) < 838
        )

    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(is_train_example)

    def get_collate_fn(encoder_processor, decoder_processor):
        prompt = """### 指示:
以下は、タスクを説明する指示と、さらなる文脈を提供する音声入力の組み合わせです。音声を書き起こし、その後、リクエストを適切に完了する応答を書いてください。

### 命令:
{}

### 応答:
### 書き起こし:
{}

### 回答:
{}<|eos|>"""

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of the following example:
                    {
                        "instruction": "",
                        "input": "",
                        "output": "",
                        "text": "",
                        "audio": {"path": None, "array": tensor([...]), "sampling_rate": tensor(16000)},
                    }
            """

            encoder_inputs = encoder_processor(
                [item["audio"]["array"].numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            decoder_inputs = decoder_processor(
                [prompt.format(item["instruction"], item["input"], item["output"]) for item in batch],
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
        dataset, batch_size, True, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
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
    )


def eval(
    encoder_id="openai/whisper-large-v3",
    decoder_id="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4",
    dataset_id="ryota-komatsu/spoken-alpaca",
    model_dir="models/LlamaForSpeechLM-ja-Instruct",
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 5,
):
    model = LlamaForSpeechLM.from_pretrained(model_dir).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoTokenizer.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    prompt = """### 指示:
以下は、タスクを説明する指示と、さらなる文脈を提供する音声入力の組み合わせです。音声を書き起こし、その後、リクエストを適切に完了する応答を書いてください。

### 命令:
{}

### 応答:
"""

    def is_test_example(example):
        return (
            len(example["audio"]["array"]) < 16000 * 30
            and 102 <= len(example["instruction"])
            and len(example["output"]) < 838
        )

    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(is_test_example)

    loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    for item in loader:
        encoder_inputs = encoder_processor(
            item["audio"]["array"].numpy(),
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            prompt.format(item["instruction"][0]),
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(
            encoder_inputs.input_features,
            decoder_inputs.input_ids,
            encoder_attention_mask=encoder_inputs.attention_mask,
            decoder_attention_mask=decoder_inputs.attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        generated_txt = decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    eval()
