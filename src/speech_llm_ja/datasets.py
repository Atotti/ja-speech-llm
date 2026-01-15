"""Dataset classes for speech LLM training."""

from typing import List, Optional

import torch
import torchaudio
from datasets import load_dataset, interleave_datasets, DownloadConfig


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


class SpokenMultiturnSFT(torch.utils.data.IterableDataset):
    """Spoken multi-turn SFT dataset (Atotti/spoken-multiturn-sft)."""

    def __init__(
        self,
        dataset_id: str = "Atotti/spoken-multiturn-sft",
        split: str = "train",
        max_duration: float = 30.0,
        max_response_length: int = 2048,
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_response_length = max_response_length

    def __iter__(self):
        import numpy as np

        for item in self.dataset:
            try:
                # Yield turn 1: q1_audio -> a1
                if len(item["a1"]) <= self.max_response_length:
                    audio_data = item["q1_audio"]
                    wav = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    if isinstance(wav, list):
                        wav = np.array(wav, dtype=np.float32)
                    if len(wav) / sr <= self.max_duration:
                        audio = torch.from_numpy(wav).float()
                        if sr != 16000:
                            audio = torchaudio.functional.resample(audio, sr, 16000)
                        yield {
                            "instruction": item["q1"],
                            "response": item["a1"],
                            "audio": audio,
                        }

                # Yield turn 2: q2_audio -> a2 (with context from turn 1)
                if len(item["a2"]) <= self.max_response_length:
                    audio_data = item["q2_audio"]
                    wav = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    if isinstance(wav, list):
                        wav = np.array(wav, dtype=np.float32)
                    if len(wav) / sr <= self.max_duration:
                        audio = torch.from_numpy(wav).float()
                        if sr != 16000:
                            audio = torchaudio.functional.resample(audio, sr, 16000)
                        # Include previous context in instruction
                        context = f"前の質問: {item['q1']}\n前の回答: {item['a1']}\n\n現在の質問: {item['q2']}"
                        yield {
                            "instruction": context,
                            "response": item["a2"],
                            "audio": audio,
                        }

            except Exception as e:
                print(f"[SpokenMultiturnSFT error] {type(e).__name__}: {e}")
                continue


class FSD50KCaptioned(torch.utils.data.IterableDataset):
    """FSD50K with Qwen3-Omni captions for AAC (Atotti/fsd50k-cc0/ccby-Qwen3-Omni-captioned)."""

    def __init__(
        self,
        dataset_id: str = "Atotti/fsd50k-cc0-Qwen3-Omni-captioned",
        split: str = "train",
        max_duration: float = 30.0,
        max_response_length: int = 2048,
        instruction: str = "音声を説明してください。",
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_response_length = max_response_length
        self.instruction = instruction

    def __iter__(self):
        import numpy as np

        for item in self.dataset:
            try:
                if len(item["caption"]) > self.max_response_length:
                    continue

                audio_data = item["audio"]
                wav = audio_data["array"]
                sr = audio_data["sampling_rate"]

                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)

                # duration filter
                if len(wav) / sr > self.max_duration:
                    continue

                audio = torch.from_numpy(wav).float()
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)

                yield {
                    "instruction": self.instruction,
                    "response": item["caption"],
                    "audio": audio,
                }

            except Exception as e:
                print(f"[FSD50KCaptioned error] {type(e).__name__}: {e}")
                continue


class LibriSpeechASR(torch.utils.data.IterableDataset):
    """LibriSpeech ASR dataset for English ASR (openslr/librispeech_asr)."""

    def __init__(
        self,
        split: str = "train.clean.100",
        max_duration: float = 30.0,
        instruction: str = "Transcribe the audio.",
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset("openslr/librispeech_asr", split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.instruction = instruction

    def __iter__(self):
        import numpy as np

        for item in self.dataset:
            try:
                audio_data = item["audio"]
                wav = audio_data["array"]
                sr = audio_data["sampling_rate"]

                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)

                # duration filter
                if len(wav) / sr > self.max_duration:
                    continue

                audio = torch.from_numpy(wav).float()
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)

                yield {
                    "instruction": self.instruction,
                    "response": item["text"],
                    "audio": audio,
                }

            except Exception as e:
                print(f"[LibriSpeechASR error] {type(e).__name__}: {e}")
                continue


class ReazonSpeechSFT(torch.utils.data.IterableDataset):
    """ReazonSpeech in SFT format (instruction-response pairs)."""

    def __init__(
        self,
        split: str = "train",
        max_duration: float = 30.0,
        instruction: str = "音声を書き起こしてください。",
    ):
        self.reazon = ReazonSpeech(split=split, max_duration=max_duration)
        self.instruction = instruction

    def __iter__(self):
        for audio, sr, transcript, *_ in self.reazon:
            yield {
                "instruction": self.instruction,
                "response": transcript,
                "audio": audio.squeeze(0),  # Remove batch dim
            }


class TextMultiturn(torch.utils.data.IterableDataset):
    """Text-only multi-turn dataset (kanhatakeyama/ramdom-to-fixed-multiturn-Calm3)."""

    def __init__(
        self,
        dataset_id: str = "kanhatakeyama/ramdom-to-fixed-multiturn-Calm3",
        split: str = "20240806filtered",
        max_response_length: int = 2048,
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_response_length = max_response_length

    def __iter__(self):
        for item in self.dataset:
            try:
                messages = item["messages"]
                # Extract user-assistant pairs
                for i in range(0, len(messages) - 1, 2):
                    if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        instruction = messages[i]["content"]
                        response = messages[i + 1]["content"]

                        if len(response) > self.max_response_length:
                            continue

                        yield {
                            "instruction": instruction,
                            "response": response,
                            "audio": None,  # Text-only
                        }

            except Exception as e:
                print(f"[TextMultiturn error] {type(e).__name__}: {e}")
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
