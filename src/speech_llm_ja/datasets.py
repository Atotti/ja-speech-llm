"""Dataset classes for speech LLM training."""

from typing import List, Optional

import torch
import torchaudio
from datasets import load_dataset, interleave_datasets, DownloadConfig

# Instruction for instruction-following tasks (audio IS the instruction)
IF_INSTRUCTION = "これはタスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。"


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
                    "instruction": IF_INSTRUCTION,  # Audio IS the instruction
                    "response": item["response"],
                    "audio": audio,
                }

            except Exception as e:
                print(f"[SpokenMagpie error] {type(e).__name__}: {e}")
                continue


class SpokenMultiturnSFT(torch.utils.data.IterableDataset):
    """Spoken multi-turn SFT dataset (Atotti/spoken-multiturn-sft).

    Yields multi-turn conversations as a single sample with all turns included.
    Format: {"turns": [{"audio": Tensor, "instruction": str, "response": str}, ...]}
    """

    def __init__(
        self,
        dataset_id: str = "Atotti/spoken-multiturn-sft",
        split: str = "train",
        max_duration: float = 30.0,
        max_response_length: int = 2048,
        multi_turn: bool = True,  # True: yield multi-turn, False: yield single turns (legacy)
    ):
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_response_length = max_response_length
        self.multi_turn = multi_turn

    def _process_audio(self, audio_data):
        """Process audio data and return tensor, or None if invalid."""
        import numpy as np

        if audio_data is None:
            return None

        wav = audio_data.get("array")
        sr = audio_data.get("sampling_rate")
        if wav is None or sr is None:
            return None

        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        if len(wav) / sr > self.max_duration:
            return None
        audio = torch.from_numpy(wav).float()
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        return audio

    def __iter__(self):
        for item in self.dataset:
            try:
                if self.multi_turn:
                    # Multi-turn mode: yield both turns as a single sample
                    turns = []

                    # Turn 1
                    if len(item["a1"]) <= self.max_response_length:
                        audio1 = self._process_audio(item["q1_audio"])
                        if audio1 is not None:
                            turns.append({
                                "audio": audio1,
                                "instruction": IF_INSTRUCTION,
                                "response": item["a1"],
                            })

                    # Turn 2
                    if len(item["a2"]) <= self.max_response_length:
                        audio2 = self._process_audio(item["q2_audio"])
                        if audio2 is not None:
                            turns.append({
                                "audio": audio2,
                                "instruction": IF_INSTRUCTION,
                                "response": item["a2"],
                            })

                    # Only yield if we have at least one valid turn
                    if turns:
                        yield {"turns": turns}

                else:
                    # Legacy single-turn mode: yield each turn separately
                    if len(item["a1"]) <= self.max_response_length:
                        audio1 = self._process_audio(item["q1_audio"])
                        if audio1 is not None:
                            yield {
                                "instruction": IF_INSTRUCTION,
                                "response": item["a1"],
                                "audio": audio1,
                            }

                    if len(item["a2"]) <= self.max_response_length:
                        audio2 = self._process_audio(item["q2_audio"])
                        if audio2 is not None:
                            yield {
                                "instruction": IF_INSTRUCTION,
                                "response": item["a2"],
                                "audio": audio2,
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
    """Interleave multiple PyTorch IterableDatasets with random sampling based on weights.

    Datasets are cycled infinitely - when a dataset exhausts, it restarts from the beginning.
    This maintains constant weight ratios throughout training.
    """

    def __init__(self, datasets: List[torch.utils.data.IterableDataset], weights: List[int] = None):
        """
        Args:
            datasets: List of IterableDatasets
            weights: List of integers for sampling ratio (e.g., [3, 1] = 75%/25% probability)
        """
        self.datasets = datasets
        self.weights = weights or [1] * len(datasets)

    def __iter__(self):
        import random

        iterators = [iter(ds) for ds in self.datasets]
        indices = list(range(len(iterators)))

        while True:
            # Randomly sample a dataset based on weights
            idx = random.choices(indices, weights=self.weights, k=1)[0]

            try:
                yield next(iterators[idx])
            except StopIteration:
                # Dataset exhausted, restart from beginning
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])


class SpokenDPO(torch.utils.data.IterableDataset):
    """Spoken DPO dataset (Atotti/spoken-dpo-49k).

    Dataset format:
        - prompt: List of conversation turns with audio [{from, value, audio: {array, sampling_rate}}]
        - chosen: Preferred response text
        - rejected: Non-preferred response text

    Yields:
        dict with keys: instruction, audio, chosen, rejected
    """

    # Estimated dataset size (for LR scheduler calculation)
    ESTIMATED_SIZE = 43000

    def __init__(
        self,
        dataset_id: str = "Atotti/spoken-dpo-49k",
        split: str = "train",
        max_duration: float = 30.0,
        max_response_length: int = 2048,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split
            max_duration: Maximum audio duration in seconds
            max_response_length: Maximum response text length (for both chosen and rejected)
        """
        dl_config = DownloadConfig(resume_download=True, max_retries=10)
        self.dataset = load_dataset(dataset_id, split=split, streaming=True, download_config=dl_config)
        self.max_duration = max_duration
        self.max_response_length = max_response_length

    def __len__(self):
        """Return estimated dataset size (used for LR scheduler)."""
        return self.ESTIMATED_SIZE

    def __iter__(self):
        import numpy as np

        for item in self.dataset:
            try:
                # Filter by response length
                if len(item["chosen"]) > self.max_response_length:
                    continue
                if len(item["rejected"]) > self.max_response_length:
                    continue

                # Extract audio from prompt (first human turn with audio)
                prompt_turns = item["prompt"]
                audio_tensor = None
                instruction_text = ""

                for turn in prompt_turns:
                    if turn.get("from") == "human":
                        instruction_text = turn.get("value", "")
                        audio_data = turn.get("audio")
                        if audio_data is not None:
                            wav = audio_data.get("array")
                            sr = audio_data.get("sampling_rate", 16000)

                            if wav is not None:
                                if isinstance(wav, list):
                                    wav = np.array(wav, dtype=np.float32)

                                # Duration filter
                                if len(wav) / sr > self.max_duration:
                                    audio_tensor = None
                                    break

                                audio_tensor = torch.from_numpy(wav).float()
                                if sr != 16000:
                                    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
                        break  # Only use first human turn

                # Skip if no valid audio found
                if audio_tensor is None:
                    continue

                yield {
                    "instruction": IF_INSTRUCTION,  # Audio IS the instruction
                    "audio": audio_tensor,
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }

            except Exception as e:
                print(f"[SpokenDPO error] {type(e).__name__}: {e}")
                continue
