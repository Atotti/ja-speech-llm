"""Custom processor for speech LLM with chat-template support."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin

from .model import AUDIO_END_TOKEN_ID, AUDIO_START_TOKEN_ID


@dataclass
class SpeechLlamaProcessorConfig:
    """Lightweight config for SpeechLlamaProcessor."""

    system_prompt: str = "あなたは音声を理解できるAIアシスタントです。"
    audio_marker: str = "<|reserved_343|><|reserved_342|>"
    use_chat_template: bool = True


class SpeechLlamaProcessor(ProcessorMixin):
    """Processor that unifies audio features + chat-template prompts."""

    config_filename = "speech_llm_ja_processor.json"
    attributes = ["encoder_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    encoder_processor_class = "WhisperProcessor"

    def __init__(
        self,
        encoder_processor,
        tokenizer,
        config: Optional[SpeechLlamaProcessorConfig] = None,
        encoder_id: Optional[str] = None,
        decoder_id: Optional[str] = None,
    ) -> None:
        super().__init__(encoder_processor, tokenizer)
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        self.config = config or SpeechLlamaProcessorConfig()
        self.config.use_chat_template = (
            self.config.use_chat_template and self.tokenizer.chat_template is not None
        )

        # Ensure marker string matches model-side reserved token IDs.
        self._audio_marker = self.config.audio_marker
        if self._audio_marker is None:
            self._audio_marker = "<|reserved_343|><|reserved_342|>"

    @classmethod
    def from_pretrained(
        cls,
        encoder_id: str,
        decoder_id: str,
        config: Optional[SpeechLlamaProcessorConfig] = None,
    ) -> "SpeechLlamaProcessor":
        encoder_processor = AutoProcessor.from_pretrained(encoder_id)
        tokenizer = AutoTokenizer.from_pretrained(decoder_id)
        return cls(
            encoder_processor=encoder_processor,
            tokenizer=tokenizer,
            config=config,
            encoder_id=encoder_id,
            decoder_id=decoder_id,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.encoder_processor.save_pretrained(save_path / "encoder")
        self.tokenizer.save_pretrained(save_path / "decoder")
        config_path = save_path / self.config_filename
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_pretrained(cls, save_directory: str | Path) -> "SpeechLlamaProcessor":
        save_path = Path(save_directory)
        config_path = save_path / cls.config_filename
        config = None
        if config_path.exists():
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            config = SpeechLlamaProcessorConfig(**config_data)
        encoder_processor = AutoProcessor.from_pretrained(save_path / "encoder")
        tokenizer = AutoTokenizer.from_pretrained(save_path / "decoder")
        processor = cls(
            encoder_processor=encoder_processor,
            tokenizer=tokenizer,
            config=config,
            encoder_id=str(save_path / "encoder"),
            decoder_id=str(save_path / "decoder"),
        )
        return processor

    def _prepare_audio(self, audio: Any) -> Any:
        if audio is None:
            return None
        if torch.is_tensor(audio):
            audio = audio.detach().cpu()
            if audio.ndim > 1:
                audio = audio.squeeze(0)
            audio = audio.numpy()
        return audio

    def _normalize_messages(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], int]:
        normalized: List[Dict[str, str]] = []
        audio_marker_count = 0
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append(part.get("text", ""))
                    elif part_type == "audio":
                        parts.append(self._audio_marker)
                        audio_marker_count += 1
                    else:
                        raise ValueError(f"Unsupported content type: {part_type}")
                content_text = "".join(parts)
            else:
                content_text = str(content)
            normalized.append({"role": role, "content": content_text})
        return normalized, audio_marker_count

    def _ensure_system_prompt(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not messages:
            return [{"role": "system", "content": self.config.system_prompt}]
        if messages[0].get("role") != "system":
            return [
                {"role": "system", "content": self.config.system_prompt},
                *messages,
            ]
        return messages

    def _build_prompt_text(
        self,
        messages: Sequence[Dict[str, Any]],
        add_generation_prompt: bool,
        audio_expected: bool,
    ) -> str:
        normalized_messages, audio_marker_count = self._normalize_messages(messages)
        normalized_messages = self._ensure_system_prompt(normalized_messages)

        if not audio_expected and audio_marker_count > 0:
            raise ValueError("Audio markers provided but no audio was passed.")

        if audio_expected and audio_marker_count == 0:
            for msg in normalized_messages:
                if msg.get("role") == "user":
                    msg["content"] = f"{self._audio_marker}{msg['content']}"
                    audio_marker_count = 1
                    break

        if audio_marker_count > 1:
            raise ValueError("Only one audio segment per sample is supported.")

        if self.config.use_chat_template:
            return self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        # Fallback template (legacy)
        system = normalized_messages[0]["content"]
        user_text = ""
        assistant_text = ""
        for msg in normalized_messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
        if self._audio_marker in user_text:
            instruction = user_text.replace(self._audio_marker, "", 1)
            marker = self._audio_marker
        else:
            instruction = user_text
            marker = self._audio_marker if audio_expected else ""
        prompt = f"{system}\n\n{marker}### 指示:\n{instruction}\n\n### 応答:\n"
        if not add_generation_prompt and assistant_text:
            prompt = f"{prompt}{assistant_text}{self.tokenizer.eos_token}"
        return prompt

    def __call__(
        self,
        messages: Sequence[Sequence[Dict[str, Any]]] | Sequence[Dict[str, Any]],
        audios: Optional[Sequence[Any]] = None,
        sampling_rate: int = 16000,
        padding: bool | str = True,
        return_tensors: str = "pt",
        add_generation_prompt: bool = False,
        return_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if messages and isinstance(messages[0], dict):
            messages_batch = [messages]  # type: ignore[assignment]
        else:
            messages_batch = list(messages)  # type: ignore[arg-type]

        batch_size = len(messages_batch)
        if audios is None:
            audios = [None] * batch_size
        if len(audios) != batch_size:
            raise ValueError("audios length must match messages batch size")

        has_audio_flags = [audio is not None for audio in audios]
        if any(has_audio_flags) and not all(has_audio_flags):
            raise ValueError("Mixed audio/text batches are not supported.")

        prompt_texts = [
            self._build_prompt_text(
                msgs, add_generation_prompt, audio_expected=any(has_audio_flags)
            )
            for msgs in messages_batch
        ]

        tokenized = self.tokenizer(
            prompt_texts,
            padding=padding,
            return_tensors=return_tensors,
        )

        result: Dict[str, torch.Tensor] = {
            "input_ids": tokenized.input_ids,
            "decoder_attention_mask": tokenized.attention_mask,
        }

        if return_labels:
            labels = tokenized.input_ids.clone()
            labels[tokenized.attention_mask == 0] = -100
            result["labels"] = labels

        if all(has_audio_flags):
            audio_arrays = [self._prepare_audio(audio) for audio in audios]
            encoder_inputs = self.encoder_processor(
                audio_arrays,
                return_tensors=return_tensors,
                return_attention_mask=True,
                sampling_rate=sampling_rate,
            )
            result["input_features"] = encoder_inputs.input_features
            result["encoder_attention_mask"] = encoder_inputs.attention_mask

        return result

    @property
    def audio_marker(self) -> str:
        return self._audio_marker

    @property
    def audio_marker_ids(self) -> Tuple[int, int]:
        return AUDIO_START_TOKEN_ID, AUDIO_END_TOKEN_ID
