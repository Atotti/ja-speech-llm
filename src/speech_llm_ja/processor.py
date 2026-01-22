"""Custom processor for speech LLM with chat-template support."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin

from .model import AUDIO_TOKEN_ID


@dataclass
class SpeechLlamaProcessorConfig:
    """Lightweight config for SpeechLlamaProcessor."""

    system_prompt: str = "あなたは音声を理解できるAIアシスタントです。"
    audio_token: str = "<|reserved_343|>"
    adapter_kernel_size: int = 4
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

        self._audio_token = self.config.audio_token or "<|reserved_343|>"
        self._audio_token_id = self.tokenizer.convert_tokens_to_ids(self._audio_token)
        if self._audio_token_id != AUDIO_TOKEN_ID:
            raise ValueError(
                "audio_token_id mismatch: tokenizer returned"
                f" {self._audio_token_id}, expected {AUDIO_TOKEN_ID}"
            )

    @classmethod
    def from_pretrained(
        cls,
        encoder_id: str,
        decoder_id: str,
        config: Optional[SpeechLlamaProcessorConfig] = None,
        **kwargs,
    ) -> "SpeechLlamaProcessor":
        kwargs.pop("trust_remote_code", None)
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
        processor_config = {
            "processor_class": self.__class__.__name__,
            "auto_map": {
                "AutoProcessor": "speech_llm_ja_processor.SpeechLlamaProcessor",
            },
        }
        processor_config_path = save_path / "processor_config.json"
        with processor_config_path.open("w", encoding="utf-8") as f:
            json.dump(processor_config, f, ensure_ascii=True, indent=2)

    @classmethod
    def load_pretrained(cls, save_directory: str | Path) -> "SpeechLlamaProcessor":
        save_path = Path(save_directory)
        config_path = save_path / cls.config_filename
        config = None
        if config_path.exists():
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            if "audio_marker" in config_data and "audio_token" not in config_data:
                config_data["audio_token"] = "<|reserved_343|>"
                config_data.pop("audio_marker", None)
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
    ) -> Tuple[List[Dict[str, str]], List[Optional[Any]]]:
        normalized: List[Dict[str, str]] = []
        audio_payloads: List[Optional[Any]] = []
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
                        parts.append(self._audio_token)
                        audio_payloads.append(part.get("audio"))
                    else:
                        raise ValueError(f"Unsupported content type: {part_type}")
                content_text = "".join(parts)
            else:
                content_text = str(content)
            normalized.append({"role": role, "content": content_text})
        return normalized, audio_payloads

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
    ) -> str:
        normalized_messages = self._ensure_system_prompt(list(messages))
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
        instruction = user_text.replace(self._audio_token, "")
        marker = self._audio_token if self._audio_token in user_text else ""
        prompt = f"{system}\n\n{marker}### 指示:\n{instruction}\n\n### 応答:\n"
        if not add_generation_prompt and assistant_text:
            prompt = f"{prompt}{assistant_text}{self.tokenizer.eos_token}"
        return prompt

    def _compute_audio_token_lengths(
        self, feature_attention_mask: torch.Tensor
    ) -> List[int]:
        feature_lengths = feature_attention_mask.sum(dim=1)
        encoder_lengths = (feature_lengths - 1) // 2 + 1
        audio_token_lengths = (
            encoder_lengths // self.config.adapter_kernel_size
        ).tolist()
        return [int(length) for length in audio_token_lengths]

    def _expand_audio_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        audio_token_lengths: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        expanded_ids: List[List[int]] = []
        expanded_masks: List[List[int]] = []
        expanded_labels: List[List[int]] = []
        pad_token_id = self.tokenizer.pad_token_id

        for batch_idx in range(input_ids.shape[0]):
            token_lengths = list(audio_token_lengths[batch_idx])
            token_iter = iter(token_lengths)
            ids_row = input_ids[batch_idx].tolist()
            mask_row = attention_mask[batch_idx].tolist()
            labels_row = labels[batch_idx].tolist() if labels is not None else None

            new_ids: List[int] = []
            new_mask: List[int] = []
            new_labels: List[int] = []
            for pos, (token, mask) in enumerate(zip(ids_row, mask_row)):
                if mask == 0:
                    break
                if token == self._audio_token_id:
                    try:
                        length = next(token_iter)
                    except StopIteration as exc:
                        raise ValueError(
                            "Too few audio lengths for placeholders."
                        ) from exc
                    new_ids.extend([self._audio_token_id] * length)
                    new_mask.extend([1] * length)
                    if labels_row is not None:
                        new_labels.extend([-100] * length)
                else:
                    new_ids.append(token)
                    new_mask.append(mask)
                    if labels_row is not None:
                        new_labels.append(labels_row[pos])

            if next(token_iter, None) is not None:
                raise ValueError("Audio lengths remain after expansion.")

            expanded_ids.append(new_ids)
            expanded_masks.append(new_mask)
            if labels_row is not None:
                expanded_labels.append(new_labels)

        max_len = max(len(row) for row in expanded_ids)
        output_ids = torch.full(
            (len(expanded_ids), max_len),
            pad_token_id,
            dtype=input_ids.dtype,
        )
        output_mask = torch.zeros(
            (len(expanded_masks), max_len), dtype=attention_mask.dtype
        )
        output_labels = None
        if labels is not None:
            output_labels = torch.full(
                (len(expanded_labels), max_len),
                -100,
                dtype=labels.dtype,
            )

        for i, row in enumerate(expanded_ids):
            output_ids[i, : len(row)] = torch.tensor(row, dtype=input_ids.dtype)
            output_mask[i, : len(row)] = 1
            if output_labels is not None:
                output_labels[i, : len(row)] = torch.tensor(
                    expanded_labels[i], dtype=labels.dtype
                )

        return output_ids, output_mask, output_labels

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
        audios_per_sample: List[List[Any]] = []
        if audios is None:
            audios_per_sample = [[] for _ in range(batch_size)]
        else:
            if len(audios) != batch_size:
                raise ValueError("audios length must match messages batch size")
            for audio_item in audios:
                if audio_item is None:
                    audios_per_sample.append([])
                elif isinstance(audio_item, (list, tuple)):
                    audios_per_sample.append(list(audio_item))
                else:
                    audios_per_sample.append([audio_item])

        prompt_texts: List[str] = []
        flat_audios: List[Any] = []
        audio_token_lengths_by_sample: List[List[int]] = []

        for batch_idx, msgs in enumerate(messages_batch):
            normalized_messages, audio_payloads = self._normalize_messages(msgs)
            normalized_messages = self._ensure_system_prompt(normalized_messages)

            if (
                any(payload is not None for payload in audio_payloads)
                and audios is not None
            ):
                raise ValueError("Audio provided both in messages and audios argument.")

            if audios_per_sample[batch_idx]:
                if not audio_payloads:
                    raise ValueError(
                        "audios provided but no audio placeholders in messages."
                    )
                fill_iter = iter(audios_per_sample[batch_idx])
                filled_payloads: List[Any] = []
                for payload in audio_payloads:
                    if payload is not None:
                        raise ValueError(
                            "Audio provided in messages and audios argument."
                        )
                    try:
                        filled_payloads.append(next(fill_iter))
                    except StopIteration as exc:
                        raise ValueError(
                            "Not enough audios provided for placeholders."
                        ) from exc
                if list(fill_iter):
                    raise ValueError("Too many audios provided for placeholders.")
                audio_payloads = filled_payloads
            else:
                if any(payload is None for payload in audio_payloads):
                    raise ValueError("Audio placeholders missing audio payloads.")

            flat_audios.extend(audio_payloads)
            prompt_texts.append(
                self._build_prompt_text(normalized_messages, add_generation_prompt)
            )

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

        if flat_audios:
            audio_arrays = [self._prepare_audio(audio) for audio in flat_audios]
            encoder_inputs = self.encoder_processor(
                audio_arrays,
                return_tensors=return_tensors,
                return_attention_mask=True,
                sampling_rate=sampling_rate,
            )

            audio_token_lengths = self._compute_audio_token_lengths(
                encoder_inputs.attention_mask
            )
            offset = 0
            for msgs in messages_batch:
                normalized_messages, audio_payloads = self._normalize_messages(msgs)
                audio_count = len(audio_payloads)
                audio_token_lengths_by_sample.append(
                    audio_token_lengths[offset : offset + audio_count]
                )
                offset += audio_count

            expanded_ids, expanded_mask, expanded_labels = self._expand_audio_tokens(
                tokenized.input_ids,
                tokenized.attention_mask,
                result.get("labels"),
                audio_token_lengths_by_sample,
            )

            result["input_ids"] = expanded_ids
            result["decoder_attention_mask"] = expanded_mask
            if expanded_labels is not None:
                result["labels"] = expanded_labels

            result["input_features"] = encoder_inputs.input_features
            result["encoder_attention_mask"] = encoder_inputs.attention_mask

        return result

    @property
    def audio_token(self) -> str:
        return self._audio_token

    @property
    def audio_token_id(self) -> int:
        return self._audio_token_id
