"""Speech LLM model classes."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    WhisperForConditionalGeneration,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder


class Adapter(nn.Module):
    """2-layer MLP adapter with average pooling."""

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


# Audio marker token IDs (using reserved tokens from the end)
AUDIO_START_TOKEN_ID = 351  # <|reserved_343|>
AUDIO_END_TOKEN_ID = 350    # <|reserved_342|>


class LlamaForSpeechLMConfig(PretrainedConfig):
    """Configuration for LlamaForSpeechLM."""

    model_type = "llama_for_speech_lm"

    def __init__(
        self,
        encoder_id: str = "openai/whisper-large-v3",
        decoder_id: str = None,
        decoder_config: dict = None,
        adapter_kernel_size: int = 4,
        adapter_linear_bias: bool = False,
        encoder_type: str = "whisper",  # "whisper", "afwhisper", or "qwen2-audio"
        **kwargs,
    ):
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self.decoder_config = decoder_config
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_linear_bias = adapter_linear_bias
        self.encoder_type = encoder_type
        super().__init__(**kwargs)


class LlamaForSpeechLM(PreTrainedModel):
    """Speech LLM combining Whisper/AFWhisper encoder + LLM decoder via trainable adapter."""

    config_class = LlamaForSpeechLMConfig
    # Note: LLM-jp does NOT tie embed_tokens and lm_head weights.
    # Do NOT add _tied_weights_keys here - it would break model loading.

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__(config)

        # Load encoder based on type
        if config.encoder_type in ("afwhisper", "qwen2-audio"):
            # Qwen2AudioEncoder-based: AFWhisper (Audio-Flamingo-3) or qwen2-audio-encoder
            self.encoder = Qwen2AudioEncoder.from_pretrained(config.encoder_id)
            encoder_hidden_size = self.encoder.config.d_model  # 1280
        else:
            # Default: Whisper encoder
            self.encoder = WhisperForConditionalGeneration.from_pretrained(config.encoder_id).model.encoder
            encoder_hidden_size = self.encoder.config.d_model  # 1280 for whisper-large-v3

        if config.decoder_config is not None:
            # Inference mode: create architecture only (weights loaded by from_pretrained)
            decoder_cfg = AutoConfig.for_model(**config.decoder_config)
            self.decoder = AutoModelForCausalLM.from_config(decoder_cfg)
        else:
            # Training mode: load full pretrained decoder
            self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_id, torch_dtype=torch.bfloat16)
            # Save decoder config for future loading without decoder_id
            config.decoder_config = self.decoder.config.to_dict()
        self.adapter = Adapter(
            encoder_hidden_size,
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

    def _encode_single_audio(self, audio_features: torch.FloatTensor) -> torch.FloatTensor:
        """Encode a single audio and apply adapter.

        Args:
            audio_features: [1, feature_size, feature_length] mel spectrogram

        Returns:
            [1, time, hidden] adapted audio embeddings
        """
        encoder_outputs = self.encoder(audio_features)
        encoder_hidden_states = encoder_outputs[0]  # [1, time, dim]
        adapted = self.adapter(encoder_hidden_states)  # [1, time', hidden]
        return adapted

    def embed(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        audios: Optional[List[List[torch.FloatTensor]]] = None,
    ):
        """
        Embed input_ids and optionally insert audio embeddings between audio markers.

        Supports two modes:
        1. Single-turn (backward compatible): input_features [batch, feature_size, feature_length]
        2. Multi-turn: audios[batch_idx][audio_idx] = [feature_size, feature_length]

        Expected prompt format (multi-turn):
            あなたは音声を理解できるAIアシスタントです。

            <|reserved_343|><|reserved_342|>### 指示:
            {instruction1}

            ### 応答:
            {response1}

            <|reserved_343|><|reserved_342|>### 指示:
            {instruction2}

            ### 応答:
            {response2}<|eos|>

        Audio embeddings are inserted between <|reserved_343|> and <|reserved_342|>.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Multi-turn mode: audios is List[List[Tensor]]
        if audios is not None:
            all_inputs_embeds = []
            all_attention_masks = []
            all_audio_lengths = []  # Track total audio tokens inserted per sample

            for batch_idx in range(batch_size):
                sample_input_ids = input_ids[batch_idx]  # [seq_len]
                sample_attention_mask = decoder_attention_mask[batch_idx]  # [seq_len]
                sample_audios = audios[batch_idx]  # List of audio tensors

                # Get text embeddings
                sample_embeds = self.decoder.get_input_embeddings()(sample_input_ids.unsqueeze(0))  # [1, seq_len, hidden]
                sample_embeds = sample_embeds.squeeze(0)  # [seq_len, hidden]

                if len(sample_audios) == 0:
                    # Text-only sample
                    all_inputs_embeds.append(sample_embeds)
                    all_attention_masks.append(sample_attention_mask)
                    all_audio_lengths.append(0)
                    continue

                # Find all audio marker positions
                marker_positions = (sample_input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0]

                if len(marker_positions) != len(sample_audios):
                    raise ValueError(
                        f"Sample {batch_idx}: Number of audio markers ({len(marker_positions)}) "
                        f"!= number of audios ({len(sample_audios)})"
                    )

                # Process from right to left to maintain positions
                total_audio_len = 0
                for audio, marker_pos in reversed(list(zip(sample_audios, marker_positions))):
                    # Encode single audio
                    audio_emb = self._encode_single_audio(audio.unsqueeze(0))  # [1, time, hidden]
                    audio_emb = audio_emb.squeeze(0)  # [time, hidden]
                    audio_len = audio_emb.shape[0]
                    total_audio_len += audio_len

                    insert_pos = marker_pos.item() + 1  # Insert after <|reserved_343|>

                    # Split and insert
                    embeds_before = sample_embeds[:insert_pos]  # includes <|reserved_343|>
                    embeds_after = sample_embeds[insert_pos:]   # starts with <|reserved_342|>
                    sample_embeds = torch.cat([embeds_before, audio_emb, embeds_after], dim=0)

                    # Update attention mask
                    mask_before = sample_attention_mask[:insert_pos]
                    mask_after = sample_attention_mask[insert_pos:]
                    audio_mask = torch.ones(audio_len, dtype=sample_attention_mask.dtype, device=device)
                    sample_attention_mask = torch.cat([mask_before, audio_mask, mask_after], dim=0)

                all_inputs_embeds.append(sample_embeds)
                all_attention_masks.append(sample_attention_mask)
                all_audio_lengths.append(total_audio_len)

            # Pad to same length within batch
            max_len = max(e.shape[0] for e in all_inputs_embeds)
            hidden_size = all_inputs_embeds[0].shape[1]

            padded_embeds = torch.zeros(batch_size, max_len, hidden_size, dtype=all_inputs_embeds[0].dtype, device=device)
            padded_masks = torch.zeros(batch_size, max_len, dtype=decoder_attention_mask.dtype, device=device)

            for i, (embeds, mask) in enumerate(zip(all_inputs_embeds, all_attention_masks)):
                seq_len = embeds.shape[0]
                padded_embeds[i, :seq_len] = embeds
                padded_masks[i, :seq_len] = mask

            return padded_embeds, padded_masks, all_audio_lengths

        # Single-turn mode (backward compatible): input_features is [batch, feature_size, feature_length]
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        if input_features is not None:
            # Audio + text: insert audio embeddings between markers
            encoder_outputs = self.encoder(input_features)
            encoder_hidden_states = encoder_outputs[0]

            # Calculate output lengths based on encoder type
            if self.config.encoder_type in ("afwhisper", "qwen2-audio"):
                # AFWhisper: 30秒固定入力、全て有効として扱う
                # encoder_attention_mask is based on input audio length before 30s padding
                seq_len = encoder_hidden_states.shape[1]
                if encoder_attention_mask is not None:
                    # Calculate effective length from mel spectrogram mask
                    # AFWhisper uses stride of 2 in feature extractor
                    mel_lengths = encoder_attention_mask.sum(dim=1, keepdim=True)
                    # Approximate output length: mel_length // 2 (conv stride)
                    lengths = (mel_lengths // 2) // self.config.adapter_kernel_size
                    lengths = lengths.clamp(min=1, max=seq_len // self.config.adapter_kernel_size)
                else:
                    # No mask = fixed 30s, use full sequence
                    lengths = torch.full(
                        (input_features.shape[0], 1),
                        seq_len // self.config.adapter_kernel_size,
                        device=input_features.device,
                    )
            else:
                # Whisper: use built-in length calculation
                lengths = self.encoder._get_feat_extract_output_lengths(encoder_attention_mask.sum(dim=1, keepdim=True))
                lengths = lengths // self.config.adapter_kernel_size

            max_len = lengths.max().item()

            encoder_hidden_states = self.adapter(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states[:, :max_len]

            # Find audio_start marker position (insert audio AFTER this token)
            # Format: ... <|reserved_343|><|reserved_342|> ...
            #                            ^ insert audio here
            audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[1]

            if len(audio_start_positions) == batch_size:
                # All samples have the marker - insert audio after <|reserved_343|>
                insert_pos = audio_start_positions[0].item() + 1  # Assume same position in batch

                # Split embeddings: [before + audio_start] | [audio_end + rest]
                embeds_before = inputs_embeds[:, :insert_pos]  # includes <|reserved_343|>
                embeds_after = inputs_embeds[:, insert_pos:]   # starts with <|reserved_342|>

                # Concatenate: [before + audio_start] + [audio] + [audio_end + rest]
                inputs_embeds = torch.cat((embeds_before, encoder_hidden_states, embeds_after), dim=1)

                # Build attention mask
                mask_before = decoder_attention_mask[:, :insert_pos]
                mask_after = decoder_attention_mask[:, insert_pos:]
                audio_mask = (
                    torch.arange(encoder_hidden_states.shape[1], device=decoder_attention_mask.device).unsqueeze(0)
                    < lengths
                ).long()
                attention_mask = torch.cat((mask_before, audio_mask, mask_after), dim=1)
            else:
                # Fallback: prepend audio (legacy behavior)
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

            # Return with audio_lengths for compatibility (single audio per sample)
            audio_lengths = [encoder_hidden_states.shape[1]] * batch_size
            return inputs_embeds, attention_mask, audio_lengths
        else:
            # Text-only
            return inputs_embeds, decoder_attention_mask, [0] * batch_size

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        audios: Optional[List[List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids: Token ids (batch_size, sequence_length)
            decoder_attention_mask: 1=non-mask, 0=mask (batch_size, sequence_length)
            input_features: Log mel spectrogram (batch_size, feature_size, feature_length), optional (single-turn)
            encoder_attention_mask: 1=non-mask, 0=mask (batch_size, feature_length), optional (single-turn)
            audios: List[List[Tensor]], audios[batch_idx][audio_idx] (multi-turn)
            labels: Labels for language modeling (batch_size, sequence_length), optional
        """
        inputs_embeds, attention_mask, audio_lengths = self.embed(
            input_ids, decoder_attention_mask, input_features, encoder_attention_mask, audios
        )

        if labels is None:
            # Auto-generate labels: mask audio positions with -100
            batch_size = input_ids.shape[0]
            device = input_ids.device

            if audios is not None:
                # Multi-turn: insert -100 at each audio marker position
                all_labels = []
                for batch_idx in range(batch_size):
                    sample_input_ids = input_ids[batch_idx]
                    sample_audios = audios[batch_idx]

                    if len(sample_audios) == 0:
                        all_labels.append(sample_input_ids)
                        continue

                    # Find all marker positions and insert -100 labels
                    marker_positions = (sample_input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0]
                    sample_labels = sample_input_ids.clone()

                    # Process from right to left
                    for audio, marker_pos in reversed(list(zip(sample_audios, marker_positions))):
                        # Calculate audio embedding length (approximate)
                        audio_emb = self._encode_single_audio(audio.unsqueeze(0))
                        audio_len = audio_emb.shape[1]

                        insert_pos = marker_pos.item() + 1
                        labels_before = sample_labels[:insert_pos]
                        labels_after = sample_labels[insert_pos:]
                        audio_labels = torch.full((audio_len,), -100, dtype=sample_labels.dtype, device=device)
                        sample_labels = torch.cat([labels_before, audio_labels, labels_after], dim=0)

                    all_labels.append(sample_labels)

                # Pad labels to match inputs_embeds
                max_len = inputs_embeds.shape[1]
                labels = torch.full((batch_size, max_len), -100, dtype=input_ids.dtype, device=device)
                for i, sample_labels in enumerate(all_labels):
                    labels[i, :len(sample_labels)] = sample_labels

            elif input_features is not None:
                # Single-turn: insert -100 for audio
                total_audio_len = sum(audio_lengths)
                audio_len_per_sample = audio_lengths[0] if audio_lengths else 0

                if audio_len_per_sample > 0:
                    audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[1]
                    if len(audio_start_positions) == batch_size:
                        insert_pos = audio_start_positions[0].item() + 1
                        labels_before = input_ids[:, :insert_pos]
                        labels_after = input_ids[:, insert_pos:]
                        audio_labels = torch.full(
                            (batch_size, audio_len_per_sample), -100, dtype=input_ids.dtype, device=device
                        )
                        labels = torch.cat((labels_before, audio_labels, labels_after), dim=1)
                    else:
                        labels = F.pad(input_ids, (audio_len_per_sample, 0), value=-100)
                else:
                    labels = input_ids
            else:
                labels = input_ids

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
        audios: Optional[List[List[torch.FloatTensor]]] = None,
        **kwargs,
    ):
        inputs_embeds, attention_mask, _ = self.embed(
            input_ids, decoder_attention_mask, input_features, encoder_attention_mask, audios
        )

        generated_ids = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return generated_ids
