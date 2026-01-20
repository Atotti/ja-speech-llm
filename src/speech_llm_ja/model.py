"""Speech LLM model classes."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    WhisperForConditionalGeneration,
)


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
        self.linear1 = nn.Linear(
            encoder_hidden_size, 2 * decoder_hidden_size, bias=bias
        )
        self.linear2 = nn.Linear(
            2 * decoder_hidden_size, decoder_hidden_size, bias=bias
        )

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
AUDIO_END_TOKEN_ID = 350  # <|reserved_342|>


class LlamaForSpeechLMConfig(PretrainedConfig):
    """Configuration for LlamaForSpeechLM."""

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
    """Speech LLM combining Whisper encoder + LLM decoder via trainable adapter."""

    config_class = LlamaForSpeechLMConfig
    # Note: LLM-jp does NOT tie embed_tokens and lm_head weights.
    # Do NOT add _tied_weights_keys here - it would break model loading.

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__(config)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(
            config.encoder_id
        ).model.encoder
        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_id, torch_dtype=torch.bfloat16
        )
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
        """
        Embed input_ids and optionally insert audio embeddings between audio markers.

        Prompt construction is handled by SpeechLlamaProcessor. Audio embeddings are
        inserted between <|reserved_343|> and <|reserved_342|> tokens.
        """
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        if input_features is not None:
            # Audio + text: insert audio embeddings between markers
            encoder_outputs = self.encoder(input_features)
            encoder_hidden_states = encoder_outputs[0]

            lengths = self.encoder._get_feat_extract_output_lengths(
                encoder_attention_mask.sum(dim=1, keepdim=True)
            )
            lengths = lengths // self.config.adapter_kernel_size
            max_len = lengths.max()

            encoder_hidden_states = self.adapter(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states[:, :max_len]

            # Find audio_start marker position (insert audio AFTER this token)
            # Format: ... <|reserved_343|><|reserved_342|> ...
            #                            ^ insert audio here
            batch_size = input_ids.shape[0]
            audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(
                as_tuple=True
            )[1]

            if len(audio_start_positions) == batch_size:
                # All samples have the marker - insert audio after <|reserved_343|>
                insert_pos = (
                    audio_start_positions[0].item() + 1
                )  # Assume same position in batch

                # Split embeddings: [before + audio_start] | [audio_end + rest]
                embeds_before = inputs_embeds[
                    :, :insert_pos
                ]  # includes <|reserved_343|>
                embeds_after = inputs_embeds[
                    :, insert_pos:
                ]  # starts with <|reserved_342|>

                # Concatenate: [before + audio_start] + [audio] + [audio_end + rest]
                inputs_embeds = torch.cat(
                    (embeds_before, encoder_hidden_states, embeds_after), dim=1
                )

                # Build attention mask
                mask_before = decoder_attention_mask[:, :insert_pos]
                mask_after = decoder_attention_mask[:, insert_pos:]
                audio_mask = (
                    torch.arange(
                        encoder_hidden_states.shape[1],
                        device=decoder_attention_mask.device,
                    ).unsqueeze(0)
                    < lengths
                ).long()
                attention_mask = torch.cat((mask_before, audio_mask, mask_after), dim=1)
            else:
                # Fallback: prepend audio (legacy behavior)
                inputs_embeds = torch.cat((encoder_hidden_states, inputs_embeds), dim=1)
                attention_mask = torch.cat(
                    (
                        (
                            torch.arange(
                                encoder_hidden_states.shape[1],
                                device=decoder_attention_mask.device,
                            ).unsqueeze(0)
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
            input_ids: Token ids (batch_size, sequence_length)
            decoder_attention_mask: 1=non-mask, 0=mask (batch_size, sequence_length)
            input_features: Log mel spectrogram (batch_size, feature_size, feature_length), optional
            encoder_attention_mask: 1=non-mask, 0=mask (batch_size, feature_length), optional
            labels: Labels for language modeling (batch_size, sequence_length), optional
        """
        inputs_embeds, attention_mask = self.embed(
            input_ids, decoder_attention_mask, input_features, encoder_attention_mask
        )

        if labels is None:
            # Auto-generate labels: mask audio positions with -100
            audio_len = inputs_embeds.shape[1] - input_ids.shape[1]
            if audio_len > 0 and input_features is not None:
                # Find insert position and create labels with -100 for audio
                audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(
                    as_tuple=True
                )[1]
                if len(audio_start_positions) == input_ids.shape[0]:
                    insert_pos = audio_start_positions[0].item() + 1
                    labels_before = input_ids[:, :insert_pos]
                    labels_after = input_ids[:, insert_pos:]
                    audio_labels = torch.full(
                        (input_ids.shape[0], audio_len),
                        -100,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    labels = torch.cat(
                        (labels_before, audio_labels, labels_after), dim=1
                    )
                else:
                    # Fallback: prepend -100 for audio
                    labels = F.pad(input_ids, (audio_len, 0), value=-100)
            else:
                labels = input_ids
        else:
            # If labels are text-only, expand them to match inserted audio embeddings.
            if input_features is not None and labels.shape[1] == input_ids.shape[1]:
                audio_len = inputs_embeds.shape[1] - input_ids.shape[1]
                if audio_len > 0:
                    audio_start_positions = (input_ids == AUDIO_START_TOKEN_ID).nonzero(
                        as_tuple=True
                    )[1]
                    if len(audio_start_positions) == input_ids.shape[0]:
                        insert_pos = audio_start_positions[0].item() + 1
                        labels_before = labels[:, :insert_pos]
                        labels_after = labels[:, insert_pos:]
                        audio_labels = torch.full(
                            (input_ids.shape[0], audio_len),
                            -100,
                            dtype=labels.dtype,
                            device=labels.device,
                        )
                        labels = torch.cat(
                            (labels_before, audio_labels, labels_after), dim=1
                        )
                    else:
                        labels = F.pad(labels, (audio_len, 0), value=-100)

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
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

        generated_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
        return generated_ids
