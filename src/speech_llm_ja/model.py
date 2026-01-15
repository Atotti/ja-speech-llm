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
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

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
