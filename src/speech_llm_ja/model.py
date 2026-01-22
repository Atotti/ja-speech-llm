"""Speech LLM model classes."""

from typing import Optional

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
from transformers.generation.utils import GenerationMixin

# Audio token ID (using reserved tokens from the end)
AUDIO_TOKEN_ID = 351  # <|reserved_343|>


class LlamaForSpeechLMConfig(PretrainedConfig):
    """Configuration for LlamaForSpeechLM with Voxtral-like structure."""

    model_type = "llama_for_speech_lm"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        encoder_id: str = "openai/whisper-large-v3",
        decoder_id: str = "/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4",
        audio_config=None,
        text_config=None,
        audio_token_id: int = AUDIO_TOKEN_ID,
        adapter_kernel_size: int = 4,
        adapter_linear_bias: bool = False,
        **kwargs,
    ):
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_linear_bias = adapter_linear_bias
        self.audio_token_id = audio_token_id

        if isinstance(audio_config, dict):
            model_type = audio_config.get("model_type", "whisper")
            audio_config = AutoConfig.for_model(
                model_type,
                **{k: v for k, v in audio_config.items() if k != "model_type"},
            )
        elif audio_config is None and encoder_id is not None:
            audio_config = AutoConfig.from_pretrained(encoder_id)
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            model_type = text_config.get("model_type", "llama")
            text_config = AutoConfig.for_model(
                model_type,
                **{k: v for k, v in text_config.items() if k != "model_type"},
            )
        elif text_config is None and decoder_id is not None:
            text_config = AutoConfig.from_pretrained(decoder_id)
        self.text_config = text_config

        if text_config is not None:
            self._sync_text_config(text_config)

        super().__init__(**kwargs)

    def _sync_text_config(self, text_config) -> None:
        self.vocab_size = getattr(text_config, "vocab_size", None)
        self.hidden_size = getattr(text_config, "hidden_size", None)
        self.num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        self.num_attention_heads = getattr(text_config, "num_attention_heads", None)
        self.pad_token_id = getattr(text_config, "pad_token_id", None)
        self.bos_token_id = getattr(text_config, "bos_token_id", None)
        self.eos_token_id = getattr(text_config, "eos_token_id", None)


class Adapter(nn.Module):
    """2-layer MLP adapter with average pooling."""

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__()
        encoder_hidden_size = getattr(config.audio_config, "d_model", None)
        if encoder_hidden_size is None:
            encoder_hidden_size = getattr(config.audio_config, "hidden_size", None)
        decoder_hidden_size = getattr(config.text_config, "hidden_size", None)
        if encoder_hidden_size is None or decoder_hidden_size is None:
            raise ValueError("audio_config/text_config must define hidden sizes.")

        self.pool = nn.AvgPool1d(config.adapter_kernel_size)
        self.linear1 = nn.Linear(
            encoder_hidden_size,
            2 * decoder_hidden_size,
            bias=config.adapter_linear_bias,
        )
        self.linear2 = nn.Linear(
            2 * decoder_hidden_size,
            decoder_hidden_size,
            bias=config.adapter_linear_bias,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.pool(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LlamaForSpeechLM(PreTrainedModel, GenerationMixin):
    """Speech LLM combining Whisper encoder + LLM decoder via trainable adapter."""

    config_class = LlamaForSpeechLMConfig
    # Note: LLM-jp does NOT tie embed_tokens and lm_head weights.
    # Do NOT add _tied_weights_keys here - it would break model loading.

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__(config)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(
            config.encoder_id
        ).model.encoder
        if config.decoder_id:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                config.decoder_id, torch_dtype=torch.bfloat16
            )
        elif config.text_config is not None:
            self.decoder = AutoModelForCausalLM.from_config(config.text_config)
        else:
            raise ValueError("decoder_id or text_config must be provided.")
        if config.audio_config is None:
            config.audio_config = self.encoder.config
        if config.text_config is None:
            config.text_config = self.decoder.config
            config._sync_text_config(config.text_config)
        self.adapter = Adapter(config)

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

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (input_features.shape[0], input_features.shape[-1]),
                device=input_features.device,
                dtype=torch.long,
            )

        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = self.adapter(encoder_hidden_states)

        lengths = self.encoder._get_feat_extract_output_lengths(
            encoder_attention_mask.sum(dim=1)
        )
        lengths = lengths // self.config.adapter_kernel_size

        audio_segments = []
        for idx, length in enumerate(lengths.tolist()):
            if length > 0:
                audio_segments.append(encoder_hidden_states[idx, :length])
        if not audio_segments:
            return None
        return torch.cat(audio_segments, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: Token ids (batch_size, sequence_length)
            decoder_attention_mask: 1=non-mask, 0=mask (batch_size, sequence_length)
            input_features: Log mel spectrogram (batch_size, feature_size, feature_length), optional
            encoder_attention_mask: 1=non-mask, 0=mask (batch_size, feature_length), optional
            labels: Labels for language modeling (batch_size, sequence_length), optional
        """
        if attention_mask is None:
            attention_mask = decoder_attention_mask

        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        if input_features is not None:
            if input_ids is None:
                raise ValueError(
                    "input_ids is required when input_features is provided."
                )
            audio_embeds = self.get_audio_features(
                input_features, encoder_attention_mask
            )
            if audio_embeds is not None:
                audio_token_mask = (input_ids == AUDIO_TOKEN_ID).unsqueeze(-1)
                if audio_token_mask.sum().item() != audio_embeds.shape[0]:
                    raise ValueError(
                        "Audio token count does not match audio embeddings length."
                    )
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_token_mask.to(inputs_embeds.device),
                    audio_embeds.to(inputs_embeds.device),
                )
            if labels is not None:
                labels = labels.clone()
                labels[input_ids == AUDIO_TOKEN_ID] = -100

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        input_features = kwargs.pop("input_features", None)
        decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)
        if decoder_attention_mask is not None and "attention_mask" not in kwargs:
            kwargs["attention_mask"] = decoder_attention_mask

        cache_position = kwargs.get("cache_position")
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if input_features is not None and (
            cache_position is None or cache_position[0] == 0
        ):
            model_inputs["input_features"] = input_features

        return model_inputs

    def generate(self, *args, **kwargs):
        if "decoder_attention_mask" in kwargs and "attention_mask" not in kwargs:
            kwargs["attention_mask"] = kwargs.pop("decoder_attention_mask")
        return super().generate(*args, **kwargs)
