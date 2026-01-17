#!/usr/bin/env python
"""Gradio Web UI for Japanese Speech LLM - ChatGPT voice mode style."""

import tempfile
from threading import Thread

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AutoTokenizer, TextIteratorStreamer

from demo2_ja import LlamaForSpeechLM, LlamaForSpeechLMConfig

# =============================================================================
# Configuration
# =============================================================================
MODEL_ID = "Atotti/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000"
DECODER_ID = "models/v4-8b-decay2m-ipt_v3.1-instruct4"
MAX_AUDIO_DURATION = 30.0

CHAT_PROMPT = """以下は、タスクを説明する音声の指示です。要求を適切に満たす応答を書きなさい。

### 指示:


### 応答:
"""

# =============================================================================
# Model Loading
# =============================================================================
print(f"Loading model: {MODEL_ID}")

config = LlamaForSpeechLMConfig.from_pretrained(MODEL_ID)
config.decoder_id = DECODER_ID

model = LlamaForSpeechLM.from_pretrained(MODEL_ID, config=config).cuda().eval()

encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
decoder_processor = AutoTokenizer.from_pretrained(DECODER_ID)
decoder_processor.pad_token = decoder_processor.eos_token
print("Model loaded successfully!")


# =============================================================================
# Inference Functions
# =============================================================================
def preprocess_audio(audio_tuple: tuple) -> torch.Tensor:
    """Convert Gradio audio input to 16kHz mono tensor."""
    sample_rate, audio_array = audio_tuple

    if audio_array.dtype in [np.int16, np.int32]:
        audio_array = audio_array.astype(np.float32) / 32768.0

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)

    audio_tensor = torch.from_numpy(audio_array).float()

    duration = len(audio_tensor) / sample_rate
    if duration > MAX_AUDIO_DURATION:
        raise gr.Error(f"音声が長すぎます（{duration:.1f}秒）。{MAX_AUDIO_DURATION}秒以下にしてください。")

    if sample_rate != 16000:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.unsqueeze(0), sample_rate, 16000
        ).squeeze(0)

    return audio_tensor


def save_audio_to_file(audio_tuple: tuple) -> str:
    """Save audio to a temporary file and return the path."""
    sample_rate, audio_array = audio_tuple

    if audio_array.dtype in [np.int16, np.int32]:
        audio_array = audio_array.astype(np.float32) / 32768.0

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)

    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(temp_file.name, audio_tensor, sample_rate)
    return temp_file.name


def chat(audio_tuple, history):
    """Process audio input and generate streaming response."""
    if audio_tuple is None:
        yield history
        return

    # Save audio to file for display
    audio_path = save_audio_to_file(audio_tuple)

    # Add user message with audio
    history = history + [{"role": "user", "content": {"path": audio_path}}]
    yield history

    # Preprocess audio
    audio_tensor = preprocess_audio(audio_tuple)

    # Prepare inputs
    encoder_inputs = encoder_processor(
        [audio_tensor.numpy()],
        return_tensors="pt",
        return_attention_mask=True,
        sampling_rate=16000,
    ).to("cuda")

    decoder_inputs = decoder_processor(
        CHAT_PROMPT,
        return_tensors="pt",
    ).to("cuda")

    # Setup streamer
    streamer = TextIteratorStreamer(
        decoder_processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Generate in separate thread
    generation_kwargs = {
        "input_features": encoder_inputs.input_features,
        "input_ids": decoder_inputs.input_ids,
        "encoder_attention_mask": encoder_inputs.attention_mask,
        "decoder_attention_mask": decoder_inputs.attention_mask,
        "max_length": 1024,
        "do_sample": False,
        "num_beams": 1,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream response
    history = history + [{"role": "assistant", "content": ""}]
    for token in streamer:
        history[-1]["content"] += token
        yield history

    thread.join()


# =============================================================================
# Gradio Interface
# =============================================================================
CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
    font-family: 'Segoe UI', 'Hiragino Sans', 'Meiryo', sans-serif !important;
}

/* Header */
.header-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

.header-subtitle {
    text-align: center;
    color: #6b7280 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1.5rem !important;
}

/* Chatbot container */
.chatbot-container {
    border: none !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08) !important;
    background: #fafafa !important;
}

/* Messages */
.message {
    border-radius: 16px !important;
    padding: 12px 16px !important;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.bot-message {
    background: white !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}

/* Audio input */
.audio-container {
    border: 2px dashed #e5e7eb !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    transition: all 0.3s ease !important;
}

.audio-container:hover {
    border-color: #667eea !important;
    background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%) !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

.clear-btn {
    background: #f3f4f6 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #6b7280 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.clear-btn:hover {
    background: #e5e7eb !important;
    color: #374151 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.85rem;
    margin-top: 1.5rem;
}
"""

def create_demo() -> gr.Blocks:
    with gr.Blocks(title="日本語音声LLM デモ") as demo:
        gr.Markdown("# 日本語音声LLM", elem_classes=["header-title"])
        gr.Markdown("音声で話しかけると、AIがリアルタイムで応答します", elem_classes=["header-subtitle"])

        chatbot = gr.Chatbot(
            height=480,
            show_label=False,
            elem_classes=["chatbot-container"],
            layout="bubble",
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎙️ マイクをクリックして話す",
                elem_classes=["audio-container"],
            )

        with gr.Row():
            clear_btn = gr.Button("クリア", elem_classes=["clear-btn"])

        gr.Markdown(
            f"Model: `{MODEL_ID}`",
            elem_classes=["footer"],
        )

        # Process audio when recording stops
        audio_input.stop_recording(
            fn=chat,
            inputs=[audio_input, chatbot],
            outputs=chatbot,
        ).then(
            fn=lambda: None,
            outputs=audio_input,
        )

        clear_btn.click(fn=lambda: [], outputs=chatbot)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
