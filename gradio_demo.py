#!/usr/bin/env python
"""Gradio Web UI for Japanese Speech LLM - ChatGPT voice mode style."""

import tempfile
from threading import Thread

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AutoTokenizer, TextIteratorStreamer

from speech_llm_ja import LlamaForSpeechLM, LlamaForSpeechLMConfig

# =============================================================================
# Configuration
# =============================================================================
MODEL_ID = "Atotti/llm-jp-4-8b-speech-chat
DECODER_ID = "models/v4-8b-decay2m-ipt_v3.1-instruct4"
MAX_AUDIO_DURATION = 30.0

# Single-turn用プロンプトテンプレート
PROMPT_TEMPLATE = """あなたは音声を理解できるAIアシスタントです。

<|reserved_343|><|reserved_342|>### 指示:
{instruction}

### 応答:
"""

# 応答モード別の指示文
RESPONSE_MODES = {
    "音声指示 (IF)": "音声の指示に従ってください。",
    "音声書き起こし (ASR)": "音声を書き起こしてください。",
    "音声説明 (AAC)": "音声を説明してください。",
    "カスタム": None,  # custom_instructionを使用
}

# プロンプトテンプレート（カスタムモード用）
PROMPT_TEMPLATES = {
    "日本語で説明": "音声の内容を日本語で詳しく説明してください。",
    "英語→日本語翻訳": "英語の音声を日本語に翻訳して書き起こしてください。",
    "日本語→英語翻訳": "日本語の音声を英語に翻訳して書き起こしてください。",
    "要約": "音声の内容を簡潔に要約してください。",
}

# 推論パラメータのデフォルト値
DEFAULT_GENERATION_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "do_sample": False,
}

# Multi-turn用システムプロンプト
SYSTEM_PROMPT = "あなたは音声を理解できるAIアシスタントです。\n\n"


def build_multiturn_prompt(turns, current_instruction, current_has_audio=True):
    """Multi-turn形式のプロンプトを構築する。

    Args:
        turns: 過去のターンのリスト [{"instruction": str, "response": str, "audio_features": ...}, ...]
        current_instruction: 現在のターンの指示文
        current_has_audio: 現在のターンに音声があるか

    Returns:
        Multi-turn形式のプロンプト文字列
    """
    prompt = SYSTEM_PROMPT
    for turn in turns:
        # 音声があるターンのみに音声マーカーを追加
        if "audio_features" in turn:
            prompt += f"<|reserved_343|><|reserved_342|>### 指示:\n{turn['instruction']}\n\n"
        else:
            prompt += f"### 指示:\n{turn['instruction']}\n\n"
        prompt += f"### 応答:\n{turn['response']}\n\n"
    # 現在のターン（応答なし）
    if current_has_audio:
        prompt += f"<|reserved_343|><|reserved_342|>### 指示:\n{current_instruction}\n\n"
    else:
        prompt += f"### 指示:\n{current_instruction}\n\n"
    prompt += "### 応答:\n"
    return prompt

# =============================================================================
# Model Loading
# =============================================================================
print(f"Loading model: {MODEL_ID}")

config = LlamaForSpeechLMConfig.from_pretrained(MODEL_ID)
config.decoder_id = DECODER_ID

model = LlamaForSpeechLM.from_pretrained(
    MODEL_ID,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval()

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


def chat(audio_tuple, history, mode, conversation_turns,
         custom_instruction, temperature, top_p, max_new_tokens, do_sample):
    """Process audio input and generate streaming response.

    Args:
        audio_tuple: Gradio audio input (sample_rate, audio_array)
        history: Gradio chat history for display
        mode: Response mode ("音声指示 (IF)", "音声書き起こし (ASR)", "音声説明 (AAC)", "カスタム")
        conversation_turns: List of past turns for multi-turn conversation
        custom_instruction: Custom instruction text (used when mode is "カスタム")
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling

    Yields:
        Updated (history, conversation_turns) tuples
    """
    if audio_tuple is None:
        yield history, conversation_turns
        return

    # Save audio to file for display
    audio_path = save_audio_to_file(audio_tuple)

    # Add user message with audio
    history = history + [{"role": "user", "content": {"path": audio_path}}]
    yield history, conversation_turns

    # Preprocess audio
    audio_tensor = preprocess_audio(audio_tuple)

    # Get instruction for current mode (use custom_instruction for カスタム mode)
    if mode == "カスタム":
        instruction = custom_instruction if custom_instruction.strip() else "音声の指示に従ってください。"
    else:
        instruction = RESPONSE_MODES[mode]

    # Prepare encoder inputs (audio features)
    encoder_inputs = encoder_processor(
        [audio_tensor.numpy()],
        return_tensors="pt",
        return_attention_mask=True,
        sampling_rate=16000,
    ).to("cuda")

    # Current audio features for multi-turn
    current_audio_features = encoder_inputs.input_features.squeeze(0)  # [feature_size, feature_length]

    # Setup streamer
    streamer = TextIteratorStreamer(
        decoder_processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Choose single-turn or multi-turn based on mode and history
    # Multi-turn is enabled for IF and custom modes when there's conversation history
    is_multiturn = mode in ("音声指示 (IF)", "カスタム") and len(conversation_turns) > 0

    if is_multiturn:
        # Multi-turn: use audios parameter
        prompt = build_multiturn_prompt(conversation_turns, instruction, current_has_audio=True)
        decoder_inputs = decoder_processor(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        # Build audios list: past audio features + current audio features
        # Filter out text-only turns that don't have audio_features
        audios = [[turn["audio_features"] for turn in conversation_turns if "audio_features" in turn] + [current_audio_features]]

        generation_kwargs = {
            "input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "audios": audios,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "num_beams": 1,
            "streamer": streamer,
            "pad_token_id": decoder_processor.eos_token_id,
        }
    else:
        # Single-turn: use input_features parameter
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        decoder_inputs = decoder_processor(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        generation_kwargs = {
            "input_features": encoder_inputs.input_features,
            "input_ids": decoder_inputs.input_ids,
            "encoder_attention_mask": encoder_inputs.attention_mask,
            "decoder_attention_mask": decoder_inputs.attention_mask,
            "max_length": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "num_beams": 1,
            "streamer": streamer,
        }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream response
    history = history + [{"role": "assistant", "content": ""}]
    generated_text = ""
    for token in streamer:
        generated_text += token
        history[-1]["content"] = generated_text
        yield history, conversation_turns

    thread.join()

    # Update conversation state for IF and custom modes
    if mode in ("音声指示 (IF)", "カスタム"):
        conversation_turns = conversation_turns + [{
            "audio_features": current_audio_features,
            "instruction": instruction,
            "response": generated_text,
        }]

    yield history, conversation_turns


def text_chat(message, history, conversation_turns,
              temperature, top_p, max_new_tokens, do_sample):
    """Process text input and generate streaming response (text-only mode).

    Args:
        message: User's text message
        history: Gradio chat history for display
        conversation_turns: List of past turns for multi-turn conversation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling

    Yields:
        Updated (history, conversation_turns) tuples
    """
    if not message or not message.strip():
        yield history, conversation_turns
        return

    # Add user message
    history = history + [{"role": "user", "content": message}]
    yield history, conversation_turns

    # Build prompt for text-only mode (no audio)
    if len(conversation_turns) > 0:
        # Multi-turn conversation
        prompt = build_multiturn_prompt(conversation_turns, message, current_has_audio=False)
    else:
        # Single-turn
        prompt = f"{SYSTEM_PROMPT}### 指示:\n{message}\n\n### 応答:\n"

    # Tokenize prompt
    decoder_inputs = decoder_processor(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    # Setup streamer
    streamer = TextIteratorStreamer(
        decoder_processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Collect past audio features for multi-turn context
    past_audio_features = [turn["audio_features"] for turn in conversation_turns if "audio_features" in turn]

    # Generate (text mode with past audio context)
    generation_kwargs = {
        "input_ids": decoder_inputs.input_ids,
        "decoder_attention_mask": decoder_inputs.attention_mask,
        "audios": [past_audio_features],  # Include past audio features for context
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "num_beams": 1,
        "streamer": streamer,
        "pad_token_id": decoder_processor.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream response
    history = history + [{"role": "assistant", "content": ""}]
    generated_text = ""
    for token in streamer:
        generated_text += token
        history[-1]["content"] = generated_text
        yield history, conversation_turns

    thread.join()

    # Update conversation state (text-only turns don't have audio_features)
    conversation_turns = conversation_turns + [{
        "instruction": message,
        "response": generated_text,
    }]

    yield history, conversation_turns


# =============================================================================
# Gradio Interface
# =============================================================================
CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    width: 60vw !important;
    margin: auto !important;
    font-family: 'Segoe UI', 'Hiragino Sans', 'Meiryo', sans-serif !important;
}

/* Hide Gradio default footer */
footer {
    display: none !important;
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
    width: 100% !important;
    max-width: 100% !important;
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

/* Sidebar styles */
.sidebar-section {
    margin-bottom: 1.5rem;
}

.sidebar-section h3 {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.75rem;
}

.sidebar-divider {
    border-top: 1px solid #e5e7eb;
    margin: 1rem 0;
}
"""

def create_demo() -> gr.Blocks:
    with gr.Blocks(title="日本語音声LLM デモ", theme=gr.themes.Soft()) as demo:
        # Sidebar for advanced settings
        with gr.Sidebar(position="right", open=False):
            gr.Markdown("## 詳細設定")

            # Response mode section
            gr.Markdown("### 応答モード", elem_classes=["sidebar-section"])
            mode_selector = gr.Radio(
                choices=list(RESPONSE_MODES.keys()),
                value="音声指示 (IF)",
                label="モード選択",
            )

            gr.HTML('<div class="sidebar-divider"></div>')

            # Custom prompt section
            gr.Markdown("### カスタムプロンプト", elem_classes=["sidebar-section"])
            template_dropdown = gr.Dropdown(
                choices=["（選択してください）"] + list(PROMPT_TEMPLATES.keys()),
                value="（選択してください）",
                label="テンプレートから選択",
            )
            custom_instruction = gr.Textbox(
                value="",
                label="カスタム指示文",
                placeholder="任意の指示文を入力...",
                lines=3,
            )

            gr.HTML('<div class="sidebar-divider"></div>')

            # Inference parameters section
            gr.Markdown("### 推論パラメータ", elem_classes=["sidebar-section"])
            do_sample = gr.Checkbox(
                value=DEFAULT_GENERATION_PARAMS["do_sample"],
                label="サンプリングを有効化",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=DEFAULT_GENERATION_PARAMS["temperature"],
                step=0.1,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_GENERATION_PARAMS["top_p"],
                step=0.05,
                label="Top-p",
            )
            max_new_tokens = gr.Slider(
                minimum=64,
                maximum=2048,
                value=DEFAULT_GENERATION_PARAMS["max_new_tokens"],
                step=64,
                label="Max tokens",
            )

            gr.HTML('<div class="sidebar-divider"></div>')

            # Clear button and model info
            clear_btn = gr.Button("クリア", elem_classes=["clear-btn"])
            gr.Markdown(
                f"Model: `{MODEL_ID}`",
                elem_classes=["footer"],
            )

        # Template selection updates custom_instruction
        template_dropdown.change(
            fn=lambda t: PROMPT_TEMPLATES.get(t, ""),
            inputs=[template_dropdown],
            outputs=[custom_instruction],
        )

        # State for multi-turn conversation
        conversation_state = gr.State([])

        chatbot = gr.Chatbot(
            height="75vh",
            show_label=False,
            elem_classes=["chatbot-container"],
            layout="bubble",
        )

        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="マイクをクリックして話す",
            elem_classes=["audio-container"],
        )

        with gr.Row():
            text_input = gr.Textbox(
                placeholder="またはテキストで入力...",
                show_label=False,
                scale=9,
            )
            text_submit = gr.Button("送信", scale=1)

        # Process audio when recording stops
        audio_input.stop_recording(
            fn=chat,
            inputs=[
                audio_input, chatbot, mode_selector, conversation_state,
                custom_instruction, temperature, top_p, max_new_tokens, do_sample
            ],
            outputs=[chatbot, conversation_state],
        ).then(
            fn=lambda: None,
            outputs=audio_input,
        )

        # Process text input when submitted
        text_submit.click(
            fn=text_chat,
            inputs=[
                text_input, chatbot, conversation_state,
                temperature, top_p, max_new_tokens, do_sample
            ],
            outputs=[chatbot, conversation_state],
        ).then(
            fn=lambda: "",
            outputs=text_input,
        )

        text_input.submit(
            fn=text_chat,
            inputs=[
                text_input, chatbot, conversation_state,
                temperature, top_p, max_new_tokens, do_sample
            ],
            outputs=[chatbot, conversation_state],
        ).then(
            fn=lambda: "",
            outputs=text_input,
        )

        # Clear both chat history and conversation state
        clear_btn.click(fn=lambda: ([], []), outputs=[chatbot, conversation_state])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
