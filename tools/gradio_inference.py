#!/usr/bin/env python
"""Gradio inference for a pushed Speech LLM model (audio/text multi-turn)."""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Tuple, Union

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2_ja import (  # noqa: E402
    LlamaForSpeechLM,
    LlamaForSpeechLMConfig,
    SpeechLlamaProcessor,
    SpeechLlamaProcessorConfig,
)

DEFAULT_MODEL_ID = "Harui-i/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000"


def _normalize_audio(audio: Any, target_sr: int = 16000) -> np.ndarray:
    if audio is None:
        return None
    sr = None
    data = None
    if isinstance(audio, dict):
        sr = audio.get("sampling_rate")
        data = audio.get("array")
    elif isinstance(audio, (list, tuple)) and len(audio) == 2:
        sr, data = audio
    else:
        data = audio
    if data is None:
        return None
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sr is None:
        sr = target_sr
    if sr != target_sr:
        wav = torch.from_numpy(data)
        data = torchaudio.functional.resample(wav, sr, target_sr).numpy()
    return data


def _load_model(model_id: str, device: torch.device):
    decoder_id = "/home/harui/ml/experi/v4-8b-decay2m-ipt_v3.1-instruct4"
    config = LlamaForSpeechLMConfig.from_pretrained(model_id)
    config.decoder_id = decoder_id
    config.text_config = AutoConfig.from_pretrained(decoder_id)
    config.vocab_size = config.text_config.vocab_size
    config.hidden_size = config.text_config.hidden_size
    config.num_hidden_layers = config.text_config.num_hidden_layers
    config.num_attention_heads = config.text_config.num_attention_heads
    config.pad_token_id = config.text_config.pad_token_id
    config.bos_token_id = config.text_config.bos_token_id
    config.eos_token_id = config.text_config.eos_token_id
    model = LlamaForSpeechLM.from_pretrained(model_id, config=config)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model = model.to(dtype=torch.float32)

    processor_config = SpeechLlamaProcessorConfig(
        adapter_kernel_size=model.config.adapter_kernel_size,
    )
    processor = SpeechLlamaProcessor.from_pretrained(
        encoder_id=model.config.encoder_id,
        decoder_id=decoder_id,
        config=processor_config,
    )
    return model, processor


def _build_user_message(user_text: str, user_audio: Any) -> Tuple[Any, str]:
    content_parts: List[dict] = []
    display = []
    if user_audio is not None:
        content_parts.append({"type": "audio", "audio": user_audio})
        display.append("音声")
    if user_text:
        content_parts.append({"type": "text", "text": user_text})
        display.append(user_text)
    if not content_parts:
        return "", ""
    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
        return user_text, user_text
    return content_parts, "\n".join(display)


def _generate_reply(
    model: LlamaForSpeechLM,
    processor: SpeechLlamaProcessor,
    device: torch.device,
    messages: list[dict[str, Union[str, list[dict[str, str]]]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
) -> str:
    print(f"[_generate_reply] messages={messages}")
    # messages format (ProcessorMixin):
    # [
    #   {"role": "system|user|assistant",
    #    "content": "text" or [{"type":"text","text":"..."},{"type":"audio","audio": np.ndarray}]},
    # ]
    model_inputs = processor(
        messages,
        add_generation_prompt=True,
        return_labels=False,
    )
    print(f"[_generate_reply] model_inputs keys={list(model_inputs.keys())}")
    prompt_text = processor.tokenizer.decode(
        model_inputs["input_ids"][0], skip_special_tokens=False
    )
    print(f"[_generate_reply] prompt_text tail={prompt_text[-400:]!r}")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            generated_ids = model.generate(
                input_ids=model_inputs["input_ids"],
                decoder_attention_mask=model_inputs["decoder_attention_mask"],
                input_features=model_inputs.get("input_features"),
                encoder_attention_mask=model_inputs.get("encoder_attention_mask"),
                **gen_kwargs,
            )

    attention_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(attention_mask[0].sum().item())
    if generated_ids.shape[1] < prompt_len:
        new_tokens = generated_ids[0]
    else:
        new_tokens = generated_ids[0, prompt_len:]
    print(
        f"[_generate_reply] generated_ids shape={tuple(generated_ids.shape)} "
        f"prompt_len={prompt_len} new_tokens_len={int(new_tokens.numel())}"
    )
    print(f"[_generate_reply] new_tokens ids={new_tokens.tolist()}")
    reply = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    if not reply:
        reply_raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        print(f"[_generate_reply] reply_raw={reply_raw!r}")
    print(f"[_generate_reply] reply={reply!r}")
    return reply


def build_ui(model_id: str, device: torch.device):
    model, processor = _load_model(model_id, device)

    with gr.Blocks(title="Speech LLM JA Chat") as demo:
        gr.Markdown("# Speech LLM JA - 音声/テキスト マルチターンチャット")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=520)
                user_text = gr.Textbox(
                    label="テキスト入力",
                    placeholder="音声だけでもOK。テキストだけでもOK。",
                )
                user_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="音声入力",
                )
                send_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("クリア")
            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    label="システムプロンプト",
                    value=processor.config.system_prompt,
                )
                max_new_tokens = gr.Slider(
                    32, 1024, value=256, step=16, label="max_new_tokens"
                )
                do_sample = gr.Checkbox(value=False, label="サンプリング")
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.1, label="temperature"
                )
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                num_beams = gr.Slider(1, 5, value=1, step=1, label="num_beams")

        messages_state = gr.State([])
        chat_state = gr.State([])

        def submit_message(
            text,
            audio,
            messages,
            chat_history,
            sys_prompt,
            max_new,
            sample,
            temp,
            p,
            beams,
        ):
            print(
                f"[submit_message] text={text!r} audio={'yes' if audio is not None else 'no'}"
            )
            audio_array = _normalize_audio(audio)
            content, display_user = _build_user_message(text, audio_array)
            if not content:
                print("[submit_message] empty content -> no-op")
                return chat_history, messages, chat_history, "", None

            processor.config.system_prompt = sys_prompt
            messages = list(messages)
            messages.append({"role": "user", "content": content})
            print(f"[submit_message] messages size={len(messages)}")

            chat_history = list(chat_history)
            chat_history.append({"role": "user", "content": display_user})
            print(f"[submit_message] chat_history size={len(chat_history)}")

            reply = _generate_reply(
                model=model,
                processor=processor,
                device=device,
                messages=messages,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                top_p=p,
                num_beams=int(beams),
            )

            messages.append({"role": "assistant", "content": reply})
            chat_history.append({"role": "assistant", "content": reply})
            print(
                f"[submit_message] reply len={len(reply)} chat_history size={len(chat_history)}"
            )
            return chat_history, messages, chat_history, "", None

        send_inputs = [
            user_text,
            user_audio,
            messages_state,
            chat_state,
            system_prompt,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            num_beams,
        ]
        send_outputs = [chatbot, messages_state, chat_state, user_text, user_audio]

        send_btn.click(submit_message, inputs=send_inputs, outputs=send_outputs)
        user_text.submit(submit_message, inputs=send_inputs, outputs=send_outputs)

        def clear_history():
            return [], [], [], "", None

        clear_btn.click(
            clear_history,
            inputs=None,
            outputs=[chatbot, messages_state, chat_state, user_text, user_audio],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face Hub model id or local path",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = build_ui(args.model_id, device)
    demo.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
