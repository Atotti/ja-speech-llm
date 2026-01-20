#!/usr/bin/env python
"""Quick inference test script"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from demo2_ja import LlamaForSpeechLM, ReazonSpeech, SpeechLlamaProcessor
import evaluate

# Load model
print("Loading model...")
model = LlamaForSpeechLM.from_pretrained(
    "models/LlamaForSpeechLM-ja-20260110-221600"
).cuda().eval()
processor = SpeechLlamaProcessor.from_pretrained(
    encoder_id=model.config.encoder_id,
    decoder_id=model.config.decoder_id,
)

# Load test samples
print("Loading test data...")
dataset = ReazonSpeech(split='test', max_duration=15.0)

print("\n" + "="*50)
hyps = []
refs = []

for i, sample in enumerate(dataset):
    if i >= 10:  # 10サンプルテスト
        break

    waveform, sr, ref = sample[0], sample[1], sample[2]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "音声を書き起こしてください。"},
            ],
        }
    ]
    model_inputs = processor(
        messages,
        audios=[waveform.squeeze(0)],
        add_generation_prompt=True,
        return_labels=False,
    )
    model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            decoder_attention_mask=model_inputs["decoder_attention_mask"],
            input_features=model_inputs["input_features"],
            encoder_attention_mask=model_inputs["encoder_attention_mask"],
            max_length=1024,
            do_sample=False,
            num_beams=1,  # greedy (validation と同じ)
        )

    hyp = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    hyps.append(hyp)
    refs.append(ref)

    print(f"[{i+1}] Reference:  {ref}")
    print(f"    Prediction: {hyp}")
    print("-"*50)

# CER計算 (日本語に適切)
cer_metric = evaluate.load("cer")
cer = cer_metric.compute(predictions=hyps, references=refs) * 100
print(f"\nCER: {cer:.2f}%")
