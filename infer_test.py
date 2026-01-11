#!/usr/bin/env python
"""Quick inference test script"""
import torch
from transformers import AutoProcessor, AutoTokenizer
from demo2_ja import LlamaForSpeechLM, ReazonSpeech
import evaluate

# Load model
print("Loading model...")
model = LlamaForSpeechLM.from_pretrained('models/LlamaForSpeechLM-ja-20260110-221600').cuda().eval()

encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
decoder_processor.pad_token = decoder_processor.eos_token

# Load test samples
print("Loading test data...")
dataset = ReazonSpeech(split='test', max_duration=15.0)

prompt = '### 指示:\n音声を書き起こしてください。\n\n### 応答:\n'

print("\n" + "="*50)
hyps = []
refs = []

for i, sample in enumerate(dataset):
    if i >= 10:  # 10サンプルテスト
        break

    waveform, sr, ref = sample[0], sample[1], sample[2]

    encoder_inputs = encoder_processor(
        [waveform.squeeze(0).numpy()],  # リスト形式（validationと同じ）
        return_tensors='pt',
        return_attention_mask=True,
        sampling_rate=16000,
    ).to('cuda')
    decoder_inputs = decoder_processor(prompt, return_tensors='pt').to('cuda')

    with torch.inference_mode():
        generated_ids = model.generate(
            encoder_inputs.input_features,
            decoder_inputs.input_ids,
            encoder_inputs.attention_mask,
            decoder_inputs.attention_mask,
            max_length=1024,
            do_sample=False,
            num_beams=1,  # greedy (validation と同じ)
        )

    hyp = decoder_processor.decode(generated_ids[0], skip_special_tokens=True)
    hyps.append(hyp)
    refs.append(ref)

    print(f"[{i+1}] Reference:  {ref}")
    print(f"    Prediction: {hyp}")
    print("-"*50)

# CER計算 (日本語に適切)
cer_metric = evaluate.load("cer")
cer = cer_metric.compute(predictions=hyps, references=refs) * 100
print(f"\nCER: {cer:.2f}%")
