import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.speech_llm_ja import LlamaForSpeechLM

model_path = "models/v2/LlamaForSpeechLM-ja-Instruct-Full-20260119-184726-step15000"
model = LlamaForSpeechLM.from_pretrained(model_path)

print("=== Model dtype before upload ===")
print(f"Adapter: {next(model.adapter.parameters()).dtype}")
print(f"Encoder: {next(model.encoder.parameters()).dtype}")
print(f"Decoder: {next(model.decoder.parameters()).dtype}")

# Uncomment to upload:
model.push_to_hub("Atotti/LlamaForSpeechLM-ja-Instruct-Full-20260119-184726-step15000", safe_serialization=True)
