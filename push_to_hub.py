from demo2_ja import LlamaForSpeechLM

model_path = "models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000"
model = LlamaForSpeechLM.from_pretrained(model_path)

print("=== Model dtype before upload ===")
print(f"Adapter: {next(model.adapter.parameters()).dtype}")
print(f"Encoder: {next(model.encoder.parameters()).dtype}")
print(f"Decoder: {next(model.decoder.parameters()).dtype}")

# Uncomment to upload:
model.push_to_hub("Atotti/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000", safe_serialization=True)
