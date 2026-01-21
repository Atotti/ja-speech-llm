from demo2_ja import LlamaForSpeechLM

model_path = "models/v2/LlamaForSpeechLM-ja-Instruct-Full-20260121-182028-step500"
model = LlamaForSpeechLM.from_pretrained(model_path)

print("=== Model dtype before upload ===")
print(f"Adapter: {next(model.adapter.parameters()).dtype}")
print(f"Encoder: {next(model.encoder.parameters()).dtype}")
print(f"Decoder: {next(model.decoder.parameters()).dtype}")

# Uncomment to upload:
model.push_to_hub("Harui-i/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000", safe_serialization=True)
