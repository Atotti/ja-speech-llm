import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2_ja import (
    LlamaForSpeechLM,
)


def _warn_if_local(model_id: str, label: str) -> None:
    if Path(model_id).exists():
        print(f"WARNING: {label}_id is local path: {model_id}")


model_path = "models/v2/LlamaForSpeechLM-ja-Instruct-Full-20260121-182028-step500"
hub_id = "Harui-i/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000"

# メモ(Note): LoRAをした場合、チェックポイントは adapter.pt, lora/が別々に分かれるけどこれはそれを想定していない
model = LlamaForSpeechLM.from_pretrained(model_path)

_warn_if_local(model.config.encoder_id, "encoder")
_warn_if_local(model.config.decoder_id, "decoder")

# Processorはアップしなくてええか笑
# processor = SpeechLlamaProcessor.from_pretrained(
#    encoder_id=model.config.encoder_id,
#    decoder_id=model.config.decoder_id,
#    config=SpeechLlamaProcessorConfig(
#        adapter_kernel_size=model.config.adapter_kernel_size
#    ),
# )
# processor.push_to_hub(hub_id)

print("=== Model dtype before upload ===")
print(f"Adapter: {next(model.adapter.parameters()).dtype}")
print(f"Encoder: {next(model.encoder.parameters()).dtype}")
print(f"Decoder: {next(model.decoder.parameters()).dtype}")

model.push_to_hub(hub_id, safe_serialization=True)
