# full-asr

Adapter-only:       `models/LlamaForSpeechLM-ja-20260114-215708-step110000`
-> Decoder-train:   `models/LlamaForSpeechLM-ja-Full-20260115-224911-step225000`
-> Encoder-train:   `wip`

## Evaluation

```bash
# JSUT
uv run python tools/eval_jsut.py -m models/LlamaForSpeechLM-ja-Full-20260115-224911-step225000 -d japanese-asr/ja_asr.jsut_basic5000

# CommonVoice
uv run python tools/eval_jsut.py -m models/LlamaForSpeechLM-ja-Full-20260115-224911-step225000 -d japanese-asr/ja_asr.common_voice_8_0

# ReazonSpeech test
uv run python tools/eval_jsut.py -m models/LlamaForSpeechLM-ja-Full-20260115-224911-step225000 -d japanese-asr/ja_asr.reazonspeech_test
```
