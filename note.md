## インタラクティブジョブ
```bash
qsub -I -P gch51701 -q rt_HG -l select=1 -l walltime=02:00:00
```

# デバッグ

## アラインメント学習
```bash
# 最初から
uv run python -c "from demo2_ja import train; train(max_steps=100, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/debug-align')"

# 再開
uv run python -c "from demo2_ja import train; train(resume_from='models/LlamaForSpeechLM-ja-20260112-121119-step45000', max_steps=100, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/debug-align')"
```

## fine-tune (Adapter のみ)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-20260112-121119-step75000', max_steps=100, batch_size=8, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/debug-adapter')"
```

## fine-tune (LoRA)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-20260112-121119-step75000', use_lora=True, max_steps=100, batch_size=8, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/debug-lora')"
```

## fine-tune (Full decoder)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-20260112-121119-step75000', unfreeze_decoder=True, max_steps=100, batch_size=4, grad_accumulation=4, warmup_steps=10, val_check_interval=50, model_dir='models/debug-full')"
```

# ジョブ投入

## アライメント学習
```bash
# 新規
qsub scripts/train_ja.sh

# 再開
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-20260112-121119-step45000 scripts/train_ja.sh
```

## fine-tune (Adapter のみ)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-20260112-121119-step45000 scripts/finetune_adapter_ja.sh

# 再開
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 scripts/finetune_adapter_ja.sh
```

## fine-tune (LoRA)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-20260112-121119-step75000 scripts/finetune_lora_ja.sh

# 再開
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-LoRA-step1000 scripts/finetune_lora_ja.sh
```

## fine-tune (Full decoder)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-20260112-121119-step75000 scripts/finetune_full_ja.sh

# 再開
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-Full-step1000 scripts/finetune_full_ja.sh
```

# 学習モード比較

| モード | スクリプト | 学習対象 | パラメータ数 |
|--------|-----------|---------|-------------|
| Adapter | `finetune_adapter_ja.sh` | Adapter | ~10M |
| LoRA | `finetune_lora_ja.sh` | Adapter + LoRA | ~10M + ~8M |
| Full | `finetune_full_ja.sh` | Adapter + Decoder | ~10M + ~8B |

# その他

## 差分確認
```bash
diff /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2.py /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2_ja.py
```
