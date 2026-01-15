
# デバッグ

## アラインメント学習 (Pretrain)
```bash
# 新規
uv run python -c "from demo2_ja import train; train(max_steps=100, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/v2/debug-align')"

# 再開
uv run python -c "from demo2_ja import train; train(resume_from='models/v2/LlamaForSpeechLM-ja-XXXXXXXX-stepXXXXX', max_steps=100, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/v2/debug-align')"
```

## fine-tune (Adapter のみ)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000', max_steps=100, batch_size=8, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/v2/debug-adapter')"
```

## fine-tune (LoRA)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000', use_lora=True, max_steps=100, batch_size=8, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/v2/debug-lora')"
```

## fine-tune (Full decoder)
```bash
uv run python -c "from demo2_ja import finetune; finetune(model_id='models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000', unfreeze_decoder=True, max_steps=100, batch_size=2, grad_accumulation=4, warmup_steps=10, val_check_interval=50, model_dir='models/v2/debug-full')"
```

---

# ジョブ投入

## アライメント学習 (Pretrain)
```bash
# 新規
qsub scripts/v2/train_ja.sh

# 再開
qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-XXXXXXXX-stepXXXXX scripts/v2/train_ja.sh
```

## fine-tune (Adapter のみ)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 scripts/v2/finetune_adapter_ja.sh

# 再開
qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-XXXXXXXX-stepXXXXX scripts/v2/finetune_adapter_ja.sh
```

## fine-tune (LoRA)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 scripts/v2/finetune_lora_ja.sh

# 再開
qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-LoRA-XXXXXXXX-stepXXXXX scripts/v2/finetune_lora_ja.sh
```

## fine-tune (Full decoder)
```bash
# 新規
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 scripts/v2/finetune_full_ja.sh

# 再開
qsub -v RESUME_FROM=models/v2/LlamaForSpeechLM-ja-Instruct-Full-XXXXXXXX-stepXXXXX scripts/v2/finetune_full_ja.sh
```

