
# デバッグ

## アラインメント学習 (Pretrain)
```bash
# 新規
uv run accelerate launch --num_processes 1 scripts/v2/train_accelerate.py \
    --max-steps 100 --batch-size 32 --grad-accumulation 2 --warmup-steps 10 \
    --val-check-interval 50 --model-dir models/v2/debug-align

# 再開
uv run accelerate launch --num_processes 1 scripts/v2/train_accelerate.py \
    --resume-from models/v2/LlamaForSpeechLM-ja-XXXXXXXX-stepXXXXX \
    --max-steps 100 --batch-size 32 --grad-accumulation 2 --warmup-steps 10 \
    --val-check-interval 50 --model-dir models/v2/debug-align
```

## fine-tune (Adapter のみ)
```bash
uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py \
    --model-id models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 \
    --max-steps 100 --batch-size 8 --grad-accumulation 2 --warmup-steps 10 \
    --val-check-interval 50 --model-dir models/v2/debug-adapter
```

## fine-tune (LoRA)
```bash
uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py \
    --use-lora --model-id models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 \
    --max-steps 100 --batch-size 8 --grad-accumulation 2 --warmup-steps 10 \
    --val-check-interval 50 --model-dir models/v2/debug-lora
```

## fine-tune (Full decoder)
```bash
uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py \
    --unfreeze-decoder --model-id models/LlamaForSpeechLM-ja-Instruct-20260112-223832-step11000 \
    --max-steps 100 --batch-size 2 --grad-accumulation 4 --warmup-steps 10 \
    --val-check-interval 50 --model-dir models/v2/debug-full
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
