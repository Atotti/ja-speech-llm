## インタラクティブジョブ
```bash
qsub -I -P gch51701 -q rt_HG -l select=1 -l walltime=02:00:00
```

# デバッグ

## アラインメント学習
最初から
```bash
uv run python -c "from demo2_ja import train; train(max_steps=1000, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=20, model_dir='models/LlamaForSpeechLM-ja-debug')"
```

再開
```bash
uv run python -c "from demo2_ja import train; train(resume_from='models/LlamaForSpeechLM-ja-20260111-120922-step20000', max_steps=1000, batch_size=32, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir='models/LlamaForSpeechLM-ja-debug')"
```

## fine-tune
アラインメントから
```bash
uv run python -c 'from demo2_ja import finetune; finetune(model_id="models/LlamaForSpeechLM-ja-20260111-120922-step20000", max_steps=100, batch_size=8, grad_accumulation=2, warmup_steps=10, val_check_interval=50, model_dir="models/LlamaForSpeechLM-ja-Instruct-debug")'
```


# 学習
## アライメント学習
新規学習
```bash
qsub scripts/train_ja.sh
```

再開
```bash
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-20260111-120922-step20000 scripts/train_ja.sh
```

## fine-tune
新規fine-tune
```bash
qsub -v MODEL_ID=models/LlamaForSpeechLM-ja-20260111-120922-step20000 scripts/finetune_ja.sh
```
再開
```bash
qsub -v RESUME_FROM=models/LlamaForSpeechLM-ja-Instruct-20260112-135311-step1000 scripts/finetune_ja.sh
```


## 差分
```bash
diff /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2.py /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2_ja.py
```
