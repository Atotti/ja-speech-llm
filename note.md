## インタラクティブジョブ
```bash
qsub -I -P gch51701 -q rt_HG -l select=1 -l walltime=02:00:00
```

## バッチジョブ
```bash
qsub scripts/train_ja.sh
```


## 差分
```bash
diff /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2.py /groups/gch51701/Team031/GitHub/ayu-slp2025/demo2_ja.py
```
