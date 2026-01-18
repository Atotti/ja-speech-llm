# Repository Guidelines

## Project Structure & Module Organization
- `src/speech_llm_ja/`: Core package (model, training, finetuning, validation, datasets).
- `scripts/v2/`: Training/finetuning entrypoints and Accelerate launchers.
- `tools/`: Utility scripts (evaluation, inference checks, hub upload).
- `eval_results/`: Generated evaluation outputs and artifacts.
- Top-level scripts: `demo2_ja.py` exposes `train`/`finetune` helpers.

## Build, Test, and Development Commands
This repo is Python-only; there is no build step.
- `uv run accelerate launch --num_processes 1 scripts/v2/train_accelerate.py`: Single-GPU training.
- `uv run accelerate launch --num_processes 8 scripts/v2/train_accelerate.py`: Multi-GPU training.
- `uv run accelerate launch --num_processes 1 scripts/v2/finetune_accelerate.py --use-lora`: LoRA finetuning.
- `bash scripts/v2/train_ja.sh` or `bash scripts/v2/train_ja_8gpu.sh`: PBS job wrappers for single/8 GPU.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check --fix src`: Auto-fix lint issues after Python changes.

Use `uv run ...` for consistent dependency resolution (Python `>=3.12`).

## Coding Style & Naming Conventions
- Python code lives under `src/` with module names in `snake_case`.
- Keep function names descriptive (`train`, `finetune`, `validate`).
- No formatter/linter is configured; follow existing style in nearby files.
- Shell scripts in `scripts/v2/` are named by task and scale (e.g., `*_8gpu.sh`).

## Testing Guidelines
There is no formal test suite or `pytest` configuration.
- Use `tools/infer_test.py` for quick inference checks.
- Use `src/speech_llm_ja/validate.py` or `tools/eval_jsut.py` for evaluation runs.
- When adding new functionality, include a lightweight script in `tools/` to verify behavior.

## Commit & Pull Request Guidelines
Commit history uses short, imperative messages and mixes English/Japanese (e.g., `fix ...`, `add ...`, `...を消去`); no strict convention is enforced.

For PRs:
- Describe the change, datasets used, and expected impact on metrics.
- Link related issues or experiments; include logs or `eval_results/` artifacts when relevant.
- Note hardware assumptions (GPU count, memory) if training behavior changes.

## Security & Configuration Tips
- Avoid committing dataset files or large checkpoints; keep them out of git.
- Prefer environment variables for tokens (e.g., `WANDB_API_KEY`) and document them in the PR.
