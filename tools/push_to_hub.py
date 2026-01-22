import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2_ja import (  # noqa: E402
    LlamaForSpeechLM,
    SpeechLlamaProcessor,
    SpeechLlamaProcessorConfig,
)


def _warn_if_local(model_id: str, label: str) -> None:
    if model_id and Path(model_id).exists():
        print(f"WARNING: {label}_id is local path: {model_id}")


def _strip_local_id(model_id: str, label: str) -> str | None:
    if model_id and Path(model_id).exists():
        print(f"INFO: stripping local {label}_id: {model_id}")
        return None
    return model_id


def _write_minimal_package(save_path: Path) -> None:
    package_dir = save_path / "speech_llm_ja"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        '"""Minimal package for remote loading."""\n',
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    model_src = repo_root / "src" / "speech_llm_ja" / "model.py"
    processor_src = repo_root / "src" / "speech_llm_ja" / "processor.py"
    shutil.copy2(model_src, package_dir)
    shutil.copy2(processor_src, package_dir)
    shutil.copy2(model_src, save_path / "speech_llm_ja_model.py")

    processor_text = processor_src.read_text(encoding="utf-8")
    replaced = processor_text.replace(
        "from .model import AUDIO_TOKEN_ID",
        "AUDIO_TOKEN_ID = 351  # <|reserved_343|>",
    )
    if replaced == processor_text:
        raise ValueError("Failed to rewrite processor import for hub module.")
    (save_path / "speech_llm_ja_processor.py").write_text(replaced, encoding="utf-8")


def _write_auto_map(save_path: Path) -> None:
    config_path = save_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = {
        "AutoConfig": "speech_llm_ja_model.LlamaForSpeechLMConfig",
        "AutoModel": "speech_llm_ja_model.LlamaForSpeechLM",
        "AutoModelForCausalLM": "speech_llm_ja_model.LlamaForSpeechLM",
    }
    config["architectures"] = ["LlamaForSpeechLM"]
    config_path.write_text(
        json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8"
    )

def _sanitize_config_paths(save_path: Path) -> None:
    config_path = save_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    config["decoder_id"] = None
    encoder_id = config.get("encoder_id")
    if encoder_id and Path(encoder_id).exists():
        config["encoder_id"] = None

    text_config = config.get("text_config") or {}
    text_name = text_config.get("_name_or_path")
    if text_name and Path(text_name).exists():
        text_config["_name_or_path"] = None
        config["text_config"] = text_config

    audio_config = config.get("audio_config") or {}
    audio_name = audio_config.get("_name_or_path")
    if audio_name and Path(audio_name).exists():
        audio_config["_name_or_path"] = None
        config["audio_config"] = audio_config

    config_path.write_text(
        json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hub-id", required=True)
    parser.add_argument("--encoder-id", default=None)
    parser.add_argument("--decoder-id", default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--commit-message", default="push model")
    args = parser.parse_args()

    model = LlamaForSpeechLM.from_pretrained(args.model_path)

    if args.encoder_id:
        model.config.encoder_id = args.encoder_id
        model.config.audio_config = AutoConfig.from_pretrained(args.encoder_id)
    if args.decoder_id:
        model.config.decoder_id = args.decoder_id
        model.config.text_config = AutoConfig.from_pretrained(args.decoder_id)

    model.config.decoder_id = _strip_local_id(model.config.decoder_id, "decoder")
    model.config.audio_config = model.config.audio_config or model.encoder.config
    model.config.text_config = model.config.text_config or model.decoder.config
    model.config._sync_text_config(model.config.text_config)

    _warn_if_local(model.config.encoder_id, "encoder")
    _warn_if_local(model.config.decoder_id, "decoder")

    tokenizer_id = args.tokenizer_id or model.config.decoder_id
    if not tokenizer_id:
        raise ValueError(
            "tokenizer_id is required when decoder_id is not set. "
            "Pass --tokenizer-id with a tokenizer repo or local path."
        )
    processor = SpeechLlamaProcessor.from_pretrained(
        encoder_id=model.config.encoder_id,
        decoder_id=tokenizer_id,
        config=SpeechLlamaProcessorConfig(
            adapter_kernel_size=model.config.adapter_kernel_size
        ),
    )

    print("=== Model dtype before upload ===")
    print(f"Adapter: {next(model.adapter.parameters()).dtype}")
    print(f"Encoder: {next(model.encoder.parameters()).dtype}")
    print(f"Decoder: {next(model.decoder.parameters()).dtype}")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        model.save_pretrained(save_path, safe_serialization=True)
        processor.save_pretrained(save_path)
        _sanitize_config_paths(save_path)
        _write_auto_map(save_path)
        _write_minimal_package(save_path)

        create_repo(args.hub_id, exist_ok=True)
        HfApi().upload_folder(
            repo_id=args.hub_id,
            folder_path=str(save_path),
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()
