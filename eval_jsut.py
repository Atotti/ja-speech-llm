"""Compute CER/WER for Japanese ASR model (LlamaForSpeechLM)."""
import json
import os
import argparse

import torch
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

from demo2_ja import LlamaForSpeechLM

pretty_dataset_names = {
    "japanese-asr/ja_asr.jsut_basic5000": "JSUT Basic 5000",
    "japanese-asr/ja_asr.common_voice_8_0": "CommonVoice 8 (Japanese test set)",
    "japanese-asr/ja_asr.reazonspeech_test": "ReazonSpeech (held out test set)",
}


def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for evaluation."""
    normalizer = BasicTextNormalizer()
    return normalizer(text).replace(" ", "").replace("。.", "。")


def run_inference(
    model: LlamaForSpeechLM,
    encoder_processor: AutoProcessor,
    decoder_processor: AutoTokenizer,
    audio_samples: list,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[str]:
    """Run batch inference on audio samples."""
    prompt = "### 指示:\n音声を書き起こしてください。\n\n### 応答:\n"
    predictions = []

    for i in tqdm(range(0, len(audio_samples), batch_size), desc="Inference"):
        batch = audio_samples[i : i + batch_size]

        # Prepare encoder inputs (pad to 30 seconds = 3000 mel frames)
        encoder_inputs = encoder_processor(
            [sample["array"] for sample in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            padding="max_length",
            max_length=480000,  # 30 seconds * 16000 Hz
            truncation=True,
        ).to(model.device)

        # Prepare decoder inputs
        decoder_inputs = decoder_processor(
            [prompt] * len(batch),
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                encoder_inputs.input_features,
                decoder_inputs.input_ids,
                encoder_inputs.attention_mask,
                decoder_inputs.attention_mask,
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=1,
            )

        batch_preds = decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Compute CER/WER for Japanese ASR model.")
    parser.add_argument(
        "-m",
        "--model",
        default="models/LlamaForSpeechLM-ja",
        type=str,
        help="Path to LlamaForSpeechLM model directory",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="japanese-asr/ja_asr.jsut_basic5000",
        type=str,
        help="Dataset name on HuggingFace",
    )
    parser.add_argument("--dataset-split", default="test", type=str)
    parser.add_argument("--dataset-config", default=None, type=str)
    parser.add_argument("--column-audio", default="audio", type=str)
    parser.add_argument("--column-text", default="transcription", type=str)
    parser.add_argument("-b", "--batch", default=8, type=int, help="Batch size for inference")
    parser.add_argument("-o", "--output-dir", default="eval_results", type=str)
    parser.add_argument("--max-length", default=256, type=int, help="Max generation length")
    parser.add_argument("--pretty-table", action="store_true", help="Display results as markdown table")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_metric_file = f"{args.output_dir}/metric.jsonl"

    # Display mode: show existing results as pretty table
    if args.pretty_table:
        if not os.path.exists(output_metric_file):
            print(f"No metric file found at {output_metric_file}")
            return

        with open(output_metric_file) as f:
            metrics = [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
        df_metric = pd.DataFrame(metrics)
        df_metric = df_metric.round(1)

        print("\n=== Evaluation Results ===\n")
        print("CER (Character Error Rate):")
        print(df_metric[["dataset", "cer_norm", "cer_raw"]].to_markdown(index=False))
        print("\nWER (Word Error Rate):")
        print(df_metric[["dataset", "wer_norm", "wer_raw"]].to_markdown(index=False))
        return

    # Check for cached predictions
    model_name = os.path.basename(args.model)
    dataset_name = os.path.basename(args.dataset)
    prediction_path = (
        f"{args.output_dir}/predictions.model-{model_name}."
        f"dataset-{dataset_name}.config-{args.dataset_config}."
        f"split-{args.dataset_split}.csv"
    )

    if os.path.exists(prediction_path):
        print(f"Loading cached predictions from {prediction_path}")
        df = pd.read_csv(prediction_path)
        prediction_norm = df["prediction_norm"].values.tolist()
        reference_norm = df["reference_norm"].values.tolist()
        prediction_raw = df["prediction_raw"].values.tolist()
        reference_raw = df["reference_raw"].values.tolist()
        audio_id = df["id"].values.tolist()
    else:
        # Load model
        print(f"Loading model from {args.model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LlamaForSpeechLM.from_pretrained(args.model).to(device).eval()

        encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
        decoder_processor = AutoTokenizer.from_pretrained(model.config.decoder_id)
        decoder_processor.pad_token = decoder_processor.eos_token

        # Load dataset
        print(f"Loading dataset: {args.dataset}...")
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset, args.dataset_config, split=args.dataset_split, trust_remote_code=True
            )
        else:
            dataset = load_dataset(args.dataset, split=args.dataset_split, trust_remote_code=True)

        # Run inference
        print(f"Running inference on {len(dataset)} samples...")
        audio_samples = dataset[args.column_audio]
        prediction_raw = run_inference(
            model,
            encoder_processor,
            decoder_processor,
            audio_samples,
            batch_size=args.batch,
            max_length=args.max_length,
        )
        reference_raw = dataset[args.column_text]
        audio_id = [sample["path"] for sample in audio_samples]

        # Normalize for Japanese
        prediction_norm = [normalize_japanese(p) for p in prediction_raw]
        reference_norm = [normalize_japanese(r) for r in reference_raw]

        # Remove empty references
        valid_indices = [i for i, r in enumerate(reference_norm) if len(r) > 0]
        reference_norm = [reference_norm[i] for i in valid_indices]
        reference_raw = [reference_raw[i] for i in valid_indices]
        prediction_norm = [prediction_norm[i] for i in valid_indices]
        prediction_raw = [prediction_raw[i] for i in valid_indices]
        audio_id = [audio_id[i] for i in valid_indices]

        # Save predictions
        df = pd.DataFrame(
            {
                "id": audio_id,
                "reference_norm": reference_norm,
                "prediction_norm": prediction_norm,
                "reference_raw": reference_raw,
                "prediction_raw": prediction_raw,
            }
        )
        df.to_csv(prediction_path, index=False)
        print(f"Predictions saved to {prediction_path}")

    # Compute metrics
    print("\nComputing metrics...")
    cer_metric = load("cer")
    wer_metric = load("wer")

    cer_norm = 100 * cer_metric.compute(predictions=prediction_norm, references=reference_norm)
    cer_raw = 100 * cer_metric.compute(predictions=prediction_raw, references=reference_raw)
    wer_norm = 100 * wer_metric.compute(predictions=prediction_norm, references=reference_norm)
    wer_raw = 100 * wer_metric.compute(predictions=prediction_raw, references=reference_raw)

    metric = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "cer_norm": cer_norm,
        "cer_raw": cer_raw,
        "wer_norm": wer_norm,
        "wer_raw": wer_raw,
    }

    # Save metrics
    metrics = []
    if os.path.exists(output_metric_file):
        with open(output_metric_file) as f:
            metrics = [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
    metrics.append(metric)

    with open(output_metric_file, "w") as f:
        f.write("\n".join([json.dumps(m) for m in metrics]))

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print("-" * 50)
    print(f"CER (normalized): {cer_norm:.2f}%")
    print(f"CER (raw):        {cer_raw:.2f}%")
    print(f"WER (normalized): {wer_norm:.2f}%")
    print(f"WER (raw):        {wer_raw:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
