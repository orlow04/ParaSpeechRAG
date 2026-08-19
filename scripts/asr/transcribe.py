"""
Batch ASR transcription of audio folders using NVIDIA Parakeet (NeMo).

Reads a YAML config file that specifies the model, batch size, and a list of
input folders with their output CSV filenames. Each CSV has columns:
arquivo, transcricao.

Designed to transcribe all 25 perturbation conditions (baseline + 8 speed +
5 MP3 + 5 Opus + 6 GenVC) in a single run.

Usage:
    python transcribe_from_config.py --config transcribe_config.yml
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from nemo.collections.asr.models import ASRModel

AUDIO_PATTERNS = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus")


def read_config(config_file):
    with Path(config_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    return config


def load_model(config):
    model_config = config.get("model") or {}
    if not isinstance(model_config, dict):
        raise ValueError("model must be a YAML mapping")
    
    model_id = model_config.get("repo_id")
    if model_id:
        model = ASRModel.from_pretrained(model_name=model_id)
        print(f"Loading model from file: {model_id}")
    else:
        raise ValueError("model.repo_id is required to load the model")

    device = model_config.get("device")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    return model


def audio_files(folder_config, transcription_config):
    input_dir = folder_config.get("input_dir")
    if not input_dir:
        raise ValueError("each folder needs input_dir")

    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    pattern = folder_config.get("pattern", transcription_config.get("pattern"))
    recursive = bool(folder_config.get("recursive", transcription_config.get("recursive", False)))
    patterns = [pattern] if pattern else list(AUDIO_PATTERNS)
    print(f"Looking for audio files in {folder} with patterns {patterns} (recursive={recursive})")

    found = {
        Path(path)
        for item in patterns
        for path in glob.glob(str(folder / (f"**/{item}" if recursive and "**" not in item else item)),
                              recursive=recursive)
        if Path(path).is_file()
    }
    return sorted(found)


def output_file(folder_config, transcription_config):

    output_dir = Path(transcription_config.get("output_dir", "."))
    
    output_csv = folder_config.get("output_csv")
    if output_csv:
        return output_dir / output_csv

    name = folder_config.get("name") or Path(folder_config["input_dir"]).name
    
    return output_dir / f"{name}_transcriptions.csv"


def text_of(result):
    if isinstance(result, str):
        return result
    return str(getattr(result, "text", result))


def transcribe_one_folder(model, files, output_csv, batch_size, chunk_size):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arquivo", "transcricao"])
        step = chunk_size or len(files) or 1
        for start in range(0, len(files), step):
            chunk = files[start:start + step]
            results = model.transcribe(audio=[str(item) for item in chunk], batch_size=batch_size)
            writer.writerows((audio_file.name, text_of(result)) for audio_file, result in zip(chunk, results))


def main():
    parser = argparse.ArgumentParser(description="Transcribe folders listed in a YAML file")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = read_config(args.config)
    transcription_config = config.get("transcription") or {}
    if not isinstance(transcription_config, dict):
        raise ValueError("transcription must be a YAML mapping")

    folders = config.get("folders")
    if not isinstance(folders, list) or not folders:
        raise ValueError("config must define a non-empty folders list")

    batch_size = int(transcription_config.get("batch_size", 32))
    chunk_size = transcription_config.get("chunk_size")
    chunk_size = int(chunk_size) if chunk_size else None

    model = load_model(config)
    for folder_config in folders:
        if not isinstance(folder_config, dict):
            raise ValueError("each folder entry must be a YAML mapping")
        files = audio_files(folder_config, transcription_config)
        csv_path = output_file(folder_config, transcription_config)
        name = folder_config.get("name") or csv_path.stem
        print(f"Processing {name}: {len(files)} file(s)")
        transcribe_one_folder(model, files, csv_path, batch_size, chunk_size)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
