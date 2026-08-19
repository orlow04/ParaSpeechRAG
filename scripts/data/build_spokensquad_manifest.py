#!/usr/bin/env python3
"""
Build a CSV manifest mapping SpokenSQuAD audio files to their transcription text.

Reads the SpokenSQuAD JSON annotation and the local audio directory, then
produces a CSV with columns: id, filename, text — one row per audio file found.
The manifest is used as ground-truth reference for ASR WER computation.

Usage:
    python build_spokensquad_manifest.py \
        --json data/spoken_test-v1.1.json \
        --audio-dir spokensquad_cache \
        --output manifests/spokensquad_dev_manifest.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path


def split_context_like_spokensquad(context: str):
    # Mantem indices compatíveis com os WAVs do Spoken-SQuAD.
    # Casos como ". . ." geram indices pulados nos arquivos de audio.
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", context.strip())
        if s.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--audio-dir", default="spokensquad_cache")
    parser.add_argument("--output", default="manifests/spokensquad_dev_manifest.csv")
    args = parser.parse_args()

    json_path = Path(args.json)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    missing = 0

    for article_idx, article in enumerate(data["data"]):
        for paragraph_idx, paragraph in enumerate(article["paragraphs"]):
            sentences = split_context_like_spokensquad(paragraph["context"])

            for sentence_idx, sentence in enumerate(sentences):
                file_id = f"{article_idx}_{paragraph_idx}_{sentence_idx}"
                filename = f"{file_id}.wav"
                audio_path = audio_dir / filename

                if not audio_path.exists():
                    missing += 1
                    continue

                rows.append({
                    "id": file_id,
                    "filename": filename,
                    "text": sentence,
                })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Missing audio files: {missing}")


if __name__ == "__main__":
    main()
