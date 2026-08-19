#!/usr/bin/env python3
"""Compute Word Error Rate (WER) for all transcription conditions.

Reference: SpokenSQuAD dev manifest (ground truth text)
Hypothesis: ASR transcriptions from Parakeet (transcriptions/*.csv)

Output: results/wer_results.csv  +  results/wer_results.json
"""

import csv
import json
import re
from pathlib import Path

import jiwer

MANIFEST = Path("manifests/spokensquad_dev_manifest.csv")
TRANSCRIPTIONS_DIR = Path("transcriptions")
RESULTS_DIR = Path("results")

TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveEmptyStrings(),
    jiwer.WordTokenize(),
])


def load_manifest(path: Path) -> dict[str, str]:
    refs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[row["filename"]] = row["text"]
    return refs


def load_transcriptions(path: Path) -> dict[str, str]:
    hyps = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hyps[row["arquivo"]] = row["transcricao"]
    return hyps


def compute_wer(refs: dict, hyps: dict) -> dict:
    common = sorted(set(refs) & set(hyps))
    if not common:
        return {"wer": None, "n": 0}

    ref_list = [refs[f] for f in common]
    hyp_list = [hyps[f] for f in common]

    measures = jiwer.compute_measures(
        ref_list, hyp_list,
        truth_transform=TRANSFORM,
        hypothesis_transform=TRANSFORM,
    )
    return {
        "wer":         round(measures["wer"] * 100, 4),
        "mer":         round(measures["mer"] * 100, 4),
        "wil":         round(measures["wil"] * 100, 4),
        "hits":        measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions":   measures["deletions"],
        "insertions":  measures["insertions"],
        "n":           len(common),
    }


def main():
    refs = load_manifest(MANIFEST)
    print(f"Reference: {len(refs)} utterances from {MANIFEST.name}")

    csv_files = sorted(TRANSCRIPTIONS_DIR.glob("*.csv"))
    print(f"Conditions: {len(csv_files)}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS_DIR / "wer_results.csv"
    output_json = RESULTS_DIR / "wer_results.json"

    all_results = []

    fieldnames = ["condition", "wer", "mer", "wil", "hits",
                  "substitutions", "deletions", "insertions", "n"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for csv_path in csv_files:
            condition = csv_path.stem
            hyps = load_transcriptions(csv_path)
            metrics = compute_wer(refs, hyps)
            row = {"condition": condition, **metrics}
            writer.writerow(row)
            all_results.append(row)
            print(f"{condition:<30} WER: {metrics['wer']:>7.2f}%  (n={metrics['n']})")

    output_json.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSalvo em:\n  {output_csv}\n  {output_json}")


if __name__ == "__main__":
    main()
