#!/usr/bin/env python3
"""
Extract 444-audio subset from all perturbation folders.

Reads passages.jsonl, collects the 444 filenames, then copies
the matching WAV from every perturbation subfolder into:

    output_subset/
        spokensquad_codec_mp3/8kbps/   (411 WAVs)
        spokensquad_codec_mp3/16kbps/  ...
        spokensquad_codec_opus/...
        spokensquad_genvc/...
        spokensquad_speed/...

Usage:
    python extract_subset.py
    python extract_subset.py --src_dir output --passages passages.jsonl --out_dir output_subset
"""

import argparse
import json
import shutil
from pathlib import Path


def load_filenames(passages_path: Path) -> list[str]:
    filenames = []
    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            filenames.append(Path(entry["audio_path"]).name)
    return filenames


def main():
    parser = argparse.ArgumentParser(description="Extract 411-audio subset from perturbation folders")
    parser.add_argument("--src_dir",   default="output",          help="Root output dir with perturbation folders")
    parser.add_argument("--passages",  default="passages.jsonl",  help="Path to passages.jsonl")
    parser.add_argument("--out_dir",   default="output_subset",   help="Destination root directory")
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    out_root = Path(args.out_dir)
    passages_path = Path(args.passages)

    filenames = load_filenames(passages_path)
    filename_set = set(filenames)
    print(f"Passages loaded : {len(filenames)} entries ({len(filename_set)} unique filenames)")

    # Collect all leaf perturbation subdirectories (2 levels deep: dataset/condition)
    leaf_dirs = sorted(
        d for d in src_root.glob("*/*") if d.is_dir()
    )
    print(f"Perturbation folders found: {len(leaf_dirs)}")

    total_copied = 0
    total_missing = 0

    for src_dir in leaf_dirs:
        rel = src_dir.relative_to(src_root)   # e.g. spokensquad_codec_mp3/8kbps
        dst_dir = out_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = missing = 0
        for fname in filenames:
            src_file = src_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_dir / fname)
                copied += 1
            else:
                missing += 1

        status = f"  {rel}: {copied} copied"
        if missing:
            status += f", {missing} MISSING"
        print(status)
        total_copied += copied
        total_missing += missing

    print(f"\nDone. Total copied: {total_copied} | Missing: {total_missing}")
    if total_missing:
        print("WARNING: some files were not found — check the log above.")


if __name__ == "__main__":
    main()
