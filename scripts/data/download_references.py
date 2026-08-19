#!/usr/bin/env python3
"""
Download the 4 LibriSpeech reference audio files used in the SpokenSQuAD evaluation.

After running this, copy azuma_0.wav and trump_0.wav manually to the references/ folder.

Usage:
    python download_references.py
    python download_references.py --output_dir references/
"""

import sys
import argparse
from pathlib import Path

TARGET_FILES = {
    "2803-154320-0012",
    "3081-166546-0023",
    "6319-275224-0006",
    "8842-302201-0002",
}

# Search order: most likely splits first to avoid downloading huge train sets
# Split names for openslr/librispeech_asr:
#   clean → ['test', 'validation', 'train.100', 'train.360']
#   other → ['test', 'validation', 'train.500']
SEARCH_ORDER = [
    ("openslr/librispeech_asr", "clean", "test"),
    ("openslr/librispeech_asr", "clean", "validation"),
    ("openslr/librispeech_asr", "clean", "train.100"),
    ("openslr/librispeech_asr", "clean", "train.360"),
    ("openslr/librispeech_asr", "other", "test"),
    ("openslr/librispeech_asr", "other", "validation"),
    ("openslr/librispeech_asr", "other", "train.500"),
]


def get_file_stem(item: dict) -> str:
    """Extract the file stem (e.g. '2803-154320-0012') from a dataset item."""
    for key in ("file", "path", "id", "audio"):
        val = item.get(key)
        if isinstance(val, str) and val:
            return Path(val).stem
        if isinstance(val, dict):
            inner = val.get("path", "") or val.get("file", "")
            if inner:
                return Path(inner).stem
    return ""


def save_audio(item: dict, out_path: Path) -> bool:
    try:
        import io
        import soundfile as sf
        import numpy as np

        audio = item["audio"]

        if "array" in audio and audio["array"] is not None:
            # Fully decoded (older datasets versions)
            arr = np.array(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        elif "bytes" in audio and audio["bytes"]:
            # Raw bytes (decode=False mode) — soundfile handles FLAC/WAV/etc.
            arr, sr = sf.read(io.BytesIO(audio["bytes"]))
            arr = arr.astype(np.float32)
        else:
            print("  No audio data in item")
            return False

        sf.write(str(out_path), arr, sr, format="FLAC")
        return True
    except Exception as e:
        print(f"  ERROR saving {out_path.name}: {e}")
        return False


def download_librispeech_refs(output_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install the 'datasets' library first:")
        print("  pip install datasets soundfile")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which files are already present
    found = {
        p.stem
        for p in output_dir.iterdir()
        if p.suffix.lower() in {".flac", ".wav"} and p.stem in TARGET_FILES
    }
    if found:
        print(f"Already downloaded: {sorted(found)}")

    remaining = TARGET_FILES - found

    for repo, config_name, split in SEARCH_ORDER:
        if not remaining:
            break

        print(f"\nSearching {split} ({config_name}) for: {sorted(remaining)} …")

        try:
            from datasets import Audio as HFAudio
            ds = load_dataset(
                repo,
                config_name,
                split=split,
                streaming=True,
            )
            # Disable automatic audio decoding to avoid needing torchcodec
            ds = ds.cast_column("audio", HFAudio(decode=False))
        except Exception as e:
            print(f"  Could not load split {split}: {e}")
            continue

        for item in ds:
            if not remaining:
                break

            stem = get_file_stem(item)
            if stem not in remaining:
                continue

            out_path = output_dir / f"{stem}.flac"
            print(f"  Found {stem} — saving …", end=" ", flush=True)
            if save_audio(item, out_path):
                print("OK")
                remaining.discard(stem)
            else:
                print("FAILED")

    print("\n" + "=" * 50)
    if not remaining:
        print(f"All 4 LibriSpeech references saved to: {output_dir}/")
    else:
        print(f"WARNING: Could not find the following files:")
        for f in sorted(remaining):
            print(f"  {f}.flac")

    print("\nNext steps:")
    print(f"  1. Copy azuma_0.wav  → {output_dir}/azuma_0.wav")
    print(f"  2. Copy trump_0.wav  → {output_dir}/trump_0.wav")
    print(f"  3. Run the batch conversion:")
    print(f"     python perturbation/batch_convert_spokensquad.py --hf_dataset --ref_dir {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech reference files")
    parser.add_argument(
        "--output_dir",
        default="references",
        help="Folder to save reference files (default: references/)",
    )
    args = parser.parse_args()
    download_librispeech_refs(Path(args.output_dir))


if __name__ == "__main__":
    main()
