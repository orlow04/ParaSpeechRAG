#!/usr/bin/env python3
"""
Create a balanced RAVDESS reference selection for expressive VC experiments.

The script parses RAVDESS filenames, computes lightweight acoustic features,
and writes:
  - a full metadata/features CSV
  - a selected subset CSV
  - optional symlinks/copies arranged by emotion/intensity
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITIES = {
    "01": "normal",
    "02": "strong",
}

STATEMENTS = {
    "01": "kids_are_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}


@dataclass
class RavdessItem:
    path: str
    filename: str
    actor: int
    gender: str
    emotion: str
    intensity: str
    statement: str
    repetition: int
    duration_s: float
    rms_db: float
    peak: float
    clipping_ratio: float
    voiced_fraction: float
    f0_mean_hz: float
    f0_std_hz: float
    score: float


def parse_filename(path: Path) -> dict | None:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01":
        return None
    actor_id = int(actor)
    return {
        "actor": actor_id,
        "gender": "female" if actor_id % 2 == 0 else "male",
        "emotion": EMOTIONS[emotion],
        "intensity": INTENSITIES[intensity],
        "statement": STATEMENTS[statement],
        "repetition": int(repetition),
    }


def audio_features(path: Path, target_sr: int = 16000) -> dict:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration_s = float(len(y) / sr)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    rms_db = float(20 * math.log10(rms))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.98)) if len(y) else 0.0

    frame_length = 1024
    hop_length = 256
    frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    active = frame_rms > max(np.percentile(frame_rms, 20), 1e-4)
    voiced_fraction = float(np.mean(active)) if len(active) else 0.0

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=hop_length)
        f0 = f0[np.isfinite(f0)]
        f0_mean_hz = float(np.mean(f0)) if len(f0) else 0.0
        f0_std_hz = float(np.std(f0)) if len(f0) else 0.0
    except Exception:
        f0_mean_hz = 0.0
        f0_std_hz = 0.0

    return {
        "duration_s": duration_s,
        "rms_db": rms_db,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "voiced_fraction": voiced_fraction,
        "f0_mean_hz": f0_mean_hz,
        "f0_std_hz": f0_std_hz,
    }


def quality_score(features: dict) -> float:
    duration = features["duration_s"]
    duration_score = max(0.0, 1.0 - abs(duration - 3.5) / 3.5)
    clipping_score = max(0.0, 1.0 - features["clipping_ratio"] * 200.0)
    peak_score = 1.0 if 0.05 <= features["peak"] <= 0.98 else 0.5
    voiced_score = min(1.0, features["voiced_fraction"] / 0.75)
    return 0.35 * duration_score + 0.30 * clipping_score + 0.20 * voiced_score + 0.15 * peak_score


def emotion_score(emotion: str, intensity: str, features: dict) -> float:
    # Acoustic proxy only. Strong/high-arousal conditions should be energetic
    # and pitch-variable; low-arousal conditions should be calmer.
    high_arousal = emotion in {"happy", "angry", "fearful", "surprised"}
    low_arousal = emotion in {"calm", "sad"}
    strong_bonus = 0.15 if intensity == "strong" else 0.0

    energy = np.clip((features["rms_db"] + 45.0) / 25.0, 0.0, 1.0)
    f0_var = np.clip(features["f0_std_hz"] / 90.0, 0.0, 1.0)

    if high_arousal:
        return float(0.45 * energy + 0.40 * f0_var + strong_bonus)
    if low_arousal:
        calm_energy = 1.0 - energy
        calm_f0 = 1.0 - f0_var
        return float(0.45 * calm_energy + 0.40 * calm_f0 + (0.10 if intensity == "normal" else 0.0))
    return float(0.5 * (1.0 - abs(energy - 0.45)) + 0.3 * (1.0 - abs(f0_var - 0.35)))


def discover_items(root: Path) -> list[RavdessItem]:
    items: list[RavdessItem] = []
    for path in sorted(root.glob("Actor_*/*.wav")):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        feats = audio_features(path)
        score = 0.55 * quality_score(feats) + 0.45 * emotion_score(
            parsed["emotion"], parsed["intensity"], feats
        )
        items.append(
            RavdessItem(
                path=str(path),
                filename=path.name,
                score=float(score),
                **parsed,
                **feats,
            )
        )
    return items


def select_balanced(
    items: list[RavdessItem],
    actors: list[int],
    emotions: set[str],
    statements: set[str],
    repetitions_per_condition: int,
) -> list[RavdessItem]:
    selected: list[RavdessItem] = []
    by_key: dict[tuple[int, str, str], list[RavdessItem]] = {}
    for item in items:
        if item.actor not in actors:
            continue
        if item.emotion not in emotions:
            continue
        if item.statement not in statements:
            continue
        by_key.setdefault((item.actor, item.emotion, item.intensity), []).append(item)

    for key in sorted(by_key):
        candidates = sorted(by_key[key], key=lambda x: x.score, reverse=True)
        selected.extend(candidates[:repetitions_per_condition])
    return selected


def write_csv(path: Path, items: list[RavdessItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(items[0]).keys()) if items else list(RavdessItem.__annotations__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def materialize(selected: list[RavdessItem], output_dir: Path, copy_files: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        src = Path(item.path)
        dst_dir = output_dir / item.emotion / item.intensity
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"actor{item.actor:02d}_{item.emotion}_{item.intensity}_{item.statement}_rep{item.repetition}.wav"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_files:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select RAVDESS references for Seed-VC experiments.")
    parser.add_argument("--ravdess-root", default="../../RAVDESS", help="Root containing Actor_* folders.")
    parser.add_argument("--out-dir", default="ravdess_selection", help="Directory for CSV outputs.")
    parser.add_argument("--references-dir", default="references_ravdess", help="Directory for selected symlinks/copies.")
    parser.add_argument("--actors", default="1,2,3,4,5,6,7,8", help="Comma-separated actor IDs.")
    parser.add_argument(
        "--emotions",
        default="neutral,calm,happy,sad,angry,fearful,surprised",
        help="Comma-separated emotions to include.",
    )
    parser.add_argument("--statements", default="kids_are_talking_by_the_door", help="Comma-separated statements.")
    parser.add_argument("--repetitions-per-condition", type=int, default=1)
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--no-materialize", action="store_true", help="Only write CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.ravdess_root).resolve()
    out_dir = Path(args.out_dir)
    references_dir = Path(args.references_dir)
    actors = [int(x) for x in args.actors.split(",") if x.strip()]
    emotions = {x.strip() for x in args.emotions.split(",") if x.strip()}
    statements = {x.strip() for x in args.statements.split(",") if x.strip()}

    items = discover_items(root)
    if not items:
        raise SystemExit(f"No RAVDESS WAV files found under {root}")

    selected = select_balanced(items, actors, emotions, statements, args.repetitions_per_condition)

    write_csv(out_dir / "ravdess_all_features.csv", items)
    write_csv(out_dir / "ravdess_selected_references.csv", selected)
    if not args.no_materialize:
        materialize(selected, references_dir, args.copy_files)

    print(f"Found {len(items)} files.")
    print(f"Selected {len(selected)} references.")
    print(f"Wrote {out_dir / 'ravdess_all_features.csv'}")
    print(f"Wrote {out_dir / 'ravdess_selected_references.csv'}")
    if not args.no_materialize:
        mode = "copies" if args.copy_files else "symlinks"
        print(f"Created {mode} in {references_dir}")


if __name__ == "__main__":
    main()
