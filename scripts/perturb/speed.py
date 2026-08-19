#!/usr/bin/env python3
"""
Speed perturbation of SpokenSQuAD dev set audios following the Speech Robust Bench (SRB) protocol.

Speed factors: 0.5, 0.625, 0.75, 0.875, 1.25, 1.5, 1.75, 2.0
Method: phase vocoder (pitch-preserving time stretch) via librosa

Output structure:
    output/spokensquad_speed/
        0.5x/   -> 10578 WAV files
        0.625x/ -> 10578 WAV files
        ...
        2.0x/   -> 10578 WAV files

Usage:
    python speed_perturb_spokensquad.py
    python speed_perturb_spokensquad.py --src_dir spokensquad_cache/ --output_dir output/spokensquad_speed/
    python speed_perturb_spokensquad.py --workers 4   # parallelizar em 4 CPUs
"""

import argparse
import logging
import sys
import concurrent.futures
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# SRB (Speech Robust Bench) speed factors — arxiv 2403.07937
SRB_SPEEDS = [0.5, 0.625, 0.75, 0.875, 1.25, 1.5, 1.75, 2.0]


def setup_logging(log_file: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def process_file(src_path: Path, speeds: list, output_dir: Path, resume: bool) -> tuple[int, int, int]:
    """Process one source file across all speed factors. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    try:
        audio, sr = librosa.load(str(src_path), sr=None, mono=True)
    except Exception as e:
        return 0, 0, len(speeds)

    for speed in speeds:
        out_path = output_dir / f"{speed}x" / src_path.name

        if resume and out_path.exists():
            skipped += 1
            continue

        try:
            stretched = librosa.effects.time_stretch(audio, rate=speed)
            sf.write(str(out_path), stretched, sr)
            done += 1
        except Exception as e:
            errors += 1

    return done, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Speed perturbation of SpokenSQuAD following SRB protocol"
    )
    parser.add_argument(
        "--src_dir",
        default="spokensquad_cache",
        help="Directory with source WAV files (default: spokensquad_cache/)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/spokensquad_speed",
        help="Root output directory (default: output/spokensquad_speed/)",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=SRB_SPEEDS,
        help="Speed factors to apply (default: SRB protocol)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel CPU workers (default: 1)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Re-process files that already exist",
    )
    parser.add_argument(
        "--log_file",
        default="speed_perturbation.log",
        help="Log file path (default: speed_perturbation.log)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("Speed Perturbation — SRB Protocol")
    logger.info(f"Speed factors : {args.speeds}")
    logger.info(f"Workers       : {args.workers}")
    logger.info("=" * 60)

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    src_files = sorted(src_dir.glob("*.wav"))
    if not src_files:
        logger.error(f"No WAV files found in {src_dir}")
        sys.exit(1)
    logger.info(f"Source files  : {len(src_files)}")

    # Create output directories
    for speed in args.speeds:
        (output_dir / f"{speed}x").mkdir(parents=True, exist_ok=True)

    total = len(src_files) * len(args.speeds)
    logger.info(f"Total         : {len(src_files)} files × {len(args.speeds)} speeds = {total}")
    logger.info("")

    done_total = skipped_total = errors_total = 0
    resume = not args.no_resume

    with tqdm(total=total, desc="Speed perturbation", unit="file") as pbar:

        if args.workers == 1:
            for src_path in src_files:
                d, s, e = process_file(src_path, args.speeds, output_dir, resume)
                done_total += d
                skipped_total += s
                errors_total += e
                pbar.update(len(args.speeds))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_file, f, args.speeds, output_dir, resume): f
                    for f in src_files
                }
                for future in concurrent.futures.as_completed(futures):
                    d, s, e = future.result()
                    done_total += d
                    skipped_total += s
                    errors_total += e
                    pbar.update(len(args.speeds))

    logger.info(
        f"\nFinished: {done_total} done | {skipped_total} skipped (resume) | {errors_total} errors"
    )


if __name__ == "__main__":
    main()
