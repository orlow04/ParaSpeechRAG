#!/usr/bin/env python3
"""
Codec compression perturbation of SpokenSQuAD dev set audios.

Simulates lossy audio transmission/storage by encoding WAV to MP3 at
different bitrates and decoding back to WAV, introducing codec artifacts.

Bitrates: 8, 16, 32, 64, 128 kbps (following SRB-inspired protocol)

Output structure:
    output/spokensquad_codec/
        8kbps/   -> 10578 WAV files
        16kbps/  -> 10578 WAV files
        32kbps/  -> 10578 WAV files
        64kbps/  -> 10578 WAV files
        128kbps/ -> 10578 WAV files

Usage:
    python codec_perturb_spokensquad.py
    python codec_perturb_spokensquad.py --src_dir spokensquad_cache/ --workers 4
"""

import argparse
import logging
import sys
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path

import soundfile as sf

BITRATES = [8, 16, 32, 64, 128]  # kbps


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


def wav_to_mp3_to_wav(src_path: Path, out_path: Path, bitrate_kbps: int) -> bool:
    """Encode WAV to MP3 at given bitrate, decode back to WAV."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_mp3 = Path(tmp.name)

    try:
        # WAV → MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path),
             "-b:a", f"{bitrate_kbps}k",
             str(tmp_mp3)],
            capture_output=True, check=True
        )
        # MP3 → WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_mp3), str(out_path)],
            capture_output=True, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        tmp_mp3.unlink(missing_ok=True)


def process_file(src_path: Path, bitrates: list, output_dir: Path, resume: bool) -> tuple[int, int, int]:
    """Process one source file across all bitrates. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    for bitrate in bitrates:
        out_path = output_dir / f"{bitrate}kbps" / src_path.name

        if resume and out_path.exists():
            skipped += 1
            continue

        success = wav_to_mp3_to_wav(src_path, out_path, bitrate)
        if success:
            done += 1
        else:
            errors += 1

    return done, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Codec compression perturbation (WAV → MP3 → WAV)"
    )
    parser.add_argument(
        "--src_dir",
        default="spokensquad_cache",
        help="Directory with source WAV files (default: spokensquad_cache/)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/spokensquad_codec",
        help="Root output directory (default: output/spokensquad_codec/)",
    )
    parser.add_argument(
        "--bitrates",
        nargs="+",
        type=int,
        default=BITRATES,
        help="MP3 bitrates in kbps (default: 8 16 32 64 128)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Re-process files that already exist",
    )
    parser.add_argument(
        "--log_file",
        default="codec_perturbation.log",
        help="Log file path (default: codec_perturbation.log)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("Codec Compression Perturbation (MP3)")
    logger.info(f"Bitrates : {args.bitrates} kbps")
    logger.info(f"Workers  : {args.workers}")
    logger.info("=" * 60)

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    src_files = sorted(src_dir.glob("*.wav"))
    if not src_files:
        logger.error(f"No WAV files found in {src_dir}")
        sys.exit(1)
    logger.info(f"Source files : {len(src_files)}")

    for bitrate in args.bitrates:
        (output_dir / f"{bitrate}kbps").mkdir(parents=True, exist_ok=True)

    total = len(src_files) * len(args.bitrates)
    logger.info(f"Total        : {len(src_files)} files × {len(args.bitrates)} bitrates = {total}")
    logger.info("")

    done_total = skipped_total = errors_total = 0
    resume = not args.no_resume

    from tqdm import tqdm
    with tqdm(total=total, desc="Codec perturbation", unit="file") as pbar:
        if args.workers == 1:
            for src_path in src_files:
                d, s, e = process_file(src_path, args.bitrates, output_dir, resume)
                done_total += d
                skipped_total += s
                errors_total += e
                pbar.update(len(args.bitrates))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_file, f, args.bitrates, output_dir, resume): f
                    for f in src_files
                }
                for future in concurrent.futures.as_completed(futures):
                    d, s, e = future.result()
                    done_total += d
                    skipped_total += s
                    errors_total += e
                    pbar.update(len(args.bitrates))

    logger.info(
        f"\nFinished: {done_total} done | {skipped_total} skipped (resume) | {errors_total} errors"
    )


if __name__ == "__main__":
    main()
