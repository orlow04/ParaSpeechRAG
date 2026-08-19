#!/usr/bin/env python3
"""
Batch voice conversion of SpokenSQuAD audios using GenVC.

Replicates the SeedVC evaluation setup:
  - ~3k source audios from SpokenSQuAD (files starting with 0_, 10_, and 11_0_* through 11_13_*)
  - 6 reference voices (azuma_0, trump_0, and 4 LibriSpeech speakers)
  - Each source audio is converted 6 times (once per reference)
  - Total: ~18k output files

Usage:
  # With pre-downloaded SpokenSQuAD audios:
  python batch_convert_spokensquad.py --src_dir path/to/spokensquad/ --ref_dir references/

  # Download from HuggingFace automatically:
  python batch_convert_spokensquad.py --hf_dataset --ref_dir references/
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# Add GenVC to path
sys.path.insert(0, str(Path(__file__).parent.parent / "GenVC"))

from inference.model_init import model_init
from utils import load_audio


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SpokenSQuAD file filter
# ---------------------------------------------------------------------------

def should_include(name: str) -> bool:
    """
    Keep files matching the ~3k subset used in the SeedVC evaluation:
      - starts with 0_   (article 0, all paragraphs/sentences)
      - starts with 10_  (article 10, all paragraphs/sentences)
      - starts with 11_  with paragraph index 0-13  (11_0_* … 11_13_*)
    """
    stem = Path(name).stem

    if re.match(r"^0_", stem):
        return True
    if re.match(r"^10_", stem):
        return True
    m = re.match(r"^11_(\d+)_", stem)
    if m and 0 <= int(m.group(1)) <= 13:
        return True
    return False


# ---------------------------------------------------------------------------
# Source audio collection
# ---------------------------------------------------------------------------

def collect_from_dir(src_dir: str, use_filter: bool = True) -> list[Path]:
    """Collect audio files from a local directory."""
    src_dir = Path(src_dir)
    all_files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        all_files.extend(src_dir.glob(ext))
    if use_filter:
        return sorted(f for f in all_files if should_include(f.name))
    return sorted(all_files)


def download_from_hf(split: str, cache_dir: str, logger: logging.Logger) -> list[Path]:
    """
    Download matching SpokenSQuAD audios from HuggingFace (alinet/spoken_squad)
    and save them as WAV files in cache_dir.
    Returns list of saved file paths.
    """
    try:
        from datasets import load_dataset, Audio as HFAudio
        import soundfile as sf
        import numpy as np
        import io
    except ImportError:
        logger.error("Install 'datasets' and 'soundfile': pip install datasets soundfile")
        sys.exit(1)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading alinet/spoken_squad ({split}) from HuggingFace (streaming) …")
    # streaming=True: downloads parquet chunks on demand instead of the full dataset at once
    ds = load_dataset("alinet/spoken_squad", split=split, streaming=True)
    # Disable automatic audio decoding to avoid needing torchcodec
    ds = ds.cast_column("audio", HFAudio(decode=False))

    saved = []
    skipped_filter = 0
    errors = 0

    for item in tqdm(ds, desc="Saving audios"):
        # Try to get the file identifier
        audio_info = item.get("audio", {}) or {}
        item_id = (
            item.get("id")
            or audio_info.get("path", "")
            or ""
        )

        if not should_include(item_id):
            skipped_filter += 1
            continue

        stem = Path(item_id).stem if item_id else f"item_{len(saved)}"
        out_path = cache_dir / f"{stem}.wav"

        if out_path.exists():
            saved.append(out_path)
            continue

        try:
            if audio_info.get("array") is not None:
                arr = np.array(audio_info["array"], dtype=np.float32)
                sr = int(audio_info["sampling_rate"])
            elif audio_info.get("bytes"):
                arr, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                arr = arr.astype(np.float32)
            else:
                raise ValueError("No audio data found in item")
            sf.write(str(out_path), arr, sr)
            saved.append(out_path)
        except Exception as e:
            logger.warning(f"Failed to save {item_id}: {e}")
            errors += 1

    logger.info(
        f"SpokenSQuAD: {len(saved)} saved, {skipped_filter} filtered out, {errors} errors"
    )
    return sorted(saved)


# ---------------------------------------------------------------------------
# GenVC synthesis helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_content_codes(model, src_wav: torch.Tensor, seg_len: float = 6.0) -> list:
    """
    Extract content codes for all segments of a source audio.
    Called once per source audio; reused across all 6 references.
    """
    total_wavlen = src_wav.shape[-1]
    min_chunk = int(0.32 * model.content_sample_rate)
    seg_samples = int(seg_len * model.content_sample_rate)
    codes_list = []

    for i in range(0, total_wavlen, seg_samples):
        seg = src_wav[:, i : i + seg_samples]
        if seg.shape[-1] < min_chunk:
            seg = torch.nn.functional.pad(seg, (0, min_chunk - seg.shape[-1]))

        feat = model.content_extractor.extract_content_features(seg)
        codes = model.content_dvae.get_codebook_indices(feat.transpose(1, 2))
        codes_list.append(codes)

    return codes_list


@torch.inference_mode()
def synthesize_from_codes(model, content_codes_list: list, cond_latent: torch.Tensor) -> torch.Tensor:
    """
    Generate audio from precomputed content codes + a precomputed reference latent.
    Equivalent to synthesize_utt() but avoids re-extracting content features.
    """
    final_latents = []

    for content_codes in content_codes_list:
        gen_codes = model.gpt.generate(
            cond_latent,
            content_codes,
            do_sample=True,
            top_p=model.config.top_p,
            top_k=model.config.top_k,
            temperature=model.config.temperature,
            num_beams=1,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            output_attentions=False,
        )[0]

        # Remove stop token and handle edge cases
        valid_mask = gen_codes != model.gpt.stop_audio_token
        if valid_mask.sum() == 0:
            continue
        gen_codes = gen_codes[valid_mask.nonzero().squeeze()]
        if gen_codes.dim() == 0:
            gen_codes = gen_codes.unsqueeze(0)

        expected_output_len = torch.tensor(
            [gen_codes.shape[-1] * model.config.model_args.gpt_code_stride_len],
            device=model.device,
        )
        content_len = torch.tensor([content_codes.shape[-1]], device=model.device)

        acoustic_latents = model.gpt(
            content_codes,
            content_len,
            gen_codes.unsqueeze(0),
            expected_output_len,
            cond_latents=cond_latent,
            return_latent=True,
        )
        final_latents.append(acoustic_latents)

    if not final_latents:
        return torch.zeros(1, device=model.device)

    final_latents = torch.cat(final_latents, dim=1)
    mel_input = torch.nn.functional.interpolate(
        final_latents.transpose(1, 2),
        scale_factor=[model.hifigan_scale_factor],
        mode="linear",
    ).squeeze(1)

    return model.hifigan(mel_input)[0].squeeze()


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def run_batch_conversion(
    model,
    config,
    src_paths: list[Path],
    ref_paths: list[Path],
    output_dir: str,
    top_k: int = 15,
    resume: bool = True,
    logger: logging.Logger = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    model.config.top_k = top_k

    # --- Precompute reference conditioning latents (once per reference) ---
    logger.info("Precomputing conditioning latents for reference audios …")
    ref_latents = []
    ref_names = []
    for ref_path in ref_paths:
        try:
            ref_audio = load_audio(str(ref_path), config.audio.sample_rate)
            ref_audio = ref_audio.to(model.device)
            latent = model.get_gpt_cond_latents(ref_audio, config.audio.sample_rate)
            ref_latents.append(latent)
            ref_names.append(Path(ref_path).stem)
            logger.info(f"  OK  {Path(ref_path).name}")
        except Exception as e:
            logger.error(f"  FAIL  {ref_path}: {e}")
            sys.exit(1)

    # Create one output folder per reference
    for name in ref_names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    # --- Count already done (for resume) ---
    total = len(src_paths) * len(ref_paths)
    done = skipped = errors = 0

    logger.info(
        f"\nStarting batch conversion: {len(src_paths)} sources × {len(ref_paths)} refs = {total} total\n"
    )

    pbar = tqdm(total=total, desc="Converting", unit="file")

    for src_path in src_paths:
        src_stem = Path(src_path).stem

        # Check if all outputs for this source already exist (fast resume skip)
        all_exist = all(
            (output_dir / rname / f"{src_stem}.wav").exists()
            for rname in ref_names
        )
        if resume and all_exist:
            skipped += len(ref_names)
            pbar.update(len(ref_names))
            continue

        # Load source audio and extract content codes ONCE for all refs
        try:
            src_wav = load_audio(str(src_path), model.content_sample_rate)
            src_wav = src_wav.to(model.device)
            content_codes_list = extract_content_codes(model, src_wav)
        except Exception as e:
            logger.warning(f"[SKIP source] {src_stem}: {e}")
            errors += len(ref_names)
            pbar.update(len(ref_names))
            continue

        for cond_latent, ref_name in zip(ref_latents, ref_names):
            out_path = output_dir / ref_name / f"{src_stem}.wav"

            if resume and out_path.exists():
                skipped += 1
                pbar.update(1)
                continue

            try:
                audio_out = synthesize_from_codes(model, content_codes_list, cond_latent)
                torchaudio.save(
                    str(out_path),
                    audio_out.unsqueeze(0).detach().cpu(),
                    config.audio.sample_rate,
                )
                done += 1
            except Exception as e:
                logger.warning(f"[ERROR] {src_stem} -> {ref_name}: {e}")
                errors += 1

            pbar.update(1)

    pbar.close()
    logger.info(
        f"\nFinished: {done} converted | {skipped} skipped (resume) | {errors} errors"
    )
    return done, skipped, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch GenVC voice conversion on SpokenSQuAD subset"
    )

    # Model
    parser.add_argument(
        "--model_path",
        default="GenVC/pre_trained/GenVC_large.pth",
        help="Path to GenVC checkpoint (default: GenVC/pre_trained/GenVC_large.pth)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="top_k for GPT decoding (default: 15)",
    )

    # Source audios (mutually exclusive)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--src_dir",
        help="Directory with pre-downloaded SpokenSQuAD WAV/FLAC files",
    )
    src_group.add_argument(
        "--hf_dataset",
        action="store_true",
        help="Download SpokenSQuAD from HuggingFace (alinet/spoken_squad)",
    )

    parser.add_argument(
        "--hf_split",
        default="train",
        help="HuggingFace dataset split to use (default: train)",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default="spokensquad_cache",
        help="Local folder to cache HuggingFace audio files (default: spokensquad_cache/)",
    )

    # References
    parser.add_argument(
        "--ref_dir",
        required=True,
        help="Directory containing the 6 reference audio files (WAV/FLAC)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default="output/spokensquad_genvc",
        help="Root output directory (default: output/spokensquad_genvc/)",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Process all audio files in --src_dir (default: only articles 0, 10, 11)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume — re-convert files that already exist",
    )
    parser.add_argument(
        "--log_file",
        default="batch_conversion.log",
        help="Log file path (default: batch_conversion.log)",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("GenVC — SpokenSQuAD Batch Conversion")
    logger.info("=" * 60)

    # --- Reference audios ---
    ref_dir = Path(args.ref_dir)
    ref_paths = sorted(
        p
        for p in ref_dir.iterdir()
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    )
    if not ref_paths:
        logger.error(f"No audio files found in --ref_dir: {ref_dir}")
        sys.exit(1)
    if len(ref_paths) != 6:
        logger.warning(
            f"Expected 6 reference files, found {len(ref_paths)}: "
            + ", ".join(p.name for p in ref_paths)
        )
    logger.info(f"References ({len(ref_paths)}): " + ", ".join(p.name for p in ref_paths))

    # --- Load model ---
    # model_init uses relative paths (e.g. pre_trained/contentVec.pt) so we
    # temporarily change to the GenVC directory before calling it.
    logger.info(f"Loading GenVC model: {args.model_path} on {args.device}")
    _orig_dir = Path.cwd()
    genvc_dir = Path(__file__).parent.parent / "GenVC"
    os.chdir(genvc_dir)
    try:
        model, config = model_init(args.model_path, args.device)
    finally:
        os.chdir(_orig_dir)
    logger.info("Model loaded successfully.")

    # --- Collect source audios ---
    if args.src_dir:
        src_paths = collect_from_dir(args.src_dir, use_filter=not args.no_filter)
        logger.info(f"Found {len(src_paths)} source audios in {args.src_dir}")
        if not src_paths:
            logger.error("No matching files found. Check --src_dir and the filter logic.")
            sys.exit(1)
    else:
        src_paths = download_from_hf(args.hf_split, args.hf_cache_dir, logger)
        if not src_paths:
            logger.error("No audio files downloaded from HuggingFace.")
            sys.exit(1)

    # --- Run batch ---
    run_batch_conversion(
        model=model,
        config=config,
        src_paths=src_paths,
        ref_paths=ref_paths,
        output_dir=args.output_dir,
        top_k=args.top_k,
        resume=not args.no_resume,
        logger=logger,
    )


if __name__ == "__main__":
    main()
