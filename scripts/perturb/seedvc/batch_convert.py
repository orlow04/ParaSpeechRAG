"""
Batch Voice Conversion Script
==============================
Converts every audio in `inputs/` using every reference voice in `references/`,
saving results to `outputs/{input_stem}__{ref_stem}.wav`.

After each conversion, computes SECS (Speaker Embedding Cosine Similarity)
between the converted audio and the reference voice using resemblyzer.

Usage:
    cd seed-vc/
    python SCRIPT_TEST/batch_convert.py [options]

Options:
    --inputs        Path to folder with source audios  (default: SCRIPT_TEST/inputs)
    --references    Path to folder with reference voices (default: SCRIPT_TEST/references)
    --outputs       Path to output folder              (default: SCRIPT_TEST/outputs)
    --diffusion-steps  Diffusion steps (default: 30)
    --length-adjust    Speed factor, 1.0 = normal      (default: 1.0)
    --cfg-rate         CFG guidance rate 0-1            (default: 0.7)
    --f0-condition     Use F0 model (for singing)       (default: False)
    --auto-f0-adjust   Auto pitch match to reference    (default: True)
    --pitch-shift      Semitone shift                   (default: 0)
    --skip-existing    Skip already converted files     (default: True)
"""

import os
import sys
import time
import argparse
import warnings

warnings.simplefilter("ignore")

# Make sure we can import from the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

# Set HF cache before any model imports
os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT_DIR, "checkpoints", "hf_cache"))

import numpy as np
import torch
import torchaudio
import librosa
from resemblyzer import preprocess_wav, VoiceEncoder

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def list_audio_files(folder: str) -> list[str]:
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    return [os.path.join(folder, f) for f in files]


def output_path(outputs_dir: str, input_path: str, ref_path: str) -> str:
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    ref_stem = os.path.splitext(os.path.basename(ref_path))[0]
    return os.path.join(outputs_dir, f"{input_stem}__{ref_stem}.wav")


def compute_secs(encoder: VoiceEncoder, audio_a: np.ndarray, sr_a: int,
                 audio_b: np.ndarray, sr_b: int) -> float:
    """Cosine similarity between speaker embeddings of two audio arrays."""
    wav_a = preprocess_wav(audio_a, source_sr=sr_a)
    wav_b = preprocess_wav(audio_b, source_sr=sr_b)
    emb_a = encoder.embed_utterance(wav_a)
    emb_b = encoder.embed_utterance(wav_b)
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))


def secs_label(secs: float) -> str:
    if secs >= 0.90:
        return "excelente"
    if secs >= 0.80:
        return "bom"
    if secs >= 0.70:
        return "razoável"
    return "fraco"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch voice conversion with Seed-VC")
    parser.add_argument("--inputs",       default=os.path.join(SCRIPT_DIR, "inputs"))
    parser.add_argument("--references",   default=os.path.join(SCRIPT_DIR, "references"))
    parser.add_argument("--outputs",      default=os.path.join(SCRIPT_DIR, "outputs"))
    parser.add_argument("--diffusion-steps", type=int,   default=30)
    parser.add_argument("--length-adjust",   type=float, default=1.0)
    parser.add_argument("--cfg-rate",        type=float, default=0.7)
    parser.add_argument("--f0-condition",    type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--auto-f0-adjust",  type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--pitch-shift",     type=int,   default=0)
    parser.add_argument("--skip-existing",   type=lambda x: x.lower() == "true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    inputs = list_audio_files(args.inputs)
    references = list_audio_files(args.references)

    if not inputs:
        print(f"[ERROR] No audio files found in: {args.inputs}")
        sys.exit(1)
    if not references:
        print(f"[ERROR] No audio files found in: {args.references}")
        sys.exit(1)

    total = len(inputs) * len(references)
    print(f"\n{'='*60}")
    print(f"  Seed-VC Batch Conversion")
    print(f"{'='*60}")
    print(f"  Inputs      : {len(inputs)} file(s)  →  {args.inputs}")
    print(f"  References  : {len(references)} file(s)  →  {args.references}")
    print(f"  Output dir  : {args.outputs}")
    print(f"  Total pairs : {total}")
    print(f"  Diffusion   : {args.diffusion_steps} steps | CFG rate: {args.cfg_rate}")
    print(f"  F0 model    : {args.f0_condition} | Skip existing: {args.skip_existing}")
    print(f"{'='*60}\n")

    # Load Seed-VC models once
    print("Loading Seed-VC models (this may take a minute)...")
    from seed_vc_wrapper import SeedVCWrapper
    wrapper = SeedVCWrapper()
    sr = 44100 if args.f0_condition else 22050
    print("Models loaded.")

    # Load resemblyzer once for SECS computation
    print("Loading resemblyzer speaker encoder...")
    voice_encoder = VoiceEncoder()
    print("Ready.\n")

    done = 0
    skipped = 0
    failed = 0
    secs_scores = []  # (input, ref, secs)

    for inp in inputs:
        inp_name = os.path.basename(inp)
        # Pre-load reference audio for SECS (reused across all inputs)
        ref_cache: dict[str, tuple[np.ndarray, int]] = {}

        for ref in references:
            ref_name = os.path.basename(ref)
            done += 1
            out = output_path(args.outputs, inp, ref)

            prefix = f"[{done}/{total}]"

            if args.skip_existing and os.path.exists(out):
                skipped += 1
                continue

            print(f"{prefix} Converting  {inp_name}  →  {ref_name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                # convert_voice is a generator (has yield internally).
                # stream_output=True yields (mp3_bytes, full_audio) where
                # the last non-None full_audio is (sample_rate, np.ndarray).
                audio = None
                out_sr = sr
                gen = wrapper.convert_voice(
                    source=inp,
                    target=ref,
                    diffusion_steps=args.diffusion_steps,
                    length_adjust=args.length_adjust,
                    inference_cfg_rate=args.cfg_rate,
                    f0_condition=args.f0_condition,
                    auto_f0_adjust=args.auto_f0_adjust,
                    pitch_shift=args.pitch_shift,
                    stream_output=False,
                )
                try:
                    while True:
                        next(gen)
                except StopIteration as e:
                    if e.value is not None:
                        audio = e.value

                elapsed = time.time() - t0

                if audio is None:
                    raise RuntimeError("convert_voice returned no audio")

                audio_tensor = torch.tensor(audio).unsqueeze(0)  # [1, samples]
                torchaudio.save(out, audio_tensor, out_sr)

                # Compute SECS
                if ref not in ref_cache:
                    ref_cache[ref] = librosa.load(ref, sr=None)
                ref_audio, ref_sr = ref_cache[ref]
                secs = compute_secs(voice_encoder, audio, out_sr, ref_audio, ref_sr)
                secs_scores.append((inp_name, ref_name, secs))

                print(f"done  ({elapsed:.1f}s)  SECS={secs:.4f} ({secs_label(secs)})  →  {os.path.basename(out)}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.1f}s)")
                print(f"       Error: {e}")
                failed += 1

    # Summary
    converted = done - skipped - failed
    print(f"\n{'='*60}")
    print(f"  Finished: {converted} converted | {skipped} skipped | {failed} failed")

    if secs_scores:
        all_secs = [s for _, _, s in secs_scores]
        print(f"\n  SECS (Speaker Embedding Cosine Similarity)")
        print(f"  {'─'*40}")
        print(f"  Média   : {np.mean(all_secs):.4f}")
        print(f"  Mínimo  : {np.min(all_secs):.4f}")
        print(f"  Máximo  : {np.max(all_secs):.4f}")

        # Per-reference average
        from collections import defaultdict
        per_ref: dict[str, list] = defaultdict(list)
        for _, ref_name, secs in secs_scores:
            per_ref[ref_name].append(secs)
        print(f"\n  Por referência:")
        for ref_name, scores in sorted(per_ref.items()):
            avg = np.mean(scores)
            print(f"    {ref_name:<30}  avg SECS={avg:.4f} ({secs_label(avg)})")

    print(f"\n  Results saved to: {args.outputs}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
