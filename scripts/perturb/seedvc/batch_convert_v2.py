#!/usr/bin/env python3
"""
Batch conversion with Seed-VC V2.

This script is meant for the two-step emotion pipeline:

1. Emotionize neutral reference voices:
   references_neutral x references_ravdess -> references_emotional

2. Convert Spoken SQuAD inputs using the emotionized references:
   inputs x references_emotional -> outputs_emotional

Run from the repository root or from SCRIPT_TEST; the script will chdir to the
Seed-VC root before loading configs/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import queue as _queue
import re
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
import yaml
import librosa
from hydra.utils import instantiate
from omegaconf import DictConfig


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from modules.commons import str2bool  # noqa: E402

WORKER_TIMEOUT_S = 600  # max seconds to wait for a single conversion


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

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


@dataclass
class ConversionRecord:
    source: str
    target: str
    output: str
    source_stem: str
    target_stem: str
    actor: str
    emotion: str
    intensity: str
    status: str
    elapsed_s: float
    error: str


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_audio_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def parse_csv_set(value: str | None, normalize: str = "lower") -> set[str] | None:
    if not value:
        return None
    values = {x.strip() for x in value.split(",") if x.strip()}
    if normalize == "lower":
        return {x.lower() for x in values}
    return values


def target_metadata(path: Path) -> dict[str, str]:
    """Extract actor/emotion/intensity from RAVDESS or generated filenames."""
    stem = path.stem
    parts = stem.split("-")
    if len(parts) == 7 and parts[0] == "03" and parts[1] == "01":
        return {
            "actor": str(int(parts[6])),
            "emotion": EMOTIONS.get(parts[2], "unknown"),
            "intensity": INTENSITIES.get(parts[3], "unknown"),
        }

    actor_match = re.search(r"(?:actor|ravactor)(\d+)", stem, re.IGNORECASE)
    emotion_match = re.search(
        r"(neutral|calm|happy|sad|angry|fearful|disgust|surprised)",
        stem,
        re.IGNORECASE,
    )
    intensity_match = re.search(r"(normal|strong)", stem, re.IGNORECASE)

    # references_ravdess uses emotion/intensity directories.
    parent_names = [p.name.lower() for p in path.parents[:3]]
    emotion = emotion_match.group(1).lower() if emotion_match else "unknown"
    intensity = intensity_match.group(1).lower() if intensity_match else "unknown"
    for name in parent_names:
        if name in {"neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"}:
            emotion = name
        if name in {"normal", "strong"}:
            intensity = name

    return {
        "actor": str(int(actor_match.group(1))) if actor_match else "unknown",
        "emotion": emotion,
        "intensity": intensity,
    }


def filter_targets(
    targets: list[Path],
    actors: set[str] | None,
    emotions: set[str] | None,
    intensities: set[str] | None,
) -> list[Path]:
    filtered = []
    for target in targets:
        meta = target_metadata(target)
        if actors is not None and meta["actor"] not in actors:
            continue
        if emotions is not None and meta["emotion"] not in emotions:
            continue
        if intensities is not None and meta["intensity"] not in intensities:
            continue
        filtered.append(target)
    return filtered


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def audio_duration_s(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        y, sr = librosa.load(str(path), sr=None)
        return float(len(y) / sr)


def output_path(outputs_dir: Path, source: Path, target: Path, preserve_tree: bool) -> Path:
    meta = target_metadata(target)
    target_label = f"actor{meta['actor']}_{meta['emotion']}_{meta['intensity']}_{safe_stem(target)}"
    filename = f"{safe_stem(source)}__{target_label}.wav"
    if preserve_tree and meta["emotion"] != "unknown" and meta["intensity"] != "unknown":
        return outputs_dir / meta["emotion"] / meta["intensity"] / filename
    return outputs_dir / filename


def load_v2_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    os.chdir(ROOT_DIR)
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint_path,
        cfm_checkpoint_path=args.cfm_checkpoint_path,
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        if hasattr(torch._inductor.config, "fx_graph_cache"):
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()

    return vc_wrapper


def convert_one(
    vc_wrapper,
    source: Path,
    target: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, object] | None:
    if args.anonymization_only:
        raise ValueError("anonymization_only requires the streaming V2 path, which depends on MP3 export.")

    if args.preserve_source_speaker and not args.convert_style:
        raise ValueError("--preserve-source-speaker requires --convert-style true")

    use_streaming = args.force_streaming or (
        args.streaming_threshold_s is not None
        and audio_duration_s(source) >= args.streaming_threshold_s
    )

    if use_streaming and args.convert_style and not args.preserve_source_speaker:
        generator = vc_wrapper.convert_voice_with_streaming(
            source_audio_path=str(source),
            target_audio_path=str(target),
            diffusion_steps=args.diffusion_steps,
            length_adjust=args.length_adjust,
            intelligebility_cfg_rate=args.intelligibility_cfg_rate,
            similarity_cfg_rate=args.similarity_cfg_rate,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            convert_style=True,
            anonymization_only=args.anonymization_only,
            device=device,
            dtype=dtype,
            stream_output=False,
        )

        full_audio = None
        try:
            while True:
                next(generator)
        except StopIteration as exc:
            full_audio = exc.value
        if full_audio is None:
            raise RuntimeError("streaming V2 conversion returned no full_audio")
        return vc_wrapper.sr, full_audio

    if args.preserve_source_speaker:
        # Pass 1: transfer target emotion/style, accepting that the target speaker
        # may leak in. Pass 2: timbre-convert back to the original source speaker
        # while keeping the emotional/prosodic contour from pass 1.
        emotional_audio = vc_wrapper.convert_voice(
            source_audio_path=str(source),
            target_audio_path=str(target),
            diffusion_steps=args.diffusion_steps,
            length_adjust=args.length_adjust,
            inference_cfg_rate=[args.intelligibility_cfg_rate, args.similarity_cfg_rate],
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            device=device,
            dtype=dtype,
        ).squeeze()

        with tempfile.NamedTemporaryFile(suffix=".wav", dir=args.tmp_dir, delete=True) as tmp:
            sf.write(tmp.name, emotional_audio, vc_wrapper.sr)
            audio = vc_wrapper.convert_timbre(
                source_audio_path=tmp.name,
                target_audio_path=str(source),
                diffusion_steps=args.diffusion_steps,
                length_adjust=1.0,
                inference_cfg_rate=args.similarity_cfg_rate,
                device=device,
                dtype=dtype,
            )
    elif args.convert_style:
        audio = vc_wrapper.convert_voice(
            source_audio_path=str(source),
            target_audio_path=str(target),
            diffusion_steps=args.diffusion_steps,
            length_adjust=args.length_adjust,
            inference_cfg_rate=[args.intelligibility_cfg_rate, args.similarity_cfg_rate],
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            device=device,
            dtype=dtype,
        )
    else:
        audio = vc_wrapper.convert_timbre(
            source_audio_path=str(source),
            target_audio_path=str(target),
            diffusion_steps=args.diffusion_steps,
            length_adjust=args.length_adjust,
            inference_cfg_rate=args.similarity_cfg_rate,
            device=device,
            dtype=dtype,
        )

    audio = audio.squeeze()
    return vc_wrapper.sr, audio


def write_manifest(path: Path, records: list[ConversionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ConversionRecord.__annotations__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _worker_entry(job_q: mp.Queue, result_q: mp.Queue, args_ns: argparse.Namespace) -> None:
    """Subprocess: loads model once and converts jobs until CUDA context dies."""
    device = get_device()
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    try:
        vc_wrapper = load_v2_model(args_ns, device, dtype)
    except Exception as exc:
        result_q.put({"type": "init_error", "error": repr(exc)})
        return
    result_q.put({"type": "ready"})

    while True:
        job = job_q.get()
        if job is None:
            return
        source = Path(job["source"])
        target = Path(job["target"])
        out_path = Path(job["out"])
        t0 = time.time()
        try:
            full_audio = convert_one(vc_wrapper, source, target, args_ns, device, dtype)
            if full_audio is None:
                raise RuntimeError("convert_one returned None")
            save_sr, audio = full_audio
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, save_sr)
            result_q.put({"type": "ok", "job": job, "elapsed": time.time() - t0})
        except Exception as exc:
            err = repr(exc)
            result_q.put({"type": "error", "job": job, "elapsed": time.time() - t0, "error": err})
            if "CUDA error" in err or "device-side assert" in err or "out of memory" in err.lower():
                # CUDA context is poisoned — exit so main process can spawn a fresh worker
                return


def _start_worker(args: argparse.Namespace):
    """Spawn a worker process and wait for it to finish loading the model."""
    job_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue(maxsize=4)
    proc = mp.Process(target=_worker_entry, args=(job_q, result_q, args), daemon=True)
    proc.start()
    try:
        msg = result_q.get(timeout=600)
    except _queue.Empty:
        proc.kill()
        proc.join()
        raise RuntimeError("Worker timed out during model load")
    if msg["type"] != "ready":
        proc.join()
        raise RuntimeError(f"Worker init failed: {msg.get('error', msg)}")
    return proc, job_q, result_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Seed-VC V2 conversion.")
    parser.add_argument("--sources", required=True, help="Source audio directory.")
    parser.add_argument("--targets", required=True, help="Target/reference audio directory.")
    parser.add_argument("--outputs", required=True, help="Output directory.")
    parser.add_argument("--recursive-targets", type=str2bool, default=True)
    parser.add_argument("--recursive-sources", type=str2bool, default=False)
    parser.add_argument("--preserve-target-tree", type=str2bool, default=True)
    parser.add_argument("--manifest", default=None, help="CSV manifest path.")
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--target-actors", default=None, help="Comma-separated actor IDs, e.g. 1,2.")
    parser.add_argument("--target-emotions", default=None, help="Comma-separated emotions.")
    parser.add_argument("--target-intensities", default=None, help="Comma-separated intensities.")
    parser.add_argument(
        "--preserve-source-speaker",
        type=str2bool,
        default=False,
        help="Two-pass mode: copy target emotion/style, then convert timbre back to source speaker.",
    )
    parser.add_argument("--tmp-dir", default=None, help="Temp directory for two-pass intermediate WAVs.")
    parser.add_argument(
        "--force-streaming",
        type=str2bool,
        default=False,
        help="Use V2 chunked/streaming path and save WAV directly. Useful for long inputs.",
    )
    parser.add_argument(
        "--streaming-threshold-s",
        type=float,
        default=None,
        help="Use chunked/streaming path when source duration is at least this many seconds.",
    )

    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--convert-style", type=str2bool, default=True)
    parser.add_argument("--anonymization-only", type=str2bool, default=False)
    parser.add_argument("--ar-checkpoint-path", type=str, default=None)
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources_dir = Path(args.sources).resolve()
    targets_dir = Path(args.targets).resolve()
    outputs_dir = Path(args.outputs).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sources = list_audio_files(sources_dir, args.recursive_sources)
    targets = list_audio_files(targets_dir, args.recursive_targets)
    targets = filter_targets(
        targets,
        actors=parse_csv_set(args.target_actors, normalize="none"),
        emotions=parse_csv_set(args.target_emotions),
        intensities=parse_csv_set(args.target_intensities),
    )

    if args.max_sources is not None:
        sources = sources[:args.max_sources]
    if args.max_targets is not None:
        targets = targets[:args.max_targets]

    if not sources:
        raise SystemExit(f"No source audio files found in {sources_dir}")
    if not targets:
        raise SystemExit(f"No target audio files found in {targets_dir}")

    total = len(sources) * len(targets)
    print("=" * 72)
    print("Seed-VC V2 Batch Conversion")
    print(f"Sources : {len(sources)} from {sources_dir}")
    print(f"Targets : {len(targets)} from {targets_dir}")
    print(f"Outputs : {outputs_dir}")
    print(f"Total   : {total}")
    print(f"Style   : convert_style={args.convert_style}")
    print("=" * 72)

    device = get_device()
    print(f"Starting worker process (device: {device})...")
    proc, job_q, result_q = _start_worker(args)
    print("Worker ready.")

    records: list[ConversionRecord] = []
    done = 0
    skipped = 0
    failed = 0

    for source in sources:
        for target in targets:
            done += 1
            meta = target_metadata(target)
            out = output_path(outputs_dir, source, target, args.preserve_target_tree)
            out.parent.mkdir(parents=True, exist_ok=True)
            prefix = f"[{done}/{total}]"

            if args.skip_existing and out.exists():
                skipped += 1
                records.append(ConversionRecord(
                    source=str(source), target=str(target), output=str(out),
                    source_stem=source.stem, target_stem=target.stem,
                    actor=meta["actor"], emotion=meta["emotion"], intensity=meta["intensity"],
                    status="skipped", elapsed_s=0.0, error="",
                ))
                continue

            print(f"{prefix} {source.name} -> {target.name}", flush=True)
            job_q.put({"source": str(source), "target": str(target), "out": str(out)})

            # Wait for result, with timeout in case the worker hangs silently
            try:
                result = result_q.get(timeout=WORKER_TIMEOUT_S)
            except _queue.Empty:
                proc.kill()
                proc.join()
                result = {
                    "type": "error",
                    "elapsed": float(WORKER_TIMEOUT_S),
                    "error": f"Worker timeout after {WORKER_TIMEOUT_S}s",
                }
                print(f"    FAILED (timeout): restarting worker...", flush=True)
                proc, job_q, result_q = _start_worker(args)

            if result["type"] == "ok":
                elapsed = result["elapsed"]
                status, error = "ok", ""
                print(f"    done in {elapsed:.1f}s -> {out.name}", flush=True)
            else:
                failed += 1
                elapsed = result.get("elapsed", 0.0)
                error = result.get("error", "unknown error")
                status = "failed"
                print(f"    FAILED: {error}", flush=True)
                # CUDA context is dead — worker will have exited; restart it
                if "CUDA error" in error or "device-side assert" in error or "out of memory" in error.lower():
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    print("    CUDA error detected — restarting worker...", flush=True)
                    proc, job_q, result_q = _start_worker(args)

            records.append(ConversionRecord(
                source=str(source), target=str(target), output=str(out),
                source_stem=source.stem, target_stem=target.stem,
                actor=meta["actor"], emotion=meta["emotion"], intensity=meta["intensity"],
                status=status, elapsed_s=elapsed, error=error,
            ))

            if args.manifest:
                write_manifest(Path(args.manifest), records)

    # Graceful shutdown
    job_q.put(None)
    proc.join(timeout=15)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if args.manifest:
        write_manifest(Path(args.manifest), records)

    converted = total - skipped - failed
    print("=" * 72)
    print(f"Finished: {converted} converted | {skipped} skipped | {failed} failed")
    if args.manifest:
        print(f"Manifest: {args.manifest}")
    print("=" * 72)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
