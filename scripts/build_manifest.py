#!/usr/bin/env python3
"""Build a checksummed manifest for a perturbed-audio directory.

The perturbed audio *is* the benchmark. Seed-VC runs a stochastic diffusion
sampler and GenVC uses top-k sampling, so the audio cannot be regenerated
bit-identically even with a fixed seed — a different torch or CUDA build gives
different waveforms and therefore different Recall@1. The reproduction path is
therefore ``released audio -> embeddings -> metrics``, which is deterministic,
and this manifest is what pins the first arrow.

Emits one CSV row per WAV:

    path, source_query_id, axis, severity, generator, sha256, bytes, sr, duration_s

``axis`` and ``severity`` are inferred from the directory layout, which differs
per axis:

    <root>/<voice>/{a}_{p}_{c}__<voice>.wav                  -> speaker
    <root>/<emotion>/<intensity>/{a}_{p}_{c}__...wav         -> emotion
    <root>/<noise_type>/{a}_{p}_{c}_<noise_type>.wav         -> noise
    <root>/{a}_{p}_{c}.wav                                   -> clean

Pass ``--axis``/``--severity``/``--generator`` to override inference when the
layout does not say it (codec bitrate and time-stretch factor, for instance,
are not recoverable from the current filenames).

``--verify`` re-reads an existing manifest and reports files that changed,
disappeared, or appeared since it was written.

Usage:
    python scripts/build_manifest.py data/datasets/spoken_squad_emotions \\
        --generator seed-vc-v2 --out data/manifests/emotion.csv
    python scripts/build_manifest.py data/datasets/spoken_squad_emotions \\
        --verify data/manifests/emotion.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

_NOISE_TYPES = {"white", "reverb", "ambient"}
_INTENSITIES = {"normal", "strong"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="Directory to walk (recursively)")
    p.add_argument("--out", type=Path, default=None, help="Manifest CSV to write")
    p.add_argument("--verify", type=Path, default=None,
                   help="Compare against an existing manifest instead of writing one")
    p.add_argument("--axis", default=None, help="Override the inferred axis")
    p.add_argument("--severity", default=None, help="Override the inferred severity")
    p.add_argument("--generator", default="unknown",
                   help="Generator identifier, e.g. seed-vc-v2, genvc-large, ffmpeg-7.1")
    p.add_argument("--no-audio-probe", action="store_true",
                   help="Skip reading sample rate / duration (much faster)")
    return p.parse_args()


def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def infer_axis_severity(wav: Path, root: Path) -> tuple[str, str]:
    """Infer (axis, severity) from the path components below ``root``."""
    parts = wav.relative_to(root).parts[:-1]  # directories only

    if len(parts) >= 2 and parts[-1] in _INTENSITIES:
        return "emotion", f"{parts[-2]}/{parts[-1]}"
    if len(parts) >= 1 and parts[-1] in _NOISE_TYPES:
        return "noise", parts[-1]
    if len(parts) >= 1:
        return "speaker", parts[-1]

    # Flat directory: the "__tag" filename suffix is the only remaining hint.
    stem_parts = wav.stem.split("__", 1)
    if len(stem_parts) == 2:
        return "speaker", stem_parts[1]
    return "clean", "none"


def source_query_id(wav: Path) -> str:
    """Recover ``{article}_{paragraph}_{chunk}`` from the filename."""
    stem = wav.stem.split("__")[0]
    for suffix in _NOISE_TYPES:
        if stem.endswith(f"_{suffix}"):
            stem = stem[: -len(suffix) - 1]
            break
    return stem


def collect(root: Path, args: argparse.Namespace) -> list[dict]:
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAV files under {root}")

    probe = None
    if not args.no_audio_probe:
        try:
            import soundfile as sf
            probe = sf
        except ImportError:
            print("soundfile unavailable; skipping sr/duration probe", file=sys.stderr)

    rows: list[dict] = []
    for i, wav in enumerate(wavs, 1):
        axis, severity = infer_axis_severity(wav, root)
        sr = duration = ""
        if probe is not None:
            try:
                info = probe.info(str(wav))
                sr, duration = info.samplerate, round(info.frames / info.samplerate, 4)
            except Exception as e:  # noqa: BLE001 - a bad file should not abort the walk
                print(f"  probe failed for {wav}: {e}", file=sys.stderr)
        rows.append({
            "path": str(wav.relative_to(root)),
            "source_query_id": source_query_id(wav),
            "axis": args.axis or axis,
            "severity": args.severity or severity,
            "generator": args.generator,
            "sha256": sha256_file(wav),
            "bytes": wav.stat().st_size,
            "sr": sr,
            "duration_s": duration,
        })
        if i % 500 == 0:
            print(f"  hashed {i}/{len(wavs)}", file=sys.stderr)
    return rows


FIELDS = ["path", "source_query_id", "axis", "severity", "generator",
          "sha256", "bytes", "sr", "duration_s"]


def do_verify(root: Path, manifest: Path, args: argparse.Namespace) -> int:
    with open(manifest, newline="", encoding="utf-8") as f:
        old = {r["path"]: r for r in csv.DictReader(f)}
    new = {r["path"]: r for r in collect(root, args)}

    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(p for p in set(old) & set(new)
                     if old[p]["sha256"] != new[p]["sha256"])

    print(f"\nmanifest : {manifest}")
    print(f"root     : {root}")
    print(f"  in manifest : {len(old)}")
    print(f"  on disk     : {len(new)}")
    for label, items in (("MISSING", missing), ("ADDED", added), ("CHANGED", changed)):
        if items:
            print(f"  {label}: {len(items)}")
            for p in items[:10]:
                print(f"    {p}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
    if not (missing or added or changed):
        print("  OK: byte-identical to the manifest.")
        return 0
    return 1


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if args.verify:
        sys.exit(do_verify(root, args.verify, args))

    if args.out is None:
        raise SystemExit("--out is required unless --verify is given")

    rows = collect(root, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_axis: dict[tuple[str, str], int] = {}
    for r in rows:
        by_axis[(r["axis"], r["severity"])] = by_axis.get((r["axis"], r["severity"]), 0) + 1

    print(f"\nWrote {len(rows)} rows -> {args.out}")
    print(f"Distinct source_query_id: {len({r['source_query_id'] for r in rows})}")
    print("Rows per (axis, severity):")
    for (axis, sev), n in sorted(by_axis.items()):
        print(f"  {axis:10s} {sev:24s} {n}")


if __name__ == "__main__":
    main()
