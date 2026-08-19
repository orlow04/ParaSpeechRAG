#!/usr/bin/env python3
"""Report, per perturbation directory, how much of the SQuAD split it covers.

Recall@k depends on the size of the candidate pool. A condition whose audio
covers 91 paragraphs is scored against 91 candidates; the clean baseline over
the full validation split is scored against ~2,067. Those two Recall@1 numbers
cannot be subtracted from each other, and a ΔR@1 built from them is not a
measurement of the perturbation.

This script walks each leaf directory of perturbed audio, recovers the
``{article}_{paragraph}`` keys from the filenames, and compares against the
paragraph list in the SQuAD JSON. Run it on whichever machine holds the
authoritative audio — the local checkout may only have a sample.

Usage:
    python scripts/sanity/audit_perturbation_coverage.py \\
        --json data/datasets/spoken_squad/spoken_test-v1.1.json \\
        data/datasets/spoken_squad_emotions \\
        data/datasets/spoken_squad_seed-vc \\
        data/datasets/spoken_squad/dev_wav_noisy_by_type

Exit code 1 if any directory covers less than --min-coverage of the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", type=Path, nargs="+",
                   help="Perturbation directories to audit (walked recursively)")
    p.add_argument("--json", type=Path, required=True,
                   help="Reference SQuAD JSON defining the full split")
    p.add_argument("--min-coverage", type=float, default=1.0,
                   help="Fail if coverage is below this fraction (default 1.0 = full)")
    return p.parse_args()


def split_paragraph_keys(json_path: Path) -> set[tuple[int, int]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        (a_idx, p_idx)
        for a_idx, article in enumerate(data["data"])
        for p_idx, _ in enumerate(article["paragraphs"])
    }


def keys_in_dir(d: Path) -> tuple[set[tuple[int, int]], int]:
    """Return (paragraph keys found, wav count) for a single directory."""
    keys: set[tuple[int, int]] = set()
    n = 0
    for wav in d.glob("*.wav"):
        n += 1
        try:
            a, p, _c = wav.stem.split("__")[0].split("_")[:3]
            keys.add((int(a), int(p)))
        except (ValueError, IndexError):
            continue
    return keys, n


def leaf_dirs(root: Path) -> list[Path]:
    """Directories that directly contain WAVs."""
    out = [d for d in sorted({p.parent for p in root.rglob("*.wav")})]
    return out or [root]


def main() -> None:
    args = parse_args()
    reference = split_paragraph_keys(args.json)
    n_ref = len(reference)
    print(f"Reference split: {args.json.name} — {n_ref} paragraphs\n")

    failures = 0
    for root in args.roots:
        if not root.exists():
            print(f"{root}: MISSING\n")
            failures += 1
            continue
        print(f"### {root}")
        for d in leaf_dirs(root):
            keys, n_wav = keys_in_dir(d)
            covered = keys & reference
            frac = len(covered) / n_ref if n_ref else 0.0
            label = str(d.relative_to(root)) or "."
            status = "OK  " if frac >= args.min_coverage else "SHORT"
            print(f"  [{status}] {label:34s} {n_wav:6d} wav  "
                  f"{len(covered):5d}/{n_ref} paragraphs  ({frac:6.1%})")
            if frac < args.min_coverage:
                failures += 1
                articles = sorted({a for a, _ in covered})
                shown = articles[:12]
                more = "" if len(articles) <= 12 else f" ... (+{len(articles)-12})"
                print(f"          articles present: {shown}{more}")
            extra = keys - reference
            if extra:
                print(f"          {len(extra)} keys NOT in this split — wrong JSON, "
                      f"or audio built from the other split")
        print()

    if failures:
        print(f"{failures} directory/directories below the coverage threshold.")
        print("Any Recall@k from those is computed over a smaller candidate pool "
              "than the full-split baseline and is not directly comparable.")
        sys.exit(1)
    print("All directories cover the reference split.")


if __name__ == "__main__":
    main()
