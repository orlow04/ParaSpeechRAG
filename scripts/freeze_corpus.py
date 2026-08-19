#!/usr/bin/env python3
"""Freeze the Spoken-SQuAD retrieval corpus and split ID lists to disk.

Why this exists: every evaluation script currently rebuilds the candidate pool
at runtime by globbing a WAV directory, so the pool size depends on which files
happen to be present on that machine. Recall@k computed against a 91-candidate
pool and Recall@k against a 2,067-candidate pool are different quantities. This
script writes the pool down once so it can be checked into the repo and
diffed.

Outputs (default under ``data/``):

    corpus/spoken_squad_<split>.jsonl   one row per paragraph:
                                        {paragraph_id, context, n_chunks, sha256_text}
    splits/spoken_squad_<split>.txt     one paragraph_id per line, sorted

``paragraph_id`` is ``"{split}:{article_idx}_{paragraph_idx}"``, matching
``scripts/build_spoken_squad_pkl.py``.

With ``--wav-dir`` the corpus is restricted to paragraphs that actually have
audio there, and the shortfall is reported. Without it, the corpus is the full
split as defined by the JSON — which is what you want for the canonical frozen
corpus.

Usage:
    python scripts/freeze_corpus.py \\
        --json data/datasets/spoken_squad/spoken_test-v1.1.json \\
        --split validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, required=True,
                   help="Spoken-SQuAD JSON (spoken_train-v1.1.json / spoken_test-v1.1.json)")
    p.add_argument("--split", required=True, choices=["train", "validation"],
                   help="Split label; becomes the paragraph_id prefix")
    p.add_argument("--wav-dir", type=Path, default=None,
                   help="Optional: keep only paragraphs with audio in this directory")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data",
                   help="Root for corpus/ and splits/ (default: data/)")
    return p.parse_args()


def _audio_index(wav_dir: Path) -> dict[tuple[int, int], int]:
    """Map (article_idx, paragraph_idx) -> chunk count found in wav_dir."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for wav in wav_dir.glob("*.wav"):
        try:
            a, p, _c = wav.stem.split("__")[0].split("_")
            counts[(int(a), int(p))] += 1
        except (ValueError, IndexError):
            continue
    return counts


def main() -> None:
    args = parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    counts = _audio_index(args.wav_dir) if args.wav_dir else None

    rows: list[dict] = []
    n_total = 0
    for a_idx, article in enumerate(data["data"]):
        for p_idx, para in enumerate(article["paragraphs"]):
            n_total += 1
            n_chunks = counts.get((a_idx, p_idx), 0) if counts is not None else None
            if counts is not None and not n_chunks:
                continue
            context = para["context"]
            rows.append({
                "paragraph_id": f"{args.split}:{a_idx}_{p_idx}",
                "context": context,
                "n_chunks": n_chunks,
                "sha256_text": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            })

    corpus_dir = args.out_dir / "corpus"
    splits_dir = args.out_dir / "splits"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_dir / f"spoken_squad_{args.split}.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ids_path = splits_dir / f"spoken_squad_{args.split}.txt"
    ids_path.write_text(
        "\n".join(sorted(r["paragraph_id"] for r in rows)) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(rows)} paragraphs -> {corpus_path}")
    print(f"Wrote {len(rows)} ids        -> {ids_path}")
    print(f"Corpus / retrieval pool size : {len(rows)}")
    if counts is not None and len(rows) < n_total:
        print(
            f"NOTE: {n_total - len(rows)} of {n_total} paragraphs in {args.json.name} "
            f"have no audio in {args.wav_dir}. Numbers from this pool are NOT "
            f"comparable to numbers from the full {n_total}-candidate pool."
        )


if __name__ == "__main__":
    main()
