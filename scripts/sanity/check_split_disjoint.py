#!/usr/bin/env python3
"""Assert that a built PKL's train and validation splits share nothing.

Two independent checks, because either alone can pass while the split is still
broken:

1. **paragraph_id overlap** — catches the ID-namespace collision. The keys
   ``{article}_{paragraph}`` are positions within a single SQuAD JSON, so
   ``0_0`` means different things in the train and test JSONs.
2. **audio path overlap** — catches the case the IDs miss: distinct IDs whose
   WAVs are the same files on disk. That is what happens when a launcher passes
   one converted-audio directory as both ``--train-wav-dir`` and
   ``--val-wav-dir``.

Also prints the split sizes and the retrieval pool size, so any metric reported
from this PKL is self-documenting: Recall@k is meaningless without the number
of candidates it was computed against.

Exit code 0 = disjoint, 1 = overlap found.

Usage:
    python scripts/sanity/check_split_disjoint.py <PKL> [<PKL> ...]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def _paths(split: dict) -> set[str]:
    """Flatten whichever audio-path column this PKL flavour uses."""
    out: set[str] = set()
    if "audio_paths" in split:  # Spoken-SQuAD: list[list[str]]
        for group in split["audio_paths"]:
            out.update(str(p) for p in group)
    elif "audio_path" in split:  # VoxPopuli: list[str]
        out.update(str(p) for p in split["audio_path"])
    return out


def check(pkl_path: Path) -> bool:
    with open(pkl_path, "rb") as f:
        total = pickle.load(f)

    train = total.get("train")
    val = total.get("validation") or total.get("test")
    if train is None or val is None:
        print(f"{pkl_path.name}: SKIP — missing a train or validation/test split")
        return True

    meta = total.get("_meta") or {}
    pooling = meta.get("pooling_mode", "<unknown>")

    n_train_rows = len(train.get("text", []))
    n_val_rows = len(val.get("text", []))
    val_pids = list(val.get("paragraph_id", []))
    pool = len(set(val_pids)) if val_pids else n_val_rows

    print(f"\n=== {pkl_path} ===")
    print(f"  pooling_mode        : {pooling}")
    print(f"  train rows          : {n_train_rows}")
    print(f"  validation rows     : {n_val_rows}")
    print(f"  retrieval pool size : {pool} candidates")
    if meta.get("train_wav_dir"):
        print(f"  train wav dir       : {meta['train_wav_dir']}")
        print(f"  val   wav dir       : {meta['val_wav_dir']}")

    ok = True

    train_pids = set(train.get("paragraph_id", []))
    val_pids_set = set(val_pids)
    if train_pids and val_pids_set:
        shared = train_pids & val_pids_set
        if shared:
            ok = False
            sample = sorted(shared)[:10]
            print(f"  FAIL: {len(shared)} paragraph_id in both splits, e.g. {sample}")
        else:
            print(f"  OK  : paragraph_id disjoint ({len(train_pids)} / {len(val_pids_set)})")
    else:
        print("  WARN: no paragraph_id column — ID check skipped")

    train_paths, val_paths = _paths(train), _paths(val)
    if train_paths and val_paths:
        shared_paths = train_paths & val_paths
        if shared_paths:
            ok = False
            sample = sorted(shared_paths)[:5]
            print(f"  FAIL: {len(shared_paths)} audio files used by BOTH splits, e.g.:")
            for s in sample:
                print(f"          {s}")
            print("        A file paired with two different texts means at least one "
                  "split is mislabelled.")
        else:
            print(f"  OK  : audio paths disjoint ({len(train_paths)} / {len(val_paths)})")
    else:
        print("  WARN: no audio path column — path check skipped")

    if meta.get("shared_wav_dir"):
        ok = False
        print("  FAIL: _meta records train_wav_dir == val_wav_dir")

    return ok


def main() -> None:
    args = sys.argv[1:]
    if {"-h", "--help"} & set(args):
        print(__doc__)
        sys.exit(0)
    if not args:
        raise SystemExit(__doc__)
    missing = [a for a in args if not Path(a).is_file()]
    if missing:
        raise SystemExit("Not a file: " + ", ".join(missing))
    results = [check(Path(p)) for p in args]
    if all(results):
        print(f"\nAll {len(results)} PKL(s) have disjoint splits.")
        sys.exit(0)
    print(f"\n{results.count(False)}/{len(results)} PKL(s) FAILED the split check.")
    sys.exit(1)


if __name__ == "__main__":
    main()
