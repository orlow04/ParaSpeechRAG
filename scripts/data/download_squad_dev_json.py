#!/usr/bin/env python3
"""Download the original SQuAD v1.1 dev set JSON from the official source.

Used as a reference to align passage texts with SpokenSQuAD audio filenames.

Usage:
    python scripts/data/download_squad_dev_json.py
    python scripts/data/download_squad_dev_json.py --out data/squad_dev-v1.1.json
"""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
DEFAULT_OUT = Path("data/squad_dev-v1.1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Destination path (default: {DEFAULT_OUT})")
    p.add_argument("--url", default=URL, help="Source URL")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the destination already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"Already present, skipping: {args.out}  (use --force to re-download)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {args.url}")
    req = Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    content = urlopen(req).read()
    args.out.write_bytes(content)

    data = json.loads(content.decode("utf-8"))
    qas = sum(
        len(paragraph["qas"])
        for article in data["data"]
        for paragraph in article["paragraphs"]
    )
    paragraphs = sum(len(a["paragraphs"]) for a in data["data"])

    print(f"Saved to  : {args.out}")
    print(f"Articles  : {len(data['data'])}")
    print(f"Paragraphs: {paragraphs}")
    print(f"Questions : {qas}")


# Guard matters here: without it, merely importing this module — or running it
# with --help — performs a 4.9 MB download as a side effect.
if __name__ == "__main__":
    main()
