#!/usr/bin/env python3
"""
Download the SpokenSQuAD annotation JSON from the official GitHub repository.

Downloads the question-answer annotations (spoken_test-v1.1.json or
spoken_train-v1.1.json) needed to build the retrieval passages manifest.

Usage:
    python download_spokensquad_json.py --split test --output data/spoken_test-v1.1.json
    python download_spokensquad_json.py --split train --output data/spoken_train-v1.1.json
"""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen, Request

URLS = {
    "test": "https://raw.githubusercontent.com/chiahsuan156/Spoken-SQuAD/master/spoken_test-v1.1.json",
    "train": "https://raw.githubusercontent.com/chiahsuan156/Spoken-SQuAD/master/spoken_train-v1.1.json",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output", default="data/spoken_test-v1.1.json")
    args = parser.parse_args()

    url = URLS[args.split]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req) as r:
        content = r.read()

    out.write_bytes(content)

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_articles = len(data["data"])
    n_qas = sum(
        len(paragraph["qas"])
        for article in data["data"]
        for paragraph in article["paragraphs"]
    )

    print(f"Saved to: {out}")
    print(f"Articles: {n_articles}")
    print(f"Questions: {n_qas}")

if __name__ == "__main__":
    main()
