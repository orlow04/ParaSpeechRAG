#!/usr/bin/env python3
"""
Download the original SQuAD v1.1 dev set JSON from the official source.

Used as a reference to align passage texts with SpokenSQuAD audio filenames.
Saves to data/squad_dev-v1.1.json by default.

Usage:
    python download_squad_dev_json.py
"""

from pathlib import Path
from urllib.request import urlopen, Request
import json

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
out = Path("data/squad_dev-v1.1.json")
out.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading: {url}")
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
content = urlopen(req).read()
out.write_bytes(content)

data = json.loads(content.decode("utf-8"))
qas = sum(
    len(paragraph["qas"])
    for article in data["data"]
    for paragraph in article["paragraphs"]
)

print(f"Saved to: {out}")
print(f"Questions: {qas}")
