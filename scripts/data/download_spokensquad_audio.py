#!/usr/bin/env python3
"""
Download only the needed Spoken-SQuAD audio files from the remote zip.

Two modes:
  --dev   : extract from the already-downloaded dev_wav.zip  (550 files)
  --train : selectively download ~3k matching files from train_wav.zip
            using HTTP range requests — downloads ~1-2 GB instead of 41 GB.

Usage:
    pip install remotezip requests tqdm
    python download_spokensquad_audio.py --train   # recommended
    python download_spokensquad_audio.py --dev
"""

import io
import re
import sys
import struct
import time
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

AUDIO_ZIP_URL  = "http://speech.ee.ntu.edu.tw/~chiahsuan/Spoken-SQuAD/Spoken-SQuAD_audio.zip"
TRAIN_ZIP_NAME = "Spoken-SQuAD_audio/train_wav.zip"
DEV_ZIP_NAME   = "Spoken-SQuAD_audio/dev_wav.zip"
MAX_HTTP_RETRIES = 10


def should_include(filename: str) -> bool:
    stem = Path(filename).stem
    if re.match(r"^0_", stem):
        return True
    if re.match(r"^10_", stem):
        return True
    m = re.match(r"^11_(\d+)_", stem)
    if m and 0 <= int(m.group(1)) <= 13:
        return True
    return False


# ---------------------------------------------------------------------------
# Seekable HTTP range file-like object
# ---------------------------------------------------------------------------

class RemoteRangeFile(io.RawIOBase):
    """Seekable file-like object backed by HTTP range requests for a byte slice."""

    def __init__(self, url: str, abs_start: int, abs_end: int):
        self.url = url
        self._start = abs_start   # absolute byte offset in the remote file
        self._end   = abs_end
        self._size  = abs_end - abs_start + 1
        self._pos   = 0
        self._session = requests.Session()

    def readable(self)  -> bool: return True
    def seekable(self)  -> bool: return True
    def writable(self)  -> bool: return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if   whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def readinto(self, b) -> int:
        n = len(b)
        if self._pos >= self._size or n == 0:
            return 0

        abs_start = self._start + self._pos
        abs_end   = min(self._start + self._pos + n - 1, self._end)

        for attempt in range(MAX_HTTP_RETRIES):
            try:
                resp = self._session.get(
                    self.url,
                    headers={"Range": f"bytes={abs_start}-{abs_end}"},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.content
                n_read = len(data)
                b[:n_read] = data
                self._pos += n_read
                return n_read
            except Exception as e:
                if attempt < MAX_HTTP_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._pos
        if size == 0 or self._pos >= self._size:
            return b""
        buf = bytearray(size)
        n = self.readinto(buf)
        return bytes(buf[:n])


# ---------------------------------------------------------------------------
# Get the byte range of an entry inside the outer zip
# ---------------------------------------------------------------------------

def get_data_range(outer_url: str, inner_name: str):
    from remotezip import RemoteZip

    print(f"Fetching zip central directory for: {inner_name} ...")
    with RemoteZip(outer_url) as zf:
        info = zf.getinfo(inner_name)
        header_offset = info.header_offset
        compress_size = info.compress_size

    # Read the 30-byte fixed local file header to find fname/extra lengths
    resp = requests.get(
        outer_url,
        headers={"Range": f"bytes={header_offset}-{header_offset + 29}"},
        timeout=30,
    )
    resp.raise_for_status()
    hdr = resp.content
    assert struct.unpack_from("<I", hdr, 0)[0] == 0x04034B50, "Bad local file header"
    fname_len  = struct.unpack_from("<H", hdr, 26)[0]
    extra_len  = struct.unpack_from("<H", hdr, 28)[0]

    data_start = header_offset + 30 + fname_len + extra_len
    data_end   = data_start + compress_size - 1

    print(f"  size : {compress_size / 1e9:.2f} GB")
    print(f"  range: bytes {data_start}–{data_end}")
    return data_start, data_end, compress_size


# ---------------------------------------------------------------------------
# Selective download from train_wav.zip via range requests
# ---------------------------------------------------------------------------

def selective_train_download(outer_url: str, output_dir: Path):
    data_start, data_end, compress_size = get_data_range(outer_url, TRAIN_ZIP_NAME)

    print("\nOpening train_wav.zip remotely (reads central directory only) ...")
    remote_file   = RemoteRangeFile(outer_url, data_start, data_end)
    buffered_file = io.BufferedReader(remote_file, buffer_size=4 * 1024 * 1024)

    with zipfile.ZipFile(buffered_file) as inner_zf:
        all_names = inner_zf.namelist()
        matching  = [n for n in all_names if should_include(Path(n).name)]

        print(f"Total files in train_wav.zip : {len(all_names)}")
        print(f"Matching filter              : {len(matching)}")

        if not matching:
            print("Filter matched nothing — check naming convention inside the zip.")
            sample = [n for n in all_names if n.endswith(".wav")][:10]
            print("Sample WAV names:", sample)
            return

        already = sum(1 for n in matching if (output_dir / Path(n).name).exists())
        todo    = [n for n in matching if not (output_dir / Path(n).name).exists()]
        print(f"Already downloaded           : {already}")
        print(f"To download                  : {len(todo)}\n")

        errors = 0
        for entry in tqdm(todo, desc="Downloading train audio"):
            out = output_dir / Path(entry).name
            try:
                out.write_bytes(inner_zf.read(entry))
            except Exception as e:
                print(f"\n  ERROR {entry}: {e}")
                errors += 1

    print(f"\nDone: {len(todo) - errors} downloaded, {already} skipped, {errors} errors")
    print(f"Audio saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Extract from local dev_wav.zip
# ---------------------------------------------------------------------------

def extract_dev(output_dir: Path):
    local_zip = output_dir / "dev_wav.zip"
    if not local_zip.exists():
        print(f"ERROR: {local_zip} not found. Run without --dev first to download it.")
        sys.exit(1)

    print(f"Extracting from {local_zip} ...")
    with zipfile.ZipFile(local_zip) as zf:
        all_names = zf.namelist()
        matching  = [n for n in all_names if should_include(Path(n).name)]
        print(f"Total: {len(all_names)}  Matching: {len(matching)}")

        already = sum(1 for n in matching if (output_dir / Path(n).name).exists())
        errors  = 0
        for entry in tqdm(matching, desc="Extracting dev audio"):
            out = output_dir / Path(entry).name
            if out.exists():
                continue
            try:
                out.write_bytes(zf.read(entry))
            except Exception as e:
                print(f"\n  ERROR {entry}: {e}")
                errors += 1

    print(f"\nDone: {len(matching) - already - errors} extracted, {already} skipped, {errors} errors")
    print(f"Audio saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="spokensquad_cache")
    parser.add_argument("--url", default=AUDIO_ZIP_URL)
    parser.add_argument("--dev",   action="store_true",
                        help="Extract from local dev_wav.zip (550 files, already downloaded)")
    parser.add_argument("--train", action="store_true",
                        help="Selectively download matching files from train_wav.zip (~1-2 GB)")
    args = parser.parse_args()

    if not args.dev and not args.train:
        parser.error("Specify --train (recommended) or --dev")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dev:
        extract_dev(output_dir)
    else:
        selective_train_download(args.url, output_dir)


if __name__ == "__main__":
    main()
