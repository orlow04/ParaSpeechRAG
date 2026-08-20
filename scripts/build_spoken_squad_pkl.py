#!/usr/bin/env python3
"""Build `total_dataset_spoken_squad.pkl` with correct text/audio pairing:

text  = ``paragraphs[*]["context"]`` (the transcript)
audio = the reading of that context, split across chunks ``{a}_{p}_*.wav``.

Pooling modes (the two CLASP variants in the paper):

* ``mean``     — one sample per paragraph. Concatenates the paragraph WAVs
                 into a single waveform, then HuBERT (mean-pooled over 20 s
                 chunks) and EfficientNet-B7 (mean-pooled over windows).
* ``chunked``  — one sample per chunk. The paragraph text is replicated
                 across its chunks; the ``paragraph_id`` column drives
                 max-sim grouping at eval time (CLASP-chunked).

Output (per split):
    text         : list[Tensor[1024]]    — LaBSE sentence embedding
    hubert-emb   : list[Tensor[1024]]    — HuBERT, mean-pooled over 20 s chunks
    image        : list[Tensor[1000]]    — EfficientNet-B7 logits (mean)
    paragraph_id : list[str]             — "{split}:{article_idx}_{paragraph_idx}"
    audio_paths  : list[list[str]]       — source WAVs backing each row

Requires: `uv sync --extra realdata`
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paraspeechrag.config.settings import get_default_device
from paraspeechrag.inference.audio_preprocess import load_mono_16k_padded
from paraspeechrag.inference.embed_audio import hubert_numpy_waveform
from paraspeechrag.inference.spectrogram_image import (
    efficientnet_embedding_from_waveform,
    load_efficientnet_b7,
)


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

def collect_paragraph_chunks(json_path: Path, wav_dir: Path, split: str) -> list[dict]:
    """For each paragraph, return {paragraph_id, context, wav_paths} sorted by chunk index.

    Paragraphs with no matching WAV are dropped, so the returned length is the
    *audio* coverage of ``wav_dir``, not the size of the SQuAD split.

    ``paragraph_id`` is ``"{split}:{article_idx}_{paragraph_idx}"``. The
    ``{split}:`` prefix is load-bearing: ``{a}_{p}`` are positions within a
    *single* SQuAD JSON, so ``0_0`` in ``spoken_train-v1.1.json`` and ``0_0`` in
    ``spoken_test-v1.1.json`` are different paragraphs with different context
    text. Without the prefix the two splits share an ID namespace, and
    paragraph-grouped retrieval merges chunks from both into one candidate.

    Filenames are matched as ``{a}_{p}_{c}[__{tag}].wav`` — the optional
    ``__{tag}`` suffix carries the perturbation label (voice, emotion,
    intensity) in the converted sets and is ignored for pairing.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # (a_idx, p_idx) -> [(chunk_idx, path), ...]
    per_paragraph: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)

    for wav in wav_dir.glob("*.wav"):
        try:
            chunk_part = wav.stem.split("__")[0]  # "0_0_0"
            a, p, c = chunk_part.split("_")
            per_paragraph[(int(a), int(p))].append((int(c), str(wav)))
        except (ValueError, IndexError):
            continue

    for k in per_paragraph:
        per_paragraph[k].sort(key=lambda x: x[0])

    out: list[dict] = []
    articles = data["data"]
    n_total = 0
    for a_idx, article in enumerate(articles):
        for p_idx, para in enumerate(article["paragraphs"]):
            n_total += 1
            chunks = per_paragraph.get((a_idx, p_idx))
            if not chunks:
                continue
            out.append({
                "paragraph_id": f"{split}:{a_idx}_{p_idx}",
                "context": para["context"],
                "wav_paths": [c[1] for c in chunks],
            })

    print(
        f"  [{split}] {len(out)}/{n_total} paragraphs in {json_path.name} have audio "
        f"in {wav_dir}"
    )
    if out and len(out) < n_total:
        print(
            f"  [{split}] WARNING: retrieval pool is {len(out)} candidates, not "
            f"{n_total}. Recall@k is not comparable against a run with a "
            f"different pool size."
        )
    return out


# --------------------------------------------------------------------------- #
# Embedding extraction                                                        #
# --------------------------------------------------------------------------- #

def _concat_waveforms(paths: list[str]) -> np.ndarray:
    pieces = [load_mono_16k_padded(p) for p in paths]
    if not pieces:
        return np.zeros(16_000, dtype=np.float32)
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in pieces])


def build_split_dict_mean(
    paragraphs: list[dict],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    *,
    chunk_samples: int,
    chunk_batch_size: int,
    text_batch_size: int,
) -> dict:
    """CLASP-mean: one sample per paragraph, mean-pooled over chunks."""
    texts: list[str] = []
    hubert_list: list[torch.Tensor] = []
    image_list: list[torch.Tensor] = []
    paragraph_ids: list[str] = []
    audio_paths: list[list[str]] = []

    for para in tqdm(paragraphs, desc="paragraphs (mean)"):
        wav = _concat_waveforms(para["wav_paths"])
        h = hubert_numpy_waveform(
            wav, hubert_processor, hubert_model, device,
            chunk_samples=chunk_samples,
            chunk_batch_size=chunk_batch_size,
            pooling="mean",
        )
        s = efficientnet_embedding_from_waveform(
            wav, vision_model, vision_preprocess, device,
            chunk_samples=chunk_samples,
            chunk_batch_size=chunk_batch_size,
            pooling="mean",
        )
        hubert_list.append(h.detach().cpu().float().contiguous())
        image_list.append(s.detach().cpu().float().contiguous())
        texts.append(para["context"])
        paragraph_ids.append(para["paragraph_id"])
        audio_paths.append(list(para["wav_paths"]))

    text_emb = sentence_model.encode(
        texts, batch_size=text_batch_size, convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    text_list = [text_emb[j].detach().cpu().float() for j in range(text_emb.size(0))]

    return {
        "text": text_list,
        "hubert-emb": hubert_list,
        "image": image_list,
        "paragraph_id": paragraph_ids,
        "audio_paths": audio_paths,
    }


def build_split_dict_chunked(
    paragraphs: list[dict],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    *,
    chunk_samples: int,
    chunk_batch_size: int,
    text_batch_size: int,
) -> dict:
    """CLASP-chunked: one sample per chunk, paragraph text replicated."""
    flat_texts: list[str] = []
    hubert_list: list[torch.Tensor] = []
    image_list: list[torch.Tensor] = []
    paragraph_ids: list[str] = []
    audio_paths: list[list[str]] = []

    for para in tqdm(paragraphs, desc="paragraphs (chunked)"):
        for wav_path in para["wav_paths"]:
            wav = load_mono_16k_padded(wav_path)
            h = hubert_numpy_waveform(
                wav, hubert_processor, hubert_model, device,
                chunk_samples=chunk_samples,
                chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            s = efficientnet_embedding_from_waveform(
                wav, vision_model, vision_preprocess, device,
                chunk_samples=chunk_samples,
                chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            hubert_list.append(h.detach().cpu().float().contiguous())
            image_list.append(s.detach().cpu().float().contiguous())
            flat_texts.append(para["context"])
            paragraph_ids.append(para["paragraph_id"])
            audio_paths.append([wav_path])

    # encode com dedup p/ economizar — mas mantém ordem original
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    for t in flat_texts:
        if t not in text_index:
            text_index[t] = len(unique_texts)
            unique_texts.append(t)
    encoded = sentence_model.encode(
        unique_texts, batch_size=text_batch_size, convert_to_tensor=True,
        show_progress_bar=len(unique_texts) > 32,
    )
    text_list = [encoded[text_index[t]].detach().cpu().float().clone() for t in flat_texts]

    return {
        "text": text_list,
        "hubert-emb": hubert_list,
        "image": image_list,
        "paragraph_id": paragraph_ids,
        "audio_paths": audio_paths,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--train-wav-dir", type=Path, required=True)
    p.add_argument("--val-json", type=Path, required=True)
    p.add_argument("--val-wav-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Caminho do PKL de saída")
    p.add_argument(
        "--pooling-mode",
        choices=["mean", "chunked"],
        required=True,
        help=(
            "mean: 1 amostra/parágrafo (concat+mean-pool). "
            "chunked: 1 amostra/chunk com paragraph_id (eval max-sim)."
        ),
    )
    p.add_argument("--max-train-paragraphs", type=int, default=None,
                   help="Cap the number of train paragraphs (debug only)")
    p.add_argument("--max-val-paragraphs", type=int, default=None,
                   help="Cap the number of validation paragraphs (debug only)")
    p.add_argument(
        "--allow-shared-wav-dir",
        action="store_true",
        help=(
            "Permit --train-wav-dir == --val-wav-dir. Off by default because a "
            "shared directory makes both splits read the SAME audio files under "
            "colliding {a}_{p} indices, contaminating train with eval audio. "
            "Only pass this if you genuinely intend it and understand why."
        ),
    )
    p.add_argument("--device", default=None)
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--chunk-samples", type=int, default=320_000,
                   help="Janela em amostras (16 kHz) para HuBERT/EfficientNet")
    p.add_argument("--chunk-batch-size", type=int, default=1)
    p.add_argument("--text-batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"Device: {device}  |  pooling-mode: {args.pooling_mode}")

    if args.train_wav_dir.resolve() == args.val_wav_dir.resolve() and not args.allow_shared_wav_dir:
        raise SystemExit(
            f"--train-wav-dir and --val-wav-dir are the same directory:\n"
            f"  {args.train_wav_dir.resolve()}\n"
            f"Paragraph keys {{article}}_{{paragraph}} are positional within each SQuAD "
            f"JSON, so a shared directory makes both splits read the SAME WAV files "
            f"while pairing them with DIFFERENT context text. That contaminates train "
            f"with eval audio and mispairs at least one of the two splits.\n"
            f"Point the two flags at separate directories, or pass "
            f"--allow-shared-wav-dir if you truly intend this."
        )

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()
    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    builder = (
        build_split_dict_mean if args.pooling_mode == "mean" else build_split_dict_chunked
    )

    print("\n[1/2] Train …")
    train_paragraphs = collect_paragraph_chunks(args.train_json, args.train_wav_dir, "train")
    if args.max_train_paragraphs is not None:
        train_paragraphs = train_paragraphs[: args.max_train_paragraphs]
    train_split = builder(
        train_paragraphs, hubert_processor, hubert_model,
        sentence_model, vision_model, vision_preprocess, device,
        chunk_samples=args.chunk_samples,
        chunk_batch_size=args.chunk_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print(f"  {len(train_split['text'])} train rows generated")

    print("\n[2/2] Validation …")
    val_paragraphs = collect_paragraph_chunks(args.val_json, args.val_wav_dir, "validation")
    if args.max_val_paragraphs is not None:
        val_paragraphs = val_paragraphs[: args.max_val_paragraphs]
    val_split = builder(
        val_paragraphs, hubert_processor, hubert_model,
        sentence_model, vision_model, vision_preprocess, device,
        chunk_samples=args.chunk_samples,
        chunk_batch_size=args.chunk_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print(f"  {len(val_split['text'])} validation rows generated")

    # Self-documenting provenance. Numbers reported off this PKL are only
    # interpretable alongside the pool size and the audio source they came from.
    total_dataset = {
        "train": train_split,
        "validation": val_split,
        "_meta": {
            "pooling_mode": args.pooling_mode,
            "train_json": str(args.train_json),
            "val_json": str(args.val_json),
            "train_wav_dir": str(args.train_wav_dir),
            "val_wav_dir": str(args.val_wav_dir),
            "shared_wav_dir": args.train_wav_dir.resolve() == args.val_wav_dir.resolve(),
            "n_train_paragraphs": len(train_paragraphs),
            "n_val_paragraphs": len(val_paragraphs),
            "n_train_rows": len(train_split["text"]),
            "n_val_rows": len(val_split["text"]),
            "hubert_model": args.hubert_model,
            "sentence_transformer": args.sentence_transformer,
            "chunk_samples": args.chunk_samples,
        },
    }

    overlap = set(train_split["paragraph_id"]) & set(val_split["paragraph_id"])
    if overlap:
        print(
            f"\nWARNING: {len(overlap)} paragraph_id values appear in BOTH splits. "
            f"Run scripts/sanity/check_split_disjoint.py for detail."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"\nPKL salvo em {args.output}")


if __name__ == "__main__":
    main()
