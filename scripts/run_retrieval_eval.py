#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paraspeechrag.config.settings import get_default_device
from paraspeechrag.data.datasets import TestDataset, build_test_metadata
from paraspeechrag.eval.metrics import (
    evaluate_matrix,
    evaluate_matrix_by_source,
    evaluate_model_on_candidates,
    evaluate_model_on_paragraph_groups,
)
from paraspeechrag.eval.ranking_metrics import compute_ranking_metrics, similarity_matrix_to_rows
from paraspeechrag.eval.retrieval_plots import save_retrieval_plot
from paraspeechrag.inference.pipeline import load_model
from paraspeechrag.retrievers.search import build_similarity_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLASP retrieval evaluation.")
    parser.add_argument(
        "--mode",
        choices=["candidate", "matrix", "paragraph_grouped"],
        default="candidate",
        help=(
            "candidate: pickle dataset, one sample per row, ranked against a "
            "candidate pool. matrix: full similarity matrix from precomputed "
            "embeddings. paragraph_grouped: 'chunked' pickle with paragraph_id; "
            "max-sim per paragraph."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a total_dataset pickle",
    )
    parser.add_argument("--model-path", help="Required for candidate and paragraph_grouped modes")
    parser.add_argument("--audio-key", default="hubert-emb")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--emb-key", default="clasp_emb", help="Embedding key for matrix mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--by-source", action="store_true")
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="If set (matrix mode only), save retrieval metrics plot as PNG to this path.",
    )
    parser.add_argument(
        "--retrieval-plot-dir",
        type=Path,
        default=None,
        help="If set (matrix mode), save retrieval_summary.png under this directory.",
    )
    parser.add_argument(
        "--hits-k",
        type=str,
        default="1,5,10,50",
        help="Comma-separated K values for Hits@K in ranking metrics and plot.",
    )
    parser.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    return parser.parse_args()


def _parse_hits_k(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    device = get_default_device()

    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)

    # support PKLs that use 'validation' instead of 'test'
    if "test" not in total_dataset and "validation" in total_dataset:
        total_dataset["test"] = total_dataset["validation"]

    if args.mode == "candidate":
        if args.plot_out is not None:
            print(
                "Warning: --plot-out is ignored in candidate mode (full similarity matrix required).",
                file=sys.stderr,
            )
        if args.retrieval_plot_dir is not None:
            print(
                "Warning: --retrieval-plot-dir is ignored in candidate mode.",
                file=sys.stderr,
            )
        if not args.model_path:
            raise ValueError("--model-path is required for candidate mode")

        test_len_data = len(total_dataset["test"][args.text_key])
        test_metadata = build_test_metadata(test_len_data, args.num_candidates)
        test_dataset = TestDataset(total_dataset["test"], test_metadata, args.audio_key, args.text_key)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        model = load_model(args.model_path, device)
        results = evaluate_model_on_candidates(model, test_loader, device, threshold=args.threshold)
        print(results)
        return

    if args.mode == "paragraph_grouped":
        if not args.model_path:
            raise ValueError("--model-path is required for paragraph_grouped mode")
        if "paragraph_id" not in total_dataset["test"]:
            raise KeyError(
                "PKL 'test/validation' split does not contain 'paragraph_id'. "
                "Re-build with --pooling-mode chunked."
            )
        model = load_model(args.model_path, device)
        ks = _parse_hits_k(args.hits_k)
        results = evaluate_model_on_paragraph_groups(
            model,
            total_dataset["test"],
            device,
            audio_key=args.audio_key,
            text_key=args.text_key,
            ks=ks,
            batch_size=max(1, args.batch_size * 16),
        )
        print(results)
        return

    query_embeddings = total_dataset["test"][args.emb_key]
    candidate_embeddings = total_dataset["test"][args.text_key]
    similarity_matrix = build_similarity_matrix(query_embeddings, candidate_embeddings)

    if args.by_source:
        results = evaluate_matrix_by_source(
            similarity_matrix,
            total_dataset["test"]["source"],
            threshold=args.threshold,
        )
    else:
        results = evaluate_matrix(similarity_matrix, threshold=args.threshold)
    print(results)

    ks = _parse_hits_k(args.hits_k)
    rows = similarity_matrix_to_rows(similarity_matrix)
    ranking_metrics, ranks = compute_ranking_metrics(rows, ks=ks)
    print("Ranking metrics:", ranking_metrics)

    if args.plot_out is not None:
        save_retrieval_plot(ranking_metrics, ranks, args.plot_out, ks=ks)

    if args.retrieval_plot_dir is not None:
        out_dir = Path(args.retrieval_plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_retrieval_plot(
            ranking_metrics,
            ranks,
            out_dir / "retrieval_summary.png",
            ks=ks,
            title="Matrix retrieval",
            hits_display="percent",
        )


if __name__ == "__main__":
    main()
