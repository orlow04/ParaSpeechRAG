# Naming map

Three names refer to overlapping things. A reader coming from the paper needs
this table to navigate the code.

| Name | What it is | Where it appears |
|---|---|---|
| **CLASP** | The retrieval *model* under test: HuBERT + EfficientNet-B7 + LaBSE with a concat fusion head. Published at ECIR 2025. | `paraspeechrag.models.fusion`, all `*.pt` checkpoints, the paper |
| **ParaSpeechRAG** | This repository: the perturbation-robustness *benchmark* wrapped around CLASP. | Repo name, Python package `paraspeechrag` |
| **SpeechRAG** | Earlier internal code name for the benchmark. Superseded. | Figure 1's legend — **this is a bug in the figure**, fix it to read CLASP or ParaSpeechRAG as appropriate |

The Python package was renamed `clasp` → `paraspeechrag` during the port. If
you are diffing against the old repo, the module moves were:

| Old | New |
|---|---|
| `clasp.audio.noise_augmentation` | `paraspeechrag.perturb.noise` |
| `clasp.evaluation.*` | `paraspeechrag.eval.*` |
| `clasp.retrieval.search` | `paraspeechrag.retrievers.search` |
| `clasp.<rest>` | `paraspeechrag.<rest>` (unchanged otherwise) |

## Metric names

The code and the paper use different words for the same quantities. This is
the mapping a replicator needs — it is also documented in the docstring of
`src/paraspeechrag/eval/ranking_metrics.py`.

| Code | Paper | Why they coincide |
|---|---|---|
| `Hits@k` | **Recall@k** | Exactly one relevant item per query, so recall at cutoff *k* is 1 iff the gold item ranks ≤ *k*. Hits@k is then the mean of that indicator — identical to Recall@k. |
| `MAP` | **MRR** | Same reason: with one relevant item, AP_i = 1/rank_i, so MAP = MRR. `compute_ranking_metrics` returns both keys with the same value. |
| `mean_rank` | mean rank (meanR) | 1-based |
| `median_rank` | median rank | 1-based |

Rank convention: `rank = 1 + |{j : score_j > score_gold}|`. Ties therefore
favour the gold item — a query where everything scores identically gets rank 1,
not rank *n*. `evaluate_matrix` and `compute_ranking_metrics` agree on this;
`evaluate_matrix_by_source` differs, it *skips* queries where a non-gold item
ties the gold score.

## Column names in built PKLs

| Key | Contents |
|---|---|
| `text` | LaBSE sentence embedding, 1024-d |
| `hubert-emb` | HuBERT-large mean-pooled over 20 s windows, 1024-d |
| `image` | EfficientNet-B7 logits over the spectrogram image, 1000-d |
| `paragraph_id` | `"{split}:{article_idx}_{paragraph_idx}"` |
| `audio_paths` | Spoken-SQuAD: `list[list[str]]`, one list per row |
| `audio_path` | VoxPopuli: `list[str]`, flat |
| `_meta` | Build provenance: pooling mode, source dirs, split sizes, encoder ids |

Split keys are inconsistent across builders: Spoken-SQuAD writes `validation`,
some others write `test`. Readers accept either (`run_retrieval_eval.py`
falls back from `test` to `validation`).
