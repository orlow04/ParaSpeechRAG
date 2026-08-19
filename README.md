# ParaSpeechRAG

A perturbation-robustness benchmark for speech retrieval, built around
**CLASP** (Contrastive Language-Speech Pretraining, ECIR 2025).

> **Do the ported numbers still match the paper?** Audited file by file in
> [`docs/REPRODUCING_PAPER_NUMBERS.md`](docs/REPRODUCING_PAPER_NUMBERS.md).
> Short answer: everything is bit-identical except the white/ambient noise
> rows, which were never reproducible in the first place (the original drew
> from an unseeded global RNG).
>
> **Read [`docs/GAPS.md`](docs/GAPS.md) first.** Four of the five perturbation
> axes have generators here, but only two are wired through to a retrieval
> evaluation; three of the five baseline retrievers are missing entirely. That
> file is the honest inventory, and several findings in it change how the
> existing numbers should be read.

---

## Paper claim → command → artifact

Two things this table separates, because the repo does too: **generating** the
perturbed audio, and **evaluating** retrieval over it.

| Paper claim | Generate | Evaluate | Status |
|---|---|---|---|
| Table 1 — clean retrieval baseline | — | `make eval-clean` | ✅ end to end |
| Figure 1 — noise robustness | in-memory during eval | `make eval-noise` | ⚠️ runnable; reverb rows are **not** on an SNR axis ([GAPS §3.1](docs/GAPS.md)) |
| Table 2 — speaker variation | `make perturb-speaker` (GenVC only) | `make eval-speaker` (Seed-VC audio only) | ⚠️ the two halves cover **different** VC systems ([GAPS §1](docs/GAPS.md)) |
| Table 3 — emotion variation | ❌ no generator | `make eval-emotion` | ⚠️ eval only; audio produced out-of-tree |
| Codec — MP3 / Opus | `make perturb-codec` | ❌ not wired up | ⚠️ generator only |
| Rate — time stretch | `make perturb-rate` | ❌ not wired up | ⚠️ generator only |
| ASR robustness (WER) | — | `make asr-transcribe && make asr-wer` | ⚠️ WER only, no retrieval |
| ASR cascade retrieval / GLAP / CLAP | — | — | ❌ not implemented |

`make help` lists every target. Exact parameters per axis live in
`configs/perturb/*.yaml`.

## Reproduction path

The perturbed audio **is** the benchmark. Seed-VC runs a stochastic diffusion
sampler and GenVC uses top-k sampling, so the audio is not bit-reproducible
even with fixed seeds — a different torch or CUDA build yields different
waveforms and therefore different Recall@1. The reproducible path is

```
released audio  →  embeddings  →  metrics
```

which is deterministic. Generation code is provenance, not the reproduction
path. `scripts/build_manifest.py` pins the first arrow with SHA256 per file.

## Quickstart

```bash
make setup                  # uv sync --extra realdata

# 1. Freeze what the numbers are computed against
make freeze-corpus          # data/corpus/*.jsonl, data/splits/*.txt
make manifests              # data/manifests/*.csv  (SHA256 per WAV)

# 2. Verify before trusting anything
make check-coverage         # do the perturbed sets cover the full split?
make check-splits           # are train and validation actually disjoint?

# 3. Evaluate
make eval-clean
make eval-noise
```

Steps 1 and 2 are not optional. `collect_paragraph_chunks` silently drops
paragraphs with no audio, so a partial perturbation directory yields a small
candidate pool with no error — and Recall@1 over 91 candidates is not
comparable to Recall@1 over 2,067. See [GAPS §3.2](docs/GAPS.md).

## Layout

```
configs/
  perturb/{noise,codec,rate,speaker}.yaml   exact parameters per axis
  retrievers/clasp_{mean,chunked}.yaml
  train/clasp.yaml              TODOs mark unrecorded run provenance
  asr/*.yml                     Parakeet transcription job definitions
src/paraspeechrag/
  perturb/       noise generation (white / ambient / reverb)
  retrievers/    similarity search
  eval/          metrics, ranking metrics, plots
  models/        CLASP fusion encoder
  inference/     HuBERT + EfficientNet-B7 embedding, audio preprocessing
  data/          dataset wrappers, path resolution
  train/         contrastive trainer
  mseb_adapter/  CLASP encoder for the MSEB / SVQ benchmark (Python 3.12)
  rag/           SVQ end-to-end RAG (retrieval + LLM generation)
scripts/
  build_*_pkl.py       dataset → embedded PKL
  run_*_eval.py        retrieval / noise-robustness evaluation
  freeze_corpus.py     write the candidate pool to disk
  build_manifest.py    SHA256 manifest of a perturbed-audio directory
  perturb/             audio generators: codec_mp3, codec_opus, speed,
                       voice_convert_genvc  (write WAV trees; no eval wired up)
  asr/                 Parakeet TDT transcription + WER scoring
  data/                dataset and reference-voice downloaders, manifest builder
  sanity/              split-disjointness and coverage checks
experiments/           shell launchers, one per condition sweep
data/
  corpus/    frozen paragraph lists      splits/     paragraph ID lists
  manifests/ per-file checksums          datasets/   audio + built PKLs (gitignored)
results/     per-condition metrics; legacy_* are carried over from the old repo
docs/        GAPS, NAMING, DATA_LICENSES, TRAINING, EVAL, MSEB, ROADMAP
```

## Metric naming

The code says `Hits@k`; the paper says `Recall@k`. They are the same number —
there is exactly one relevant item per query, so recall at cutoff *k* is the
indicator that the gold item ranked ≤ *k*. For the same reason `MAP` equals
`MRR`, and `compute_ranking_metrics` returns both keys with one value. Full
table in [`docs/NAMING.md`](docs/NAMING.md), which also covers the
`SpeechRAG` → `ParaSpeechRAG` rename (Figure 1's legend still says
`SpeechRAG`).

## Environment

```bash
uv sync                       # base
uv sync --extra realdata      # Spoken-SQuAD / SpeechBrown  (Python 3.10)
uv sync --extra voxpopuli     # VoxPopuli
uv sync --extra rag           # SVQ end-to-end RAG (needs CUDA)
```

The MSEB / SVQ extra needs Python ≥ 3.12 and a separate virtualenv — see
[`docs/MSEB.md`](docs/MSEB.md).

## Checkpoints

```bash
huggingface-cli download llm-lab/CLASP CLASP_Concat_Final_Fusion_Encoder.pt \
  --local-dir models/checkpoints
```

Record the SHA256 in `configs/retrievers/*.yaml` — every headline number
depends on this one file.

## Licences

Code is under [`LICENSE`](LICENSE). **Data is not.** RAVDESS (CC BY-NC-SA 4.0)
and ESC-50 (CC BY-NC 3.0) are non-commercial, and derived audio inherits those
terms, so the benchmark cannot ship as one permissively licensed blob. See
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) — it also covers the
identifiable-public-figure target voice, which needs a decision before the
audio is frozen.

## Citation

```bibtex
@inproceedings{10.1007/978-3-031-88717-8_2,
  author    = {Abootorabi, Mohammad Mahdi and Asgari, Ehsaneddin},
  title     = {CLASP: Contrastive Language-Speech Pretraining for Multilingual
               Multimodal Information Retrieval},
  year      = {2025},
  publisher = {Springer-Verlag},
  doi       = {10.1007/978-3-031-88717-8_2},
  booktitle = {Advances in Information Retrieval: ECIR 2025, Lucca, Italy},
  pages     = {10-20},
}
```
