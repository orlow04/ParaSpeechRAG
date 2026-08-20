# Reproducing the paper's numbers after the port

Audited by diffing every ported file against its pre-port original.

## Summary

| Path | Deviation from the paper |
|---|---|
| Clean retrieval (Table 1) | **none** — bit-identical code |
| Speaker / emotion retrieval (Tables 2, 3) | **none** — bit-identical code |
| Noise: **reverb** rows | **none** — deterministic, no RNG |
| Noise: **white** and **ambient** rows | **unavoidable, and was already unavoidable before the port** — see below |
| Training | **none** — `trainer.py` and `train.py` unchanged |
| Dependency versions | **none** — zero drift, verified below |

## What was verified

**20 library modules: zero logic changes.** `diff` against the originals shows
only `from clasp.` → `from paraspeechrag.` import-line renames in
`config/settings.py`, `models/fusion.py`, `data/*`, `inference/*`,
`retrievers/search.py`, `eval/metrics.py`, `eval/ranking_metrics.py`,
`eval/qa_metrics.py`, `eval/retrieval_plots.py`, `eval/spiral_max_sim.py`,
`eval/spiral_runner.py`, `train/trainer.py`, `mseb_adapter/`, `rag/`.

Nothing in the scoring path was touched: `evaluate_model_on_candidates`,
`evaluate_matrix`, `evaluate_matrix_by_source`,
`evaluate_model_on_paragraph_groups`, `compute_ranking_metrics`, the rank
convention, and the tie-breaking rule are all byte-identical.

**11 of 14 scripts: import renames only** — including `run_retrieval_eval.py`
and `train.py`, which produce Tables 1–3.

**Dependency versions: zero drift.** Re-locking changed no numerically relevant
version:

```
torch 2.11.0  numpy 2.2.6  scipy 1.15.3  librosa 0.11.0  transformers 5.4.0
scikit-learn 1.7.2  soundfile 0.13.1  torchvision 0.26.0  numba 0.65.0
```

identical in both lockfiles, and still identical after the scope trim.

Two non-numeric packages did move when the `rag` extra was dropped (it had
pulled `accelerate`, which floored them): `packaging` 26.0 → 24.2 and `fsspec`
2026.2.0 → 2025.12.0. Neither participates in any computation — `packaging`
parses version strings, `fsspec` abstracts filesystem access. Verified by
isolating the cause: removing the `vc` extra does not change them.

The `asr` dependencies (NeMo, jiwer) live in `requirements-asr.txt`, not as an
extra, precisely to keep them out of this resolution — adding NeMo to
`pyproject.toml` dragged `fsspec` and `packaging` down for *every* environment,
extras or not.

**`paragraph_id` prefixing: zero numeric effect.** `evaluate_model_on_paragraph_groups`
groups rows by `paragraph_id` in first-seen order, so prefixing every ID in a
split with the same string is a bijective relabeling. Verified by running the
function on identical data under both schemes:

```
MRR 0.13004413887821736 == 0.13004413887821736
mean_rank 19.725 == 19.725     Hits@1 0.05 == 0.05
Hits@5 0.15 == 0.15            Hits@10 0.225 == 0.225
ALL METRICS IDENTICAL: True
```

## The one real deviation: white and ambient noise

The change was `np.random.randn` → `rng.standard_normal` and
`np.random.randint` → `rng.integers`. **The arithmetic is unchanged** — same
SNR formula, same power scaling, same clipping. Only the source of the random
draw differs (legacy MT19937 vs PCG64).

This changes the noise waveform, and therefore the white and ambient rows.

**But those rows were never reproducible.** The original code drew from the
unseeded global `numpy` RNG, and no script in the repo called
`np.random.seed()` on that path. Running the *original* function twice on the
same input:

```
white  identical across two runs?  False
max abs difference:                0.1627
reverb identical across two runs?  True
```

So the paper's white/ambient numbers came from one particular unseeded run and
cannot be reproduced exactly by any code — including the original. The port
does not introduce the deviation; it makes future runs reproducible, which the
original could not be.

Reverb is unaffected: it uses no RNG and is bit-identical.

### If you need the original RNG stream anyway

Re-running with the legacy generator will still not match the paper (the
original run's stream is unrecoverable), but if you want the old code path
verbatim, revert three lines in `src/paraspeechrag/perturb/noise.py`:

```python
noise = np.random.randn(len(audio)).astype(np.float32)          # add_white_noise
start_idx = np.random.randint(0, len(noise_audio) - len(audio) + 1)  # add_ambient_noise
wav_path = esc50_files[np.random.randint(len(esc50_files))]     # load_esc50_clip
```

The honest fix is the opposite direction: report the noise results as produced
by the seeded pipeline, state the seeds (`--ambient-seed`, `--noise-seed`), and
note in the paper that they supersede an unseeded earlier run.

## Guards that could block a reproduction run

Two checks were added. Neither alters a number, but both can stop a run:

1. **`build_spoken_squad_pkl.py` refuses `--train-wav-dir == --val-wav-dir`.**
   The original launchers did exactly this. To reproduce that behaviour, pass
   `--allow-shared-wav-dir`. Note this reproduces a genuine contamination bug
   (docs/GAPS.md §3.3) — do it only to check what the old numbers were.
2. **The launchers run `check_split_disjoint.py`.** It is advisory: it prints a
   warning and continues, never aborting the eval.

## Reproducing the pre-port launcher behaviour exactly

The ported launchers changed one thing: `--train-wav-dir` now points at clean
train audio instead of the perturbed directory. **This does not affect any
reported metric** — evaluation reads only the validation split, and
`--val-wav-dir` is unchanged. It changes only the unused train split in the
PKL, and shortens build time.

To restore the old command verbatim:

```bash
python scripts/build_spoken_squad_pkl.py \
    --train-json "$TRAIN_JSON" --train-wav-dir "$WAV_DIR" \
    --val-json   "$VAL_JSON"   --val-wav-dir   "$WAV_DIR" \
    --output "$PKL" --pooling-mode "$POOLING_MODE" \
    --allow-shared-wav-dir
```

## Checklist before comparing against the paper

- [ ] Confirm the checkpoint SHA256 matches the one used for the paper —
      a different checkpoint moves every number and nothing here detects that
- [ ] Confirm the candidate pool size matches (`n_candidates_in_pool` in the
      `.meta.json` sidecar); see docs/GAPS.md §3.2
- [ ] Expect white/ambient noise rows to differ; expect everything else to match
- [ ] Reverb rows should match exactly — if they do not, something other than
      this port changed
