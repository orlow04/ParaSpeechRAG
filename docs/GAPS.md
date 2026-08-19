# Gaps between the paper and this repository

This file is the honest inventory. It records what the code actually does,
what is missing, and which findings would change a number in the paper.

Everything here was checked against the code at port time. Line references are
into this repository.

---

## 1. Perturbation axes

The paper reports five perturbation axes. **Four have generators here; one
does not.**

The codec, rate and GenVC generators came from a second codebase
(`codigos_matheus/`) that was merged in during the port. They were never part
of the CLASP repo's history, which is why the launcher scripts in
`experiments/` know nothing about them.

| Axis | Generator | Audio | Notes |
|---|---|---|---|
| Noise (white / reverb / ambient) | ✅ `src/paraspeechrag/perturb/noise.py` | applied in-memory at eval time, **never written to disk** | Only axis implemented as a library module rather than a standalone script |
| Codec — MP3 | ✅ `scripts/perturb/codec_mp3.py` | written to `<out>/<N>kbps/` | ffmpeg + libmp3lame, 8/16/32/64/128 kbps, decoded at source rate |
| Codec — Opus | ✅ `scripts/perturb/codec_opus.py` | written to `<out>/<N>kbps/` | ffmpeg + libopus, same bitrates, decoded at **24 kHz** |
| Rate (time-stretch) | ✅ `scripts/perturb/speed.py` | written to `<out>/<factor>x/` | librosa phase vocoder, SRB 8 factors (arXiv:2403.07937) |
| Speaker — GenVC | ✅ `scripts/perturb/voice_convert_genvc.py` | written to `<out>/<ref>/` | `GenVC_large`, `top_k=15`, 6 reference voices |
| Speaker — Seed-VC | ❌ none | consumed from `data/datasets/spoken_squad_seed-vc/<voice>/` | Generation happened out-of-tree; no code, config or seed preserved |
| Emotion (Seed-VC v2) | ❌ none | consumed from `data/datasets/spoken_squad_emotions/<emotion>/<intensity>/` | Same |

Exact parameters for each axis are recorded in `configs/perturb/*.yaml`.

Consequences:

- **No perturbation axis is bit-reproducible.** GenVC samples with `top_k=15`
  and Seed-VC runs a 30-step diffusion sampler, so neither is deterministic
  even with a fixed seed; neither driver sets or logs one. The codec path
  depends on the ffmpeg/libmp3lame/libopus build, and the rate path on the
  librosa version. This is why the released **audio** is the artifact and the
  generator code is provenance. Pin it with `scripts/build_manifest.py`.
- **The Seed-VC speaker and emotion conditions cannot be regenerated at all**
  from this repository — only re-evaluated against existing audio.
- **The noise conditions have no released audio.** They are generated in-memory
  by `scripts/run_noise_robustness_eval.py`, which re-embeds the noisy waveform
  and discards it. Generation is now seeded (§3.5) but still depends on the
  ESC-50 release and the local numpy version.
- **The eval side of codec, rate and GenVC is not wired up.** The generators
  write perturbed WAV directories, but no launcher in `experiments/` builds a
  PKL from them or evaluates retrieval over them. Those directories are
  flat-per-condition (`<N>kbps/`, `<factor>x/`, `<ref>/`), which
  `build_spoken_squad_pkl.py` can consume as a `--val-wav-dir`, so wiring is
  mechanical — but it has not been done, and no retrieval results for those
  axes exist in this repo.

### 1.1 The two VC systems used different target voices — CONFIRMED

| | Seed-VC (`experiments/run_seedvc_all_voices.sh`) | GenVC (`scripts/data/download_references.py`) |
|---|---|---|
| LibriSpeech 1 | `2803-154320-0012` | `2803-154320-0012` |
| LibriSpeech 2 | `3081-166546-0023` | `3081-166546-0023` |
| LibriSpeech 3 | `6319-275224-0006` | `6319-275224-0006` |
| LibriSpeech 4 | **`1089-134686-0000`** | **`8842-302201-0002`** |
| Non-corpus | `azuma`, `trump` | `azuma_0`, `trump_0` |

Three of four LibriSpeech speakers match; the fourth does not. If Table 2
reports a mean over Seed-VC and GenVC on "the same six target voices", that
premise is false. Either regenerate one system on the other's voice set, or
report per-system results rather than the mean — which the review flagged as a
near-certain reviewer question anyway.

### 1.2 The GenVC source set is a deliberate ~3k subset

`should_include()` in `scripts/perturb/voice_convert_genvc.py` keeps only
article 0 (all paragraphs), article 10 (all), and article 11 paragraphs 0–13.
That is ~3k source audios × 6 references ≈ 18k outputs — **not** full coverage
of the dev split. See §3.2: this makes the candidate pool for the GenVC
conditions smaller than the clean baseline's by construction, and the pool size
must be reported alongside any Recall@k from it.

---

## 2. Baseline retrievers

The paper compares five systems. **Two are complete, one is half-built.**

| System | Status |
|---|---|
| CLASP-mean | ✅ `--pooling-mode mean` + `--mode candidate` |
| CLASP-chunked (max-sim) | ✅ `--pooling-mode chunked` + `--mode paragraph_grouped` |
| ASR cascade (Parakeet TDT → LaBSE) | ⚠️ **ASR half only** — see below |
| GLAP | ❌ not present |
| CLAP | ❌ not present |

### 2.1 The ASR cascade stops at transcription

`scripts/asr/transcribe.py` runs **NVIDIA Parakeet TDT 0.6b v2** (via NeMo)
over a list of audio folders defined in `configs/asr/*.yml`, writing one CSV of
`(filename, transcription)` per condition.
`scripts/asr/compute_wer.py` then scores those against the SpokenSQuAD dev
manifest with `jiwer`, producing `results/wer_results.{csv,json}`.

That gives **WER per condition**, which is a useful robustness measurement in
its own right — but it is not the retrieval baseline. The second half of the
cascade, embedding the transcripts with LaBSE and ranking them against the
paragraph corpus, does not exist. Nothing joins `transcriptions/*.csv` back to
`paragraph_id` or produces Recall@k.

`configs/asr/transcribe_all_conditions.yml` is set up for 25 conditions
(baseline + 8 speed + 5 MP3 + 5 Opus + 6 GenVC), which is a useful record of
which conditions were actually generated.

Note the paths in that config are absolute `/workspace/...` paths from the
machine it ran on; they will not resolve elsewhere.

### 2.2 No shared retriever interface

`CLASP-mean` and `CLASP-chunked` are two code paths through
`scripts/run_retrieval_eval.py` selected by flags, not two implementations of a
common protocol. Adding GLAP, CLAP and the retrieval half of the ASR cascade
means either writing that interface or adding three more branches.

Related: the paper's CLAP Recall@1 ≈ 0.000 is indistinguishable from a broken
pipeline. Without a positive control (e.g. CLAP zero-shot on ESC-50
reproducing its published accuracy), that row reads as a bug rather than a
finding. No such control exists here.

---

## 3. Findings that affect reported numbers

### 3.1 Reverberation is not evaluated at any SNR — CONFIRMED

`scripts/run_noise_robustness_eval.py` maps the SNR flag onto a decay time:

```python
if noise_type == "reverb":
    return add_reverberation(y, decay_time_ms=snr * 10)
```

A results row tagged `reverb_20.0` is a **200 ms decay tail**, not a 20 dB SNR.
The reverb series therefore does not share an axis with the white and ambient
series, and plotting all three against a common "SNR (dB)" axis mislabels the
reverb curve. Additionally `add_reverberation` peak-normalises its output to
0.95, so it changes signal gain — the additive noise functions do not.

The behaviour is preserved as-is (changing it would invalidate existing
results) but is now documented at the call site, in
`src/paraspeechrag/perturb/noise.py`, and in `configs/perturb/noise.yaml`.

### 3.2 Perturbed audio may cover a far smaller corpus than the clean baseline — NEEDS VERIFICATION ON THE GPU BOX

In the local checkout, every perturbed set covers **91 paragraphs** drawn from
articles 0 and 1 only:

```
data/datasets/spoken_squad_emotions/<emotion>/<intensity>/   444 WAVs →  91 paragraphs
data/datasets/spoken_squad/dev_wav_noisy_by_type/<type>/            →  91 paragraphs
```

The paper's clean baseline uses a 2,067-paragraph corpus. Recall@1 over 91
candidates and Recall@1 over 2,067 candidates are different quantities, and a
ΔR@1 built by subtracting one from the other does not measure the perturbation
— it mostly measures the pool-size difference.

The local copy may simply be a sample. Run this where the authoritative audio
lives:

```bash
python scripts/sanity/audit_perturbation_coverage.py \
    --json data/datasets/spoken_squad/spoken_test-v1.1.json \
    data/datasets/spoken_squad_emotions \
    data/datasets/spoken_squad_seed-vc \
    data/datasets/spoken_squad/dev_wav_noisy_by_type
```

Note that `collect_paragraph_chunks` silently drops paragraphs with no audio,
so a partial directory produces a small pool with no error — that is exactly
how this can go unnoticed. The builder now prints the pool size and warns on
shortfall, and `run_noise_robustness_eval.py` writes
`n_candidates_in_pool` into a `.meta.json` sidecar next to every results CSV.

### 3.3 Train/eval contamination in the perturbation launchers — FIXED

`experiments/run_emotions_all_voices.sh` and `run_seedvc_all_voices.sh`
previously passed the same perturbed directory as **both** `--train-wav-dir`
and `--val-wav-dir`.

Paragraph keys `{article}_{paragraph}` are positions within a *single* SQuAD
JSON, so `0_0` in `spoken_train-v1.1.json` and `0_0` in `spoken_test-v1.1.json`
are different paragraphs with different context text. With one shared
directory, both splits read the **same WAV files** paired with **different
text** — train was contaminated with eval audio, and at least one split was
mispaired.

Fixed three ways:

- `scripts/build_spoken_squad_pkl.py` now **refuses** identical train/val WAV
  directories unless `--allow-shared-wav-dir` is passed explicitly.
- `paragraph_id` is now namespaced as `"{split}:{article}_{paragraph}"`.
- Both launchers take the clean train audio via `TRAIN_WAV_DIR` and pass the
  perturbed directory only as `--val-wav-dir`, then run
  `scripts/sanity/check_split_disjoint.py` on the built PKL.

**Any PKL built before this fix is suspect.** Rebuild, or at minimum run the
checker over the existing ones.

### 3.4 `spiral_runner` has never been importable — PRE-EXISTING, NOT FIXED

`src/paraspeechrag/eval/spiral_runner.py:16` imports
`paraspeechrag.data.spiral`, which does not exist in this repository (it did
not exist in the source repo either). The module raises `ModuleNotFoundError`
at import time, so `run_retrieval_eval.py --mode spiral` and
`scripts/eval_spiral_retrieval.py` cannot run.

Ported unchanged rather than deleted, because the SPIRAL results may have been
produced with a `data/spiral.py` that was never committed. It needs
`load_spiral_jsonl` and `spiral_temporal_bin_indices`.

### 3.5 SNR uses full-signal RMS, not P.56 active level — DOCUMENTED

```python
P_signal = mean(x**2)                 # includes silence
P_noise  = P_signal / 10**(snr_db/10)
```

Verified correct to 0.01 dB *under this convention*. But Spoken-SQuAD
paragraph readings contain substantial silence, so full-signal RMS understates
the speech level and the effective speech-to-noise ratio is **higher** than the
nominal `snr_db`. Results will not line up against work that uses the ITU-T
P.56 active-speech level. Do not change the convention without regenerating
every noise result.

Both additive functions also clip to `[-1, 1]`, which at low SNR adds a
nonlinear distortion on top of the additive noise.

### 3.6 Two different training temperatures — UNRESOLVED

- `src/paraspeechrag/train/trainer.py:23` defaults to `temperature=np.log(0.07)`
  ≈ −2.66
- `scripts/train.py` passes `--temperature` defaulting to `np.log(1/0.07)`
  ≈ +2.66

Anything launched through the CLI gets the positive value; a direct call to
`train_the_model()` without an explicit temperature gets the negative one.
Which produced the released checkpoint is not recorded. Fill in
`configs/train/clasp.yaml`.

### 3.7 No global seed in training — NOT FIXED

`scripts/train.py` sets no `torch`/`numpy`/`random` seed, so training runs are
not reproducible even on identical hardware. Left alone because adding a seed
now would not reproduce any existing checkpoint; record the checkpoint hash
instead.

---

## 4. Documentation and release items still open

These are from the review and are **not** addressed by this port — they need
decisions, not code:

- **Number of evaluation queries is never stated.** Only the 2,067-paragraph
  corpus size is given. Recall@k is uninterpretable without the query count.
  `run_noise_robustness_eval.py` now emits `n_queries` into its `.meta.json`.
- **SNR levels conflict across the paper**: §3.3 lists four (20/15/10/~5),
  §4.2 lists three (20/10/5), Figure 1's axis reads 0/5/10/20. The code default
  is `20,15,10,5`; `experiments/run_noise_ambient.sh` overrides to `20,10,5`.
- **Noise classes conflict**: §3.3 describes white + reverb, §4.2 describes
  three classes including ESC-50 ambient. The code supports all three.
- **CLASP-mean MRR** is 0.971 in Table 1 and 0.972 in §5.2 / Table 2.
- **Figure 1's legend says "SpeechRAG"** — internal code name leaking into the
  figure. See `docs/NAMING.md`.
- **Figure 1's caption is a placeholder.**
- **Sampling-rate story is inconsistent** — now verifiable in code: Opus decodes
  with a forced `-ar 24000` (`scripts/perturb/codec_opus.py`), MP3 decodes at
  the source rate with no `-ar` flag (`codec_mp3.py`), the rate axis preserves
  the source rate (`speed.py`), and GenVC writes at its own config's
  `audio.sample_rate`. All of them are then resampled to 16 kHz mono by
  `paraspeechrag.inference.audio_preprocess` before reaching the retriever, so
  these differ in *resampling history*, not in the rate the model sees. State
  that explicitly rather than leaving three rates unexplained in the text.
- **Per-system VC breakdown.** Table 2 reports the mean over Seed-VC and GenVC.
  The per-voice PKLs and logs are separate artifacts, so the breakdown is
  recoverable — keep them. §1.1 makes this more pressing than a presentation
  preference: the two systems did not use the same voice set, so the mean is
  over mismatched conditions.
- **VC-control decomposition.** Neutral-normal conversion alone drops CLASP
  0.954 → 0.799, so much of the emotion table may be the conversion process
  rather than emotion. The eval code cannot currently report deltas against a
  neutral-converted baseline as well as against clean.
- **Licensing.** RAVDESS is CC BY-NC-SA 4.0 and ESC-50 is CC BY-NC 3.0. Derived
  audio inherits those terms — the benchmark cannot ship as one permissively
  licensed blob. See `docs/DATA_LICENSES.md`.
- **The "trump" target voice.** Redistributing voice-converted audio in the
  cloned voice of an identifiable living political figure is a problem for a
  public artifact. It is still in the default `VOICES` list in
  `experiments/run_seedvc_all_voices.sh`; dropping it means regenerating that
  condition and re-running Table 2, so decide before freezing.
