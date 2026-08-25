# Gaps between the paper and this repository

This file is the honest inventory. It records what the code actually does,
what is missing, and which findings would change a number in the paper.

Everything here was checked against the code at port time. Line references are
into this repository.

---

## 1. Perturbation axes

The paper reports five perturbation axes. **All five now have generators
here.** Two of them are still not wired to any evaluation.

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
| Speaker — Seed-VC | ✅ `scripts/perturb/seedvc/batch_convert.py` | consumed from `data/datasets/spoken_squad_seed-vc/<voice>/` | 30 diffusion steps, CFG 0.7, length-adjust 1.0. Runs from a Seed-VC checkout — see §1.3 |
| Emotion (Seed-VC v2) | ✅ `scripts/perturb/seedvc/batch_convert_v2.py` | consumed from `data/datasets/spoken_squad_emotions/<emotion>/<intensity>/` | `convert_style=True`, RAVDESS Actor 01, 7 conditions — see §1.3 |

Exact parameters for each axis are recorded in `configs/perturb/*.yaml`.

Consequences:

- **No perturbation axis is bit-reproducible.** GenVC samples with `top_k=15`
  and Seed-VC runs a 30-step diffusion sampler, so neither is deterministic
  even with a fixed seed; neither driver sets or logs one. The codec path
  depends on the ffmpeg/libmp3lame/libopus build, and the rate path on the
  librosa version. This is why the released **audio** is the artifact and the
  generator code is provenance. Pin it with `scripts/build_manifest.py`.
- **The Seed-VC generators must run from a Seed-VC checkout**, not from this
  repo — they import upstream modules and read `configs/v2/vc_wrapper.yaml`
  relative to that root. See `scripts/perturb/seedvc/README.md`.
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

### 1.1 GenVC used a voice the paper never mentions — CONFIRMED AGAINST THE PAPER

§4.4 states the target speakers are "four LibriSpeech voices (IDs 1089, 2803,
3081, and 6319) alongside two additional Seed-VC reference voices (Azuma and
Trump)". Table 2 lists exactly those six rows.

| | Paper §4.4 / Table 2 | Seed-VC launcher | GenVC `download_references.py` |
|---|---|---|---|
| LibriSpeech 1 | 2803 | `2803-154320-0012` | `2803-154320-0012` |
| LibriSpeech 2 | 3081 | `3081-166546-0023` | `3081-166546-0023` |
| LibriSpeech 3 | 6319 | `6319-275224-0006` | `6319-275224-0006` |
| LibriSpeech 4 | **1089** | `1089-134686-0000` | **`8842-302201-0002`** |
| Non-corpus | Azuma, Trump | `azuma`, `trump` | `azuma_0`, `trump_0` |

Seed-VC matches the paper. **GenVC does not**: it converted to speaker 8842,
which appears nowhere in the paper, and never converted to 1089.

Table 2's caption reads "Mean of Models" — i.e. averaged over Seed-VC and
GenVC. If GenVC never produced a 1089 condition, the `1089` row cannot be a
mean over both systems, and the other rows average over a voice set that
differs between the two systems. Either regenerate GenVC on 1089, or report
per-system results instead of the mean and say which voices each system used.

### 1.2 The GenVC source set is a deliberate ~3k subset

`should_include()` in `scripts/perturb/voice_convert_genvc.py` keeps only
article 0 (all paragraphs), article 10 (all), and article 11 paragraphs 0–13.
That is ~3k source audios × 6 references ≈ 18k outputs — **not** full coverage
of the dev split. See §3.2: this makes the candidate pool for the GenVC
conditions smaller than the clean baseline's by construction, and the pool size
must be reported alongside any Recall@k from it.

### 1.3 Seed-VC generators, vendored and verified

`scripts/perturb/seedvc/` holds the project-specific scripts from a Seed-VC
checkout (`seed-vc-test/`, ~1.1 GB, kept outside this repo). Upstream Seed-VC
code, checkpoints and reference audio are deliberately not vendored — see that
folder's README for why and for how to run them.

Reading this code settles four open questions:

| Paper claim | Verdict |
|---|---|
| §3.3 Seed-VC: 30 diffusion steps, CFG 0.7, length-adjust 1.0 | ✅ confirmed (`batch_convert.py:91-93` defaults) |
| §4.4 speaker targets 1089 / 2803 / 3081 / 6319 + Azuma + Trump | ✅ confirmed — `references_neutral/` holds exactly those six, **including 1089** |
| §4.5 emotion: Seed-VC v2, Actor 01, four categories, seven conditions | ✅ confirmed — `run_ravdess_actor01_background.sh` passes `--target-actors 1 --target-emotions angry,happy,neutral,sad --convert-style true`; `references_ravdess/` gives neutral×1 + happy×2 + sad×2 + angry×2 = 7 |
| §3.3 / §4.5: converted audio is 16 kHz | ❌ **false** — see below |

**The 16 kHz claim is wrong.** `batch_convert.py:132` is

```python
sr = 44100 if args.f0_condition else 22050
```

`--f0-condition` defaults to `False`, and the result is written with
`torchaudio.save(out, audio_tensor, out_sr)` where `out_sr = sr`. Seed-VC
writes **22,050 Hz and never resamples**, which matches the released audio
(every WAV under `data/datasets/spoken_squad_emotions/*/*/` is 22,050 Hz).
Results are unaffected — `load_mono_16k_padded` resamples on the way into the
retriever — but §4.5's "All converted audio is resampled to 16 kHz" and §3.3's
"output → 16 kHz" both need correcting.

This also sharpens §1.1: Seed-VC's reference set matches the paper exactly, so
GenVC's `8842-302201-0002` is unambiguously the outlier, not a documentation
slip on the Seed-VC side.

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

### 3.1 The reverb axis is inverted and near-inert, and §5.3 interprets the artifact — CONFIRMED AGAINST THE PAPER

This is the most consequential finding. It is not just a labelling problem.

`scripts/run_noise_robustness_eval.py` maps the SNR flag onto a decay time:

```python
if noise_type == "reverb":
    return add_reverberation(y, decay_time_ms=snr * 10)
```

So the sweep is:

| Results row | Actual decay tail | Early reflection? | RMS deviation from clean |
|---|---|---|---|
| `reverb_20.0` | 200 ms | yes | 0.2965 |
| `reverb_15.0` | 150 ms | yes | 0.2936 |
| `reverb_10.0` | 100 ms | yes | 0.2854 |
| `reverb_5.0`  |  50 ms | **no** | 0.2824 |

Three separate problems:

1. **Severity runs backwards.** As the label falls 20 → 5 — which every reader
   parses as *increasing* degradation — the reverb tail *shortens*, so the
   perturbation gets *milder*.
2. **The swept parameter barely does anything.** Deviation moves only 5% across
   the whole sweep (0.2965 → 0.2824). Compare white noise over the same labels:
   0.0100 → 0.0560, a 5.6× increase. The reverb curve is flat because the knob
   is inert in this range, not because reverberation saturates.
3. **The endpoint is structurally different.** At 50 ms, `early_idx` (800
   samples) is not `< len(rir)` (800), so the early reflection is silently
   dropped. The mildest condition uses a different impulse response shape from
   the other three.

The large *constant* offset — the paper reports Recall@1 pinned between 0.420
and 0.435 across all reverb severities — is consistent with
`add_reverberation` peak-normalising its output to 0.95. That is a gain change
the additive noise functions never apply, and it dominates the transform.

**§5.3 of the paper explains the flat curve physically:** "reverberation
preserves much of the underlying speech content while redistributing it
temporally. Consequently, retrieval degradation saturates early instead of
collapsing progressively with severity." That conclusion is drawn from an axis
that is inverted, near-inert, and inconsistent at its endpoint. §4.2's
"Reverberation is simulated through delayed room reflections at equivalent
severity levels" is the only description given, and "equivalent severity
levels" is not a specification.

The code behaviour is preserved as-is — changing it would invalidate existing
results — but it is documented at the call site, in
`src/paraspeechrag/perturb/noise.py`, and in `configs/perturb/noise.yaml`.
**A real reverb sweep needs a decay-time axis (or measured RIRs) and a
re-run**; the current Figure 1 reverb panel should not ship as is.

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

All of these were re-checked against the PDF at port time. They need decisions,
not code.

- **Number of evaluation queries is never stated.** §3.4 gives only the
  2,067-paragraph corpus and "each query is ranked against the entire
  validation pool". Recall@k is uninterpretable without the query count.
  `run_noise_robustness_eval.py` now emits `n_queries` into its `.meta.json`.
- **SNR levels conflict** — VERIFIED. §3.3 says "four SNR levels: 20, 15, 10,
  and ∼5 dB"; §4.2 says "20, 10, and 5 dB SNR levels, corresponding to mild,
  moderate, and severe". The code default is `20,15,10,5`;
  `experiments/run_noise_ambient.sh` overrides to `20,10,5`.
- **Noise classes conflict** — VERIFIED. §3.3 says "white noise and convolutive
  reverberation"; §4.2 says "white noise, ambient environmental noise, and
  reverberation". The code supports all three.
- **CLASP-mean MRR** — VERIFIED. Table 1 reports 0.971; §5.2 and Table 2's
  "WAV (Original/Baseline)" row both report 0.972.
- **Figure 1's caption is a placeholder** — VERIFIED. It currently reads: "It
  describes the 5 subframes (codec, rate, white noise, reverb, ambient),
  identifies solid lines (MRR) vs. dashed lines (Recall@1), maps CLASP vs. ASR,
  and explains the x-axis orientation in each subframe." That is a description
  of what the caption should contain, not a caption.
- **Figure 1's legend reportedly says "SpeechRAG"** — NOT VERIFIABLE from the
  PDF's extracted text (legends live inside the figure image). Flagged in the
  original review. The paper's title is *ParaSpeechRAG*, so a `SpeechRAG`
  legend would be a stale internal code name. Check the plotting source. See
  `docs/NAMING.md`.
- **Recall@k ≡ Hits@k is already documented** — footnote 1 states it, and that
  MAP coincides with MRR. `docs/NAMING.md` agrees with the paper here; no
  action needed beyond keeping them in sync.
- **Emotion audio is 22.05 kHz, not 16 kHz** — VERIFIED IN CODE (§1.3). §4.5
  states "All converted audio is resampled to 16 kHz"; §3.3 says the same for
  Seed-VC. `scripts/perturb/seedvc/batch_convert.py:132` writes 22050 Hz and
  never resamples, and every WAV under
  `data/datasets/spoken_squad_emotions/*/*/` is 22050 Hz. (The noise sets under
  `dev_wav_noisy_by_type/` *are* 16 kHz.) Harmless for results —
  `load_mono_16k_padded` resamples on the way in — but both sentences are false
  as written. Fix the text, or regenerate with an explicit resample.
- **Training epochs disagree with the code** — NEW. §4.1 says "50 epochs";
  `scripts/train.py` defaults to `--num-epochs 100` with early-stopping
  patience 10. Whether the released checkpoint ran 50 or stopped early at some
  other count is unrecorded. See `configs/train/clasp.yaml`.
- **Results exist for systems with no code here** — NEW. Table 1 reports CLAP
  (0.000/0.001/0.002/0.001) and GLAP (0.209/0.366/0.442/0.289); §5.3 reports
  codec numbers (MP3 8 kbps → 0.885/0.927) and rate numbers (0.5× → 0.674,
  2.0× → 0.630); §5.3 reports ambient noise at 5 dB → 0.301/0.389. Those runs
  happened somewhere. Nothing in this repository can regenerate the CLAP,
  GLAP, or ASR-retrieval numbers, and the codec/rate generators here have no
  evaluation attached (§1). Recover that code or the numbers are unbacked.
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




n sync at e3c4f33, working tree clean. Clone on the 4090 will pick up everything.

  When you go looking for the remaining code on that box, here's what docs/GAPS.md currently says is missing — worth grabbing in one pass rather than discovering
  piecemeal:

  When you go looking for the remaining code on that box, here's what docs/GAPS.md currently says is missing — worth grabbing in one pass rather than discovering piecemeal:

  Produces numbers in the paper, no code anywhere here:
  - CLAP retriever — Table 1 row 0.000 / 0.001 / 0.002 / 0.001
  When you go looking for the remaining code on that box, here's what docs/GAPS.md currently says is missing — worth grabbing in one pass rather than discovering
  piecemeal:

  Produces numbers in the paper, no code anywhere here:
  - CLAP retriever — Table 1 row 0.000 / 0.001 / 0.002 / 0.001
  - GLAP retriever — Table 1 row 0.209 / 0.366 / 0.442 / 0.289
  - ASR cascade retrieval half — transcripts → LaBSE → ranking. scripts/asr/transcribe.py gets you Parakeet transcripts and compute_wer.py scores WER, but
  nothing turns those into Recall@k. This is the entire Cascade column of Tables 2 and 3.
  - Codec eval — §5.3 quotes MP3 8 kbps → 0.885 / 0.927. Generators are here; nothing evaluates their output.
  - Rate eval — §5.3 quotes 0.5× → 0.674, 2.0× → 0.630. Same situation.
  - Figure 1 plotting code — worth checking whether the legend really says SpeechRAG; I couldn't read it from the PDF text layer.

  Would settle open questions:
  - Seed-VC generation script — the emotion and Seed-VC speaker audio was made out-of-tree
  - The 18,896-pair Spoken-SQuAD training subset list
  - The checkpoint that actually produced Table 1, plus its hash

  Also worth running there, since it answers the biggest unknown cheaply:

  make check-coverage

  If the perturbed sets on that box cover all 2,067 paragraphs, GAPS §3.2 closes. If they cover 91 like the local copies, the perturbed Recall@k figures are on a
  different candidate pool than the clean baseline, and the ΔR@1 values in the paper need recomputing.

  Send me whatever you pull back and I'll wire it in and update GAPS accordingly — leaving the commits to you.