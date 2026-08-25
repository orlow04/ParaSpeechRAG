# Seed-VC generators (speaker + emotion axes)

Vendored from the `SCRIPT_TEST/` folder of a Seed-VC checkout
(`seed-vc-test/`, ~1.1 GB) that lives outside this repo. Only the
project-specific scripts are here — **not** the upstream Seed-VC code, its
checkpoints, or the reference audio.

## Running these

They import from the Seed-VC repo root, so they must execute from inside a
Seed-VC checkout, not from here:

```bash
cd /path/to/seed-vc            # upstream: https://github.com/Plachtaa/seed-vc
cp -r /path/to/ParaSpeechRAG/scripts/perturb/seedvc SCRIPT_TEST
python SCRIPT_TEST/batch_convert.py --help
```

`batch_convert_v2.py` additionally reads `configs/v2/vc_wrapper.yaml` relative
to the Seed-VC root.

| File | Axis | What it does |
|---|---|---|
| `batch_convert.py` | speaker | Seed-VC v1 conversion to each target voice |
| `batch_convert_v2.py` | emotion | Seed-VC v2 with `convert_style=True` |
| `select_ravdess_references.py` | emotion | Picks RAVDESS reference clips per emotion/intensity |
| `run_ravdess_actor01_background.sh` | emotion | The exact invocation used for the paper's emotion set |
| `run_ravdess_actor01_invalids_background.sh` | emotion | Re-run helper for failed conversions |

## What this code settles about the paper

**Confirms §3.3 / §4.4 / §4.5:**

- Speaker: `--diffusion-steps 30`, `--cfg-rate 0.7`, `--length-adjust 1.0`
  (`batch_convert.py:91-93`) — exactly as the paper states.
- Speaker targets: `references_neutral/` holds `1089-134686-0000`,
  `2803-154320-0012`, `3081-166546-0023`, `6319-275224-0006`, `azuma_0`,
  `trump_0` — the paper's six. **Seed-VC matches the paper; GenVC does not**
  (it used `8842-302201-0002` and never 1089). See `docs/GAPS.md` §1.1.
- Emotion: `run_ravdess_actor01_background.sh` passes `--target-actors 1`,
  `--target-emotions angry,happy,neutral,sad`, `--convert-style true` — Actor 01
  and four categories, as §4.5 states.
- Seven conditions: `references_ravdess/` holds neutral×1 + happy×2 + sad×2 +
  angry×2 = 7, matching "seven condition combinations". (`calm`, `fearful` and
  `surprised` references also exist but are filtered out by the run script.)

**Contradicts §3.3 and §4.5 on sample rate:**

`batch_convert.py:132` is

```python
sr = 44100 if args.f0_condition else 22050
```

with `--f0-condition` defaulting to `False`, and the output is written with
`torchaudio.save(out, audio_tensor, out_sr)` where `out_sr = sr`. **Seed-VC
writes 22,050 Hz and never resamples.** §4.5's "All converted audio is
resampled to 16 kHz" and §3.3's "output → 16 kHz" are both false as written.

This is confirmed against the released audio: every WAV under
`data/datasets/spoken_squad_emotions/*/*/` is 22,050 Hz. It does not change any
result — `paraspeechrag.inference.audio_preprocess.load_mono_16k_padded`
resamples to 16 kHz on the way into the retriever — but the sentence needs
fixing or the audio needs regenerating.

## Not vendored

- **Reference audio** (`references_neutral/`, `references_ravdess/`, ~2.7 MB).
  Deliberate: `trump_0.wav` is a cloned voice of an identifiable living public
  figure, and the RAVDESS clips are CC BY-NC-SA 4.0. See
  `docs/DATA_LICENSES.md`.
- **Manifests** (`references_emotional*.csv`) — per-conversion records that
  belong with the audio release, not the code.
- **Upstream Seed-VC** and its checkpoints.

## Reproducibility

Seed-VC runs a 30-step diffusion sampler and neither script sets or logs a
seed, so conversions are not bit-reproducible. The released **audio** is the
artifact; pin it with `scripts/build_manifest.py`.
