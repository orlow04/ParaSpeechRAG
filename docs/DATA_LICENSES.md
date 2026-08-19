# Data licences

**The benchmark cannot be released as a single permissively-licensed blob.**
Two of the upstream sources are non-commercial, and audio derived from them
inherits those terms. Segment the release by source licence, or distribute
recipes instead of audio for the NC-derived portions.

This file is a checklist, not legal advice. Confirm each entry against the
current upstream terms before publishing.

---

## Sources

| Source | Used for | Licence | Commercial use | Derived audio |
|---|---|---|---|---|
| **Spoken-SQuAD** | Base corpus: paragraph text + read audio | Check upstream — derived from SQuAD 1.1 (CC BY-SA 4.0) | Check | Every perturbed set is a derivative |
| **VoxPopuli** (EN) | Training pairs (184,219 in the paper) | CC0 for the audio; transcripts from EU Parliament records | Yes | Training only, not redistributed |
| **LibriSpeech** | Speaker-VC target voices `1089-134686-0000`, `2803-154320-0012`, `3081-166546-0023`, `6319-275224-0006` | CC BY 4.0 | Yes | Attribution required on converted audio |
| **RAVDESS** | Emotion-VC reference (Actor 01, 7 conditions) | **CC BY-NC-SA 4.0** | **No** | **NC + ShareAlike propagate to all emotion audio** |
| **ESC-50** | Ambient noise | **CC BY-NC 3.0** | **No** | **NC propagates to any released ambient-noise audio** |
| **Azuma** (VC target) | Speaker-VC target voice | ⚠️ **UNRESOLVED** — identify the source and its terms | ? | ? |
| **"trump"** (VC target) | Speaker-VC target voice | ⚠️ **See below — not primarily a licensing question** | ? | ? |

## What this means for the release

**Split the artifact into at least three buckets:**

1. **Redistributable** — clean Spoken-SQuAD-derived audio and any
   LibriSpeech-voice conversions, under terms compatible with Spoken-SQuAD and
   CC BY 4.0 (attribution required).
2. **Non-commercial, ShareAlike** — everything derived from RAVDESS, i.e. the
   entire emotion axis. CC BY-NC-SA 4.0's ShareAlike clause means the derived
   set must itself be released under CC BY-NC-SA 4.0.
3. **Non-commercial** — anything containing ESC-50 ambient noise. In this repo
   the noise conditions are generated in-memory and never written to disk
   (see `docs/GAPS.md` §1), so as long as that stays true, nothing
   ESC-50-derived is redistributed — only the recipe plus a pointer to
   `scripts/download_esc50.sh`. **If you start releasing rendered noise audio,
   bucket 3 becomes real.**

The code in this repository is under the repository `LICENSE`. That licence
covers the code only and does not extend to any data or model weights.

## Model weights

| Artifact | Source | Terms |
|---|---|---|
| `CLASP_Concat_Final_Fusion_Encoder.pt` | https://huggingface.co/llm-lab/CLASP | Check the model card |
| `facebook/hubert-large-ls960-ft` | HuggingFace | Apache-2.0 (confirm) |
| `sentence-transformers/LaBSE` | HuggingFace | Apache-2.0 (confirm) |
| `efficientnet_b7` | torchvision, ImageNet weights | BSD-3-Clause code; ImageNet weights carry their own research-use terms |
| Seed-VC / Seed-VC v2 | Upstream repo | ⚠️ Confirm — used to generate released audio |
| GenVC (`GenVC_large`) | Upstream repo | ⚠️ Confirm — used to generate released audio |

Seed-VC and GenVC terms matter more than usual here: they were used to
*generate* audio that would be redistributed as the benchmark, which is a
stronger claim on the upstream licence than merely running inference.

## The "trump" target voice

This is an ethics-statement problem before it is a licensing one.
Redistributing voice-converted audio in the cloned voice of an identifiable
living political figure is difficult to justify in a public research artifact,
regardless of what the reference recording's licence permits.

Options, in the order they cost least:

1. Drop the condition and re-run Table 2 without it.
2. Replace it with another bundled reference voice and regenerate.
3. Keep it in the paper, exclude it from the public artifact, and say so
   explicitly in the ethics statement and the data card.

All three require a decision before the audio is frozen, because 1 and 2 change
Table 2. The voice is still in the default `VOICES` list in
`experiments/run_seedvc_all_voices.sh`.

## Required before publication

- [ ] Confirm Spoken-SQuAD's redistribution terms for derived audio
- [ ] Identify the "azuma" reference voice and its licence
- [ ] Decide on the "trump" condition (see above)
- [ ] Confirm Seed-VC and GenVC licences permit redistributing generated audio
- [ ] Split the artifact into licence buckets and label each one
- [ ] Write a data card per bucket, listing sources and attribution
- [ ] Confirm the CLASP checkpoint's licence permits redistribution
