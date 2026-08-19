# ParaSpeechRAG — one target per reproducible step.
#
# Targets are grouped: setup, freeze (corpus + manifests), check (sanity),
# build (PKLs), eval (metrics). Run `make help` for the list.
#
# Override any variable on the command line:
#     make eval-speaker MODEL=models/checkpoints/my.pt POOLING=chunked

SHELL      := /bin/bash
PY         ?= python
ROOT       := $(shell pwd)

DATA       ?= $(ROOT)/data/datasets
RESULTS    ?= $(ROOT)/results
MANIFESTS  ?= $(ROOT)/data/manifests

MODEL      ?= $(ROOT)/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt
POOLING    ?= mean

TRAIN_JSON ?= $(DATA)/spoken_squad/spoken_train-v1.1.json
VAL_JSON   ?= $(DATA)/spoken_squad/spoken_test-v1.1.json
TRAIN_WAV  ?= $(DATA)/spoken_squad/train_wav
VAL_WAV    ?= $(DATA)/spoken_squad/dev_wav

EMOTION_DIR ?= $(DATA)/spoken_squad_emotions
SEEDVC_DIR  ?= $(DATA)/spoken_squad_seed-vc
NOISY_DIR   ?= $(DATA)/spoken_squad/dev_wav_noisy_by_type
ESC50_DIR   ?= $(ROOT)/data/noise_sources/esc50

WORKERS     ?= 4
ASR_CONFIG  ?= $(ROOT)/configs/asr/transcribe_all_conditions.yml

SNR_LEVELS  ?= 20,15,10,5
NOISE_TYPES ?= white,reverb,ambient
HITS_K      ?= 1,5,10,50

CLEAN_PKL   := $(DATA)/total_dataset_spoken_squad_$(POOLING).pkl

.DEFAULT_GOAL := help
.PHONY: help setup freeze-corpus manifests verify-manifests check-splits \
        check-coverage build-clean eval-clean eval-noise eval-speaker \
        eval-emotion perturb-codec perturb-rate perturb-speaker \
        asr-transcribe asr-wer clean-results

## ---------------------------------------------------------------- meta ----

help:  ## Show this help
	@echo "ParaSpeechRAG targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Read docs/GAPS.md before trusting any number this produces."

setup:  ## Install the package and base dependencies
	uv sync --extra realdata

## ------------------------------------------------------ freeze artifacts ----

freeze-corpus:  ## Write the frozen corpus + split ID lists to data/corpus, data/splits
	$(PY) scripts/freeze_corpus.py --json $(TRAIN_JSON) --split train
	$(PY) scripts/freeze_corpus.py --json $(VAL_JSON)   --split validation

manifests:  ## Checksum every perturbed-audio directory into data/manifests/
	@mkdir -p $(MANIFESTS)
	$(PY) scripts/build_manifest.py $(EMOTION_DIR) --generator seed-vc-v2 \
	    --out $(MANIFESTS)/emotion.csv
	$(PY) scripts/build_manifest.py $(SEEDVC_DIR)  --generator seed-vc \
	    --out $(MANIFESTS)/speaker.csv
	$(PY) scripts/build_manifest.py $(NOISY_DIR)   --generator offline-noise \
	    --out $(MANIFESTS)/noise_prerendered.csv

verify-manifests:  ## Re-hash the audio and diff against data/manifests/
	$(PY) scripts/build_manifest.py $(EMOTION_DIR) --verify $(MANIFESTS)/emotion.csv
	$(PY) scripts/build_manifest.py $(SEEDVC_DIR)  --verify $(MANIFESTS)/speaker.csv
	$(PY) scripts/build_manifest.py $(NOISY_DIR)   --verify $(MANIFESTS)/noise_prerendered.csv

## ------------------------------------------------------------- sanity ----

check-splits:  ## Assert train/validation are disjoint in every built PKL
	$(PY) scripts/sanity/check_split_disjoint.py $(DATA)/total_dataset_*.pkl

check-coverage:  ## Report paragraph coverage of each perturbation directory
	$(PY) scripts/sanity/audit_perturbation_coverage.py --json $(VAL_JSON) \
	    $(EMOTION_DIR) $(SEEDVC_DIR) $(NOISY_DIR)

## -------------------------------------------------------------- build ----

build-clean: $(CLEAN_PKL)  ## Build the clean Spoken-SQuAD PKL

$(CLEAN_PKL):
	$(PY) scripts/build_spoken_squad_pkl.py \
	    --train-json $(TRAIN_JSON) --train-wav-dir $(TRAIN_WAV) \
	    --val-json   $(VAL_JSON)   --val-wav-dir   $(VAL_WAV) \
	    --output $@ --pooling-mode $(POOLING)
	$(PY) scripts/sanity/check_split_disjoint.py $@

## ------------------------------------------------------ generate audio ----
## These WRITE perturbed WAV trees. Nothing downstream consumes them yet —
## see docs/GAPS.md section 1. Point --src_dir at the clean dev audio.

perturb-codec:  ## Generate MP3 + Opus round-trips at 8/16/32/64/128 kbps
	$(PY) scripts/perturb/codec_mp3.py  --src_dir $(VAL_WAV) \
	    --output_dir $(DATA)/perturb/codec_mp3  --workers $(WORKERS)
	$(PY) scripts/perturb/codec_opus.py --src_dir $(VAL_WAV) \
	    --output_dir $(DATA)/perturb/codec_opus --workers $(WORKERS)

perturb-rate:  ## Generate time-stretched audio at the 8 SRB speed factors
	$(PY) scripts/perturb/speed.py --src_dir $(VAL_WAV) \
	    --output_dir $(DATA)/perturb/rate --workers $(WORKERS)

perturb-speaker:  ## Voice-convert with GenVC (needs a GenVC checkout + GPU)
	$(PY) scripts/perturb/voice_convert_genvc.py \
	    --src_dir $(VAL_WAV) --ref_dir $(DATA)/references \
	    --output_dir $(DATA)/perturb/genvc --top_k 15
	@echo "NOTE: GenVC's reference voice set differs from Seed-VC's."
	@echo "      See docs/GAPS.md section 1.1 before averaging the two."

## ---------------------------------------------------------------- asr ----

asr-transcribe:  ## Transcribe every condition with Parakeet TDT (needs NeMo + GPU)
	$(PY) scripts/asr/transcribe.py --config $(ASR_CONFIG)

asr-wer:  ## Score transcriptions against the dev manifest -> results/wer_results.*
	$(PY) scripts/asr/compute_wer.py

## --------------------------------------------------------------- eval ----

eval-clean: build-clean  ## Clean baseline retrieval (paper: Table 1)
	@mkdir -p $(RESULTS)/clean
	SKIP_NOISE=1 bash experiments/eval_spoken_squad.sh $(CLEAN_PKL) $(MODEL)

eval-noise: build-clean  ## Noise robustness (paper: Figure 1)
	@mkdir -p $(RESULTS)/noise
	$(PY) scripts/run_noise_robustness_eval.py \
	    --dataset-path $(CLEAN_PKL) --model-path $(MODEL) \
	    --audio-key hubert-emb --text-key text \
	    --snr-levels $(SNR_LEVELS) --noise-types $(NOISE_TYPES) \
	    --esc50-dir $(ESC50_DIR) --hits-k $(HITS_K) \
	    --output-csv $(RESULTS)/noise/noise_$(POOLING).csv
	@echo "NOTE: reverb rows use --snr-levels as a decay-time knob, NOT an SNR."
	@echo "      See docs/GAPS.md section 3.1 before plotting a shared axis."

eval-speaker:  ## Speaker variation, all target voices (paper: Table 2)
	POOLING_MODE=$(POOLING) MODEL=$(MODEL) TRAIN_WAV_DIR=$(TRAIN_WAV) \
	    bash experiments/run_seedvc_all_voices.sh

eval-emotion:  ## Emotion variation, all conditions (paper: Table 3)
	POOLING_MODE=$(POOLING) MODEL=$(MODEL) TRAIN_WAV_DIR=$(TRAIN_WAV) \
	    bash experiments/run_emotions_all_voices.sh

clean-results:  ## Delete generated results (keeps results/legacy_*)
	find $(RESULTS) -type f -not -name 'legacy_*' -delete
