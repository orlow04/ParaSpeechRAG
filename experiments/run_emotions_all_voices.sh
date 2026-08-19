#!/usr/bin/env bash
# =============================================================================
# experiments/run_emotions_all_voices.sh
#
# Build a PKL + run eval for every emotion/intensity combination under
# data/datasets/spoken_squad_emotions/{emotion}/{intensity}/.
#
# USAGE:
#   bash experiments/run_emotions_all_voices.sh
#
# SPLIT INTEGRITY — read before changing TRAIN_WAV_DIR:
#   The perturbed audio is EVALUATION audio, so it is passed only as
#   --val-wav-dir. The train split reads the CLEAN Spoken-SQuAD train audio
#   from TRAIN_WAV_DIR.
#
#   An earlier version of this script passed the same perturbed directory as
#   BOTH --train-wav-dir and --val-wav-dir. Paragraph keys {article}_{paragraph}
#   are positions within a single SQuAD JSON, so "0_0" in spoken_train-v1.1.json
#   and "0_0" in spoken_test-v1.1.json are different paragraphs — pointing both
#   at one directory made the two splits read the SAME WAV files paired with
#   DIFFERENT context text. build_spoken_squad_pkl.py now refuses that by
#   default. Verify any PKL with:
#       python scripts/sanity/check_split_disjoint.py <PKL>
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   COMBOS         — space-separated "emotion/intensity" list (default: all)
#   POOLING_MODE   — mean or chunked (default: mean)
#   MODEL          — checkpoint path
#   TRAIN_WAV_DIR  — clean train audio (default: spoken_squad/train_wav)
#   SKIP_NOISE     — "1" to skip the noise eval (default: 1)
#   MAX_PARALLEL   — parallel workers (default: 3; drop to 2 on OOM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

COMBOS="${COMBOS:-angry/normal angry/strong happy/normal happy/strong neutral/normal sad/normal sad/strong}"
POOLING_MODE="${POOLING_MODE:-mean}"
MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_spoken_squad%2Bvoxpopuli.pt}"
SKIP_NOISE="${SKIP_NOISE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

TRAIN_JSON="${ROOT}/data/datasets/spoken_squad/spoken_train-v1.1.json"
VAL_JSON="${ROOT}/data/datasets/spoken_squad/spoken_test-v1.1.json"
TRAIN_WAV_DIR="${TRAIN_WAV_DIR:-${ROOT}/data/datasets/spoken_squad/train_wav}"
EMOTIONS_DIR="${ROOT}/data/datasets/spoken_squad_emotions"
mkdir -p "${ROOT}/logs"

[[ -f "$MODEL" ]]           || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -f "$TRAIN_JSON" ]]      || { echo "ERROR: JSON not found: $TRAIN_JSON"; exit 1; }
[[ -f "$VAL_JSON" ]]        || { echo "ERROR: JSON not found: $VAL_JSON"; exit 1; }
[[ -d "$TRAIN_WAV_DIR" ]]   || { echo "ERROR: clean train audio not found: $TRAIN_WAV_DIR
Set TRAIN_WAV_DIR to the directory holding the unperturbed Spoken-SQuAD train WAVs."; exit 1; }

echo "============================================================"
echo "  emotions all-combos pipeline (parallel)"
echo "  Combos        : ${COMBOS}"
echo "  Pooling       : ${POOLING_MODE}"
echo "  Eval model    : ${MODEL}"
echo "  Train wav dir : ${TRAIN_WAV_DIR}  (clean)"
echo "  Max parallel  : ${MAX_PARALLEL}"
echo "  Skip noise    : ${SKIP_NOISE}"
echo "============================================================"
echo ""

pids=()
running=0

for COMBO in $COMBOS; do
    EMOTION="${COMBO%%/*}"
    INTENSITY="${COMBO##*/}"
    WAV_DIR="${EMOTIONS_DIR}/${EMOTION}/${INTENSITY}"
    TAG="${EMOTION}_${INTENSITY}"
    PKL="${ROOT}/data/datasets/total_dataset_emotions_${TAG}_${POOLING_MODE}.pkl"
    LOG="${ROOT}/logs/emotions_${TAG}.log"

    if [[ ! -d "$WAV_DIR" ]]; then
        echo "[${TAG}] WARNING: directory not found, skipping."
        continue
    fi

    echo "[${TAG}] Starting — log: ${LOG}"

    (
        set -euo pipefail
        if [[ -f "$PKL" ]]; then
            echo "[${TAG}] PKL already exists, skipping build."
        else
            echo "[${TAG}] Building PKL ..."
            python scripts/build_spoken_squad_pkl.py \
                --train-json    "$TRAIN_JSON" \
                --train-wav-dir "$TRAIN_WAV_DIR" \
                --val-json      "$VAL_JSON" \
                --val-wav-dir   "$WAV_DIR" \
                --output        "$PKL" \
                --pooling-mode  "$POOLING_MODE"
            echo "[${TAG}] PKL ready."
        fi

        # Advisory, never fatal: this reports contamination and pool size but
        # must not block a run that is deliberately reproducing older numbers.
        echo "[${TAG}] Split check (advisory) ..."
        python scripts/sanity/check_split_disjoint.py "$PKL" \
            || echo "[${TAG}] WARNING: split check FAILED — see docs/GAPS.md 3.3. Continuing."

        echo "[${TAG}] Eval ..."
        SKIP_NOISE="$SKIP_NOISE" bash experiments/eval_spoken_squad.sh "$PKL" "$MODEL"
        echo "[${TAG}] DONE."
    ) > "$LOG" 2>&1 &

    pids+=($!)
    running=$(( running + 1 ))

    if [[ $running -ge $MAX_PARALLEL ]]; then
        wait -n
        running=$(( running - 1 ))
    fi
done

wait "${pids[@]}"

echo ""
echo "============================================================"
echo "  All combinations finished."
echo "  Logs: ${ROOT}/logs/emotions_<emotion>_<intensity>.log"
echo "============================================================"
