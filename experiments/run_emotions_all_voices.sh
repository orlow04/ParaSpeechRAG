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
# The checkpoint ships under two spellings depending on how it was fetched:
# a literal "+" or the URL-encoded "%2B". Accept either.
if [[ -z "${MODEL:-}" ]]; then
    for _cand in "clasp_spoken_squad+voxpopuli.pt" "clasp_spoken_squad%2Bvoxpopuli.pt"; do
        if [[ -f "${ROOT}/models/checkpoints/${_cand}" ]]; then
            MODEL="${ROOT}/models/checkpoints/${_cand}"; break
        fi
    done
    MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_spoken_squad+voxpopuli.pt}"
fi
SKIP_NOISE="${SKIP_NOISE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

# ALLOW_SHARED_WAV_DIR=1 reproduces the pre-fix behaviour: the perturbed audio
# is passed as BOTH --train-wav-dir and --val-wav-dir. That is what produced the
# existing numbers, and it is a genuine contamination bug (docs/GAPS.md 3.3).
# Use it only to check what the old numbers were.
ALLOW_SHARED_WAV_DIR="${ALLOW_SHARED_WAV_DIR:-0}"
SHARED_ARGS=()
if [[ "$ALLOW_SHARED_WAV_DIR" == "1" ]]; then
    SHARED_ARGS=(--allow-shared-wav-dir)
fi

TRAIN_JSON="${ROOT}/data/datasets/spoken_squad/spoken_train-v1.1.json"
VAL_JSON="${ROOT}/data/datasets/spoken_squad/spoken_test-v1.1.json"
TRAIN_WAV_DIR="${TRAIN_WAV_DIR:-${ROOT}/data/datasets/spoken_squad/train_wav}"
EMOTIONS_DIR="${ROOT}/data/datasets/spoken_squad_emotions"
mkdir -p "${ROOT}/logs"

[[ -f "$MODEL" ]]           || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -f "$TRAIN_JSON" ]]      || { echo "ERROR: JSON not found: $TRAIN_JSON"; exit 1; }
[[ -f "$VAL_JSON" ]]        || { echo "ERROR: JSON not found: $VAL_JSON"; exit 1; }
[[ -d "$TRAIN_WAV_DIR" || "$ALLOW_SHARED_WAV_DIR" == "1" ]] || { echo "ERROR: clean train audio not found: $TRAIN_WAV_DIR
Set TRAIN_WAV_DIR to the unperturbed Spoken-SQuAD train WAVs, e.g.:
    TRAIN_WAV_DIR=/path/to/train_wav bash $0

Or, to reproduce the pre-fix behaviour exactly (perturbed audio as BOTH splits,
which is what produced the existing numbers -- see docs/GAPS.md 3.3):
    TRAIN_WAV_DIR=<the same perturbed dir> ALLOW_SHARED_WAV_DIR=1 bash $0"; exit 1; }

echo "============================================================"
echo "  emotions all-combos pipeline (parallel)"
echo "  Combos        : ${COMBOS}"
echo "  Pooling       : ${POOLING_MODE}"
echo "  Eval model    : ${MODEL}"
echo "  Train wav dir : ${TRAIN_WAV_DIR}  (clean)"
echo "  Max parallel  : ${MAX_PARALLEL}"
echo "  Skip noise    : ${SKIP_NOISE}"
echo "  Shared wav dir: ${ALLOW_SHARED_WAV_DIR}  (1 = reproduce pre-fix contamination)"
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
            # Pre-fix reproduction reads the SAME perturbed dir for both splits.
            EFFECTIVE_TRAIN_WAV_DIR="$TRAIN_WAV_DIR"
            [[ "$ALLOW_SHARED_WAV_DIR" == "1" ]] && EFFECTIVE_TRAIN_WAV_DIR="$WAV_DIR"
            python scripts/build_spoken_squad_pkl.py \
                --train-json    "$TRAIN_JSON" \
                --train-wav-dir "$EFFECTIVE_TRAIN_WAV_DIR" \
                --val-json      "$VAL_JSON" \
                --val-wav-dir   "$WAV_DIR" \
                --output        "$PKL" \
                --pooling-mode  "$POOLING_MODE" \
                ${SHARED_ARGS[@]+"${SHARED_ARGS[@]}"}
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
