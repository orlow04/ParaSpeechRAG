#!/usr/bin/env bash
# =============================================================================
# experiments/run_seedvc_all_voices.sh
#
# Build a PKL + run eval for every target voice under
# data/datasets/spoken_squad_seed-vc/{voice}/.
#
# USAGE:
#   bash experiments/run_seedvc_all_voices.sh
#
# SPLIT INTEGRITY — read before changing TRAIN_WAV_DIR:
#   The converted audio is EVALUATION audio, so it is passed only as
#   --val-wav-dir. The train split reads the CLEAN Spoken-SQuAD train audio
#   from TRAIN_WAV_DIR. See the same note in run_emotions_all_voices.sh for
#   why passing one directory to both flags silently corrupts the split.
#
# NOTE ON THE "trump" TARGET VOICE:
#   Redistributing voice-converted audio in the cloned voice of an identifiable
#   living political figure is a problem for a public artifact release. It is
#   left in VOICES here because removing it would silently change Table 2, but
#   see docs/GAPS.md before publishing the audio.
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   VOICES         — space-separated voice list (default: all six)
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

VOICES="${VOICES:-1089-134686-0000 2803-154320-0012 3081-166546-0023 6319-275224-0006 azuma trump}"
POOLING_MODE="${POOLING_MODE:-mean}"
# The checkpoint ships under two spellings depending on how it was fetched:
# a literal "+" or the URL-encoded "%2B". Accept either rather than failing on
# an encoding artifact.
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
SEEDVC_DIR="${ROOT}/data/datasets/spoken_squad_seed-vc"
mkdir -p "${ROOT}/logs"

[[ -f "$MODEL" ]]         || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -f "$TRAIN_JSON" ]]    || { echo "ERROR: JSON not found: $TRAIN_JSON"; exit 1; }
[[ -f "$VAL_JSON" ]]      || { echo "ERROR: JSON not found: $VAL_JSON"; exit 1; }
[[ -d "$TRAIN_WAV_DIR" || "$ALLOW_SHARED_WAV_DIR" == "1" ]] || { echo "ERROR: clean train audio not found: $TRAIN_WAV_DIR
Set TRAIN_WAV_DIR to the unperturbed Spoken-SQuAD train WAVs, e.g.:
    TRAIN_WAV_DIR=/path/to/train_wav bash $0

Or, to reproduce the pre-fix behaviour exactly (perturbed audio as BOTH splits,
which is what produced the existing numbers -- see docs/GAPS.md 3.3):
    TRAIN_WAV_DIR=<the same perturbed dir> ALLOW_SHARED_WAV_DIR=1 bash $0"; exit 1; }

echo "============================================================"
echo "  seed-vc all-voices pipeline (parallel)"
echo "  Voices        : ${VOICES}"
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

for VOICE in $VOICES; do
    WAV_DIR="${SEEDVC_DIR}/${VOICE}"
    PKL="${ROOT}/data/datasets/total_dataset_seedvc_${VOICE}_${POOLING_MODE}.pkl"
    LOG="${ROOT}/logs/seedvc_${VOICE}.log"

    if [[ ! -d "$WAV_DIR" ]]; then
        echo "[${VOICE}] WARNING: directory not found, skipping."
        continue
    fi

    echo "[${VOICE}] Starting — log: ${LOG}"

    (
        set -euo pipefail
        if [[ -f "$PKL" ]]; then
            echo "[${VOICE}] PKL already exists, skipping build."
        else
            echo "[${VOICE}] Building PKL ..."
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
            echo "[${VOICE}] PKL ready."
        fi

        # Advisory, never fatal: this reports contamination and pool size but
        # must not block a run that is deliberately reproducing older numbers.
        echo "[${VOICE}] Split check (advisory) ..."
        python scripts/sanity/check_split_disjoint.py "$PKL" \
            || echo "[${VOICE}] WARNING: split check FAILED — see docs/GAPS.md 3.3. Continuing."

        echo "[${VOICE}] Eval ..."
        SKIP_NOISE="$SKIP_NOISE" bash experiments/eval_spoken_squad.sh "$PKL" "$MODEL"
        echo "[${VOICE}] DONE."
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
echo "  All voices finished."
echo "  Logs: ${ROOT}/logs/seedvc_<voice>.log"
echo "============================================================"
