#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/SCRIPT_TEST/logs"
PID_FILE="${LOG_DIR}/ravdess_actor01.pid"
LOG_FILE="${LOG_DIR}/ravdess_actor01.log"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/SCRIPT_TEST/manifests"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="python"
fi

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

start() {
  if is_running; then
    echo "Already running with PID $(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi

  cd "${ROOT_DIR}"
  echo "Starting RAVDESS actor01 conversion..."
  echo "Log: ${LOG_FILE}"

  nohup "${PYTHON}" -u SCRIPT_TEST/batch_convert_v2.py \
    --sources SCRIPT_TEST/inputs \
    --targets SCRIPT_TEST/references_ravdess \
    --outputs SCRIPT_TEST/outputs_ravdess_v2 \
    --manifest SCRIPT_TEST/manifests/outputs_ravdess_v2_actor01_emotions_only.csv \
    --target-actors 1 \
    --target-emotions angry,happy,neutral,sad \
    --recursive-targets true \
    --convert-style true \
    --skip-existing true \
    >> "${LOG_FILE}" 2>&1 &

  echo "$!" > "${PID_FILE}"
  echo "Started with PID $(cat "${PID_FILE}")"
}

status() {
  if is_running; then
    echo "Running with PID $(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
  else
    echo "Not running"
    if [[ -f "${PID_FILE}" ]]; then
      echo "Stale PID file: ${PID_FILE}"
    fi
  fi
}

tail_log() {
  touch "${LOG_FILE}"
  tail -f "${LOG_FILE}"
}

stop() {
  if is_running; then
    kill "$(cat "${PID_FILE}")"
    echo "Stopped PID $(cat "${PID_FILE}")"
  else
    echo "Not running"
  fi
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  tail) tail_log ;;
  stop) stop ;;
  *)
    echo "Usage: $0 {start|status|tail|stop}"
    exit 1
    ;;
esac
