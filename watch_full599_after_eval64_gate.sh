#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks outputs/fixed_blocks_eval

EVAL_RUN_NAME=${EVAL_RUN_NAME:-fixed_blocks_train64_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000}
FULL_RUN_NAME=${FULL_RUN_NAME:-fixed_blocks_full599_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step100000}
SAMPLING_MODE=${SAMPLING_MODE:-argmax}
STEPS=${STEPS:-128}
LIMIT=${LIMIT:-64}
WATCH_INTERVAL=${WATCH_INTERVAL:-600}

MIN_MATCH=${MIN_MATCH:-0.90}
MIN_EOS=${MIN_EOS:-0.90}
MAX_EMPTY=${MAX_EMPTY:-0.05}

EVAL_JSONL=${EVAL_JSONL:-outputs/fixed_blocks_eval/${EVAL_RUN_NAME}_train64_${SAMPLING_MODE}${STEPS}.jsonl}
EVAL_WATCH_LOG=${EVAL_WATCH_LOG:-logs/fixed_blocks/watch_eval_${EVAL_RUN_NAME}.log}
PID_FILE=${PID_FILE:-logs/fixed_blocks/watch_full599_after_${EVAL_RUN_NAME}.pid}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ -f "${PID_FILE}" ]; then
  old_pid=$(cat "${PID_FILE}")
  if [ -n "${old_pid}" ] && ps -p "${old_pid}" >/dev/null 2>&1; then
    log "gate_watcher_already_running pid=${old_pid} pid_file=${PID_FILE}"
    exit 0
  fi
fi
echo "$$" > "${PID_FILE}"

log "gate_watcher_start eval_run=${EVAL_RUN_NAME} full_run=${FULL_RUN_NAME} interval=${WATCH_INTERVAL}s"
log "eval_jsonl=${EVAL_JSONL}"
log "gate_thresholds min_match=${MIN_MATCH} min_eos=${MIN_EOS} max_empty=${MAX_EMPTY}"

while true; do
  if pgrep -f "[t]rain_sft.py.*${FULL_RUN_NAME}" >/dev/null 2>&1; then
    log "full599_already_running full_run=${FULL_RUN_NAME}"
    exit 0
  fi

  eval_done=0
  if [ -f "${EVAL_WATCH_LOG}" ] && grep -F "watcher_done run=${EVAL_RUN_NAME}" "${EVAL_WATCH_LOG}" >/dev/null 2>&1; then
    eval_done=1
  elif [ -f "${EVAL_JSONL}" ]; then
    rows=$(wc -l < "${EVAL_JSONL}" | tr -d ' ')
    if [ "${rows}" -ge "${LIMIT}" ] && ! pgrep -f "[e]val_fixed_blocks.py.*${EVAL_JSONL}" >/dev/null 2>&1; then
      eval_done=1
    fi
  fi

  if [ "${eval_done}" = "1" ]; then
    if [ ! -f "${EVAL_JSONL}" ]; then
      log "eval_done_but_jsonl_missing ${EVAL_JSONL}"
      exit 1
    fi

    rows=$(wc -l < "${EVAL_JSONL}" | tr -d ' ')
    log "eval64_complete rows=${rows} jsonl=${EVAL_JSONL}"
    if [ "${rows}" -lt "${LIMIT}" ]; then
      log "gate_fail reason=incomplete_eval_rows rows=${rows} expected=${LIMIT}"
      exit 1
    fi

    if python tools/check_fixed_eval_gate.py \
      "${EVAL_JSONL}" \
      --min-match "${MIN_MATCH}" \
      --min-eos "${MIN_EOS}" \
      --max-empty "${MAX_EMPTY}" \
      --show 12; then
      log "gate_pass_launching_full599 full_run=${FULL_RUN_NAME}"
      RUN_NAME="${FULL_RUN_NAME}" bash launch_fixed_blocks_full599.sh
      log "gate_watcher_done launched_full599=${FULL_RUN_NAME}"
      exit 0
    else
      log "gate_fail_not_launching_full599 full_run=${FULL_RUN_NAME}"
      exit 1
    fi
  fi

  sleep "${WATCH_INTERVAL}"
done
