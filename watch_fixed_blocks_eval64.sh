#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks

RUN_NAME=${RUN_NAME:-fixed_blocks_train64_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000}
PID_FILE=${PID_FILE:-logs/fixed_blocks/${RUN_NAME}.pid}
WATCH_INTERVAL=${WATCH_INTERVAL:-600}
WATCH_LOG=${WATCH_LOG:-logs/fixed_blocks/watch_eval_${RUN_NAME}.log}
LORA_CKPT=${LORA_CKPT:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}/lora_final.pt}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "watcher_start run=${RUN_NAME} pid_file=${PID_FILE} interval=${WATCH_INTERVAL}s"

while true; do
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}")
    if ps -p "${pid}" >/dev/null 2>&1; then
      sleep "${WATCH_INTERVAL}"
      continue
    fi
  fi

  if pgrep -f "[t]rain_sft.py.*${RUN_NAME}" >/dev/null 2>&1; then
    sleep "${WATCH_INTERVAL}"
    continue
  fi
  break
done

log "train_process_finished run=${RUN_NAME}"

if [ ! -f "${LORA_CKPT}" ]; then
  log "missing_final_checkpoint ${LORA_CKPT}"
  exit 1
fi

log "starting_eval64 ckpt=${LORA_CKPT}"
RUN_NAME="${RUN_NAME}" LORA_CKPT="${LORA_CKPT}" STEPS="${STEPS:-128}" SAMPLING_MODE="${SAMPLING_MODE:-argmax}" \
  bash run_fixed_blocks_eval64.sh

OUT_JSONL="outputs/fixed_blocks_eval/${RUN_NAME}_train64_${SAMPLING_MODE:-argmax}${STEPS:-128}.jsonl"
if [ -f "${OUT_JSONL}" ]; then
  log "summarizing_eval64 out=${OUT_JSONL}"
  python tools/summarize_fixed_eval.py "${OUT_JSONL}" --show 12
  if [ -f tools/analyze_fixed_eval_failures.py ]; then
    log "analyzing_eval64_failures out=${OUT_JSONL}"
    python tools/analyze_fixed_eval_failures.py "${OUT_JSONL}" --show 12
  fi
  if [ -f tools/check_fixed_eval_gate.py ]; then
    log "checking_eval64_gate out=${OUT_JSONL}"
    python tools/check_fixed_eval_gate.py "${OUT_JSONL}" --show 12 || true
  fi
else
  log "missing_eval_jsonl ${OUT_JSONL}"
fi
log "watcher_done run=${RUN_NAME}"
