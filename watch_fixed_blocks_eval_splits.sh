#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks outputs/fixed_blocks_eval

RUN_NAME=${RUN_NAME:-fixed_blocks_full599_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step100000}
PID_FILE=${PID_FILE:-logs/fixed_blocks/${RUN_NAME}.pid}
WATCH_INTERVAL=${WATCH_INTERVAL:-600}
WATCH_LOG=${WATCH_LOG:-logs/fixed_blocks/watch_eval_${RUN_NAME}.log}
LORA_CKPT=${LORA_CKPT:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}/lora_final.pt}
SPLITS=${SPLITS:-valid test}
STEPS=${STEPS:-128}
SAMPLING_MODE=${SAMPLING_MODE:-argmax}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "watcher_start run=${RUN_NAME} pid_file=${PID_FILE} interval=${WATCH_INTERVAL}s splits=${SPLITS}"

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

log "starting_split_eval ckpt=${LORA_CKPT} splits=${SPLITS}"
RUN_NAME="${RUN_NAME}" \
LORA_CKPT="${LORA_CKPT}" \
SPLITS="${SPLITS}" \
STEPS="${STEPS}" \
SAMPLING_MODE="${SAMPLING_MODE}" \
  bash run_fixed_blocks_eval_splits.sh

for split in ${SPLITS}; do
  OUT_JSONL="outputs/fixed_blocks_eval/${RUN_NAME}_${split}_${SAMPLING_MODE}${STEPS}.jsonl"
  if [ -f "${OUT_JSONL}" ]; then
    log "summarizing_eval split=${split} out=${OUT_JSONL}"
    python tools/summarize_fixed_eval.py "${OUT_JSONL}" --show 12
    if [ -f tools/analyze_fixed_eval_failures.py ]; then
      log "analyzing_eval_failures split=${split} out=${OUT_JSONL}"
      python tools/analyze_fixed_eval_failures.py "${OUT_JSONL}" --show 12
    fi
  else
    log "missing_eval_jsonl split=${split} out=${OUT_JSONL}"
  fi
done

log "watcher_done run=${RUN_NAME}"
