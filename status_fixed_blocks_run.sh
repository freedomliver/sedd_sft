#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"

RUN_NAME=${RUN_NAME:-fixed_blocks_train64_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}}
LOG_FILE=${LOG_FILE:-logs/fixed_blocks/nohup_${RUN_NAME}.log}
WATCH_LOG=${WATCH_LOG:-logs/fixed_blocks/watch_eval_${RUN_NAME}.log}
PID_FILE=${PID_FILE:-logs/fixed_blocks/${RUN_NAME}.pid}
WATCH_PID_FILE=${WATCH_PID_FILE:-logs/fixed_blocks/watch_eval_${RUN_NAME}.pid}
GATE_WATCH_PID_FILE=${GATE_WATCH_PID_FILE:-logs/fixed_blocks/watch_full599_after_${RUN_NAME}.pid}
GATE_WATCH_LOG=${GATE_WATCH_LOG:-logs/fixed_blocks/watch_full599_after_${RUN_NAME}.log}
SAMPLING_MODE=${SAMPLING_MODE:-argmax}
STEPS=${STEPS:-128}
CHECK_GATE=${CHECK_GATE:-auto}

if [ -z "${EVAL_JSONLS:-}" ]; then
  if [[ "${RUN_NAME}" == *full599* ]]; then
    EVAL_SPLITS=${EVAL_SPLITS:-valid test}
    EVAL_JSONLS=""
    for split in ${EVAL_SPLITS}; do
      EVAL_JSONLS="${EVAL_JSONLS} outputs/fixed_blocks_eval/${RUN_NAME}_${split}_${SAMPLING_MODE}${STEPS}.jsonl"
    done
  else
    EVAL_JSONLS="outputs/fixed_blocks_eval/${RUN_NAME}_train64_${SAMPLING_MODE}${STEPS}.jsonl"
  fi
fi

echo "now=$(date '+%F %T %Z')"
echo "run_name=${RUN_NAME}"

pid=$(cat "${PID_FILE}" 2>/dev/null || true)
if [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1; then
  echo "train_process=running pid=${pid}"
else
  echo "train_process=stopped_or_missing pid=${pid}"
fi

watch_pid=$(cat "${WATCH_PID_FILE}" 2>/dev/null || true)
if [ -n "${watch_pid}" ] && ps -p "${watch_pid}" >/dev/null 2>&1; then
  echo "watcher=running pid=${watch_pid}"
else
  echo "watcher=stopped_or_missing pid=${watch_pid}"
fi

gate_watch_pid=$(cat "${GATE_WATCH_PID_FILE}" 2>/dev/null || true)
if [ -n "${gate_watch_pid}" ] && ps -p "${gate_watch_pid}" >/dev/null 2>&1; then
  echo "gate_watcher=running pid=${gate_watch_pid}"
elif [ -n "${gate_watch_pid}" ]; then
  echo "gate_watcher=stopped_or_missing pid=${gate_watch_pid}"
fi

echo "latest_steps:"
if [ -f "${LOG_FILE}" ]; then
  grep -E "step=[0-9]+ loss=" "${LOG_FILE}" | tail -20 || true
else
  echo "log_missing=${LOG_FILE}"
fi

echo "recent_errors:"
if [ -f "${LOG_FILE}" ]; then
  grep -Ei "traceback|error|cuda out|nan|inf" "${LOG_FILE}" | tail -10 || true
fi

echo "checkpoint_files:"
if [ -d "${OUT_DIR}" ]; then
  find "${OUT_DIR}" -maxdepth 1 -type f \( -name "*.pt" -o -name "args.json" \) \
    -printf "%TY-%Tm-%Td %TH:%TM:%TS %f %s\n" | sort | tail -80
else
  echo "out_dir_missing=${OUT_DIR}"
fi

echo "watch_tail:"
if [ -f "${WATCH_LOG}" ]; then
  tail -80 "${WATCH_LOG}"
else
  echo "watch_log_missing=${WATCH_LOG}"
fi

if [ -f "${GATE_WATCH_LOG}" ]; then
  echo "gate_watch_tail:"
  tail -80 "${GATE_WATCH_LOG}"
fi

for eval_jsonl in ${EVAL_JSONLS}; do
  if [ -f "${eval_jsonl}" ]; then
    echo "eval_jsonl=${eval_jsonl}"
    wc -l "${eval_jsonl}"
    python tools/summarize_fixed_eval.py "${eval_jsonl}" --show 12
    if [ -f tools/analyze_fixed_eval_failures.py ]; then
      python tools/analyze_fixed_eval_failures.py "${eval_jsonl}" --show 8 || true
    fi
    if [ -f tools/check_fixed_eval_gate.py ]; then
      should_check_gate=0
      if [ "${CHECK_GATE}" = "1" ]; then
        should_check_gate=1
      elif [ "${CHECK_GATE}" = "auto" ] && [[ "${eval_jsonl}" == *"_train64_"* ]]; then
        should_check_gate=1
      fi
      if [ "${should_check_gate}" = "1" ]; then
        python tools/check_fixed_eval_gate.py "${eval_jsonl}" --show 12 || true
      fi
    fi
  else
    echo "eval_jsonl=missing ${eval_jsonl}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  echo -n "gpu="
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | head -1
fi
