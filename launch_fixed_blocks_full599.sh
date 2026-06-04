#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks /root/autodl-tmp/sedd_outputs

SEED=${SEED:-10}
RUN_NAME=${RUN_NAME:-fixed_blocks_full599_q231_r530_a263_dwdseonly_seed${SEED}_r32a64_b16ga8_step100000}
PID_FILE=${PID_FILE:-logs/fixed_blocks/${RUN_NAME}.pid}
WATCH_PID_FILE=${WATCH_PID_FILE:-logs/fixed_blocks/watch_eval_${RUN_NAME}.pid}
NOHUP_LOG=${NOHUP_LOG:-logs/fixed_blocks/nohup_${RUN_NAME}.log}
WATCH_LOG=${WATCH_LOG:-logs/fixed_blocks/watch_eval_${RUN_NAME}.log}
WATCH_INTERVAL=${WATCH_INTERVAL:-600}
SPLITS=${SPLITS:-valid test}
STEPS=${STEPS:-128}
SAMPLING_MODE=${SAMPLING_MODE:-argmax}
ALLOW_CONCURRENT_TRAIN=${ALLOW_CONCURRENT_TRAIN:-0}

if [ -f "${PID_FILE}" ]; then
  old_pid=$(cat "${PID_FILE}")
  if ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "training already running: pid=${old_pid} pid_file=${PID_FILE}" >&2
    exit 1
  fi
fi

if pgrep -f "[t]rain_sft.py.*${RUN_NAME}" >/dev/null 2>&1; then
  echo "training process already found for RUN_NAME=${RUN_NAME}" >&2
  exit 1
fi

if [ "${ALLOW_CONCURRENT_TRAIN}" != "1" ]; then
  other_train=$(pgrep -af "[t]rain_sft.py" | grep -v -- "${RUN_NAME}" || true)
  if [ -n "${other_train}" ]; then
    echo "another train_sft.py process is running; set ALLOW_CONCURRENT_TRAIN=1 to override" >&2
    echo "${other_train}" >&2
    exit 1
  fi
fi

echo "launching full599 fixed-block training"
echo "run=${RUN_NAME}"
echo "log=${NOHUP_LOG}"

RUN_NAME="${RUN_NAME}" SEED="${SEED}" nohup bash run_fixed_blocks_full599_sft.sh > "${NOHUP_LOG}" 2>&1 &
train_pid=$!
echo "${train_pid}" > "${PID_FILE}"
echo "train_pid=${train_pid}"
echo "pid_file=${PID_FILE}"

RUN_NAME="${RUN_NAME}" \
SEED="${SEED}" \
PID_FILE="${PID_FILE}" \
WATCH_INTERVAL="${WATCH_INTERVAL}" \
LORA_CKPT="${LORA_CKPT:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}/lora_final.pt}" \
SPLITS="${SPLITS}" \
STEPS="${STEPS}" \
SAMPLING_MODE="${SAMPLING_MODE}" \
  nohup bash watch_fixed_blocks_eval_splits.sh >> "${WATCH_LOG}" 2>&1 &
watch_pid=$!
echo "${watch_pid}" > "${WATCH_PID_FILE}"
echo "watch_pid=${watch_pid}"
echo "watch_pid_file=${WATCH_PID_FILE}"
echo "watch_log=${WATCH_LOG}"
