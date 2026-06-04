#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks outputs/fixed_blocks_eval /root/autodl-tmp/sedd_outputs

BASE_RUN=${BASE_RUN:-fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000}
DATA_JSON=${DATA_JSON:-data/simple_manual_synthetic/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json}
TRAIN_SIZE=${TRAIN_SIZE:-64}
VALID_SIZE=${VALID_SIZE:-0}
TEST_SIZE=${TEST_SIZE:-0}
SEED=${SEED:-10}

COMMON_TAG=${COMMON_TAG:-simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8}
BASE_OUT=/root/autodl-tmp/sedd_outputs/lora_${BASE_RUN}
BASE_CKPT=${BASE_OUT}/checkpoint_step_1000.pt
BASE_EVAL_JSON=outputs/fixed_blocks_eval/${BASE_RUN}_step1000_train16_argmax64.jsonl

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_file() {
  local path=$1
  local label=$2
  log "waiting_for_${label} ${path}"
  while [ ! -f "${path}" ]; do
    sleep 30
  done
  log "found_${label} ${path}"
}

stop_run_if_alive() {
  local run=$1
  local pids
  pids=$(pgrep -f "[t]rain_sft.py.*${run}" || true)
  if [ -z "${pids}" ]; then
    log "no_live_train_process run=${run}"
    return
  fi
  log "stopping_train run=${run} pids=${pids//$'\n'/,}"
  kill ${pids} || true
  sleep 10
  pids=$(pgrep -f "[t]rain_sft.py.*${run}" || true)
  if [ -n "${pids}" ]; then
    log "force_stopping_train run=${run} pids=${pids//$'\n'/,}"
    kill -9 ${pids} || true
  fi
}

eval_ckpt() {
  local run=$1
  local ckpt=$2
  local step_label=$3
  local out_json=outputs/fixed_blocks_eval/${run}_${step_label}_train16_argmax64.jsonl
  local log_file=logs/fixed_blocks/eval_${run}_${step_label}_train16_argmax64.log
  if [ -f "${out_json}" ]; then
    log "eval_exists ${out_json}"
  else
    log "starting_eval run=${run} ckpt=${ckpt}"
    RUN_NAME="${run}" \
      DATA_JSON="${DATA_JSON}" \
      LORA_CKPT="${ckpt}" \
      LIMIT=16 \
      EVAL_BATCH_SIZE=1 \
      STEPS=64 \
      SAMPLING_MODE=argmax \
      REASONING_FIELD=compressed_reasoning \
      FINAL_ANSWER_FIELD=compressed_answer \
      OUT_JSONL="${out_json}" \
      LOG_FILE="${log_file}" \
      bash run_fixed_blocks_eval64.sh
  fi
  /root/miniconda3/bin/python tools/summarize_fixed_eval.py "${out_json}" --show 16
}

run_multit_experiment() {
  local run=$1
  local max_steps=$2
  local num_t=$3
  local t_grid=$4
  local ckpt=/root/autodl-tmp/sedd_outputs/lora_${run}/checkpoint_step_${max_steps}.pt

  if [ -f "${ckpt}" ]; then
    log "checkpoint_exists run=${run} ckpt=${ckpt}"
  else
    log "launch_train run=${run} max_steps=${max_steps} num_t=${num_t} t_grid=${t_grid:-random}"
    RUN_NAME="${run}" \
      DATA_JSON="${DATA_JSON}" \
      TRAIN_SIZE="${TRAIN_SIZE}" \
      VALID_SIZE="${VALID_SIZE}" \
      TEST_SIZE="${TEST_SIZE}" \
      SEED="${SEED}" \
      MAX_STEPS="${max_steps}" \
      EPOCHS=100000 \
      BATCH_SIZE=16 \
      GRAD_ACCUM=8 \
      LR=3e-4 \
      WARMUP_STEPS=50 \
      SAVE_FREQ="${max_steps}" \
      LOG_FREQ=10 \
      EVAL_FREQ=0 \
      EVAL_BATCHES=10 \
      REASONING_FIELD=compressed_reasoning \
      FINAL_ANSWER_FIELD=compressed_answer \
      FIXED_LAYOUT_SUPERVISE_PAD=1 \
      FIXED_LAYOUT_FINAL_ANSWER_WEIGHT=4.0 \
      FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT=0.25 \
      SFT_NUM_T_PER_SAMPLE="${num_t}" \
      SFT_T_GRID="${t_grid}" \
      bash run_fixed_blocks_sft.sh
  fi
  eval_ckpt "${run}" "${ckpt}" "step${max_steps}"
}

log "post1000_controller_start base_run=${BASE_RUN}"
wait_for_file "${BASE_CKPT}" "base_ckpt1000"
stop_run_if_alive "${BASE_RUN}"
eval_ckpt "${BASE_RUN}" "${BASE_CKPT}" "step1000"

run_multit_experiment \
  "fixed_blocks_${COMMON_TAG}_random4_samefwd250" \
  250 \
  4 \
  ""

run_multit_experiment \
  "fixed_blocks_${COMMON_TAG}_tgrid001_01_1_6_samefwd250" \
  250 \
  1 \
  "0.001,0.01,0.1,0.6"

log "post1000_controller_done"
