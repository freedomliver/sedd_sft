#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

cd /root/sedd
mkdir -p logs/fixed_blocks /root/autodl-tmp/sedd_outputs

DATA_JSON=data/simple_manual_synthetic/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json
BASE_RUN=fixed_blocks_simple400_pos256_q48_rs96_r96_as213_a16_activeonly_freshmid5_fa4pad025_seed10_train320_valid40_test40_r32a64_b160ga1_step3000

checkpoint_path() {
  local run_name=$1
  echo "/root/autodl-tmp/sedd_outputs/lora_${run_name}/checkpoint_step_3000.pt"
}

wait_for_checkpoint() {
  local run_name=$1
  local ckpt
  ckpt=$(checkpoint_path "${run_name}")
  echo "WAIT checkpoint ${ckpt}"
  while [ ! -f "${ckpt}" ]; do
    if ! pgrep -f "lora_${run_name}" >/dev/null 2>&1; then
      echo "ERROR: no process for ${run_name}, checkpoint missing: ${ckpt}" >&2
      return 1
    fi
    sleep 30
  done
  echo "READY ${ckpt}"
}

eval_run() {
  local run_name=$1
  local ckpt
  ckpt=$(checkpoint_path "${run_name}")
  for split_limit in train:64 valid:40 test:40 train:320; do
    local split=${split_limit%%:*}
    local limit=${split_limit##*:}
    local elog=logs/fixed_blocks/eval_${run_name}_step3000_${split}${limit}_argmax64.log
    local out=logs/fixed_blocks/eval_${run_name}_step3000_${split}${limit}_argmax64.jsonl
    echo "EVAL run=${run_name} split=${split} limit=${limit}"
    /root/miniconda3/bin/python -u eval_fixed_blocks.py \
      --pretrained pretrained/sedd-small \
      --lora_ckpt "${ckpt}" \
      --data_json "${DATA_JSON}" \
      --split "${split}" --train_size 320 --valid_size 40 --test_size 40 --seed 10 \
      --max_length 256 \
      --question_block_len 48 \
      --fixed_layout_reasoning_start 96 \
      --reasoning_block_len 96 \
      --fixed_layout_final_answer_start 213 \
      --final_answer_block_len 16 \
      --reasoning_field compressed_reasoning \
      --final_answer_field compressed_answer \
      --fixed_layout_supervise_pad \
      --steps 64 --sampling_mode argmax --eval_batch_size 16 --limit "${limit}" \
      --offline --out_jsonl "${out}" > "${elog}" 2>&1
    tail -n 8 "${elog}"
  done
}

train_run() {
  local run_name=$1
  local final_answer_weight=$2
  local final_answer_pad_weight=$3
  local ckpt
  ckpt=$(checkpoint_path "${run_name}")
  if [ -f "${ckpt}" ]; then
    echo "SKIP train existing checkpoint ${ckpt}"
    return 0
  fi
  echo "TRAIN run=${run_name} final_answer_weight=${final_answer_weight} final_answer_pad_weight=${final_answer_pad_weight}"
  env \
    PRETRAINED=pretrained/sedd-small \
    DATA_JSON="${DATA_JSON}" \
    RUN_NAME="${run_name}" \
    OUT_DIR="/root/autodl-tmp/sedd_outputs/lora_${run_name}" \
    TRAIN_SIZE=320 VALID_SIZE=40 TEST_SIZE=40 SEED=10 \
    MAX_LENGTH=256 \
    QUESTION_BLOCK_LEN=48 \
    FIXED_LAYOUT_REASONING_START=96 \
    REASONING_BLOCK_LEN=96 \
    FIXED_LAYOUT_FINAL_ANSWER_START=213 \
    FINAL_ANSWER_BLOCK_LEN=16 \
    REASONING_FIELD=compressed_reasoning \
    FINAL_ANSWER_FIELD=compressed_answer \
    FIXED_LAYOUT_SUPERVISE_PAD=1 \
    FIXED_LAYOUT_FINAL_ANSWER_WEIGHT="${final_answer_weight}" \
    FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT="${final_answer_pad_weight}" \
    BATCH_SIZE=160 GRAD_ACCUM=1 MAX_STEPS=3000 \
    SAVE_FREQ=250 LOG_FREQ=10 EVAL_FREQ=250 EVAL_BATCHES=5 \
    LR=3e-4 WARMUP_STEPS=50 \
    LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
    SFT_NUM_T_PER_SAMPLE=1 REPLAY_BATCH_SIZE=0 \
    FIXED_BATCH_T_GRID=0.03,0.1,0.3,0.6,0.9 \
    bash run_fixed_blocks_sft.sh > "logs/fixed_blocks/nohup_${run_name}.log" 2>&1
}

echo "controller_start $(date)"
wait_for_checkpoint "${BASE_RUN}"
eval_run "${BASE_RUN}"

RUN_W8P025=fixed_blocks_simple400_pos256_q48_rs96_r96_as213_a16_activeonly_freshmid5_fa8pad025_seed10_train320_valid40_test40_r32a64_b160ga1_step3000
train_run "${RUN_W8P025}" 8.0 0.25
eval_run "${RUN_W8P025}"

RUN_W8P0=fixed_blocks_simple400_pos256_q48_rs96_r96_as213_a16_activeonly_freshmid5_fa8pad0_seed10_train320_valid40_test40_r32a64_b160ga1_step3000
train_run "${RUN_W8P0}" 8.0 0.0
eval_run "${RUN_W8P0}"

echo "controller_done $(date)"
