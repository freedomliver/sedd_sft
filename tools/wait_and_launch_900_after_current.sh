#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

cd /root/sedd

CURRENT_RUN=fixed_blocks_simple400_pos256_q48_rs96_r96_as213_a16_activeonly_freshmid5_fa8pad0_seed10_train320_valid40_test40_r32a64_b160ga1_step3000
RUN=fixed_blocks_manual900_pos256_q48_rs64_r128_as213_a16_activeonly_freshmid5_fa8pad025_seed10_train720_valid90_test90_r32a64_b160ga1_step3000
LOG=logs/fixed_blocks/nohup_${RUN}.log

while pgrep -f "lora_${CURRENT_RUN}" >/dev/null 2>&1; do
  sleep 30
done

if pgrep -f '[t]rain_sft.py' >/dev/null 2>&1; then
  echo "another train_sft.py is still running; not launching ${RUN}"
  pgrep -af '[t]rain_sft.py'
  exit 1
fi

env \
  PRETRAINED=pretrained/sedd-small \
  DATA_JSON=data/simple_manual_synthetic/s1K_manual_synthetic_900_simple400_medium500_seed10_train720_valid90_test90.json \
  RUN_NAME=${RUN} OUT_DIR=/root/autodl-tmp/sedd_outputs/lora_${RUN} \
  TRAIN_SIZE=720 VALID_SIZE=90 TEST_SIZE=90 SEED=10 \
  MAX_LENGTH=256 QUESTION_BLOCK_LEN=48 FIXED_LAYOUT_REASONING_START=64 \
  REASONING_BLOCK_LEN=128 FIXED_LAYOUT_FINAL_ANSWER_START=213 FINAL_ANSWER_BLOCK_LEN=16 \
  REASONING_FIELD=compressed_reasoning FINAL_ANSWER_FIELD=compressed_answer \
  FIXED_LAYOUT_SUPERVISE_PAD=1 FIXED_LAYOUT_FINAL_ANSWER_WEIGHT=8.0 FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT=0.25 \
  BATCH_SIZE=160 GRAD_ACCUM=1 MAX_STEPS=3000 \
  SAVE_FREQ=250 LOG_FREQ=10 EVAL_FREQ=250 EVAL_BATCHES=5 \
  LR=3e-4 WARMUP_STEPS=50 LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
  SFT_NUM_T_PER_SAMPLE=1 REPLAY_BATCH_SIZE=0 FIXED_BATCH_T_GRID=0.03,0.1,0.3,0.6,0.9 \
  bash run_fixed_blocks_sft.sh > "${LOG}" 2>&1
