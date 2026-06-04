#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks outputs/fixed_blocks_eval

PRETRAINED=${PRETRAINED:-pretrained/sedd-medium}
DATA_JSON=${DATA_JSON:-data/compression_pilot/deepseek_q220_r512_parts_v3/merged_train.json}
RUN_NAME=${RUN_NAME:-fixed_blocks_train64_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000}
LORA_CKPT=${LORA_CKPT:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}/lora_final.pt}
SEED=${SEED:-10}

LIMIT=${LIMIT:-64}
START_IDX=${START_IDX:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
STEPS=${STEPS:-128}
SAMPLING_MODE=${SAMPLING_MODE:-argmax}

MAX_LENGTH=${MAX_LENGTH:-1024}
QUESTION_BLOCK_LEN=${QUESTION_BLOCK_LEN:-231}
REASONING_BLOCK_LEN=${REASONING_BLOCK_LEN:-530}
FINAL_ANSWER_BLOCK_LEN=${FINAL_ANSWER_BLOCK_LEN:-263}
REASONING_FIELD=${REASONING_FIELD:-solution}
FINAL_ANSWER_FIELD=${FINAL_ANSWER_FIELD:-answer}

OUT_JSONL=${OUT_JSONL:-outputs/fixed_blocks_eval/${RUN_NAME}_train64_${SAMPLING_MODE}${STEPS}.jsonl}
LOG_FILE=${LOG_FILE:-logs/fixed_blocks/eval_${RUN_NAME}_train64_${SAMPLING_MODE}${STEPS}.log}

echo "=========================================="
echo "Fixed-block eval64"
echo "data=${DATA_JSON}"
echo "ckpt=${LORA_CKPT}"
echo "limit=${LIMIT} steps=${STEPS} sampling=${SAMPLING_MODE}"
echo "out=${OUT_JSONL}"
echo "=========================================="

python -u eval_fixed_blocks.py \
  --pretrained "${PRETRAINED}" \
  --lora_ckpt "${LORA_CKPT}" \
  --data_json "${DATA_JSON}" \
  --split train \
  --train_size 64 \
  --valid_size 0 \
  --test_size 0 \
  --seed "${SEED}" \
  --start_idx "${START_IDX}" \
  --limit "${LIMIT}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --steps "${STEPS}" \
  --sampling_mode "${SAMPLING_MODE}" \
  --max_length "${MAX_LENGTH}" \
  --question_block_len "${QUESTION_BLOCK_LEN}" \
  --reasoning_block_len "${REASONING_BLOCK_LEN}" \
  --final_answer_block_len "${FINAL_ANSWER_BLOCK_LEN}" \
  --reasoning_field "${REASONING_FIELD}" \
  --final_answer_field "${FINAL_ANSWER_FIELD}" \
  --dtype bfloat16 \
  --offline \
  --out_jsonl "${OUT_JSONL}" \
  2>&1 | tee "${LOG_FILE}"
