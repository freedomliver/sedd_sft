#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs outputs/lora_sft

PRETRAINED=${PRETRAINED:-pretrained/sedd-medium}
DATA_JSON=${DATA_JSON:-}
OUT_DIR=${OUT_DIR:-outputs/lora_sft}
MAX_STEPS=${MAX_STEPS:-1000}
RESUME_LORA_CKPT=${RESUME_LORA_CKPT:-}
TRAIN_SIZE=${TRAIN_SIZE:-800}
VALID_SIZE=${VALID_SIZE:-100}
TEST_SIZE=${TEST_SIZE:-100}

echo "=========================================="
echo "Training LoRA response-only SEDD SFT"
echo "base=${PRETRAINED} out=${OUT_DIR} max_steps=${MAX_STEPS}"
echo "=========================================="

DATA_ARG=()
if [ -n "${DATA_JSON}" ]; then
  DATA_ARG=(--data_json "${DATA_JSON}")
fi

RESUME_ARG=()
if [ -n "${RESUME_LORA_CKPT}" ]; then
  RESUME_ARG=(--resume_lora_ckpt "${RESUME_LORA_CKPT}")
fi

python -u train_sft.py \
  --pretrained "${PRETRAINED}" \
  "${DATA_ARG[@]}" \
  "${RESUME_ARG[@]}" \
  --out_dir "${OUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --train_size "${TRAIN_SIZE}" \
  --valid_size "${VALID_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 5e-5 \
  --warmup_steps 100 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_answer_len 512 \
  --offline \
  2>&1 | tee logs/train_lora_sft.log

echo ""
echo "=========================================="
echo "Evaluating generation"
echo "=========================================="

python -u eval_correct.py \
  --pretrained "${PRETRAINED}" \
  --lora_ckpt "${OUT_DIR}/lora_final.pt" \
  "${DATA_ARG[@]}" \
  --split test \
  --train_size "${TRAIN_SIZE}" \
  --valid_size "${VALID_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --limit 100 \
  --steps 128 \
  --max_answer_len 512 \
  --out_jsonl logs/eval_lora_sft.jsonl \
  --offline \
  2>&1 | tee logs/eval_lora_sft.log
