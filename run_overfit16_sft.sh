#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs outputs

PRETRAINED=${PRETRAINED:-pretrained/sedd-medium}
DATA_JSON=${DATA_JSON:-data/s1K_train_answer_inner_le32.json}
SEED=${SEED:-0}
RUN_NAME=${RUN_NAME:-overfit16_seed${SEED}_boxed_prompt_noout_cefix}
OUT_DIR=${OUT_DIR:-outputs/lora_${RUN_NAME}}
TRAIN_SIZE=${TRAIN_SIZE:-16}
VALID_SIZE=${VALID_SIZE:-0}
TEST_SIZE=${TEST_SIZE:-0}
EVAL_LIMIT=${EVAL_LIMIT:-${TRAIN_SIZE}}
ANSWER_PREFIX=${ANSWER_PREFIX:-Answer:}
MAX_LENGTH=${MAX_LENGTH:-1024}
MIN_ANSWER_LEN=${MIN_ANSWER_LEN:-32}
MAX_ANSWER_LEN=${MAX_ANSWER_LEN:-32}
BOXED_PROMPT=${BOXED_PROMPT:-1}
LORA_TARGETS=${LORA_TARGETS:-attn_qkv,attn_out,mlp.0,mlp.2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
ALL_MASK_CE_ANSWER_WEIGHT=${ALL_MASK_CE_ANSWER_WEIGHT:-4.0}
ALL_MASK_CE_WEIGHT_STAGE1=${ALL_MASK_CE_WEIGHT_STAGE1:-30.0}
ALL_MASK_CE_WEIGHT_STAGE2=${ALL_MASK_CE_WEIGHT_STAGE2:-1.0}
STAGE1_LR=${STAGE1_LR:-3e-4}
STAGE2_LR=${STAGE2_LR:-5e-5}
STAGE1_STEPS=${STAGE1_STEPS:-300}
STAGE2_STEPS=${STAGE2_STEPS:-1000}
SAVE_FREQ=${SAVE_FREQ:-0}
EVAL_JSONL=${EVAL_JSONL:-logs/eval_${RUN_NAME}.jsonl}
EVAL_SAMPLE4_JSONL=${EVAL_SAMPLE4_JSONL:-logs/eval_${RUN_NAME}_sample4_steps128.jsonl}

COMMON_ARGS=(
  --pretrained "${PRETRAINED}"
  --data_json "${DATA_JSON}"
  --train_size "${TRAIN_SIZE}"
  --valid_size "${VALID_SIZE}"
  --test_size "${TEST_SIZE}"
  --seed "${SEED}"
  --max_length "${MAX_LENGTH}"
  --min_answer_len "${MIN_ANSWER_LEN}"
  --max_answer_len "${MAX_ANSWER_LEN}"
  --answer_field final_boxed
  --answer_prefix "${ANSWER_PREFIX}"
  --answer_leading_newline
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_targets "${LORA_TARGETS}"
  --all_mask_ce_t_values 0.0001,0.001,0.01,0.1,0.5,1.0
  --all_mask_ce_cropped
  --all_mask_ce_answer_weight "${ALL_MASK_CE_ANSWER_WEIGHT}"
  --eval_freq 0
  --save_freq "${SAVE_FREQ}"
  --offline
)

if [[ "${BOXED_PROMPT}" == "1" ]]; then
  COMMON_ARGS+=(--boxed_prompt)
fi

echo "== Stage 1: ${TRAIN_SIZE}-sample overfit, SEDD 1024 context =="
rm -rf "${OUT_DIR}"
python -u train_sft.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_DIR}/stage1" \
  --max_steps "${STAGE1_STEPS}" \
  --epochs 100000 \
  --lr "${STAGE1_LR}" \
  --warmup_steps 5 \
  --all_mask_ce_weight "${ALL_MASK_CE_WEIGHT_STAGE1}" \
  --log_freq 10 \
  2>&1 | tee "logs/${RUN_NAME}_stage1.log"

echo "== Stage 2: EOS/answer-window polish =="
python -u train_sft.py \
  "${COMMON_ARGS[@]}" \
  --resume_lora_ckpt "${OUT_DIR}/stage1/lora_final.pt" \
  --out_dir "${OUT_DIR}/final" \
  --max_steps "${STAGE2_STEPS}" \
  --epochs 100000 \
  --lr "${STAGE2_LR}" \
  --warmup_steps 5 \
  --all_mask_ce_weight "${ALL_MASK_CE_WEIGHT_STAGE2}" \
  --log_freq 5 \
  2>&1 | tee "logs/${RUN_NAME}_stage2.log"

echo "== Eval: target boxed correctness, exact target, EOS =="
EVAL_ARGS=(
  --pretrained "${PRETRAINED}" \
  --lora_ckpt "${OUT_DIR}/final/lora_final.pt" \
  --data_json "${DATA_JSON}" \
  --split train \
  --train_size "${TRAIN_SIZE}" \
  --valid_size "${VALID_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --seed "${SEED}" \
  --limit "${EVAL_LIMIT}" \
  --num_samples 1 \
  --steps 64 \
  --predictor analytic \
  --sampling_mode argmax \
  --max_length "${MAX_LENGTH}" \
  --min_answer_len "${MIN_ANSWER_LEN}" \
  --max_answer_len "${MAX_ANSWER_LEN}" \
  --answer_field final_boxed \
  --answer_prefix "${ANSWER_PREFIX}" \
  --answer_leading_newline \
  --out_jsonl "${EVAL_JSONL}" \
  --offline
)
if [[ "${BOXED_PROMPT}" == "1" ]]; then
  EVAL_ARGS+=(--boxed_prompt)
fi

python -u eval_correct.py "${EVAL_ARGS[@]}" 2>&1 | tee "logs/eval_${RUN_NAME}.log"

if ! grep -q "pass@1_boxed_match: ${EVAL_LIMIT}/${EVAL_LIMIT}" "logs/eval_${RUN_NAME}.log"; then
  echo "== Eval: sample pass@4, steps=128 =="
  EVAL_SAMPLE4_ARGS=("${EVAL_ARGS[@]}")
  EVAL_SAMPLE4_ARGS+=(
    --num_samples 4
    --steps 128
    --sampling_mode sample
    --out_jsonl "${EVAL_SAMPLE4_JSONL}"
  )
  python -u eval_correct.py "${EVAL_SAMPLE4_ARGS[@]}" 2>&1 | tee "logs/eval_${RUN_NAME}_sample4_steps128.log"
fi
