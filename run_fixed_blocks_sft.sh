#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs/fixed_blocks /root/autodl-tmp/sedd_outputs

PRETRAINED=${PRETRAINED:-pretrained/sedd-medium}
DATA_JSON=${DATA_JSON:-data/compression_pilot/deepseek_q220_r512_parts_v3/merged_train.json}
SEED=${SEED:-10}
RUN_NAME=${RUN_NAME:-fixed_blocks_599_q231_r530_a263_dwdseonly_seed${SEED}_r32a64_b8ga4_step1600}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}}
RESUME_LORA_CKPT=${RESUME_LORA_CKPT:-}

# train/valid/test all zero means: use all rows in DATA_JSON as train.
TRAIN_SIZE=${TRAIN_SIZE:-0}
VALID_SIZE=${VALID_SIZE:-0}
TEST_SIZE=${TEST_SIZE:-0}
MAX_STEPS=${MAX_STEPS:-1600}
EPOCHS=${EPOCHS:-100000}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-50}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
SAVE_FREQ=${SAVE_FREQ:-400}
LOG_FREQ=${LOG_FREQ:-10}
EVAL_FREQ=${EVAL_FREQ:-0}
EVAL_BATCHES=${EVAL_BATCHES:-10}

MAX_LENGTH=${MAX_LENGTH:-1024}
QUESTION_BLOCK_LEN=${QUESTION_BLOCK_LEN:-231}
REASONING_BLOCK_LEN=${REASONING_BLOCK_LEN:-530}
FINAL_ANSWER_BLOCK_LEN=${FINAL_ANSWER_BLOCK_LEN:-263}
FIXED_LAYOUT_REASONING_START=${FIXED_LAYOUT_REASONING_START:-}
FIXED_LAYOUT_FINAL_ANSWER_START=${FIXED_LAYOUT_FINAL_ANSWER_START:-}
REASONING_FIELD=${REASONING_FIELD:-solution}
FINAL_ANSWER_FIELD=${FINAL_ANSWER_FIELD:-answer}
FIXED_LAYOUT_SUPERVISE_PAD=${FIXED_LAYOUT_SUPERVISE_PAD:-0}
FIXED_LAYOUT_FINAL_ANSWER_WEIGHT=${FIXED_LAYOUT_FINAL_ANSWER_WEIGHT:-1.0}
FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT=${FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT:-1.0}

LORA_TARGETS=${LORA_TARGETS:-attn_qkv,attn_out,mlp.0,mlp.2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
FINETUNE_MODE=${FINETUNE_MODE:-lora}
SFT_T_MIN=${SFT_T_MIN:-0.0}
SFT_T_MAX=${SFT_T_MAX:-}
SFT_HIGH_T_FRAC=${SFT_HIGH_T_FRAC:-0.0}
SFT_HIGH_T_MIN=${SFT_HIGH_T_MIN:-0.75}
SFT_HIGH_T_MAX=${SFT_HIGH_T_MAX:-}
SFT_NUM_T_PER_SAMPLE=${SFT_NUM_T_PER_SAMPLE:-1}
SFT_T_GRID=${SFT_T_GRID:-}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-0}
REPLAY_T_GRID=${REPLAY_T_GRID:-}
FIXED_BATCH_T_GRID=${FIXED_BATCH_T_GRID:-}
REPLAY_CACHE_SIZE=${REPLAY_CACHE_SIZE:-4096}

RESUME_ARG=()
if [ -n "${RESUME_LORA_CKPT}" ]; then
  RESUME_ARG=(--resume_lora_ckpt "${RESUME_LORA_CKPT}")
fi
T_RANGE_ARG=(--sft_t_min "${SFT_T_MIN}")
if [ -n "${SFT_T_MAX}" ]; then
  T_RANGE_ARG+=(--sft_t_max "${SFT_T_MAX}")
fi
if [ "${SFT_HIGH_T_FRAC}" != "0.0" ] && [ "${SFT_HIGH_T_FRAC}" != "0" ]; then
  T_RANGE_ARG+=(--sft_high_t_frac "${SFT_HIGH_T_FRAC}" --sft_high_t_min "${SFT_HIGH_T_MIN}")
  if [ -n "${SFT_HIGH_T_MAX}" ]; then
    T_RANGE_ARG+=(--sft_high_t_max "${SFT_HIGH_T_MAX}")
  fi
fi
T_RANGE_ARG+=(--num_t_per_sample "${SFT_NUM_T_PER_SAMPLE}")
if [ -n "${SFT_T_GRID}" ]; then
  T_RANGE_ARG+=(--sft_t_grid "${SFT_T_GRID}")
fi
PAD_ARG=()
if [ "${FIXED_LAYOUT_SUPERVISE_PAD}" != "0" ]; then
  PAD_ARG=(--fixed_layout_supervise_pad)
fi
SPARSE_LAYOUT_ARG=()
if [ -n "${FIXED_LAYOUT_REASONING_START}" ]; then
  SPARSE_LAYOUT_ARG=(--fixed_layout_reasoning_start "${FIXED_LAYOUT_REASONING_START}")
fi
if [ -n "${FIXED_LAYOUT_FINAL_ANSWER_START}" ]; then
  SPARSE_LAYOUT_ARG+=(
    --fixed_layout_final_answer_start "${FIXED_LAYOUT_FINAL_ANSWER_START}"
  )
fi

echo "=========================================="
echo "Fixed-block compressed reasoning LoRA SFT"
echo "repo=${REPO_DIR}"
echo "base=${PRETRAINED}"
echo "data=${DATA_JSON}"
echo "out=${OUT_DIR}"
echo "train_size=${TRAIN_SIZE} valid_size=${VALID_SIZE} test_size=${TEST_SIZE}"
echo "max_steps=${MAX_STEPS} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} lr=${LR}"
echo "sft_num_t_per_sample=${SFT_NUM_T_PER_SAMPLE} sft_t_grid=${SFT_T_GRID:-random}"
echo "replay_batch_size=${REPLAY_BATCH_SIZE} replay_t_grid=${REPLAY_T_GRID:-none} fixed_batch_t_grid=${FIXED_BATCH_T_GRID:-none} replay_cache_size=${REPLAY_CACHE_SIZE}"
if [ -n "${FIXED_LAYOUT_REASONING_START}" ]; then
  REASONING_START_FOR_LOG=${FIXED_LAYOUT_REASONING_START}
else
  REASONING_START_FOR_LOG=${QUESTION_BLOCK_LEN}
fi
if [ -n "${FIXED_LAYOUT_FINAL_ANSWER_START}" ]; then
  ANSWER_START_FOR_LOG=${FIXED_LAYOUT_FINAL_ANSWER_START}
else
  ANSWER_START_FOR_LOG=$((REASONING_START_FOR_LOG + REASONING_BLOCK_LEN))
fi
echo "layout question=[0,${QUESTION_BLOCK_LEN}) reasoning=[${REASONING_START_FOR_LOG},$((REASONING_START_FOR_LOG + REASONING_BLOCK_LEN))) answer=[${ANSWER_START_FOR_LOG},$((ANSWER_START_FOR_LOG + FINAL_ANSWER_BLOCK_LEN))) max_length=${MAX_LENGTH}"
echo "fields reasoning=${REASONING_FIELD} final_answer=${FINAL_ANSWER_FIELD}"
echo "all_mask_ce_weight=0 fixed_layout_supervise_pad=${FIXED_LAYOUT_SUPERVISE_PAD} fixed_layout_final_answer_weight=${FIXED_LAYOUT_FINAL_ANSWER_WEIGHT} fixed_layout_final_answer_pad_weight=${FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT}"
echo "=========================================="

python -u train_sft.py \
  --pretrained "${PRETRAINED}" \
  --data_json "${DATA_JSON}" \
  "${RESUME_ARG[@]}" \
  --out_dir "${OUT_DIR}" \
  --train_size "${TRAIN_SIZE}" \
  --valid_size "${VALID_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --seed "${SEED}" \
  --max_length "${MAX_LENGTH}" \
  --fixed_layout \
  --question_block_len "${QUESTION_BLOCK_LEN}" \
  --reasoning_block_len "${REASONING_BLOCK_LEN}" \
  --final_answer_block_len "${FINAL_ANSWER_BLOCK_LEN}" \
  "${SPARSE_LAYOUT_ARG[@]}" \
  --reasoning_field "${REASONING_FIELD}" \
  --final_answer_field "${FINAL_ANSWER_FIELD}" \
  "${PAD_ARG[@]}" \
  --fixed_layout_final_answer_weight "${FIXED_LAYOUT_FINAL_ANSWER_WEIGHT}" \
  --fixed_layout_final_answer_pad_weight "${FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --all_mask_ce_weight 0 \
  "${T_RANGE_ARG[@]}" \
  --replay_batch_size "${REPLAY_BATCH_SIZE}" \
  --replay_t_grid "${REPLAY_T_GRID}" \
  --batch_t_grid "${FIXED_BATCH_T_GRID}" \
  --replay_cache_size "${REPLAY_CACHE_SIZE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_targets "${LORA_TARGETS}" \
  --finetune_mode "${FINETUNE_MODE}" \
  --eval_freq "${EVAL_FREQ}" \
  --eval_batches "${EVAL_BATCHES}" \
  --save_freq "${SAVE_FREQ}" \
  --log_freq "${LOG_FREQ}" \
  --offline \
  2>&1 | tee "logs/fixed_blocks/train_${RUN_NAME}.log"
