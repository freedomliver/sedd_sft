#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"
mkdir -p logs /root/autodl-tmp/sedd_outputs

PRETRAINED=${PRETRAINED:-pretrained/sedd-medium}
DATA_JSON=${DATA_JSON:-data/s1K_train_answer_inner_le32.json}
SEED=${SEED:-10}
RUN_NAME=${RUN_NAME:-full_le32_seed${SEED}_dwdse_only}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/sedd_outputs/lora_${RUN_NAME}}
RESUME_LORA_CKPT=${RESUME_LORA_CKPT:-}

TRAIN_SIZE=${TRAIN_SIZE:-0}
VALID_SIZE=${VALID_SIZE:-0}
TEST_SIZE=${TEST_SIZE:-0}
MAX_STEPS=${MAX_STEPS:-1600}
EPOCHS=${EPOCHS:-100000}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-400}
LOG_FREQ=${LOG_FREQ:-10}
EVAL_FREQ=${EVAL_FREQ:-0}
EVAL_BATCHES=${EVAL_BATCHES:-10}

MAX_LENGTH=${MAX_LENGTH:-1024}
MIN_ANSWER_LEN=${MIN_ANSWER_LEN:-32}
MAX_ANSWER_LEN=${MAX_ANSWER_LEN:-32}
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

echo "=========================================="
echo "Full le32 LoRA SFT"
echo "repo=${REPO_DIR}"
echo "base=${PRETRAINED}"
echo "data=${DATA_JSON}"
echo "out=${OUT_DIR}"
echo "train_size=${TRAIN_SIZE} valid_size=${VALID_SIZE} test_size=${TEST_SIZE}"
echo "max_steps=${MAX_STEPS} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} lr=${LR}"
echo "sft_t_min=${SFT_T_MIN} sft_t_max=${SFT_T_MAX:-default}"
echo "sft_high_t_frac=${SFT_HIGH_T_FRAC} sft_high_t_min=${SFT_HIGH_T_MIN} sft_high_t_max=${SFT_HIGH_T_MAX:-default}"
echo "=========================================="

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
  --min_answer_len "${MIN_ANSWER_LEN}" \
  --max_answer_len "${MAX_ANSWER_LEN}" \
  --answer_field final_boxed \
  --answer_prefix Answer: \
  --answer_leading_newline \
  --boxed_prompt \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --all_mask_ce_weight 0 \
  "${T_RANGE_ARG[@]}" \
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
  2>&1 | tee "logs/train_${RUN_NAME}.log"
