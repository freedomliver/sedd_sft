#!/bin/bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-data/compression_pilot/deepseek_q220_r512_parts_v3}
PASS_NAME=${PASS_NAME:-repair_pass_1}
INPUT=${INPUT:-data/s1K_train_599.json}
BATCH_REPAIR_ATTEMPTS=${BATCH_REPAIR_ATTEMPTS:-4}

cd "${REPO_DIR:-/root/sedd}"
source .api_env
export S1K_COMPRESS_API=${S1K_COMPRESS_API:-anthropic}
export S1K_COMPRESS_DISABLE_THINKING=${S1K_COMPRESS_DISABLE_THINKING:-1}
export S1K_COMPRESS_MAX_OUTPUT_TOKENS=${S1K_COMPRESS_MAX_OUTPUT_TOKENS:-1600}

REPAIR_DIR="${BASE_DIR}/${PASS_NAME}"
mkdir -p "${REPAIR_DIR}"

/root/miniconda3/bin/python tools/audit_merge_compressed.py \
  --base_dir "${BASE_DIR}" \
  --input "${INPUT}" \
  --max_question_tokens 220 \
  --max_reasoning_tokens 512 || true

FAILED="${BASE_DIR}/failed_indices.txt"
if [ ! -s "${FAILED}" ]; then
  echo "No failed indices to repair."
  exit 0
fi

OUT="${REPAIR_DIR}/output.jsonl"
ERR="${REPAIR_DIR}/errors.jsonl"
TRAIN="${REPAIR_DIR}/train.json"
LOG="${REPAIR_DIR}/repair.log"

echo "Repairing failed indices from ${FAILED}" | tee "${LOG}"
while read -r IDX; do
  [ -n "${IDX}" ] || continue
  echo "REPAIR_START index=${IDX}" | tee -a "${LOG}"
  /root/miniconda3/bin/python -u tools/compress_s1k.py \
    --input "${INPUT}" \
    --output_jsonl "${OUT}" \
    --errors_jsonl "${ERR}" \
    --train_out "${TRAIN}" \
    --start "${IDX}" --limit 1 \
    --max_question_tokens 220 --max_reasoning_tokens 512 \
    --temperature 0 \
    --max_retries 1 --retry_sleep 2 \
    --repair_attempts "${BATCH_REPAIR_ATTEMPTS}" \
    --resume 2>&1 | tee -a "${LOG}" || true
  echo "REPAIR_END index=${IDX}" | tee -a "${LOG}"
done < "${FAILED}"

/root/miniconda3/bin/python tools/audit_merge_compressed.py \
  --base_dir "${BASE_DIR}" \
  --input "${INPUT}" \
  --max_question_tokens 220 \
  --max_reasoning_tokens 512
