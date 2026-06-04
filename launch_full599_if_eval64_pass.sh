#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:${PATH}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "${REPO_DIR}"

EVAL_RUN_NAME=${EVAL_RUN_NAME:-fixed_blocks_train64_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000}
EVAL_JSONL=${EVAL_JSONL:-outputs/fixed_blocks_eval/${EVAL_RUN_NAME}_train64_argmax128.jsonl}
FULL_RUN_NAME=${FULL_RUN_NAME:-fixed_blocks_full599_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step100000}
DRY_RUN=${DRY_RUN:-0}

MIN_MATCH=${MIN_MATCH:-0.90}
MIN_EOS=${MIN_EOS:-0.90}
MAX_EMPTY=${MAX_EMPTY:-0.05}

if [ ! -f "${EVAL_JSONL}" ]; then
  echo "missing_eval_jsonl=${EVAL_JSONL}" >&2
  exit 2
fi

echo "checking_eval64_gate=${EVAL_JSONL}"
python tools/check_fixed_eval_gate.py \
  "${EVAL_JSONL}" \
  --min-match "${MIN_MATCH}" \
  --min-eos "${MIN_EOS}" \
  --max-empty "${MAX_EMPTY}" \
  --show 12

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_full599=${FULL_RUN_NAME}"
  exit 0
fi

echo "launching_full599=${FULL_RUN_NAME}"
RUN_NAME="${FULL_RUN_NAME}" bash launch_fixed_blocks_full599.sh
