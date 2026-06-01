#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /root/sedd

echo "=========================================="
echo "Training v9: 200000 steps, boxed-only loss, sigma annealing, clip=1000"
echo "=========================================="
python train_sft.py 200000 2>&1 | tee logs/train_v9.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training done. Starting evaluation (200 samples)..."
    echo "=========================================="
    python eval_correct.py 2>&1 | tee logs/eval_v9.log
else
    echo "Training failed!"
    exit 1
fi
