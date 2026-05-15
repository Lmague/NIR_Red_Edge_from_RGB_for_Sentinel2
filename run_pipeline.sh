#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-MicroNIR}"
TRAIN_DATA="${TRAIN_DATA:-gs://YOUR_BUCKET/dataset_training_mst/train/*.tfrecord.gz}"
VAL_DATA="${VAL_DATA:-gs://YOUR_BUCKET/dataset_training_mst/test/*.tfrecord.gz}"
CKPT_DIR="${CKPT_DIR:-checkpoints_${MODEL_NAME}}"

echo "===================================================="
echo " ${MODEL_NAME} - phase 1 (L1 + SAM)"
echo "===================================================="

python train.py \
    --model_name   "$MODEL_NAME" \
    --train_data   "$TRAIN_DATA" \
    --val_data     "$VAL_DATA" \
    --lambda_l1    1.0 \
    --lambda_freq  0.0 \
    --lambda_wavelet 0.0 \
    --checkpoint_dir "${CKPT_DIR}/phase1"

PHASE1_CKPT="${CKPT_DIR}/phase1/best.weights.h5"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "Phase 1 best checkpoint not found at $PHASE1_CKPT - aborting." >&2
    exit 1
fi

echo
echo "===================================================="
echo " ${MODEL_NAME} - phase 2 (+ freq + wavelet, fine-tune)"
echo "===================================================="

python train.py \
    --model_name   "$MODEL_NAME" \
    --train_data   "$TRAIN_DATA" \
    --val_data     "$VAL_DATA" \
    --lambda_l1    1.0 \
    --lambda_freq  0.1 \
    --lambda_wavelet 0.1 \
    --learning_rate 1e-5 \
    --epochs 10 \
    --warmup_epochs 1 \
    --checkpoint_dir "${CKPT_DIR}/phase2" \
    --resume_from "$PHASE1_CKPT"

echo
echo "Done. Best phase 2 weights at ${CKPT_DIR}/phase2/best.weights.h5"
