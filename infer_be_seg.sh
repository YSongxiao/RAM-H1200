#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MODE="${MODE:-test}"
MODEL="${MODEL:-UMambaEnc}"
CHECKPOINT="${CHECKPOINT:-/home/yafei/code/RAM-H1200/ckpts/Baseline_BESeg_umambaenc_20260405003952}"
DATA_PATH="${DATA_PATH:-/home/yafei/data/RAM-H1200/split_dataset_via_be_mask_remapped}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
SEED="${SEED:-2026}"
USE_COORDS="${USE_COORDS:-1}"
SAVE_NPY="${SAVE_NPY:-0}"
SAVE_OVERLAY="${SAVE_OVERLAY:-0}"
SAVE_UNCERTAINTY_OVERLAY="${SAVE_UNCERTAINTY_OVERLAY:-0}"
SAVE_CSV="${SAVE_CSV:-1}"

if [[ "${MODE}" != "infer" && "${MODE}" != "test" ]]; then
    echo "MODE must be 'infer' or 'test', got: ${MODE}" >&2
    exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Please set CHECKPOINT to a BE experiment directory containing model_best.pth or model_latest.pth." >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT directory does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ -z "${DATA_PATH}" ]]; then
    if [[ "${MODE}" == "infer" ]]; then
        echo "Please set DATA_PATH to the image directory for inference." >&2
    else
        echo "Please set DATA_PATH to the dataset root containing test/." >&2
    fi
    exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
    echo "DATA_PATH directory does not exist: ${DATA_PATH}" >&2
    exit 1
fi

cmd=(
    "${PYTHON_BIN}"
    main_be_seg.py
    --launcher none
    --mode "${MODE}"
    --model "${MODEL}"
    --checkpoint "${CHECKPOINT}"
    --data_path "${DATA_PATH}"
    --image_size "${IMAGE_SIZE}"
    --val_batch_size "${VAL_BATCH_SIZE}"
    --seed "${SEED}"
)

if [[ "${USE_COORDS}" == "1" ]]; then
    cmd+=(--use_coords)
else
    cmd+=(--no-use_coords)
fi

if [[ "${SAVE_NPY}" == "1" ]]; then
    cmd+=(--save_npy)
fi

if [[ "${SAVE_OVERLAY}" == "1" ]]; then
    cmd+=(--save_overlay)
fi

if [[ "${SAVE_UNCERTAINTY_OVERLAY}" == "1" ]]; then
    cmd+=(--save_uncertainty_overlay)
fi

if [[ "${SAVE_CSV}" == "1" ]]; then
    cmd+=(--save_csv)
fi

cmd+=("$@")

echo "=================================================="
echo "Launching single-GPU BE segmentation ${MODE}"
echo "GPU_ID=${GPU_ID}"
echo "MODEL=${MODEL}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"
echo "Command: ${cmd[*]}"
echo "=================================================="

"${cmd[@]}"
