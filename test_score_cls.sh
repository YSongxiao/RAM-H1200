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
SCORE_TYPE="${SCORE_TYPE:-BE}"
MODEL="${MODEL:-ResNet34}"
CHECKPOINT="${CHECKPOINT:-/mnt/data1/songxiao/RAM-H1200/ckpts/Benchmark_BEScoring_be_resnet34_20260406022214}"
DATA_PATH="${DATA_PATH:-/mnt/data2/datasx/FullHand/NIPS26/be_joints_dataset_remapped_renamed}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
SEED="${SEED:-2026}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"
TO_RGB="${TO_RGB:-0}"
SAVE_CSV="${SAVE_CSV:-1}"

if [[ "${MODE}" != "test" ]]; then
    echo "MODE must be 'test', got: ${MODE}" >&2
    exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Please set CHECKPOINT to a scoring experiment directory containing model_best.pth." >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT directory does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}/model_best.pth" ]]; then
    echo "model_best.pth not found under CHECKPOINT: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ -z "${DATA_PATH}" ]]; then
    echo "Please set DATA_PATH to the scoring dataset root containing test/." >&2
    exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
    echo "DATA_PATH directory does not exist: ${DATA_PATH}" >&2
    exit 1
fi

cmd=(
    "${PYTHON_BIN}"
    main_score_cls.py
    --mode "${MODE}"
    --score_type "${SCORE_TYPE}"
    --model "${MODEL}"
    --checkpoint "${CHECKPOINT}"
    --data_path "${DATA_PATH}"
    --image_size "${IMAGE_SIZE}"
    --val_batch_size "${VAL_BATCH_SIZE}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
    --no-amp
    --no-to_rgb
)

if [[ "${PIN_MEMORY}" == "1" ]]; then
    cmd+=(--pin_memory)
else
    cmd+=(--no-pin_memory)
fi

if [[ "${TO_RGB}" == "1" ]]; then
    cmd+=(--to_rgb)
fi

if [[ "${SAVE_CSV}" == "1" ]]; then
    cmd+=(--save_csv)
fi

cmd+=("$@")

echo "=================================================="
echo "Launching scoring ordinal-regression test"
echo "GPU_ID=${GPU_ID}"
echo "SCORE_TYPE=${SCORE_TYPE}"
echo "MODEL=${MODEL}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"
echo "Command: ${cmd[*]}"
echo "=================================================="

"${cmd[@]}"
