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

MODE="${MODE:-train}"
SCORE_TYPE="${SCORE_TYPE:-BE}"
MODEL="${MODEL:-ResNet34}"
TRIAL_NAME="${TRIAL_NAME:-Benchmark_BEScoring}"
CHECKPOINT="${CHECKPOINT:-./ckpts}"
DATA_PATH="${DATA_PATH:-/home/yafei/data/RAM-H1200/be_joints_dataset_remapped_renamed}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_EPOCH="${MAX_EPOCH:-200}"
LR="${LR:-1e-4}"
SCHEDULER="${SCHEDULER:-CosineAnnealing}"
SEED="${SEED:-2026}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MONITOR_METRIC="${MONITOR_METRIC:-qwk}"
EARLYSTOP_PATIENCE="${EARLYSTOP_PATIENCE:-20}"
OVERSAMPLE_POWER="${OVERSAMPLE_POWER:-1.0}"

AMP="${AMP:-0}"
TO_RGB="${TO_RGB:-0}"
OVERSAMPLE="${OVERSAMPLE:-0}"
USE_CLASS_WEIGHT="${USE_CLASS_WEIGHT:-0}"
EARLYSTOP="${EARLYSTOP:-0}"
PIN_MEMORY="${PIN_MEMORY:-1}"

# 单卡顺序实验。格式：
#   "最终TrialName::额外参数"
# 例如：
# EXPERIMENTS=(
#   "Baseline_BEScore::--model ResNet34 --score_type BE"
#   "MedMamba_BEScore::--model MedMamba --score_type BE --amp"
# )
# 留空时按上面的默认变量只跑一个实验。
EXPERIMENTS=(
    "Baseline_BEScore::--model ConvNeXtV2 --score_type BE"
    "Baseline_BEScore::--model EfficientNetV2 --score_type BE"
    # "Baseline_BEScore::--model MambaVisionT --score_type BE"
    # "Baseline_BEScore::--model ResNet34 --score_type BE"
    # "Baseline_BEScore::--model DenseNet --score_type BE"
    # "Baseline_BEScore::--model MedMamba --score_type BE"
    # "Baseline_BEScore::--model MambaVisionT --score_type BE"
    # "Baseline_BEScore::--model EfficientFormer --score_type BE"
    # "Baseline_BEScore::--model LeViT --score_type BE"
    # "Baseline_BEScore::--model MobileViT --score_type BE"
)

run_experiment() {
    local exp_name="$1"
    shift

    local exp_trial_name="${TRIAL_NAME}"
    if [[ -n "${exp_name}" ]]; then
        exp_trial_name="${exp_name}"
    fi

    local cmd=(
        "${PYTHON_BIN}"
        main_score_cls.py
        --mode "${MODE}"
        --score_type "${SCORE_TYPE}"
        --model "${MODEL}"
        --trial_name "${exp_trial_name}"
        --checkpoint "${CHECKPOINT}"
        --data_path "${DATA_PATH}"
        --image_size "${IMAGE_SIZE}"
        --train_batch_size "${TRAIN_BATCH_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}"
        --max_epoch "${MAX_EPOCH}"
        --lr "${LR}"
        --scheduler "${SCHEDULER}"
        --seed "${SEED}"
        --num_workers "${NUM_WORKERS}"
        --monitor_metric "${MONITOR_METRIC}"
        --earlystop_patience "${EARLYSTOP_PATIENCE}"
        --oversample_power "${OVERSAMPLE_POWER}"
    )

    if [[ "${AMP}" == "1" ]]; then
        cmd+=(--amp)
    else
        cmd+=(--no-amp)
    fi

    if [[ "${TO_RGB}" == "1" ]]; then
        cmd+=(--to_rgb)
    else
        cmd+=(--no-to_rgb)
    fi

    if [[ "${OVERSAMPLE}" == "1" ]]; then
        cmd+=(--oversample)
    fi

    if [[ "${USE_CLASS_WEIGHT}" == "1" ]]; then
        cmd+=(--use_class_weight)
    fi

    if [[ "${EARLYSTOP}" == "1" ]]; then
        cmd+=(--earlystop)
    fi

    if [[ "${PIN_MEMORY}" == "1" ]]; then
        cmd+=(--pin_memory)
    else
        cmd+=(--no-pin_memory)
    fi

    cmd+=("$@")

    echo "=================================================="
    echo "Launching single-GPU cls experiment: ${exp_trial_name}"
    echo "GPU_ID=${GPU_ID}"
    echo "MODEL=${MODEL} | SCORE_TYPE=${SCORE_TYPE}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    echo "Launching single classification experiment"
    echo "GPU_ID=${GPU_ID}"
    echo "MODEL=${MODEL}"
    echo "SCORE_TYPE=${SCORE_TYPE}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "TRIAL_NAME=${TRIAL_NAME}"
    echo "CHECKPOINT=${CHECKPOINT}"
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} classification experiments sequentially"
echo "GPU_ID=${GPU_ID}"
echo "Fallback TRIAL_NAME=${TRIAL_NAME}"
echo "SCORE_TYPE=${SCORE_TYPE}"
echo "DATA_PATH=${DATA_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"

failed_experiments=()

for idx in "${!EXPERIMENTS[@]}"; do
    spec="${EXPERIMENTS[idx]}"
    exp_idx=$((idx + 1))
    exp_name="exp$(printf '%02d' "${exp_idx}")"
    exp_args_str="${spec}"

    if [[ "${spec}" == *"::"* ]]; then
        exp_name="${spec%%::*}"
        exp_args_str="${spec#*::}"
    fi

    if [[ -z "${exp_name}" ]]; then
        exp_name="exp$(printf '%02d' "${exp_idx}")"
    fi

    read -r -a exp_args <<< "${exp_args_str}"

    echo "[${exp_idx}/${#EXPERIMENTS[@]}] Starting ${exp_name}"
    if run_experiment "${exp_name}" "${exp_args[@]}" "$@"; then
        echo "[${exp_idx}/${#EXPERIMENTS[@]}] Finished ${exp_name}"
    else
        echo "[${exp_idx}/${#EXPERIMENTS[@]}] Failed ${exp_name}"
        failed_experiments+=("${exp_name}")
        exit 1
    fi
done

echo "All classification experiments finished successfully."
