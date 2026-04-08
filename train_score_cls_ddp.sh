#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

count_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local devices="${CUDA_VISIBLE_DEVICES// /}"
        IFS=',' read -r -a gpu_ids <<< "${devices}"
        echo "${#gpu_ids[@]}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --list-gpus | wc -l
        return
    fi

    echo 1
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS// /}"

NUM_GPUS="${NUM_GPUS:-$(count_visible_gpus)}"
MASTER_PORT="${MASTER_PORT:-29500}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

MODE="${MODE:-train}"
SCORE_TYPE="${SCORE_TYPE:-BE}"
MODEL="${MODEL:-ResNet34}"
TRIAL_NAME="${TRIAL_NAME:-Benchmark_BEScoring_DDP}"
CHECKPOINT="${CHECKPOINT:-./ckpts}"
DATA_PATH="${DATA_PATH:-/home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_EPOCH="${MAX_EPOCH:-200}"
LR="${LR:-1e-4}"
SCHEDULER="${SCHEDULER:-CosineAnnealing}"
SEED="${SEED:-2026}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MONITOR_METRIC="${MONITOR_METRIC:-qwk}"
EARLYSTOP_PATIENCE="${EARLYSTOP_PATIENCE:-20}"
OVERSAMPLE_POWER="${OVERSAMPLE_POWER:-1.0}"

AMP="${AMP:-0}"
TO_RGB="${TO_RGB:-0}"
OVERSAMPLE="${OVERSAMPLE:-0}"
USE_CLASS_WEIGHT="${USE_CLASS_WEIGHT:-0}"
EARLYSTOP="${EARLYSTOP:-0}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"

# 多卡顺序实验。格式：
#   "最终TrialName::额外参数"
# 例如：
# EXPERIMENTS=(
#   "ResNet34_JSN_DDP::--model ResNet34 --score_type JSN"
#   "ConvNeXtV2_JSN_DDP::--model ConvNeXtV2 --score_type JSN --amp"
# )
# 留空时按上面的默认变量只跑一个实验。
EXPERIMENTS=(
    # "ResNet34_JSN::--model ResNet34 --score_type JSN"
    # "DenseNet_JSN::--model DenseNet --score_type JSN"
    # "MedMamba_JSN::--model MedMamba --score_type JSN"
    # "EfficientFormer_JSN::--model EfficientFormer --score_type JSN"
    "LeViT_JSN::--model LeViT --score_type JSN"
    "MobileViT_JSN::--model MobileViT --score_type JSN"
    "ConvNeXtV2_JSN::--model ConvNeXtV2 --score_type JSN"
    "EfficientNetV2_JSN::--model EfficientNetV2 --score_type JSN"
    "MambaVisionT_JSN::--model MambaVisionT --score_type JSN"
)

python_module_available() {
    local module_name="$1"
    "${PYTHON_BIN}" -c "import importlib.util; print(importlib.util.find_spec('${module_name}') is not None)" 2>/dev/null | grep -qx "True"
}

model_is_available() {
    local model_name="$1"
    case "${model_name}" in
        MedMamba)
            python_module_available "mamba_ssm" || python_module_available "selective_scan"
            ;;
        MambaVisionT|MambaVisionT2|MambaVisionS)
            python_module_available "mambavision"
            ;;
        *)
            return 0
            ;;
    esac
}

extract_model_name() {
    local -a spec_args=("$@")
    local idx
    for ((idx = 0; idx < ${#spec_args[@]}; idx++)); do
        if [[ "${spec_args[idx]}" == "--model" ]] && (( idx + 1 < ${#spec_args[@]} )); then
            echo "${spec_args[idx + 1]}"
            return 0
        fi
    done
    echo "${MODEL}"
}

run_experiment() {
    local exp_name="$1"
    shift

    local exp_trial_name="${TRIAL_NAME}"
    if [[ -n "${exp_name}" ]]; then
        exp_trial_name="${exp_name}"
    fi

    local cmd=(
        "${TORCHRUN_BIN}"
        --nproc_per_node="${NUM_GPUS}"
        --master_port="${MASTER_PORT}"
        main_score_cls.py
        --launcher ddp
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
        --prefetch_factor "${PREFETCH_FACTOR}"
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

    if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
        cmd+=(--persistent_workers)
    else
        cmd+=(--no-persistent_workers)
    fi

    cmd+=("$@")

    echo "=================================================="
    echo "Launching multi-GPU cls experiment: ${exp_trial_name}"
    echo "GPU(s): ${NUM_GPUS} | GPU_IDS=${CUDA_VISIBLE_DEVICES:-all} | MASTER_PORT=${MASTER_PORT}"
    echo "MODEL=${MODEL} | SCORE_TYPE=${SCORE_TYPE}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    echo "Launching single multi-GPU classification experiment"
    echo "GPU_IDS=${CUDA_VISIBLE_DEVICES:-all}"
    echo "NUM_GPUS=${NUM_GPUS}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "MODEL=${MODEL}"
    echo "SCORE_TYPE=${SCORE_TYPE}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "TRIAL_NAME=${TRIAL_NAME}"
    echo "CHECKPOINT=${CHECKPOINT}"
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} classification experiments sequentially with ${NUM_GPUS} GPU(s)"
echo "GPU_IDS=${CUDA_VISIBLE_DEVICES:-all}"
echo "MASTER_PORT=${MASTER_PORT}"
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
    exp_model="$(extract_model_name "${exp_args[@]}")"

    if ! model_is_available "${exp_model}"; then
        echo "[${exp_idx}/${#EXPERIMENTS[@]}] Skipping ${exp_name} because dependency is unavailable for model=${exp_model}"
        continue
    fi

    echo "[${exp_idx}/${#EXPERIMENTS[@]}] Starting ${exp_name}"
    if run_experiment "${exp_name}" "${exp_args[@]}" "$@"; then
        echo "[${exp_idx}/${#EXPERIMENTS[@]}] Finished ${exp_name}"
    else
        echo "[${exp_idx}/${#EXPERIMENTS[@]}] Failed ${exp_name}"
        failed_experiments+=("${exp_name}")
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
            echo "STOP_ON_ERROR=1, stopping batch run."
            exit 1
        fi
    fi
done

if [[ "${#failed_experiments[@]}" -gt 0 ]]; then
    echo "Batch run finished with failures: ${failed_experiments[*]}"
    exit 1
fi

echo "All classification experiments finished successfully."
