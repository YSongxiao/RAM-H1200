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

STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

DATASET_ID="${DATASET_ID:-120}"
DATASET_NAME="${DATASET_NAME:-RAMH1200BESeg}"
NNUNET_DATA_ROOT="${NNUNET_DATA_ROOT:-${SCRIPT_DIR}/models/nnUNet/DATASET}"
TRAINER="${TRAINER:-nnUNetTrainerBE}"
PLANS="${PLANS:-nnUNetPlans}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLDS="${FOLDS:-0}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
MODEL_FOLDER="${MODEL_FOLDER:-}"
INPUT_DIR="${INPUT_DIR:-}"
GT_DIR="${GT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

STEP_SIZE="${STEP_SIZE:-0.5}"
NPP="${NPP:-3}"
NPS="${NPS:-3}"
METRIC_NP="${METRIC_NP:-4}"
DEVICE="${DEVICE:-cuda}"

DISABLE_TTA="${DISABLE_TTA:-0}"
DISABLE_PROGRESS_BAR="${DISABLE_PROGRESS_BAR:-0}"
NOT_ON_DEVICE="${NOT_ON_DEVICE:-0}"
CONTINUE_PREDICTION="${CONTINUE_PREDICTION:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SAVE_CSV="${SAVE_CSV:-1}"
SAVE_OVERLAY="${SAVE_OVERLAY:-0}"
SAVE_UNCERTAINTY_OVERLAY="${SAVE_UNCERTAINTY_OVERLAY:-0}"
SAVE_NPZ="${SAVE_NPZ:-${SAVE_NPY:-0}}"
SAVE_PRED="${SAVE_PRED:-0}"
CHILL="${CHILL:-0}"

EXPERIMENTS=(
    "Baseline_BESeg_nnUNet::--trainer nnUNetTrainerBE --plans nnUNetPlans --configuration 2d --folds 0 --checkpoint_name checkpoint_best.pth"
)

run_experiment() {
    local exp_name="$1"
    shift

    local cmd=(
        "${PYTHON_BIN}"
        nnunet_be_test.py
        --dataset_id "${DATASET_ID}"
        --dataset_name "${DATASET_NAME}"
        --nnunet_data_root "${NNUNET_DATA_ROOT}"
        --trainer "${TRAINER}"
        --plans "${PLANS}"
        --configuration "${CONFIGURATION}"
        --checkpoint_name "${CHECKPOINT_NAME}"
        --step_size "${STEP_SIZE}"
        --npp "${NPP}"
        --nps "${NPS}"
        --metric_np "${METRIC_NP}"
        --device "${DEVICE}"
    )

    read -r -a fold_args <<< "${FOLDS}"
    if [[ "${#fold_args[@]}" -gt 0 ]]; then
        cmd+=(--folds "${fold_args[@]}")
    fi

    if [[ -n "${MODEL_FOLDER}" ]]; then
        cmd+=(--model_folder "${MODEL_FOLDER}")
    fi
    if [[ -n "${INPUT_DIR}" ]]; then
        cmd+=(--input_dir "${INPUT_DIR}")
    fi
    if [[ -n "${GT_DIR}" ]]; then
        cmd+=(--gt_dir "${GT_DIR}")
    fi
    if [[ -n "${OUTPUT_ROOT}" ]]; then
        cmd+=(--output_root "${OUTPUT_ROOT}")
    fi

    if [[ "${DISABLE_TTA}" == "1" ]]; then
        cmd+=(--disable_tta)
    fi
    if [[ "${DISABLE_PROGRESS_BAR}" == "1" ]]; then
        cmd+=(--disable_progress_bar)
    fi
    if [[ "${NOT_ON_DEVICE}" == "1" ]]; then
        cmd+=(--not_on_device)
    fi
    if [[ "${CONTINUE_PREDICTION}" == "1" ]]; then
        cmd+=(--continue_prediction)
    fi
    if [[ "${SKIP_INFERENCE}" == "1" ]]; then
        cmd+=(--skip_inference)
    fi
    if [[ "${SAVE_CSV}" == "1" ]]; then
        cmd+=(--save_csv)
    else
        cmd+=(--no-save_csv)
    fi
    if [[ "${SAVE_OVERLAY}" == "1" ]]; then
        cmd+=(--save_overlay)
    fi
    if [[ "${SAVE_UNCERTAINTY_OVERLAY}" == "1" ]]; then
        cmd+=(--save_uncertainty_overlay)
    fi
    if [[ "${SAVE_NPZ}" == "1" ]]; then
        cmd+=(--save_npz)
    fi
    if [[ "${SAVE_PRED}" == "1" ]]; then
        cmd+=(--save_pred)
    fi
    if [[ "${CHILL}" == "1" ]]; then
        cmd+=(--chill)
    fi

    cmd+=("$@")

    echo "=================================================="
    echo "Launching nnUNet BE test: ${exp_name:-single}"
    echo "GPU_ID=${GPU_ID}"
    echo "DATASET_ID=${DATASET_ID}"
    echo "DATASET_NAME=${DATASET_NAME}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} nnUNet BE tests sequentially"
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

    read -r -a exp_args <<< "${exp_args_str}"

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

echo "All nnUNet BE tests finished successfully."
