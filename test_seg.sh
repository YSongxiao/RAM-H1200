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

MODE="${MODE:-test}"
MODEL="${MODEL:-SwinUMamba}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_PATH="${DATA_PATH:-/home/yafei/data/RAM-H1200/split_dataset_via_be_mask_remapped}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
SEED="${SEED:-2026}"
USE_COORDS="${USE_COORDS:-1}"
SAVE_NPZ="${SAVE_NPZ:-${SAVE_NPY:-0}}"
SAVE_OVERLAY="${SAVE_OVERLAY:-0}"
SAVE_UNCERTAINTY_OVERLAY="${SAVE_UNCERTAINTY_OVERLAY:-0}"
SAVE_CSV="${SAVE_CSV:-1}"
SAVE_PRED="${SAVE_PRED:-0}"
SAVE_MASK="${SAVE_MASK:-0}"

if [[ "${MODE}" != "test" ]]; then
    echo "MODE must be 'test', got: ${MODE}" >&2
    exit 1
fi

if [[ "${USE_COORDS}" != "1" ]]; then
    echo "main_seg.py currently defaults to use_coords=True and does not expose --no-use_coords." >&2
    echo "Please keep USE_COORDS=1 for test_seg.sh." >&2
    exit 1
fi

# Batch experiments run sequentially. Format:
#   "DisplayName::--model SwinUMamba --checkpoint ./ckpts/your_experiment_dir"
# Leave empty to run a single test with the default variables above.
EXPERIMENTS=(
    # "Baseline_BoneSeg_SwinUMamba::--model SwinUMamba --checkpoint ./ckpts/Baseline_BoneSeg_swinumamba_20260331000732 --save_overlay --save_overlay --save_npz"
    # "Baseline_BoneSeg_SwinUNETR::--model SwinUNETR --checkpoint ./ckpts/Baseline_BoneSeg_swinunetr_202603291751 --save_overlay --save_overlay --save_npz"
    "Baseline_BoneSeg_TansUNet::--model TransUnet --checkpoint ./ckpts/Baseline_BoneSeg_transunet_20260401113613 --save_overlay --save_overlay --save_npz"
    "Baseline_BoneSeg_UMambaEnc::--model UMambaEnc --checkpoint ./ckpts/Baseline_BoneSeg_umambaenc_20260331224356 --save_overlay --save_overlay --save_npz"
    "Baseline_BoneSeg_Unet::--model Unet --checkpoint ./ckpts/Baseline_BoneSeg_unet_20260401195552 --save_overlay --save_overlay --save_npz"
    "Baseline_BoneSeg_Unet++::--model Unet++ --checkpoint ./ckpts/Baseline_BoneSeg_unet++_20260402182806 --save_overlay --save_overlay --save_npz"
)

has_checkpoint_arg() {
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "--checkpoint" ]]; then
            return 0
        fi
    done
    return 1
}

run_experiment() {
    local exp_name="$1"
    shift

    if [[ -z "${CHECKPOINT}" ]] && ! has_checkpoint_arg "$@"; then
        echo "Please set CHECKPOINT or add --checkpoint in the experiment args." >&2
        return 1
    fi

    local cmd=(
        "${PYTHON_BIN}"
        main_seg.py
        --launcher none
        --mode "${MODE}"
        --model "${MODEL}"
        --checkpoint "${CHECKPOINT}"
        --data_path "${DATA_PATH}"
        --image_size "${IMAGE_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}"
        --seed "${SEED}"
        --use_coords
    )

    if [[ "${SAVE_NPZ}" == "1" ]]; then
        cmd+=(--save_npz)
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

    if [[ "${SAVE_PRED}" == "1" ]]; then
        cmd+=(--save_pred)
    fi

    if [[ "${SAVE_MASK}" == "1" ]]; then
        cmd+=(--save_mask)
    fi

    cmd+=("$@")

    echo "=================================================="
    echo "Launching bone segmentation test: ${exp_name:-single}"
    echo "GPU_ID=${GPU_ID}"
    echo "MODEL=${MODEL}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    echo "Launching single bone segmentation test"
    echo "GPU_ID=${GPU_ID}"
    echo "MODEL=${MODEL}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "DATA_PATH=${DATA_PATH}"
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} bone segmentation tests sequentially"
echo "GPU_ID=${GPU_ID}"
echo "Fallback MODEL=${MODEL}"
echo "Fallback CHECKPOINT=${CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"

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

echo "All bone segmentation tests finished successfully."
