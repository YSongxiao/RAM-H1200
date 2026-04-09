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
SCORE_TYPE="${SCORE_TYPE:-BE}"
MODEL="${MODEL:-ResNet34}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_PATH="${DATA_PATH:-}"

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

# Batch experiments run sequentially. Format:
#   "DisplayName::--score_type BE --model ResNet34 --checkpoint ./ckpts/your_experiment_dir --data_path /path/to/data"
# Leave empty to run a single test with the default variables above.
EXPERIMENTS=(
    # ---- BE scoring ----
    "Baseline_BEScore_ConvNeXtV2::--score_type BE --model ConvNeXtV2 --checkpoint ./ckpts/Baseline_BEScore_be_convnextv2_20260406214233 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_DenseNet::--score_type BE --model DenseNet --checkpoint ./ckpts/Baseline_BEScore_be_densenet_20260406043912 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_EfficientFormer::--score_type BE --model EfficientFormer --checkpoint ./ckpts/Baseline_BEScore_be_efficientformer_20260406124424 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_EfficientNetV2::--score_type BE --model EfficientNetV2 --checkpoint ./ckpts/Baseline_BEScore_be_efficientnetv2_20260407010854 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_LeViT::--score_type BE --model LeViT --checkpoint ./ckpts/Baseline_BEScore_be_levit_20260406151143 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_MambaVisionT::--score_type BE --model MambaVisionT --checkpoint ./ckpts/Baseline_BEScore_be_mambavisiont_20260406192728 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_MedMamba::--score_type BE --model MedMamba --checkpoint ./ckpts/Baseline_BEScore_be_medmamba_20260406065848 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_MobileViT::--score_type BE --model MobileViT --checkpoint ./ckpts/Baseline_BEScore_be_mobilevit_20260406171657 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"
    "Baseline_BEScore_ResNet34::--score_type BE --model ResNet34 --checkpoint ./ckpts/Baseline_BEScore_be_resnet34_20260406034451 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring"

    # ---- JSN scoring ----
    "Baseline_JSNScore_DenseNet::--score_type JSN --model DenseNet --checkpoint ./ckpts/Baseline_JSNScore_jsn_densenet_20260408145311 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_ConvNeXtV2::--score_type JSN --model ConvNeXtV2 --checkpoint ./ckpts/Baseline_JSNScore_jsn_convnextv2_20260408235834 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_EfficientFormer::--score_type JSN --model EfficientFormer --checkpoint ./ckpts/Baseline_JSNScore_jsn_efficientformer_20260408193624 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_EfficientNetV2::--score_type JSN --model EfficientNetV2 --checkpoint ./ckpts/Baseline_JSNScore_jsn_efficientnetv2_20260409112019 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_LeViT::--score_type JSN --model LeViT --checkpoint ./ckpts/Baseline_JSNScore_jsn_levit_20260408210652 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_MambaVisionT::--score_type JSN --model MambaVisionT --checkpoint ./ckpts/Baseline_JSNScore_jsn_mambavisiont_20260409221059 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_MedMamba::--score_type JSN --model MedMamba --checkpoint ./ckpts/Baseline_JSNScore_jsn_medmamba_20260408161409 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_MobileViT::--score_type JSN --model MobileViT --checkpoint ./ckpts/Baseline_JSNScore_jsn_mobilevit_20260408222750 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"
    "Baseline_JSNScore_ResNet34::--score_type JSN --model ResNet34 --checkpoint ./ckpts/Baseline_JSNScore_jsn_resnet34_20260408140357 --data_path /home/yafei/data/RAM-H1200/SvdH_Scoring/SvdH_JSN_Scoring/"

)

has_arg() {
    local target="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

run_experiment() {
    local exp_name="$1"
    shift

    if [[ -z "${CHECKPOINT}" ]] && ! has_arg "--checkpoint" "$@"; then
        echo "Please set CHECKPOINT or add --checkpoint in the experiment args." >&2
        return 1
    fi

    if [[ -z "${DATA_PATH}" ]] && ! has_arg "--data_path" "$@"; then
        echo "Please set DATA_PATH or add --data_path in the experiment args." >&2
        return 1
    fi

    local cmd=(
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
    echo "Launching scoring test: ${exp_name:-single}"
    echo "GPU_ID=${GPU_ID}"
    echo "SCORE_TYPE=${SCORE_TYPE}"
    echo "MODEL=${MODEL}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    echo "Launching single scoring test"
    echo "GPU_ID=${GPU_ID}"
    echo "SCORE_TYPE=${SCORE_TYPE}"
    echo "MODEL=${MODEL}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "DATA_PATH=${DATA_PATH}"
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} scoring tests sequentially"
echo "GPU_ID=${GPU_ID}"
echo "Fallback SCORE_TYPE=${SCORE_TYPE}"
echo "Fallback MODEL=${MODEL}"
echo "Fallback CHECKPOINT=${CHECKPOINT}"
echo "Fallback DATA_PATH=${DATA_PATH}"

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

echo "All scoring tests finished successfully."
