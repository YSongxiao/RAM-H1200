#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

GPU_IDS="0, 1, 2, 3"
export CUDA_VISIBLE_DEVICES="${GPU_IDS// /}"

NUM_GPUS="${NUM_GPUS:-$(count_visible_gpus)}"
MASTER_PORT="${MASTER_PORT:-29500}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

MODE="${MODE:-train}"
MODEL="${MODEL:-SwinUMamba}"
TRIAL_NAME="${TRIAL_NAME:-Benchmark_BoneSeg}"
CHECKPOINT="${CHECKPOINT:-./ckpts}"
DATA_PATH="${DATA_PATH:-/path/to/segmentation_data}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-./ckpts/Annotation_swinumamba_202601210504/model_best_dice.pth1}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
TR_PATCHES_PER_IMG="${TR_PATCHES_PER_IMG:-16}"
MAX_EPOCH="${MAX_EPOCH:-200}"
LR="${LR:-1e-4}"
SCHEDULER="${SCHEDULER:-CosineAnnealing}"
EARLYSTOP_PATIENCE="${EARLYSTOP_PATIENCE:-15}"
SEED="${SEED:-2026}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"

# Specify which physical GPUs to use.
# Example:
#   GPU_IDS="0,1,3"
# Update this value here if you want to switch GPUs later.

# Define sequential batch experiments here. Format:
#   "FinalTrialName::extra arguments"
# Example:
# EXPERIMENTS=(
#   "swinumamba_lr1e4::--model SwinUMamba --lr 1e-4 --train_batch_size 8"
#   "unetpp_lr5e5::--model Unet++ --lr 5e-5 --train_batch_size 8"
# )
# To resume training, pass --resume or --resume_from when launching the script.
# If left empty, the script runs a single experiment with the default variables above.
EXPERIMENTS=(
    "Baseline_BoneSeg::--model SegFormer --train_batch_size 8"
    # "Baseline_BoneSeg::--model MambaVisionT --train_batch_size 8"
    # "Baseline_BoneSeg::--model SwinUMamba --train_batch_size 8"
    # "Baseline_BoneSeg::--model UMambaEnc --train_batch_size 8"
    # "Baseline_BoneSeg::--model TransUnet --train_batch_size 8"
    # "Baseline_BoneSeg::--model Unet++ --train_batch_size 8"
#    "Baseline_BoneSeg::--model TransUnet --train_batch_size 8 --resume"
)

run_experiment() {
    local exp_name="$1"
    shift

    local exp_trial_name="${TRIAL_NAME}"
    if [[ -n "${exp_name}" ]]; then
        exp_trial_name="${exp_name}"
    fi

    local cmd=(
        torchrun
        --nproc_per_node="${NUM_GPUS}"
        --master_port="${MASTER_PORT}"
        main_seg.py
        --launcher ddp
        --mode "${MODE}"
        --model "${MODEL}"
        --trial_name "${exp_trial_name}"
        --checkpoint "${CHECKPOINT}"
        --data_path "${DATA_PATH}"
        --image_size "${IMAGE_SIZE}"
        --train_batch_size "${TRAIN_BATCH_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}"
        --tr_patches_per_img "${TR_PATCHES_PER_IMG}"
        --max_epoch "${MAX_EPOCH}"
        --lr "${LR}"
        --scheduler "${SCHEDULER}"
        --earlystop_patience "${EARLYSTOP_PATIENCE}"
        --seed "${SEED}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
    )

    if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
        cmd+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
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
    echo "Launching experiment: ${exp_trial_name}"
    echo "GPU(s): ${NUM_GPUS} | GPU_IDS=${CUDA_VISIBLE_DEVICES:-all} | MASTER_PORT=${MASTER_PORT}"
    echo "Command: ${cmd[*]}"
    echo "=================================================="

    "${cmd[@]}"
}

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    echo "Launching single experiment with ${NUM_GPUS} GPU(s)"
    echo "MODEL=${MODEL}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "TRIAL_NAME=${TRIAL_NAME}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "GPU_IDS=${CUDA_VISIBLE_DEVICES:-all}"
    echo "MASTER_PORT=${MASTER_PORT}"
    run_experiment "" "$@"
    exit 0
fi

echo "Launching ${#EXPERIMENTS[@]} experiments sequentially with ${NUM_GPUS} GPU(s)"
echo "Fallback TRIAL_NAME=${TRIAL_NAME}"
echo "DATA_PATH=${DATA_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "GPU_IDS=${CUDA_VISIBLE_DEVICES:-all}"
echo "MASTER_PORT=${MASTER_PORT}"

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

echo "All experiments finished successfully."
