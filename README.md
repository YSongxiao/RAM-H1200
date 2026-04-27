# RAM-H1200 Benchmark

This repository contains benchmark code for RAM-H1200 tasks, including:

- Bone segmentation
- BE segmentation
- Scoring of SvdH BE / JSN

Agent-related updates are maintained on the `agent-test-svdh` branch.

## Setup

First download the RAM-H1200 dataset from Hugging Face:

- https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200

For example:

```bash
git clone https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200
```

Then prepare your Python environment and update the placeholder paths in the shell scripts, or pass them through environment variables.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

This repository also includes local packages for some components. Install them when needed:

```bash
pip install -e ./models/nnUNet
pip install -e ./models/segment-anything
pip install -e ./models/MedSAM
```

Please install a CUDA-compatible PyTorch build first if you are using GPU training or inference.

Common path placeholders:

- `/path/to/segmentation_data`
- `/path/to/be_scoring_data`
- `/path/to/jsn_scoring_data`

Outputs are typically written under `./ckpts/`.

## Benchmark Entry Points

Main benchmark scripts in the repository root:

- `train_seg.sh`: bone segmentation training
- `test_seg.sh`: bone segmentation testing
- `train_be_seg.sh`: BE segmentation training
- `test_be_seg.sh`: BE segmentation testing
- `train_score_cls.sh`: BE/JSN scoring training
- `train_score_cls_ddp.sh`: BE/JSN scoring distributed training
- `test_score_cls.sh`: BE/JSN scoring testing
- `test_be_seg_nnunet.sh`: nnU-Net BE segmentation testing

Each script contains an `EXPERIMENTS` array for batch benchmark runs. Uncomment or edit the entries you want to run.

## Typical Usage

### Bone Segmentation

Training:

```bash
GPU_ID=0 DATA_PATH=/path/to/segmentation_data bash train_seg.sh
```

Testing:

```bash
GPU_ID=0 DATA_PATH=/path/to/segmentation_data bash test_seg.sh
```

### BE Segmentation

Training:

```bash
GPU_ID=0 DATA_PATH=/path/to/segmentation_data bash train_be_seg.sh
```

Testing:

```bash
GPU_ID=0 DATA_PATH=/path/to/segmentation_data bash test_be_seg.sh
```

### Scoring

BE scoring:

```bash
GPU_ID=0 SCORE_TYPE=BE DATA_PATH=/path/to/be_scoring_data bash train_score_cls.sh
GPU_ID=0 SCORE_TYPE=BE DATA_PATH=/path/to/be_scoring_data bash test_score_cls.sh
```

JSN scoring:

```bash
GPU_ID=0 SCORE_TYPE=JSN DATA_PATH=/path/to/jsn_scoring_data bash train_score_cls.sh
GPU_ID=0 SCORE_TYPE=JSN DATA_PATH=/path/to/jsn_scoring_data bash test_score_cls.sh
```

## nnU-Net for BE Segmentation

nnU-Net requires the BE segmentation dataset to be converted to the nnU-Net directory layout first.

Step 1: convert the dataset:

```bash
python convert_be_seg_to_nnunet.py
```

This script prepares the dataset under:

```text
models/nnUNet/DATASET/
```

Step 2: follow the nnU-Net workflow inside `models/nnUNet`.

Typical next steps are:

1. Set the nnU-Net environment variables to the directories under `models/nnUNet/DATASET`.
2. Run planning and preprocessing.
3. Run nnU-Net training.
4. Run inference or testing.

Example command templates:

```bash
cd models/nnUNet

export nnUNet_raw="/path/to/RAM-H1200/models/nnUNet/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/path/to/RAM-H1200/models/nnUNet/DATASET/nnUNet_preprocessed"
export nnUNet_results="/path/to/RAM-H1200/models/nnUNet/DATASET/nnUNet_trained_models"
```

Planning and preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 120 -c 2d --verify_dataset_integrity
```

Training:

```bash
nnUNetv2_train 120 2d 0 -tr nnUNetTrainerBE
```

Inference:

```bash
nnUNetv2_predict \
    -i /path/to/input_images \
    -o /path/to/output_predictions \
    -d 120 \
    -c 2d \
    -f 0 \
    -tr nnUNetTrainerBE
```

The nnU-Net codebase used by this repository is located in:

```text
models/nnUNet
```

If you want to use the provided repository-level evaluation entry for BE segmentation inference, you can run:

```bash
bash test_be_seg_nnunet.sh
```

If you want the full native nnU-Net pipeline, continue from `models/nnUNet` after conversion and use the commands and documentation there.

## Notes

- Most scripts support single-run mode through environment variables such as `MODEL`, `CHECKPOINT`, and `DATA_PATH`.
- Batch benchmark mode is controlled by the `EXPERIMENTS` array inside each script.
- Some evaluation and summary utilities are kept as local helper scripts and may not be tracked in Git.
