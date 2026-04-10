#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
NNUNET_PACKAGE_ROOT = PROJECT_ROOT / "models" / "nnUNet"
if str(NNUNET_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_PACKAGE_ROOT))


BE_CREDIT_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{name: default})


def get_args():
    parser = argparse.ArgumentParser(description="Run BE nnUNet test inference and export aligned metrics/artifacts.")
    parser.add_argument("--dataset_id", type=int, default=120, help="nnUNet dataset id.")
    parser.add_argument("--dataset_name", type=str, default="RAMH1200BESeg", help="nnUNet dataset suffix.")
    parser.add_argument(
        "--nnunet_data_root",
        type=str,
        default=str(PROJECT_ROOT / "models" / "nnUNet" / "DATASET"),
        help="Root containing nnUNet_raw, nnUNet_preprocessed and nnUNet_trained_models.",
    )
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerBE", help="nnUNet trainer name.")
    parser.add_argument("--plans", type=str, default="nnUNetPlans", help="nnUNet plans identifier.")
    parser.add_argument("--configuration", type=str, default="2d", help="nnUNet configuration name.")
    parser.add_argument("--folds", nargs="+", default=["0"], help="Fold ids to use, for example: 0 or all.")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="",
        help="Explicit nnUNet model folder. Defaults to nnUNet_trained_models/DatasetXXX/.../trainer__plans__config.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Inference input folder. Defaults to nnUNet_raw/DatasetXXX/imagesTs.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="",
        help="Ground-truth folder. Defaults to nnUNet_raw/DatasetXXX/tmp_labelsTs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Output root. Defaults to <model_folder>/test_predictions.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint_best.pth",
        help="Checkpoint file name inside each fold folder.",
    )
    parser.add_argument("--step_size", type=float, default=0.5, help="Sliding window step size.")
    parser.add_argument("--npp", type=int, default=3, help="nnUNet preprocessing workers.")
    parser.add_argument("--nps", type=int, default=3, help="nnUNet export workers.")
    parser.add_argument("--metric_np", type=int, default=4, help="Metric worker count.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Inference device.")
    add_bool_arg(parser, "disable_tta", False, "Disable mirror TTA.")
    add_bool_arg(parser, "disable_progress_bar", False, "Disable progress bar.")
    add_bool_arg(parser, "not_on_device", False, "Disable perform_everything_on_device.")
    add_bool_arg(parser, "continue_prediction", False, "Continue a previous prediction run.")
    add_bool_arg(parser, "skip_inference", False, "Skip nnUNet inference and only export metrics/artifacts.")
    add_bool_arg(parser, "save_csv", True, "Save test metrics csv.")
    add_bool_arg(parser, "save_overlay", False, "Save overlay pdfs.")
    add_bool_arg(parser, "save_uncertainty_overlay", False, "Save uncertainty overlays.")
    add_bool_arg(parser, "save_npz", False, "Save pred/image/gt bundles as npz.")
    add_bool_arg(parser, "save_pred", False, "Save separated pred/gt overlay pdfs.")
    add_bool_arg(parser, "chill", False, "Allow missing prediction files during summary generation.")
    return parser.parse_args()


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def resolve_paths(args):
    data_root = Path(args.nnunet_data_root).resolve()
    dataset_folder = dataset_folder_name(args.dataset_id, args.dataset_name)
    raw_root = data_root / "nnUNet_raw" / dataset_folder
    results_root = data_root / "nnUNet_trained_models" / dataset_folder

    model_folder = Path(args.model_folder).resolve() if args.model_folder else (
        results_root / f"{args.trainer}__{args.plans}__{args.configuration}"
    )
    input_dir = Path(args.input_dir).resolve() if args.input_dir else (raw_root / "imagesTs")
    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else (raw_root / "tmp_labelsTs")
    output_root = Path(args.output_root).resolve() if args.output_root else (model_folder / "test_predictions")
    pred_dir = output_root / "pred"
    return dataset_folder, model_folder, input_dir, gt_dir, output_root, pred_dir


def ensure_exists(path: Path, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_dataset_meta(model_folder: Path):
    dataset_json = None
    candidate_paths = [model_folder / "dataset.json"]
    candidate_paths.extend(sorted((fold_dir / "dataset.json") for fold_dir in model_folder.glob("fold_*")))

    for candidate in candidate_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                dataset_json = json.load(f)
            break

    if dataset_json is None:
        raise FileNotFoundError(f"Could not find dataset.json under {model_folder}")

    label_items = []
    for name, value in dataset_json["labels"].items():
        if isinstance(value, (list, tuple)):
            continue
        label_items.append((int(value), name))
    label_items.sort(key=lambda item: item[0])
    channel_to_name = {idx: name for idx, name in label_items}
    class_names = [channel_to_name[idx] for idx in sorted(channel_to_name)]
    return dataset_json, channel_to_name, class_names


def build_predictor(args):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if args.device == "cpu":
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=not args.not_on_device,
        device=device,
        verbose=False,
        allow_tqdm=not args.disable_progress_bar,
        verbose_preprocessing=False,
    )
    folds = [fold if fold == "all" else int(fold) for fold in args.folds]
    return predictor, folds


def expected_pending_cases(input_dir: Path, pred_dir: Path, need_probabilities: bool):
    image_files = sorted(input_dir.glob("*_0000.png"))
    pending = []
    for image_path in image_files:
        case_id = image_path.stem.replace("_0000", "")
        pred_png = pred_dir / f"{case_id}.png"
        pred_npz = pred_dir / f"{case_id}.npz"
        ready = pred_png.exists() and ((not need_probabilities) or pred_npz.exists())
        if not ready:
            pending.append(case_id)
    return image_files, pending


def save_inference_time_summary(output_root: Path, elapsed_seconds: float, num_cases: int):
    if num_cases <= 0:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    avg_ms = elapsed_seconds / num_cases * 1000.0
    with (output_root / "inference_time.txt").open("w", encoding="utf-8") as f:
        f.write(f"Average inference time per case: {avg_ms:.4f} ms\n")
    print(f"Average inference time per case: {avg_ms:.2f} ms")


def load_label_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label png, got shape {arr.shape} from {path}")
    return arr.astype(np.uint8, copy=False)


def load_gray_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def label_to_onehot(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.uint8)[label_map].transpose(2, 0, 1)


def save_prediction_bundle(output_path: Path, pred, image=None, gt=None, probabilities=None):
    payload = {"pred": np.asarray(pred)}
    if image is not None:
        payload["image"] = np.asarray(image)
    if gt is not None:
        payload["gt"] = np.asarray(gt)
    if probabilities is not None:
        payload["probabilities"] = np.asarray(probabilities, dtype=np.float32)
    np.savez_compressed(output_path, **payload)


class BENnUNetEvaluator:
    def __init__(self, output_root: Path, channel_to_name: dict[int, str]):
        from evaluations.metrics import BESemanticSegmentationMetrics
        from utils import show_mask

        self.output_root = output_root
        self.channel_to_name = channel_to_name
        self.show_mask = show_mask
        self.colors = [
            np.array([0.931, 0.341, 0.215]),
            np.array([0.985, 0.725, 0.188]),
            np.array([0.176, 0.533, 0.855]),
        ]
        self.alpha = 0.35
        self.metrics = BESemanticSegmentationMetrics(
            num_classes=len(channel_to_name),
            credit_matrix=[row[: len(channel_to_name)] for row in BE_CREDIT_MATRIX[: len(channel_to_name)]],
        )

    def foreground_channel_indices(self):
        if 0 in self.channel_to_name and self.channel_to_name[0].lower() == "background":
            return list(range(1, len(self.channel_to_name)))
        return list(range(len(self.channel_to_name)))

    def foreground_channel_names(self):
        return [self.channel_to_name[idx] for idx in self.foreground_channel_indices()]

    def create_overlay(self, image, pred, gt, case_name: str):
        save_dir = self.output_root / "overlay"
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 3, figsize=(22, 6))
        titles = ["Image", "Prediction", "Ground Truth"]
        for i in range(3):
            ax[i].imshow(image, cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        for color_idx, channel_idx in enumerate(self.foreground_channel_indices()):
            self.show_mask(pred[channel_idx], ax[1], mask_color=self.colors[color_idx], alpha=self.alpha)
            self.show_mask(gt[channel_idx], ax[2], mask_color=self.colors[color_idx], alpha=self.alpha)

        plt.tight_layout()
        plt.savefig(save_dir / f"{case_name}.pdf", dpi=600)
        plt.close()

    def create_overlay_separate(self, image, pred, gt, case_name: str):
        save_dir = self.output_root / "overlay_single"
        save_dir.mkdir(parents=True, exist_ok=True)

        fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
        ax_pred.imshow(image, cmap="gray")
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        for color_idx, channel_idx in enumerate(self.foreground_channel_indices()):
            self.show_mask(pred[channel_idx], ax_pred, mask_color=self.colors[color_idx], alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_dir / f"{case_name}_pred.pdf", dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
        ax_gt.imshow(image, cmap="gray")
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        for color_idx, channel_idx in enumerate(self.foreground_channel_indices()):
            self.show_mask(gt[channel_idx], ax_gt, mask_color=self.colors[color_idx], alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_dir / f"{case_name}_gt.pdf", dpi=600)
        plt.close()

    def create_uncertainty_overlay(self, image, probabilities, case_name: str):
        save_dir = self.output_root / "uncertainty_overlay"
        save_dir.mkdir(parents=True, exist_ok=True)
        uncertainty = 1.0 - 2.0 * np.abs(probabilities - 0.5)

        for channel_idx in self.foreground_channel_indices():
            class_name = self.channel_to_name[channel_idx]
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Image")
            ax[0].axis("off")

            ax[1].imshow(image, cmap="gray")
            im1 = ax[1].imshow(probabilities[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[1].set_title(f"Probability {class_name}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            ax[2].imshow(image, cmap="gray")
            im2 = ax[2].imshow(uncertainty[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[2].set_title(f"Uncertainty {class_name}")
            ax[2].axis("off")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(save_dir / f"{case_name}_{class_name}.png", dpi=300)
            plt.close()

    def create_npz(self, image, pred, gt, case_name: str, probabilities=None):
        save_dir = self.output_root / "npz"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_prediction_bundle(
            save_dir / f"{case_name}.npz",
            pred.astype(np.uint8),
            image=image,
            gt=gt.astype(np.uint8),
            probabilities=probabilities,
        )

    def update_metrics(self, pred, gt, case_name: str):
        self.metrics.update_metrics(pred[np.newaxis, ...], gt[np.newaxis, ...], f"{case_name}.png")

    def create_csv(self):
        metrics_dict = self.metrics.get_metrics()

        def _safe_row_nanmean(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            valid = np.isfinite(arr)
            sums = np.where(valid, arr, 0.0).sum(axis=1)
            counts = valid.sum(axis=1)
            out = np.full(arr.shape[0], np.nan, dtype=float)
            nonzero = counts > 0
            out[nonzero] = sums[nonzero] / counts[nonzero]
            return out

        class_names = self.foreground_channel_names()
        num_fg_classes = len(class_names)

        dsc_pc = np.asarray(metrics_dict["dsc_pc"], dtype=float)[:, :num_fg_classes]
        nsd_pc = np.asarray(metrics_dict["nsd_pc"], dtype=float)[:, :num_fg_classes]
        voe_pc = np.asarray(metrics_dict["voe_pc"], dtype=float)[:, :num_fg_classes]
        msd_pc = np.asarray(metrics_dict["msd_pc"], dtype=float)[:, :num_fg_classes]
        ravd_pc = np.asarray(metrics_dict["ravd_pc"], dtype=float)[:, :num_fg_classes]
        precision_pc = np.asarray(metrics_dict["precision_pc"], dtype=float)[:, :num_fg_classes]
        recall_pc = np.asarray(metrics_dict["recall_pc"], dtype=float)[:, :num_fg_classes]
        f1_pc = np.asarray(metrics_dict["f1_pc"], dtype=float)[:, :num_fg_classes]
        tp_pc = np.asarray(metrics_dict["tp_pc"], dtype=float)[:, :num_fg_classes]
        tn_pc = np.asarray(metrics_dict["tn_pc"], dtype=float)[:, :num_fg_classes]
        fp_pc = np.asarray(metrics_dict["fp_pc"], dtype=float)[:, :num_fg_classes]
        fn_pc = np.asarray(metrics_dict["fn_pc"], dtype=float)[:, :num_fg_classes]

        metric_df = pd.concat(
            [
                pd.DataFrame(metrics_dict["fname"], columns=["Case"]),
                pd.DataFrame(dsc_pc, columns=[f"DSC {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(dsc_pc), columns=["Mean DSC"]),
                pd.DataFrame(nsd_pc, columns=[f"NSD {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(nsd_pc), columns=["Mean NSD"]),
                pd.DataFrame(voe_pc, columns=[f"VOE {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(voe_pc), columns=["Mean VOE"]),
                pd.DataFrame(msd_pc, columns=[f"MSD {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(msd_pc), columns=["Mean MSD"]),
                pd.DataFrame(ravd_pc, columns=[f"RAVD {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(ravd_pc), columns=["Mean RAVD"]),
                pd.DataFrame(precision_pc, columns=[f"Precision {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(precision_pc), columns=["Mean Precision"]),
                pd.DataFrame(recall_pc, columns=[f"Recall {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(recall_pc), columns=["Mean Recall"]),
                pd.DataFrame(f1_pc, columns=[f"F1 {name}" for name in class_names]),
                pd.DataFrame(_safe_row_nanmean(f1_pc), columns=["Mean F1"]),
                pd.DataFrame(tp_pc, columns=[f"TP {name}" for name in class_names]),
                pd.DataFrame(tn_pc, columns=[f"TN {name}" for name in class_names]),
                pd.DataFrame(fp_pc, columns=[f"FP {name}" for name in class_names]),
                pd.DataFrame(fn_pc, columns=[f"FN {name}" for name in class_names]),
            ],
            axis=1,
        )

        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        sums = np.where(valid, vals, 0.0).sum(axis=0)
        counts = valid.sum(axis=0)
        means = np.full(vals.shape[1], np.nan, dtype=float)
        np.divide(sums, counts, out=means, where=counts > 0)
        average_row = pd.DataFrame([["Average"] + means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv(self.output_root / "test_metrics.csv", index=False)


def export_summary(pred_dir: Path, gt_dir: Path, output_root: Path, num_classes: int, args):
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple

    labels = list(range(1, num_classes))
    if not labels:
        raise ValueError("No foreground labels found for summary export.")
    compute_metrics_on_folder_simple(
        str(gt_dir),
        str(pred_dir),
        labels=labels,
        output_file=str(output_root / "summary.json"),
        num_processes=args.metric_np,
        ignore_label=None,
        chill=args.chill,
    )


def main():
    args = get_args()
    dataset_folder, model_folder, input_dir, gt_dir, output_root, pred_dir = resolve_paths(args)

    ensure_exists(model_folder, "Model folder")
    ensure_exists(input_dir, "Input dir")
    ensure_exists(gt_dir, "GT dir")
    output_root.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset_json, channel_to_name, class_names = load_dataset_meta(model_folder)
    print(f"Dataset: {dataset_folder}")
    print(f"Model folder: {model_folder}")
    print(f"Input dir: {input_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Output root: {output_root}")
    print(f"Classes: {class_names}")

    need_probabilities = args.save_uncertainty_overlay
    image_files, pending_cases = expected_pending_cases(input_dir, pred_dir, need_probabilities)
    if not image_files:
        raise FileNotFoundError(f"No *_0000.png files found under {input_dir}")

    if not args.skip_inference:
        predictor, folds = build_predictor(args)
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            folds,
            checkpoint_name=args.checkpoint_name,
        )
        print(f"Using folds: {folds}")
        print(f"Checkpoint: {args.checkpoint_name}")
        start_time = time.perf_counter()
        predictor.predict_from_files(
            str(input_dir),
            str(pred_dir),
            save_probabilities=need_probabilities,
            overwrite=not args.continue_prediction,
            num_processes_preprocessing=args.npp,
            num_processes_segmentation_export=args.nps,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        elapsed_seconds = time.perf_counter() - start_time
        num_cases_for_timing = len(pending_cases) if args.continue_prediction else len(image_files)
        save_inference_time_summary(output_root, elapsed_seconds, num_cases_for_timing)
    else:
        print("Skipping inference and only exporting metrics/artifacts.")

    export_summary(pred_dir, gt_dir, output_root, len(channel_to_name), args)

    if not any(
        [
            args.save_csv,
            args.save_overlay,
            args.save_uncertainty_overlay,
            args.save_npz,
            args.save_pred,
        ]
    ):
        return

    evaluator = BENnUNetEvaluator(output_root, channel_to_name)
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction png files found under {pred_dir}")

    for pred_file in pred_files:
        case_id = pred_file.stem
        gt_file = gt_dir / f"{case_id}.png"
        image_file = input_dir / f"{case_id}_0000.png"

        if not gt_file.exists():
            if args.chill:
                continue
            raise FileNotFoundError(f"Missing GT for prediction {pred_file.name}: {gt_file}")
        if not image_file.exists():
            raise FileNotFoundError(f"Missing input image for prediction {pred_file.name}: {image_file}")

        pred_label = load_label_png(pred_file)
        gt_label = load_label_png(gt_file)
        raw_image = load_gray_png(image_file)
        probabilities = None
        probability_file = pred_dir / f"{case_id}.npz"
        if probability_file.exists():
            probabilities = np.load(probability_file)["probabilities"]

        pred_onehot = label_to_onehot(pred_label, len(channel_to_name))
        gt_onehot = label_to_onehot(gt_label, len(channel_to_name))

        if args.save_uncertainty_overlay:
            if probabilities is None:
                raise FileNotFoundError(
                    f"Missing probability npz for uncertainty overlay: {probability_file}. "
                    "Run inference without --no-save_uncertainty_overlay so probabilities are exported."
                )
            evaluator.create_uncertainty_overlay(raw_image, probabilities, case_id)
        if args.save_overlay:
            evaluator.create_overlay(raw_image, pred_onehot, gt_onehot, case_id)
        if args.save_pred:
            evaluator.create_overlay_separate(raw_image, pred_onehot, gt_onehot, case_id)
        if args.save_npz:
            evaluator.create_npz(raw_image, pred_onehot, gt_onehot, case_id, probabilities=probabilities)
        if args.save_csv:
            evaluator.update_metrics(pred_onehot, gt_onehot, case_id)

    if args.save_csv:
        evaluator.create_csv()


if __name__ == "__main__":
    main()
