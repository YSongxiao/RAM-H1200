import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.boneseg import FullHandPatchDataset
from evaluations.metrics import FullHandSegmentationMetrics
from utils import get_transform, seed_everything, show_mask
from segment_anything import SamPredictor, sam_model_registry


OVERLAY_COLORS = [
    np.array([0.1522, 0.4717, 0.9685]),
    np.array([0.3178, 0.0520, 0.8333]),
    np.array([0.3834, 0.3823, 0.6784]),
    np.array([0.8525, 0.1303, 0.4139]),
    np.array([0.9948, 0.8252, 0.3384]),
    np.array([0.8476, 0.7147, 0.2453]),
    np.array([0.2865, 0.8411, 0.0877]),
    np.array([0.1558, 0.4940, 0.4668]),
    np.array([0.9199, 0.5882, 0.5113]),
    np.array([0.1335, 0.5433, 0.6149]),
    np.array([0.0629, 0.7343, 0.0943]),
    np.array([0.8183, 0.2786, 0.3053]),
    np.array([0.1789, 0.5083, 0.6787]),
    np.array([0.9746, 0.1909, 0.4295]),
    np.array([0.1586, 0.8670, 0.6994]),
    np.array([0.9156, 0.1241, 0.3829]),
    np.array([0.2998, 0.3054, 0.4242]),
    np.array([0.7719, 0.7786, 0.1164]),
    np.array([0.8033, 0.9278, 0.7621]),
    np.array([0.1085, 0.5155, 0.4145]),
    np.array([0.6523, 0.2197, 0.9011]),
    np.array([0.2457, 0.8125, 0.3928]),
    np.array([0.9124, 0.4632, 0.7821]),
    np.array([0.4912, 0.6399, 0.1223]),
    np.array([0.2105, 0.9311, 0.8420]),
    np.array([0.9734, 0.5843, 0.1028]),
    np.array([0.4188, 0.2911, 0.7555]),
    np.array([0.6876, 0.8352, 0.2764]),
    np.array([0.5579, 0.1447, 0.5348]),
    np.array([0.8662, 0.3569, 0.2287]),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", type=str, default="test", choices=["test", "infer"])
    parser.add_argument("--prompt_type", type=str, default="box", choices=["box", "point"])
    parser.add_argument("--sam_type", type=str, default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parent / "ckpts/sam_vit_h_4b8939.pth"),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data2/datasx/FullHand/NIPS26/RAM-H1200/Segmentation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpts/Baseline_BoneSeg_sam",
    )
    parser.add_argument("--flip_left_by_name", action="store_true", default=False)
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--save_pred", action="store_true", default=False)
    parser.add_argument("--save_npz", action="store_true", default=False)
    parser.add_argument("--save_csv", action="store_true", default=True)
    parser.add_argument("--box_relax_ratio", type=float, default=0.08)
    parser.add_argument("--box_relax_pixels", type=int, default=6)
    parser.add_argument("--point_jitter_ratio", type=float, default=0.15)
    parser.add_argument("--point_jitter_pixels", type=int, default=8)
    return parser.parse_args()


def _safe_nanmean(arr, axis):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    sums = np.where(valid, arr, 0.0).sum(axis=axis, dtype=float)
    counts = valid.sum(axis=axis)
    out = np.full(np.shape(sums), np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _save_prediction_npz(output_path: Path, pred, image=None, gt=None):
    payload = {"pred": np.asarray(pred)}
    if image is not None:
        payload["image"] = np.asarray(image)
    if gt is not None:
        payload["gt"] = np.asarray(gt)
    np.savez_compressed(output_path, **payload)


def _synchronize_device(device):
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_model_forward(forward_fn, device):
    _synchronize_device(device)
    start = time.perf_counter()
    output = forward_fn()
    _synchronize_device(device)
    return output, time.perf_counter() - start


def _build_case_inference_time_df(case_time_records):
    rows = []
    for case_name, times_ms in case_time_records.items():
        if not times_ms:
            continue
        rows.append(
            {
                "Case": case_name,
                "Mean Inference Time (ms)": float(np.mean(times_ms)),
                "Std Inference Time (ms)": float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else np.nan,
                "Num Samples": int(len(times_ms)),
            }
        )
    rows.sort(key=lambda item: item["Case"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    average_row = pd.DataFrame(
        [
            {
                "Case": "Average",
                "Mean Inference Time (ms)": float(df["Mean Inference Time (ms)"].mean()),
                "Std Inference Time (ms)": float(df["Mean Inference Time (ms)"].std(ddof=1)) if len(df) > 1 else np.nan,
                "Num Samples": int(df["Num Samples"].sum()),
            }
        ]
    )
    return pd.concat([df, average_row], ignore_index=True)


def _save_case_inference_time_summary(output_dir, case_time_records, filename="inference_time.txt"):
    inference_time_df = _build_case_inference_time_df(case_time_records)
    if inference_time_df.empty:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("w", encoding="utf-8") as f:
        avg_time_ms = float(inference_time_df.iloc[-1]["Mean Inference Time (ms)"])
        f.write(f"Average inference time per case: {avg_time_ms:.4f} ms\n")


def _print_case_inference_time_summary(case_time_records):
    inference_time_df = _build_case_inference_time_df(case_time_records)
    if inference_time_df.empty:
        return
    avg_time_ms = float(inference_time_df.iloc[-1]["Mean Inference Time (ms)"])
    print(f"Average inference time per case: {avg_time_ms:.2f} ms")


def build_dataset(args):
    transform = get_transform(split="test", image_size=512)
    if args.mode == "test":
        test_root = Path(args.data_path) / "test"
        return FullHandPatchDataset(
            data_root=test_root,
            annotation_path=test_root / "_annotations_bone_rle.coco.json",
            mode="test",
            transform=transform,
            use_coords=False,
            flip_left_by_name=args.flip_left_by_name,
            expected_num_classes=14,
        )
    infer_root = Path(args.data_path)
    return FullHandPatchDataset(
        data_root=infer_root,
        annotation_path=None,
        mode="infer",
        transform=transform,
        use_coords=False,
        flip_left_by_name=args.flip_left_by_name,
        expected_num_classes=14,
    )


def compute_relaxed_box(mask: np.ndarray, relax_ratio: float, relax_pixels: int) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Cannot build a box prompt from an empty mask.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask.shape
    box_h = max(1, y_max - y_min + 1)
    box_w = max(1, x_max - x_min + 1)
    pad_y = max(int(round(box_h * relax_ratio)), int(relax_pixels))
    pad_x = max(int(round(box_w * relax_ratio)), int(relax_pixels))
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w - 1, x_max + pad_x)
    y1 = min(h - 1, y_max + pad_y)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def compute_random_centroid_point(mask: np.ndarray, jitter_ratio: float, jitter_pixels: int, rng: np.random.Generator):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Cannot build a point prompt from an empty mask.")
    centroid_yx = coords.mean(axis=0)
    centroid_y, centroid_x = float(centroid_yx[0]), float(centroid_yx[1])
    area = max(1, int(coords.shape[0]))
    base_radius = np.sqrt(area / np.pi)
    radius = max(int(round(base_radius * jitter_ratio)), int(jitter_pixels), 1)
    dist = np.sqrt((coords[:, 0] - centroid_y) ** 2 + (coords[:, 1] - centroid_x) ** 2)
    candidate_mask = dist <= radius
    candidate_coords = coords[candidate_mask] if np.any(candidate_mask) else coords
    yx = candidate_coords[rng.integers(0, len(candidate_coords))]
    point = np.array([[float(yx[1]), float(yx[0])]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    return point, label


class SAMBoneSegTester:
    def __init__(self, args, dataset, predictor, sam_model, device):
        self.args = args
        self.dataset = dataset
        self.predictor = predictor
        self.sam_model = sam_model
        self.device = device
        self.checkpoint_dir = Path(args.checkpoint)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_overlay = args.save_overlay
        self.save_pred = args.save_pred
        self.save_npz = args.save_npz
        self.save_csv = args.save_csv and args.mode == "test"
        self.channel_to_name = dataset.channel_to_name
        self.metrics = FullHandSegmentationMetrics(num_classes=len(self.channel_to_name)) if self.save_csv else None
        self.alpha = 0.5
        self.rng = np.random.default_rng(args.seed)

    def _channel_indices(self):
        return list(range(len(self.channel_to_name)))

    def _overlay_color(self, color_idx):
        return OVERLAY_COLORS[color_idx % len(OVERLAY_COLORS)]

    def _build_prompt(self, binary_mask):
        if self.args.prompt_type == "box":
            return {
                "box": compute_relaxed_box(
                    binary_mask,
                    relax_ratio=self.args.box_relax_ratio,
                    relax_pixels=self.args.box_relax_pixels,
                )
            }
        point_coords, point_labels = compute_random_centroid_point(
            binary_mask,
            jitter_ratio=self.args.point_jitter_ratio,
            jitter_pixels=self.args.point_jitter_pixels,
            rng=self.rng,
        )
        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    def run(self):
        self.sam_model.eval()
        case_inference_times = {}

        with torch.no_grad():
            for index in tqdm(range(len(self.dataset))):
                sample = self.dataset[index]
                fname = sample["fname"]
                img_tensor = sample["img"]
                raw_img = (img_tensor[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
                gt = sample.get("gt")
                gt_np = gt.numpy().astype(np.uint8) if gt is not None else None
                rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)

                _, set_image_elapsed = _time_model_forward(
                    lambda: self.predictor.set_image(rgb),
                    self.device,
                )
                total_elapsed = set_image_elapsed

                channel_preds = []
                for channel_idx in self._channel_indices():
                    if gt_np is None:
                        raise RuntimeError("Point/box prompts require ground-truth masks, so infer mode is unsupported.")
                    binary_gt = gt_np[channel_idx]
                    if binary_gt.sum() == 0:
                        channel_preds.append(np.zeros_like(binary_gt, dtype=np.uint8))
                        continue
                    prompt_kwargs = self._build_prompt(binary_gt)
                    pred_masks, decode_elapsed = _time_model_forward(
                        lambda kwargs=prompt_kwargs: self.predictor.predict(
                            point_coords=kwargs.get("point_coords"),
                            point_labels=kwargs.get("point_labels"),
                            box=kwargs.get("box"),
                            multimask_output=False,
                        ),
                        self.device,
                    )
                    total_elapsed += decode_elapsed
                    channel_preds.append(pred_masks[0][0].astype(np.uint8))

                pred = np.stack(channel_preds, axis=0).astype(np.uint8)
                case_inference_times.setdefault(fname, []).append(total_elapsed * 1000.0)

                if self.save_overlay and gt_np is not None:
                    self.create_overlay(raw_img, pred, gt_np, fname)
                if self.save_pred and gt_np is not None:
                    self.create_overlay_separate(raw_img, pred, gt_np, fname)
                if self.save_npz:
                    self.create_npz(raw_img, pred, gt_np, fname)
                if self.metrics is not None and gt_np is not None:
                    self.metrics.update_metrics(pred[np.newaxis, ...], gt_np[np.newaxis, ...], fname)

        _print_case_inference_time_summary(case_inference_times)
        _save_case_inference_time_summary(self.checkpoint_dir, case_inference_times)
        if self.metrics is not None:
            self.create_csv()

    def create_npz(self, image, pred, gt, fname):
        save_path = self.checkpoint_dir / "npz"
        save_path.mkdir(parents=True, exist_ok=True)
        _save_prediction_npz(
            save_path / f"{Path(fname).stem}.npz",
            pred.astype(np.uint8),
            image=image,
            gt=None if gt is None else gt.astype(np.uint8),
        )

    def create_overlay(self, image, pred, gt, fname):
        save_path = self.checkpoint_dir / "overlay"
        save_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(22, 6))
        titles = ["Image", "Prediction", "Ground Truth"]
        for i in range(3):
            ax[i].imshow(image, cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].axis("off")
        for color_idx, channel_idx in enumerate(self._channel_indices()):
            color = self._overlay_color(color_idx)
            show_mask(pred[channel_idx], ax[1], mask_color=color, alpha=self.alpha)
            show_mask(gt[channel_idx], ax[2], mask_color=color, alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_path / f"{Path(fname).stem}.pdf", dpi=600)
        plt.close()

    def create_overlay_separate(self, image, pred, gt, fname):
        save_path = self.checkpoint_dir / "overlay_single"
        save_path.mkdir(parents=True, exist_ok=True)

        fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
        ax_pred.imshow(image, cmap="gray")
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        for color_idx, channel_idx in enumerate(self._channel_indices()):
            show_mask(pred[channel_idx], ax_pred, mask_color=self._overlay_color(color_idx), alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_path / f"{Path(fname).stem}_pred.pdf", dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
        ax_gt.imshow(image, cmap="gray")
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        for color_idx, channel_idx in enumerate(self._channel_indices()):
            show_mask(gt[channel_idx], ax_gt, mask_color=self._overlay_color(color_idx), alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_path / f"{Path(fname).stem}_gt.pdf", dpi=600)
        plt.close()

    def create_csv(self):
        metrics_dict = self.metrics.get_metrics()
        class_names = [self.channel_to_name[i] for i in range(len(self.channel_to_name))]
        overlap_pairs = metrics_dict["overlap_pairs"]
        overlap_pair_names = [
            f"{self.channel_to_name[i]}&{self.channel_to_name[j]}"
            for i, j in overlap_pairs
        ]
        dsc_pc = np.asarray(metrics_dict["dsc_pc"], dtype=float)
        nsd_pc = np.asarray(metrics_dict["nsd_pc"], dtype=float)
        voe_pc = np.asarray(metrics_dict["voe_pc"], dtype=float)
        msd_pc = np.asarray(metrics_dict["msd_pc"], dtype=float)
        ravd_pc = np.asarray(metrics_dict["ravd_pc"], dtype=float)
        overlap_dsc_pc = np.asarray(metrics_dict["overlap_dsc_per_pair"], dtype=float)
        overlap_nsd_pc = np.asarray(metrics_dict["overlap_nsd_per_pair"], dtype=float)
        overlap_voe_pc = np.asarray(metrics_dict["overlap_voe_per_pair"], dtype=float)
        overlap_msd_pc = np.asarray(metrics_dict["overlap_msd_per_pair"], dtype=float)
        overlap_ravd_pc = np.asarray(metrics_dict["overlap_ravd_per_pair"], dtype=float)

        fname_df = pd.DataFrame(metrics_dict["fname"], columns=["Case"])
        metric_df = pd.concat(
            [
                fname_df,
                pd.DataFrame(overlap_dsc_pc, columns=[f"Overlap DSC {name}" for name in overlap_pair_names]),
                pd.DataFrame(_safe_nanmean(overlap_dsc_pc, axis=1), columns=["Mean Overlap DSC"]),
                pd.DataFrame(overlap_nsd_pc, columns=[f"Overlap NSD {name}" for name in overlap_pair_names]),
                pd.DataFrame(_safe_nanmean(overlap_nsd_pc, axis=1), columns=["Mean Overlap NSD"]),
                pd.DataFrame(overlap_voe_pc, columns=[f"Overlap VOE {name}" for name in overlap_pair_names]),
                pd.DataFrame(_safe_nanmean(overlap_voe_pc, axis=1), columns=["Mean Overlap VOE"]),
                pd.DataFrame(overlap_msd_pc, columns=[f"Overlap MSD {name}" for name in overlap_pair_names]),
                pd.DataFrame(_safe_nanmean(overlap_msd_pc, axis=1), columns=["Mean Overlap MSD"]),
                pd.DataFrame(overlap_ravd_pc, columns=[f"Overlap RAVD {name}" for name in overlap_pair_names]),
                pd.DataFrame(_safe_nanmean(overlap_ravd_pc, axis=1), columns=["Mean Overlap RAVD"]),
                pd.DataFrame(dsc_pc, columns=[f"DSC {name}" for name in class_names]),
                pd.DataFrame(_safe_nanmean(dsc_pc, axis=1), columns=["Mean DSC"]),
                pd.DataFrame(nsd_pc, columns=[f"NSD {name}" for name in class_names]),
                pd.DataFrame(_safe_nanmean(nsd_pc, axis=1), columns=["Mean NSD"]),
                pd.DataFrame(voe_pc, columns=[f"VOE {name}" for name in class_names]),
                pd.DataFrame(_safe_nanmean(voe_pc, axis=1), columns=["Mean VOE"]),
                pd.DataFrame(msd_pc, columns=[f"MSD {name}" for name in class_names]),
                pd.DataFrame(_safe_nanmean(msd_pc, axis=1), columns=["Mean MSD"]),
                pd.DataFrame(ravd_pc, columns=[f"RAVD {name}" for name in class_names]),
                pd.DataFrame(_safe_nanmean(ravd_pc, axis=1), columns=["Mean RAVD"]),
            ],
            axis=1,
        )
        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        finite_means = _safe_nanmean(vals, axis=0)
        average_row = pd.DataFrame([["Average"] + finite_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv(self.checkpoint_dir / "test_metrics.csv", index=False)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_model = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(device)
    sam_model.eval()
    n_params = sum(p.numel() for p in sam_model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")
    print(f"Prompt type: {args.prompt_type}")

    predictor = SamPredictor(sam_model)
    dataset = build_dataset(args)
    tester = SAMBoneSegTester(args, dataset, predictor, sam_model, device)
    tester.run()


if __name__ == "__main__":
    main()
