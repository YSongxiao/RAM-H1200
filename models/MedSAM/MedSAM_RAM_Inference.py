from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from pathlib import Path
import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch import amp
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import show_mask
import monai
import pandas as pd
from evaluations.metrics import SegmentationMetrics, ClassificationMetrics, bone_name_dict
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import time
from typing import Tuple
from utils import AdaptiveLossBalancer, LossBalancer
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def compute_point_prompt(mask: np.ndarray) -> np.ndarray:
    """
    计算用于 SAM 的点 prompt（重心坐标）。

    Args:
        mask (np.ndarray): 二值mask，形状为(H, W)

    Returns:
        np.ndarray: 形状为 (1, 2)，表示 [[x, y]]，用于SAM的point_coords
    """
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Mask is empty")
    cy, cx = indices.mean(axis=0)
    return np.array([[cx, cy]], dtype=np.float32)  # 注意顺序：x是列，y是行


def compute_box_prompt(mask: np.ndarray) -> np.ndarray:
    """
    计算用于 SAM 的 box prompt（边界框坐标）。

    Args:
        mask (np.ndarray): 二值mask，形状为(H, W)

    Returns:
        np.ndarray: 形状为 (4,) 的数组，表示 [x0, y0, x1, y1]
    """
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Mask is empty")
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


data_root = "/mnt/data2/datasx/Carpal/ExportedDataset/ExtendedExportVersion2/BoneSegmentation/images"
annotation_path = "/mnt/data2/datasx/Carpal/ExportedDataset/ExtendedExportVersion2/BoneSegmentation/masks/test"
filenames = [str(fname.stem) for fname in Path(annotation_path).rglob("*.npy")]
masks = []

for filename in filenames:
    tmp_mask = np.load(Path(annotation_path) / (filename + ".npy"))
    masks.append(tmp_mask)

class SegTester:
    def __init__(self):
        # self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.save_overlay = False
        self.save_csv = True
        self.save_pred = False
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        self.metrics_pt = SegmentationMetrics(num_classes=14)
        self.metrics_box = SegmentationMetrics(num_classes=14)

    def test(self):
        total_infer_time_box = 0.0
        total_items = 0
        device = "cuda:0"
        medsam_model = sam_model_registry["vit_b"](checkpoint="./work_dir/MedSAM/medsam_vit_b.pth")
        medsam_model = medsam_model.to(device)
        medsam_model.eval()
        n_params = sum(p.numel() for p in medsam_model.parameters())
        print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")

        for idx in tqdm(range(len(masks))):
            path = Path(data_root) / (filenames[idx] + ".bmp")
            fname = path.stem
            img_np = cv2.imread(str(path))
            img_np_512 = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = masks[idx]
            if filenames[idx][-1] == "L":
                img_np = cv2.flip(img_np, 1)  # Horizontal Flip
                mask = masks[idx][:, :, ::-1]
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )

            resized_mask = np.array(F.interpolate(torch.tensor(mask.copy())[None].float(), size=(512, 512), mode='nearest')).squeeze(0)
            preds_box = []
            with torch.no_grad():
                start_time_box = time.time()  # ⏱️ Start timing
                image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
                end_time_box = time.time()  # ⏱️ End timing
                infer_time_box = end_time_box - start_time_box
                total_infer_time_box += infer_time_box
            for i in range(mask.shape[0]):
                box_np = compute_box_prompt(mask[i])[np.newaxis]
                # transfer box_np t0 1024x1024 scale
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
                start_time_box = time.time()  # ⏱️ Start timing
                medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
                end_time_box = time.time()  # ⏱️ End timing
                infer_time_box = end_time_box - start_time_box
                total_infer_time_box += infer_time_box
                preds_box.append(medsam_seg)
            resized_mask = torch.tensor(resized_mask)[None]
            preds_box = torch.tensor(np.array(preds_box))[None]
            preds_box = F.interpolate(preds_box.float(), size=(512, 512), mode='nearest')
            img_tch = torch.tensor(img_np_512)[None][None]
            # preds_box[preds_box >= 0.5] = 1
            # preds_box[preds_box < 0.5] = 0
            self.metrics_box.update_metrics(preds_box, resized_mask, fname)

            if self.save_overlay:
                # self.create_overlay("./ckpts/PT", image=img_tch, pred=preds_pt, mask=resized_masks, fname=fname)
                self.create_overlay("./ckpts/Baseline_Ext_BOX", image=img_tch, pred=preds_box, mask=resized_mask, fname=fname)
            if self.save_pred:
                # self.create_pred("./ckpts/PT", image=img_tch, pred=preds_pt, mask=mask, fname=fname)
                # self.create_overlay_single("./ckpts/PT", image=img_tch, pred=preds_pt, mask=mask, fname=fname)
                self.create_pred("./ckpts/Baseline_Ext_BOX", image=img_tch, pred=preds_box, mask=resized_mask, fname=fname)
                self.create_overlay_single("./ckpts/Baseline_Ext_BOX", image=img_tch, pred=preds_box, mask=resized_mask, fname=fname)
            total_items += 1
        if self.save_csv:
            # self.create_csv("./ckpts/PT", type="PT")
            self.create_csv("./ckpts/Baseline_Ext_BOX", type="BOX")

        metrics_dict_box = self.metrics_box.get_metrics()
        dsc_reduced = metrics_dict_box["dsc"].mean()
        print("BOX Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict_box["nsd"].mean()
        print("BOX Mean NSD: ", nsd_reduced)

        avg_infer_time = total_infer_time_box / total_items
        print(f"[BOX] Average inference time per item: {avg_infer_time * 1000:.2f} ms")


    def create_overlay(self, save_root, image, pred, mask, fname):
        save_path = Path(save_root) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            # color = np.random.rand(3)
            # seg = torch.sigmoid(pred_mask[0][i]).cpu().numpy()
            # seg[seg > 0.5] = 1
            # seg[seg <= 0.5] = 0
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(self.colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname + '.pdf'), dpi=600)
        plt.close()


    def create_pred(self, save_root, image, pred, mask, fname):
        save_path = Path(save_root) / "pred"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname + '_gt.pdf'), dpi=600)
        plt.close()


    def create_overlay_single(self, save_root, image, pred, mask, fname):
        save_path = Path(save_root) / "overlay_single"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname + '_gt.pdf'), dpi=600)
        plt.close()

    def create_csv(self, save_root, type="PT"):
        save_path = Path(save_root)
        if type == "PT":
            metrics_dict = self.metrics_pt.get_metrics()
            num_classes = self.metrics_pt.num_labels
        else:
            metrics_dict = self.metrics_box.get_metrics()
            num_classes = self.metrics_box.num_labels
        overlap_dsc_mean_df = pd.DataFrame(metrics_dict["overlap_dsc"], columns=["Mean Overlap DSC"])
        overlap_dsc_df = pd.DataFrame(metrics_dict["overlap_dsc_per_pair"],
                                      columns=[f"Overlap DSC {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for
                                               pair in metrics_dict["overlap_pairs"]])
        overlap_nsd_mean_df = pd.DataFrame(metrics_dict["overlap_nsd"], columns=["Mean Overlap NSD"])
        overlap_nsd_df = pd.DataFrame(metrics_dict["overlap_nsd_per_pair"],
                                      columns=[f"Overlap NSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for
                                               pair in metrics_dict["overlap_pairs"]])
        overlap_voe_mean_df = pd.DataFrame(metrics_dict["overlap_voe"], columns=["Mean Overlap VOE"])
        overlap_voe_df = pd.DataFrame(metrics_dict["overlap_voe_per_pair"],
                                      columns=[f"Overlap VOE {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for
                                               pair in metrics_dict["overlap_pairs"]])
        overlap_msd_mean_df = pd.DataFrame(metrics_dict["overlap_msd"], columns=["Mean Overlap MSD"])
        overlap_msd_df = pd.DataFrame(metrics_dict["overlap_msd_per_pair"],
                                      columns=[f"Overlap MSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for
                                               pair in metrics_dict["overlap_pairs"]])
        overlap_ravd_mean_df = pd.DataFrame(metrics_dict["overlap_ravd"], columns=["Mean Overlap RAVD"])
        overlap_ravd_df = pd.DataFrame(metrics_dict["overlap_ravd_per_pair"],
                                       columns=[f"Overlap RAVD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for
                                                pair in metrics_dict["overlap_pairs"]])

        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"],
                               columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])

        # acc_df = pd.DataFrame(metrics_dict["accuracy_pc"], columns=[f"Accuracy {bone_name_dict[i]}" for i in range(num_classes)])
        # acc_mean_df = pd.DataFrame(metrics_dict["accuracy"], columns=["Mean Accuracy"])
        # precision_df = pd.DataFrame(metrics_dict["precision_pc"], columns=[f"Precision {bone_name_dict[i]}" for i in range(num_classes)])
        # precision_mean_df = pd.DataFrame(metrics_dict["precision"], columns=["Mean Precision"])
        # recall_df = pd.DataFrame(metrics_dict["recall_pc"], columns=[f"Recall {bone_name_dict[i]}" for i in range(num_classes)])
        # recall_mean_df = pd.DataFrame(metrics_dict["recall"], columns=["Mean Recall"])
        # f1_df = pd.DataFrame(metrics_dict["f1score_pc"], columns=[f"F1-score {bone_name_dict[i]}" for i in range(num_classes)])
        # f1_mean_df = pd.DataFrame(metrics_dict["f1score"], columns=["Mean F1-score"])
        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
            [fname_df, overlap_dsc_df, overlap_dsc_mean_df, overlap_nsd_df, overlap_nsd_mean_df, overlap_voe_df,
             overlap_voe_mean_df, overlap_msd_df, overlap_msd_mean_df, overlap_ravd_df, overlap_ravd_mean_df, dsc_df,
             dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df, ravd_df, ravd_mean_df], axis=1)
        # metric_df = pd.concat(
        #     [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, hd95_df, hd95_mean_df, acc_df, acc_mean_df,
        #      precision_df, precision_mean_df, recall_df, recall_mean_df, f1_df,f1_mean_df], axis=1)
        # 仅在有限值上求均值：跳过 inf 和 NaN
        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        finite_means = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=0)
        column_means = pd.Series(finite_means, index=metric_df.columns[1:])
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


if __name__ == '__main__':
    tester = SegTester()
    tester.test()

















