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


data_root = "/mnt/data2/datasx/Carpal/ExportedDataset/FHM-W400_v1/Segmentation/image"
annotation_path = "/mnt/data2/datasx/Carpal/ExportedDataset/FHM-W400_v1/Segmentation/mask/test"
filenames = [str(fname.stem) for fname in Path(annotation_path).rglob("*.npy")]
masks = []

for filename in filenames:
    tmp_mask = np.load(Path(annotation_path) / (filename + ".npy"))
    masks.append(tmp_mask)


def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


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

def test_batch():
    total_infer_time_pt = 0.0
    total_infer_time_box = 0.0
    total_items = 0
    for idx in range(len(masks)):
        path = Path(data_root) / (filenames[idx] + ".bmp")
        fname = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = masks[idx]
        if filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = masks[idx][:, :, ::-1]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        resized_masks = []
        preds_pt = []
        preds_box = []
        for i in range(mask.shape[0]):
            resized_mask = cv2.resize(mask[i], (512, 512), interpolation=cv2.INTER_NEAREST)
            resized_mask[resized_mask > 0] = 1
            resized_masks.append(resized_mask)
            bbox = compute_box_prompt(resized_mask)
            pt = compute_point_prompt(resized_mask)
            pt_label = np.array([1])
            sam = sam_model_registry["vit_h"](checkpoint="./ckpts/sam_vit_h_4b8939.pth")
            predictor = SamPredictor(sam)
            predictor.set_image(rgb)

            start_time_box = time.time()  # ⏱️ Start timing
            pred_masks_box, _, _ = predictor.predict(box=bbox)  # point_coords
            end_time_box = time.time()  # ⏱️ End timing
            infer_time_box = end_time_box - start_time_box
            total_infer_time_box += infer_time_box

            start_time_pt = time.time()  # ⏱️ Start timing
            pred_masks_pt, _, _ = predictor.predict(point_coords=pt, point_labels=pt_label)  # point_coords
            end_time_pt = time.time()  # ⏱️ End timing
            infer_time_pt = end_time_pt - start_time_pt
            total_infer_time_pt += infer_time_pt

            preds_pt.append(pred_masks_pt[0])
            preds_box.append(pred_masks_box[0])
        resized_masks = np.array(resized_masks)
        preds_pt = np.array(preds_pt)
        preds_box = np.array(preds_box)


        total_items += 1


class SegTester:
    def __init__(self):
        # self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.save_overlay = True
        self.save_csv = True
        self.save_pred = True
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
        # self.bone_dsc = []
        # self.dsc_reduced = []
        # self.bone_nsd = []
        # self.nsd_reduced = []
        # self.bone_hd95 = []
        # self.hd95_reduced = []
        # self.bone_acc = []
        # self.acc_reduced = []
        # self.bone_prec = []
        # self.prec_reduced = []
        # self.bone_recall = []
        # self.recall_reduced = []
        # self.bone_f1 = []
        # self.f1_reduced = []
        # self.DSC = monai.metrics.DiceMetric(reduction="none")
        # self.NSD = monai.metrics.SurfaceDistanceMetric(include_background=True, reduction="none")
        # self.HD95 = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
        self.metrics_pt = SegmentationMetrics(num_classes=14)
        self.metrics_box = SegmentationMetrics(num_classes=14)

    def test(self):
        total_infer_time_pt = 0.0
        total_infer_time_box = 0.0
        total_items = 0
        sam = sam_model_registry["vit_h"](checkpoint="./ckpts/sam_vit_h_4b8939.pth")
        n_params = sum(p.numel() for p in sam.parameters())
        print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")
        predictor = SamPredictor(sam.cuda())
        for idx in tqdm(range(len(masks))):
            path = Path(data_root) / (filenames[idx] + ".bmp")
            fname = path.stem
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = masks[idx]
            if filenames[idx][-1] == "L":
                img = cv2.flip(img, 1)  # Horizontal Flip
                mask = masks[idx][:, :, ::-1]
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            resized_masks = []
            preds_pt = []
            preds_box = []
            for i in range(mask.shape[0]):
                resized_mask = cv2.resize(mask[i], (512, 512), interpolation=cv2.INTER_NEAREST)
                resized_mask[resized_mask > 0] = 1
                resized_masks.append(resized_mask)
                bbox = compute_box_prompt(resized_mask)
                pt = compute_point_prompt(resized_mask)
                pt_label = np.array([1])
                predictor.set_image(rgb)

                start_time_box = time.time()  # ⏱️ Start timing
                pred_masks_box, _, _ = predictor.predict(box=bbox)  # point_coords
                end_time_box = time.time()  # ⏱️ End timing
                infer_time_box = end_time_box - start_time_box
                total_infer_time_box += infer_time_box

                start_time_pt = time.time()  # ⏱️ Start timing
                pred_masks_pt, _, _ = predictor.predict(point_coords=pt, point_labels=pt_label)  # point_coords
                end_time_pt = time.time()  # ⏱️ End timing
                infer_time_pt = end_time_pt - start_time_pt
                total_infer_time_pt += infer_time_pt

                preds_pt.append(pred_masks_pt[0])
                preds_box.append(pred_masks_box[0])
            resized_masks = torch.tensor(np.array(resized_masks))[None]
            preds_pt = torch.tensor(np.array(preds_pt))[None]
            preds_box = torch.tensor(np.array(preds_box))[None]
            img_tch = torch.tensor(img)[None][None]
            preds_pt[preds_pt >= 0.5] = 1
            preds_pt[preds_pt < 0.5] = 0
            preds_box[preds_box >= 0.5] = 1
            preds_box[preds_box < 0.5] = 0
            self.metrics_pt.update_metrics(preds_pt, resized_masks, fname)
            self.metrics_box.update_metrics(preds_box, resized_masks, fname)

            if self.save_overlay:
                self.create_overlay("./ckpts/PT", image=img_tch, pred=preds_pt, mask=resized_masks, fname=fname)
                self.create_overlay("./ckpts/BOX", image=img_tch, pred=preds_box, mask=resized_masks, fname=fname)
            if self.save_pred:
                self.create_pred("./ckpts/PT", image=img_tch, pred=preds_pt, mask=resized_masks, fname=fname)
                self.create_overlay_single("./ckpts/PT", image=img_tch, pred=preds_pt, mask=resized_masks, fname=fname)
                self.create_pred("./ckpts/BOX", image=img_tch, pred=preds_box, mask=resized_masks, fname=fname)
                self.create_overlay_single("./ckpts/BOX", image=img_tch, pred=preds_box, mask=resized_masks, fname=fname)
            total_items += 1
        if self.save_csv:
            self.create_csv("./ckpts/PT", type="PT")
            self.create_csv("./ckpts/BOX", type="BOX")

        metrics_dict_pt = self.metrics_pt.get_metrics()
        dsc_reduced = metrics_dict_pt["dsc"].mean()
        print("PT Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict_pt["nsd"].mean()
        print("PT Mean NSD: ", nsd_reduced)
        avg_infer_time = total_infer_time_pt / total_items
        print(f"[PT] Average inference time per item: {avg_infer_time * 1000:.2f} ms")

        metrics_dict_box = self.metrics_box.get_metrics()
        dsc_reduced = metrics_dict_box["dsc"].mean()
        print("BOX Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict_box["nsd"].mean()
        print("BOX Mean NSD: ", nsd_reduced)

        avg_infer_time = total_infer_time_box / total_items
        print(f"[BOX] Average inference time per item: {avg_infer_time * 1000:.2f} ms")
    # def test(self, pred, mask):
    #
    #
    #     self.metrics.update_metrics(pred_bin, gt, batch["fname"][0])
    #     pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #     if self.save_overlay:
    #         self.create_overlay(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
    #     if self.save_pred:
    #         self.create_pred(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
    #         self.create_overlay_single(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
    #
    # if self.save_csv:
    #     self.create_csv(self.args)
    # metrics_dict = self.metrics.get_metrics()
    # dsc_reduced = metrics_dict["dsc"].mean()
    # print("Mean DSC: ", dsc_reduced)
    # nsd_reduced = metrics_dict["nsd"].mean()
    # print("Mean NSD: ", nsd_reduced)
    #
    # avg_infer_time = total_infer_time / total_items
    # print(f"Average inference time per item: {avg_infer_time * 1000:.2f} ms")
    #

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
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
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
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
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
                                      columns=[f"Overlap DSC {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair
                                               in metrics_dict["overlap_pairs"]])
        overlap_nsd_mean_df = pd.DataFrame(metrics_dict["overlap_nsd"], columns=["Mean Overlap NSD"])
        overlap_nsd_df = pd.DataFrame(metrics_dict["overlap_nsd_per_pair"],
                                      columns=[f"Overlap NSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair
                                               in metrics_dict["overlap_pairs"]])

        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
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
            [fname_df, overlap_dsc_df, overlap_dsc_mean_df, overlap_nsd_df, overlap_nsd_mean_df, dsc_df, dsc_mean_df,
             nsd_df, nsd_mean_df, voe_df,
             voe_mean_df, msd_df, msd_mean_df, ravd_df, ravd_mean_df], axis=1)
        # metric_df = pd.concat(
        #     [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, hd95_df, hd95_mean_df, acc_df, acc_mean_df,
        #      precision_df, precision_mean_df, recall_df, recall_mean_df, f1_df,f1_mean_df], axis=1)
        column_means = metric_df.iloc[:, 1:].mean()
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


if __name__ == '__main__':
    tester = SegTester()
    tester.test()
