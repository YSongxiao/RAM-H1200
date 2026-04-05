from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
import monai
import numpy as np
import torch
from itertools import combinations
from monai.metrics.metric import CumulativeIterationMetric
from sklearn.metrics import cohen_kappa_score
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

bone_name_dict = {0: "Capitate", 1: "DistalRadius", 2: "DistalUlna", 3: "Hamate", 4: "Lunate", 5: "Pisifrom&Triquetrum",
                  6: "Scaphoid", 7: "Trapzium", 8: "Trapzoid", 9: "metacarpal1st", 10: "metacarpal2nd",
                  11: "metacarpal3rd", 12: "metacarpal4th", 13: "metacarpal5th"}

lesion_name_dict = {0: "Bone Erosion"}

overlap_pairs = [(1, 6), (1, 4), (6, 7), (0, 6), (7, 9), (0, 11), (3, 12), (4, 6), (7, 8), (0, 8),
                 (3, 13), (7, 10), (8, 10), (10, 11)]  # , (1, 2), (6, 8), (9, 10)

fullhand_bone_name_dict = {
    0: "Capitate",
    1: "DP1",
    2: "DP2",
    3: "DP3",
    4: "DP4",
    5: "DP5",
    6: "Hamate",
    7: "Lunate",
    8: "MC1",
    9: "MC2",
    10: "MC3",
    11: "MC4",
    12: "MC5",
    13: "MP2",
    14: "MP3",
    15: "MP4",
    16: "MP5",
    17: "PP1",
    18: "PP2",
    19: "PP3",
    20: "PP4",
    21: "PP5",
    22: "Pisifrom_Triquetrum",
    23: "Radius",
    24: "Scaphoid",
    25: "Sesamoid",
    26: "SoftTissue",
    27: "Trapezium",
    28: "Trapezoid",
    29: "Ulna",
}

_bone_to_fullhand_name = {
    "Capitate": "Capitate",
    "DistalRadius": "Radius",
    "DistalUlna": "Ulna",
    "Hamate": "Hamate",
    "Lunate": "Lunate",
    "Pisifrom&Triquetrum": "Pisifrom_Triquetrum",
    "Scaphoid": "Scaphoid",
    "Trapzium": "Trapezium",
    "Trapzoid": "Trapezoid",
    "metacarpal1st": "MC1",
    "metacarpal2nd": "MC2",
    "metacarpal3rd": "MC3",
    "metacarpal4th": "MC4",
    "metacarpal5th": "MC5",
}


def build_fullhand_overlap_pairs():
    fullhand_name_to_channel = {name: ch for ch, name in fullhand_bone_name_dict.items()}
    resolved_pairs = []
    for i, j in overlap_pairs:
        src_name_i = bone_name_dict[i]
        src_name_j = bone_name_dict[j]
        dst_name_i = _bone_to_fullhand_name[src_name_i]
        dst_name_j = _bone_to_fullhand_name[src_name_j]
        resolved_pairs.append(
            (fullhand_name_to_channel[dst_name_i], fullhand_name_to_channel[dst_name_j])
        )
    return resolved_pairs


fullhand_overlap_pairs = build_fullhand_overlap_pairs()


class RAVDMetric(CumulativeIterationMetric):
    """
    MONAI-compatible RAVD metric (Relative Absolute Volume Difference).
    """

    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        B, C = y_pred.shape[:2]
        results = []

        for b in range(B):
            sample_result = []
            for c in range(C):
                if not self.include_background and c == 0:
                    continue
                pred_bin = (y_pred[b, c] > 0.5).astype(np.uint8)
                gt_bin = (y[b, c] > 0.5).astype(np.uint8)
                gt_vol = int(gt_bin.sum())
                if gt_vol == 0:
                    val = 0.0
                else:
                    pred_vol = int(pred_bin.sum())
                    val = abs(pred_vol - gt_vol) / (gt_vol + 1e-8)
                sample_result.append(val)
            results.append(sample_result)
        return torch.tensor(results)

    def reset(self) -> None:
        super().reset()


class VOEMetric:
    def __init__(self, iou_func):
        self.iou = iou_func

    def __call__(self, pred_bin, gt_bin):
        assert pred_bin.shape == gt_bin.shape, "Shape mismatch between pred and gt"
        iou = self.iou(pred_bin, gt_bin)
        voe = 1 - iou
        return voe


class CostAwareDiceMetric:
    def __init__(
        self,
        cost_matrix,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act=None,
        reduction: str = "mean",
        smooth_nr: float = 1e-8,
        smooth_dr: float = 1e-8,
        threshold: float = 0.5,
    ):
        cost_matrix = torch.as_tensor(cost_matrix, dtype=torch.float32)
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError("cost_matrix must be a square matrix.")
        self.cost_matrix = cost_matrix
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.threshold = threshold

        act_count = int(sigmoid) + int(softmax) + int(other_act is not None)
        if act_count > 1:
            raise ValueError("At most one of sigmoid=True, softmax=True, or other_act is not None is allowed.")

    def __call__(self, pred_bin, gt_bin):
        if isinstance(pred_bin, np.ndarray):
            pred_bin = torch.from_numpy(pred_bin)
        if isinstance(gt_bin, np.ndarray):
            gt_bin = torch.from_numpy(gt_bin)

        pred_bin = pred_bin.detach().cpu()
        gt_bin = gt_bin.detach().cpu()

        if pred_bin.dim() == 3:
            pred_bin = pred_bin.unsqueeze(0)
        if gt_bin.dim() == 3:
            gt_bin = gt_bin.unsqueeze(0)

        if self.sigmoid:
            pred_bin = (torch.sigmoid(pred_bin) > self.threshold).int()
        elif self.softmax:
            if pred_bin.shape[1] == 1:
                pred_bin = pred_bin.int()
            else:
                pred_idx = torch.argmax(torch.softmax(pred_bin, dim=1), dim=1)
                pred_bin = torch.nn.functional.one_hot(pred_idx, num_classes=pred_bin.shape[1]).permute(0, 3, 1, 2).int()
        elif self.other_act is not None:
            pred_bin = (self.other_act(pred_bin) > self.threshold).int()
        else:
            pred_bin = pred_bin.int()

        if self.to_onehot_y:
            if pred_bin.shape[1] == 1:
                gt_bin = gt_bin.int()
            else:
                gt_idx = gt_bin.long().squeeze(1) if gt_bin.dim() == 4 and gt_bin.shape[1] == 1 else gt_bin.long()
                gt_bin = torch.nn.functional.one_hot(gt_idx, num_classes=pred_bin.shape[1]).permute(0, 3, 1, 2).int()
        else:
            gt_bin = gt_bin.int()

        assert pred_bin.shape == gt_bin.shape, "Shape mismatch between pred and gt after preprocessing."

        B, C = pred_bin.shape[:2]
        results = []

        for b in range(B):
            sample_result = []
            for c in range(C):
                if not self.include_background and c == 0:
                    continue

                p = pred_bin[b, c].bool()
                g = gt_bin[b, c].bool()

                tp = torch.logical_and(p, g).sum().item()
                fn = torch.logical_and(~p, g).sum().item()
                fp_cost = 0.0

                for gt_class in range(C):
                    if gt_class == c:
                        continue
                    penalty = float(self.cost_matrix[gt_class, c].item())
                    fp_cost += penalty * torch.logical_and(p, gt_bin[b, gt_class].bool()).sum().item()

                val = (2.0 * tp + self.smooth_nr) / (2.0 * tp + fn + fp_cost + self.smooth_dr)
                sample_result.append(val)

            results.append(sample_result)

        results = torch.tensor(results, dtype=torch.float32)
        if self.reduction == "mean":
            return results.mean()
        if self.reduction == "sum":
            return results.sum()
        if self.reduction == "none":
            return results
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class CreditAwareDiceScore:
    def __init__(
        self,
        credit_matrix,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act=None,
        reduction: str = "mean",
        smooth_nr: float = 0.0,   # �� MONAI 默认就是 0
        smooth_dr: float = 0.0,
    ):
        credit_matrix = torch.as_tensor(credit_matrix, dtype=torch.float32)
        if credit_matrix.ndim != 2 or credit_matrix.shape[0] != credit_matrix.shape[1]:
            raise ValueError("credit_matrix must be a square matrix.")

        self.credit_matrix = credit_matrix
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        act_count = int(sigmoid) + int(softmax) + int(other_act is not None)
        if act_count > 1:
            raise ValueError("At most one activation allowed.")

    def __call__(self, pred_logits, gt):
        if isinstance(pred_logits, np.ndarray):
            pred_logits = torch.from_numpy(pred_logits)
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)

        pred_logits = pred_logits.detach()
        gt = gt.detach()

        if pred_logits.dim() == 3:
            pred_logits = pred_logits.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        device = pred_logits.device
        credit_matrix = self.credit_matrix.to(device)

        # =========================
        # �� 1. hard prediction（关键）
        # =========================
        if self.softmax:
            probs = torch.softmax(pred_logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            pred = torch.nn.functional.one_hot(pred_class, num_classes=probs.shape[1])
            pred = pred.permute(0, 3, 1, 2).float()
        elif self.sigmoid:
            probs = torch.sigmoid(pred_logits)
            pred = (probs > 0.5).float()
        elif self.other_act is not None:
            probs = self.other_act(pred_logits)
            pred = (probs > 0.5).float()
        else:
            pred = pred_logits.float()

        # =========================
        # �� 2. GT one-hot
        # =========================
        if self.to_onehot_y:
            gt_idx = gt.long().squeeze(1) if gt.dim() == 4 and gt.shape[1] == 1 else gt.long()
            gt = torch.nn.functional.one_hot(gt_idx, num_classes=pred.shape[1])
            gt = gt.permute(0, 3, 1, 2).float()
        else:
            gt = gt.float()

        # =========================
        # �� 3. remove background
        # =========================
        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:, ...]
            gt = gt[:, 1:, ...]
            credit_matrix = credit_matrix[1:, 1:]

        # =========================
        # �� 4. credit-aware mapping
        # =========================
        credited_pred = torch.einsum("bd...,cd->bc...", pred, credit_matrix)

        reduce_dims = tuple(range(2, gt.ndim))

        intersection = (gt * credited_pred).sum(dim=reduce_dims)

        target_mass = gt.sum(dim=reduce_dims)

        # �� 用原始 pred mass（不是 credited）
        pred_mass = pred.sum(dim=reduce_dims)

        # =========================
        # �� 5. MONAI-style Dice
        # =========================
        denominator = target_mass + pred_mass

        # 关键：denominator==0 → NaN（自动产生）
        score = (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr
        )

        # =========================
        # �� 6. reduction（忽略 NaN）
        # =========================
        if self.reduction == "mean":
            return torch.nanmean(score)
        if self.reduction == "sum":
            return torch.nansum(score)
        if self.reduction == "none":
            return score

        raise ValueError(f"Unsupported reduction: {self.reduction}")
    

class OverlapMetric:
    def __init__(self, nsd_tolerance, num_classes):
        self.nsd_tolerance = nsd_tolerance
        self.num_classes = num_classes

        self.dsc = monai.metrics.DiceMetric(include_background=self.include_background, reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[self.nsd_tolerance],
            include_background=True,
            reduction="none"
        )

    def __call__(self, pred_bin, gt_bin):
        dsc_per_pair = []
        nsd_per_pair = []
        avg_dsc = []
        avg_nsd = []
        valid_pairs = []

        for b in range(gt_bin.shape[0]):
            dsc_per_pair_ = []
            nsd_per_pair_ = []
            valid_pairs_ = []

            pred_overlap_list = []
            gt_overlap_list = []

            for i, j in combinations(range(self.num_classes), 2):
                gt_i = gt_bin[b][i] > 0
                gt_j = gt_bin[b][j] > 0
                gt_overlap = torch.logical_and(gt_i, gt_j)

                if not gt_overlap.any():
                    continue  # skip non-overlapping pairs in GT

                pred_i = pred_bin[b][i] > 0
                pred_j = pred_bin[b][j] > 0
                pred_overlap = torch.logical_and(pred_i, pred_j)

                # 构造 [1, 1, H, W] 格式以兼容 MONAI Metric
                gt_tensor = gt_overlap[None, None].float()
                pred_tensor = pred_overlap[None, None].float()

                gt_overlap_list.append(gt_tensor)
                pred_overlap_list.append(pred_tensor)
                valid_pairs_.append((i, j))

            if len(gt_overlap_list) == 0:
                # 当前样本没有任何 valid pair
                avg_dsc.append(0.0)
                avg_nsd.append(0.0)
                dsc_per_pair.append([])
                nsd_per_pair.append([])
                valid_pairs.append([])
                continue

            # 堆叠成 [N, 1, H, W]
            pred_stack = torch.cat(pred_overlap_list, dim=0)
            gt_stack = torch.cat(gt_overlap_list, dim=0)

            # 计算 Dice
            dsc_values = self.dsc(pred_stack, gt_stack).detach().cpu()

            # 计算 NSD
            nsd_values = self.nsd(pred_stack, gt_stack).detach().cpu()

            # 处理 NSD 中的 nan 值（替换为 0.0）
            nsd_values = torch.nan_to_num(nsd_values, nan=0.0)

            for i in range(len(valid_pairs_)):
                dsc_per_pair_.append(dsc_values[i].item())
                nsd_per_pair_.append(nsd_values[i].item())

            avg_dsc_ = dsc_values.mean().item()
            avg_nsd_ = nsd_values.mean().item()

            dsc_per_pair.append(dsc_per_pair_)
            nsd_per_pair.append(nsd_per_pair_)
            avg_dsc.append(avg_dsc_)
            avg_nsd.append(avg_nsd_)
            valid_pairs.append(valid_pairs_)

        return avg_dsc, avg_nsd, dsc_per_pair, nsd_per_pair, valid_pairs

    def get_valid_pairs(self, gt_bin):
        valid_pairs = []
        for i, j in combinations(range(self.num_classes), 2):
            if torch.logical_and(gt_bin[i] > 0, gt_bin[j] > 0).any():
                valid_pairs.append((i, j))
        return valid_pairs


class NewOverlapMetric:
    def __init__(self, nsd_tolerance, num_classes, overlap_pairs):
        self.nsd_tolerance = nsd_tolerance
        self.num_classes = num_classes

        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[self.nsd_tolerance for _ in range(len(overlap_pairs))],
            include_background=True,
            reduction="none",
            get_not_nans=True
        )
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none", get_not_nans=True)
        self.ravd = RAVDMetric(include_background=True)
        self.valid_pairs = overlap_pairs

    def __call__(self, pred_bin, gt_bin):
        """
        pred_bin, gt_bin: [B, num_classes, H, W]
        overlap_pairs: list of (i, j) tuples specifying which bone pairs to evaluate
        """
        dsc_per_pair = None
        nsd_per_pair = None
        voe_per_pair = None
        msd_per_pair = None
        ravd_per_pair = None
        avg_dsc = []
        avg_nsd = []
        avg_voe = []
        avg_msd = []
        avg_ravd = []

        new_pred_bin = None
        new_gt_bin = None
        for pair in self.valid_pairs:
            if new_pred_bin is None:
                new_pred_bin = torch.logical_and(pred_bin[:, pair[0], ...], pred_bin[:, pair[1], ...]).unsqueeze(1)
                new_gt_bin = torch.logical_and(gt_bin[:, pair[0], ...], gt_bin[:, pair[1], ...]).unsqueeze(1)
            else:
                pred_overlap_bin = torch.logical_and(pred_bin[:, pair[0], ...], pred_bin[:, pair[1], ...]).unsqueeze(1)
                gt_overlap_bin = torch.logical_and(gt_bin[:, pair[0], ...], gt_bin[:, pair[1], ...]).unsqueeze(1)
                new_pred_bin = torch.cat([new_pred_bin, pred_overlap_bin], dim=1)
                new_gt_bin = torch.cat([new_gt_bin, gt_overlap_bin], dim=1)

        dsc_values = self.dsc(new_pred_bin, new_gt_bin).detach().cpu()
        nsd_values = self.nsd(new_pred_bin, new_gt_bin).detach().cpu()

        voe_values = self.voe(new_pred_bin, new_gt_bin).squeeze()
        msd_values = self.msd(new_pred_bin, new_gt_bin).squeeze()
        # 删除：不再把 inf 改成 nan
        # msd_values = torch.where(torch.isfinite(msd_values), msd_values,
        #                          torch.tensor(float('nan'), device=msd_values.device, dtype=msd_values.dtype))
        ravd_values = self.ravd(new_pred_bin, new_gt_bin).squeeze()

        avg_dsc_ = dsc_values.nanmean().item()
        avg_nsd_ = nsd_values.nanmean().item()

        avg_voe_ = voe_values.nanmean().item()
        # 这里用 isfinite 跳过 inf 与 nan；若没有有限值则返回 nan
        _finite = torch.isfinite(msd_values)
        avg_msd_ = msd_values[_finite].mean().item() if _finite.any() else float('nan')
        avg_ravd_ = ravd_values.nanmean().item()

        if dsc_per_pair is None:
            dsc_per_pair = dsc_values
        else:
            dsc_per_pair = torch.cat([dsc_per_pair, dsc_values], dim=0)

        if nsd_per_pair is None:
            nsd_per_pair = nsd_values
        else:
            nsd_per_pair = torch.cat([nsd_per_pair, nsd_values], dim=0)

        if voe_per_pair is None:
            voe_per_pair = voe_values
        else:
            voe_per_pair = torch.cat([voe_per_pair, voe_values], dim=0)

        if msd_per_pair is None:
            msd_per_pair = msd_values
        else:
            msd_per_pair = torch.cat([msd_per_pair, msd_values], dim=0)

        if ravd_per_pair is None:
            ravd_per_pair = ravd_values
        else:
            ravd_per_pair = torch.cat([ravd_per_pair, ravd_values], dim=0)
        avg_dsc.append(avg_dsc_)
        avg_nsd.append(avg_nsd_)

        avg_voe.append(avg_voe_)
        avg_msd.append(avg_msd_)
        avg_ravd.append(avg_ravd_)

        return (np.array(avg_dsc), np.array(avg_nsd), np.array(avg_voe), np.array(avg_msd), np.array(avg_ravd),
                dsc_per_pair.squeeze().numpy(), nsd_per_pair.squeeze().numpy(), voe_per_pair.squeeze().numpy(),
                msd_per_pair.squeeze().numpy(), ravd_per_pair.squeeze().numpy(), self.valid_pairs)


class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_labels = num_classes
        self.fnames = []
        # self.acc_per_channel = []
        # self.acc_reduced = []
        # self.prec_per_channel = []
        # self.prec_reduced = []
        # self.recall_per_channel = []
        # self.recall_reduced = []
        # self.f1_per_channel = []
        # self.f1_reduced = []
        self.dsc_per_channel = []
        self.dsc_reduced = []
        self.nsd_per_channel = []
        self.nsd_reduced = []
        # self.hd95_per_channel = []
        # self.hd95_reduced = []
        self.voe_per_channel = []
        self.voe_reduced = []
        self.msd_per_channel = []
        self.msd_reduced = []
        self.ravd_per_channel = []
        self.ravd_reduced = []

        self.overlap_dsc_reduced = []
        self.overlap_nsd_reduced = []
        self.overlap_voe_reduced = []
        self.overlap_msd_reduced = []
        self.overlap_ravd_reduced = []
        self.overlap_dsc_per_pair = []
        self.overlap_nsd_per_pair = []
        self.overlap_voe_per_pair = []
        self.overlap_msd_per_pair = []
        self.overlap_ravd_per_pair = []
        self.overlap_pairs = []
        # self.accuracy = MultilabelAccuracy(num_labels=num_labels, average="none")
        # self.precision = MultilabelPrecision(num_labels=num_labels, average="none")
        # self.recall = MultilabelRecall(num_labels=num_labels, average="none")
        # self.f1 = MultilabelF1Score(num_labels=num_labels, average="none")
        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(class_thresholds=[2 for _ in range(num_classes)],include_background=True, reduction="none")
        # self.nsd = monai.metrics.SurfaceDistanceMetric(include_background=True, reduction="none") # compute_surface_dice
        # self.hd95 = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none")
        self.ravd = RAVDMetric(include_background=True)
        self.overlap_metric = NewOverlapMetric(nsd_tolerance=2, num_classes=num_classes, overlap_pairs=overlap_pairs)

    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)
        if isinstance(pred_bin, np.ndarray):
            pred_bin = torch.from_numpy(pred_bin)
        else:
            pred_bin = pred_bin.detach().cpu()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        else:
            gt = gt.detach().cpu()

        dsc_pc = self.dsc(pred_bin, gt).squeeze()
        nsd_pc = self.nsd(pred_bin, gt).squeeze()
        voe_pc = self.voe(pred_bin, gt).squeeze()
        msd_pc = self.msd(pred_bin, gt).squeeze()
        ravd_pc = self.ravd(pred_bin, gt).squeeze()
        (avg_dsc, avg_nsd, avg_voe, avg_msd, avg_ravd, dsc_per_pair, nsd_per_pair, voe_per_pair, msd_per_pair,
         ravd_per_pair, valid_pairs) = self.overlap_metric(pred_bin, gt)

        if not self.include_background and pred_bin.shape[0] > 1:
            dsc_pc = dsc_pc[-pred_eval.shape[0]:]
            nsd_pc = nsd_pc[-pred_eval.shape[0]:]
            voe_pc = voe_pc[-pred_eval.shape[0]:]
            msd_pc = msd_pc[-pred_eval.shape[0]:]
            ravd_pc = ravd_pc[-pred_eval.shape[0]:]

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)
        self.overlap_dsc_per_pair.append(dsc_per_pair)
        self.overlap_nsd_per_pair.append(nsd_per_pair)
        self.overlap_voe_per_pair.append(voe_per_pair)
        self.overlap_msd_per_pair.append(msd_per_pair)
        self.overlap_ravd_per_pair.append(ravd_per_pair)

        self.dsc_reduced.append(dsc_pc.mean())
        self.nsd_reduced.append(nsd_pc.mean())
        self.voe_reduced.append(voe_pc.mean())
        self.msd_reduced.append(msd_pc[np.isfinite(msd_pc)].mean() if np.isfinite(msd_pc).any() else np.nan)
        self.ravd_reduced.append(ravd_pc.mean())
        self.overlap_dsc_reduced.append(avg_dsc)
        self.overlap_nsd_reduced.append(avg_nsd)
        self.overlap_voe_reduced.append(avg_voe)
        self.overlap_msd_reduced.append(avg_msd)
        self.overlap_ravd_reduced.append(avg_ravd)

        # self.overlap_pairs.append(valid_pairs)
        self.overlap_pairs = valid_pairs

    def get_metrics(self):
        metrics = {

            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),
            "overlap_dsc_per_pair": self.overlap_dsc_per_pair,
            "overlap_nsd_per_pair": self.overlap_nsd_per_pair,
            "overlap_voe_per_pair": self.overlap_voe_per_pair,
            "overlap_msd_per_pair": self.overlap_msd_per_pair,
            "overlap_ravd_per_pair": self.overlap_ravd_per_pair,

            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.where(np.isfinite(self.msd_reduced), self.msd_reduced, np.nan),
            "ravd": np.array(self.ravd_reduced),
            "overlap_dsc": np.array(self.overlap_dsc_reduced),
            "overlap_nsd": np.array(self.overlap_nsd_reduced),
            "overlap_voe": np.array(self.overlap_voe_reduced),
            "overlap_msd": np.array(self.overlap_msd_reduced),
            "overlap_ravd": np.array(self.overlap_ravd_reduced),
            "overlap_pairs": self.overlap_pairs,
            # "hd95": np.array(self.hd95_reduced),
            "fname": self.fnames,
        }
        return metrics


class FullHandSegmentationMetrics:
    def __init__(self, num_classes):
        self.num_labels = num_classes
        self.fnames = []
        self.dsc_per_channel = []
        self.dsc_reduced = []
        self.nsd_per_channel = []
        self.nsd_reduced = []
        self.voe_per_channel = []
        self.voe_reduced = []
        self.msd_per_channel = []
        self.msd_reduced = []
        self.ravd_per_channel = []
        self.ravd_reduced = []
        self.overlap_dsc_reduced = []
        self.overlap_nsd_reduced = []
        self.overlap_voe_reduced = []
        self.overlap_msd_reduced = []
        self.overlap_ravd_reduced = []
        self.overlap_dsc_per_pair = []
        self.overlap_nsd_per_pair = []
        self.overlap_voe_per_pair = []
        self.overlap_msd_per_pair = []
        self.overlap_ravd_per_pair = []
        self.overlap_pairs = fullhand_overlap_pairs
        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(class_thresholds=[2 for _ in range(num_classes)],include_background=True, reduction="none")
        # self.nsd = monai.metrics.SurfaceDistanceMetric(include_background=True, reduction="none") # compute_surface_dice
        # self.hd95 = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none")
        self.ravd = RAVDMetric(include_background=True)
        self.overlap_metric = NewOverlapMetric(
            nsd_tolerance=2,
            num_classes=num_classes,
            overlap_pairs=fullhand_overlap_pairs,
        )

    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)
        if isinstance(pred_bin, np.ndarray):
            pred_bin = torch.from_numpy(pred_bin)
        else:
            pred_bin = pred_bin.detach().cpu()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        else:
            gt = gt.detach().cpu()

        dsc_pc = self.dsc(pred_bin, gt).squeeze()
        nsd_pc = self.nsd(pred_bin, gt).squeeze()
        voe_pc = self.voe(pred_bin, gt).squeeze()
        msd_pc = self.msd(pred_bin, gt).squeeze()
        ravd_pc = self.ravd(pred_bin, gt).squeeze()
        (avg_dsc, avg_nsd, avg_voe, avg_msd, avg_ravd, dsc_per_pair, nsd_per_pair, voe_per_pair, msd_per_pair,
         ravd_per_pair, valid_pairs) = self.overlap_metric(pred_bin, gt)

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)
        self.overlap_dsc_per_pair.append(dsc_per_pair)
        self.overlap_nsd_per_pair.append(nsd_per_pair)
        self.overlap_voe_per_pair.append(voe_per_pair)
        self.overlap_msd_per_pair.append(msd_per_pair)
        self.overlap_ravd_per_pair.append(ravd_per_pair)

        self.dsc_reduced.append(float(torch.nanmean(dsc_pc).item()))
        self.nsd_reduced.append(float(torch.nanmean(nsd_pc).item()))
        self.voe_reduced.append(float(torch.nanmean(voe_pc).item()))
        finite_msd = torch.isfinite(msd_pc)
        self.msd_reduced.append(float(msd_pc[finite_msd].mean().item()) if finite_msd.any() else np.nan)
        self.ravd_reduced.append(float(torch.nanmean(ravd_pc).item()))
        self.overlap_dsc_reduced.append(float(np.asarray(avg_dsc).squeeze()))
        self.overlap_nsd_reduced.append(float(np.asarray(avg_nsd).squeeze()))
        self.overlap_voe_reduced.append(float(np.asarray(avg_voe).squeeze()))
        self.overlap_msd_reduced.append(float(np.asarray(avg_msd).squeeze()))
        self.overlap_ravd_reduced.append(float(np.asarray(avg_ravd).squeeze()))
        self.overlap_pairs = valid_pairs

    def get_metrics(self):
        metrics = {

            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),
            "overlap_dsc_per_pair": self.overlap_dsc_per_pair,
            "overlap_nsd_per_pair": self.overlap_nsd_per_pair,
            "overlap_voe_per_pair": self.overlap_voe_per_pair,
            "overlap_msd_per_pair": self.overlap_msd_per_pair,
            "overlap_ravd_per_pair": self.overlap_ravd_per_pair,

            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.where(np.isfinite(self.msd_reduced), self.msd_reduced, np.nan),
            "ravd": np.array(self.ravd_reduced),
            "overlap_dsc": np.array(self.overlap_dsc_reduced),
            "overlap_nsd": np.array(self.overlap_nsd_reduced),
            "overlap_voe": np.array(self.overlap_voe_reduced),
            "overlap_msd": np.array(self.overlap_msd_reduced),
            "overlap_ravd": np.array(self.overlap_ravd_reduced),
            "overlap_pairs": self.overlap_pairs,

            "fname": self.fnames,
        }
        return metrics


class NonOverlapSegmentationMetrics:
    def __init__(self, num_classes):
        self.num_labels = num_classes
        self.fnames = []

        # Per-channel metrics
        self.dsc_per_channel = []
        self.dsc_reduced = []
        self.nsd_per_channel = []
        self.nsd_reduced = []
        self.voe_per_channel = []
        self.voe_reduced = []
        self.msd_per_channel = []
        self.msd_reduced = []
        self.ravd_per_channel = []
        self.ravd_reduced = []

        # ====== MONAI Metrics ======
        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[2 for _ in range(num_classes)],
            include_background=True,
            reduction="none"
        )
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none")
        self.ravd = RAVDMetric(include_background=True)

    # =======================================================
    #                   UPDATE METRICS
    # =======================================================
    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)
        pred_bin = pred_bin.detach().cpu()
        gt = gt.detach().cpu()

        # Per-class values
        dsc_pc = self.dsc(pred_bin, gt).squeeze()
        nsd_pc = self.nsd(pred_bin, gt).squeeze()
        voe_pc = self.voe(pred_bin, gt).squeeze()
        msd_pc = self.msd(pred_bin, gt).squeeze()
        ravd_pc = self.ravd(pred_bin, gt).squeeze()

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)

        # Reduced (mean across classes)
        self.dsc_reduced.append(dsc_pc.mean())
        self.nsd_reduced.append(nsd_pc.mean())
        self.voe_reduced.append(voe_pc.mean())

        if np.isfinite(msd_pc).any():
            self.msd_reduced.append(msd_pc[np.isfinite(msd_pc)].mean())
        else:
            self.msd_reduced.append(np.nan)

        self.ravd_reduced.append(ravd_pc.mean())

    # =======================================================
    #                   EXPORT METRICS
    # =======================================================
    def get_metrics(self):
        return {
            "fname": self.fnames,

            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),

            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.where(np.isfinite(self.msd_reduced), self.msd_reduced, np.nan),
            "ravd": np.array(self.ravd_reduced),
        }


class BESemanticSegmentationMetrics:
    def __init__(self, num_classes, credit_matrix=None):
        self.num_labels = num_classes
        self.include_background = False
        self.num_eval_labels = num_classes - 1 if not self.include_background and num_classes > 1 else num_classes
        self.fnames = []

        # ====== Segmentation metrics ======
        self.dsc_per_channel = []
        self.dsc_reduced = []
        self.nsd_per_channel = []
        self.nsd_reduced = []
        self.voe_per_channel = []
        self.voe_reduced = []
        self.msd_per_channel = []
        self.msd_reduced = []
        self.ravd_per_channel = []
        self.ravd_reduced = []
        self.credit_aware_dice_per_channel = []
        self.credit_aware_dice_reduced = []

        # ====== BE confusion (foreground, no background channel) ======
        self.tp_per_channel = []
        self.tn_per_channel = []
        self.fp_per_channel = []
        self.fn_per_channel = []

        self.tp_reduced = []
        self.tn_reduced = []
        self.fp_reduced = []
        self.fn_reduced = []

        # ====== Derived BE metrics ======
        self.precision_per_channel = []
        self.recall_per_channel = []
        self.f1_per_channel = []

        self.precision_reduced = []
        self.recall_reduced = []
        self.f1_reduced = []

        # ====== MONAI Metrics ======
        self.dsc = monai.metrics.DiceMetric(include_background=True, reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[2 for _ in range(self.num_eval_labels)],
            include_background=True,
            reduction="none"
        )
        self.voe = VOEMetric(
            monai.metrics.MeanIoU(include_background=True, reduction="none")
        )
        self.msd = monai.metrics.SurfaceDistanceMetric(
            include_background=True,
            symmetric=True,
            reduction="none"
        )
        self.ravd = RAVDMetric(include_background=True)
        if credit_matrix is None:
            credit_matrix = np.eye(num_classes, dtype=np.float32)
        if not self.include_background and num_classes > 1:
            credit_matrix = np.asarray(credit_matrix, dtype=np.float32)[1:, 1:]
        self.credit_aware_dice = CreditAwareDiceScore(
            credit_matrix=credit_matrix,
            include_background=True,
            reduction="none",
        )

    # =======================================================
    #                   UPDATE METRICS
    # =======================================================
    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)

        if isinstance(pred_bin, np.ndarray):
            pred_bin = torch.from_numpy(pred_bin)
        else:
            pred_bin = pred_bin.detach().cpu()

        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        else:
            gt = gt.detach().cpu()

        pred_bin = pred_bin.int()
        gt = gt.int()

        # -------- normalize shape to (C, H, W) --------
        if pred_bin.dim() == 4:
            pred_bin = pred_bin.squeeze(0)
        if gt.dim() == 4:
            gt = gt.squeeze(0)

        if pred_bin.dim() == 2:
            pred_bin = pred_bin.unsqueeze(0)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)

        if not self.include_background and pred_bin.shape[0] > 1:
            pred_eval = pred_bin[1:]
            gt_eval = gt[1:]
        else:
            pred_eval = pred_bin
            gt_eval = gt

        # ================= MONAI metrics =================
        pred_bchw = pred_eval.unsqueeze(0)
        gt_bchw = gt_eval.unsqueeze(0)

        dsc_pc = self.dsc(pred_bchw, gt_bchw).squeeze(0)
        nsd_pc = self.nsd(pred_bchw, gt_bchw).squeeze(0)
        voe_pc = self.voe(pred_bchw, gt_bchw).squeeze(0)
        msd_pc = self.msd(pred_bchw, gt_bchw).squeeze(0)
        ravd_pc = self.ravd(pred_bchw, gt_bchw).squeeze(0)

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)

        self.dsc_reduced.append(dsc_pc.mean())
        self.nsd_reduced.append(nsd_pc.mean())
        self.voe_reduced.append(voe_pc.mean())

        finite_mask = torch.isfinite(msd_pc)

        if finite_mask.any():
            self.msd_reduced.append(msd_pc[finite_mask].mean().item())
        else:
            self.msd_reduced.append(float("nan"))

        self.ravd_reduced.append(ravd_pc.mean())

        # ================= BE confusion =================
        tp, tn, fp, fn = [], [], [], []

        for c in range(pred_eval.shape[0]):
            p = pred_eval[c].bool()
            g = gt_eval[c].bool()

            tp.append(torch.logical_and(p, g).sum().item())
            tn.append(torch.logical_and(~p, ~g).sum().item())
            fp.append(torch.logical_and(p, ~g).sum().item())
            fn.append(torch.logical_and(~p, g).sum().item())

        tp = np.array(tp)
        tn = np.array(tn)
        fp = np.array(fp)
        fn = np.array(fn)

        self.tp_per_channel.append(tp)
        self.tn_per_channel.append(tn)
        self.fp_per_channel.append(fp)
        self.fn_per_channel.append(fn)

        self.tp_reduced.append(tp.mean())
        self.tn_reduced.append(tn.mean())
        self.fp_reduced.append(fp.mean())
        self.fn_reduced.append(fn.mean())

        # ================= Precision / Recall / F1 =================
        eps = 1e-8

        precision_pc = tp / (tp + fp + eps)
        recall_pc = tp / (tp + fn + eps)
        f1_pc = 2 * tp / (2 * tp + fp + fn + eps)

        self.precision_per_channel.append(precision_pc)
        self.recall_per_channel.append(recall_pc)
        self.f1_per_channel.append(f1_pc)

        self.precision_reduced.append(precision_pc.mean())
        self.recall_reduced.append(recall_pc.mean())
        self.f1_reduced.append(f1_pc.mean())

        # ================= Credit-aware Dice =================
        credit_aware_dice_score = self.credit_aware_dice(pred_bchw, gt_bchw)
        credit_aware_dice_pc = np.asarray(credit_aware_dice_score.detach().cpu().numpy(), dtype=float)
        if credit_aware_dice_pc.ndim == 0:
            credit_aware_dice_pc = credit_aware_dice_pc[None]
        if credit_aware_dice_pc.ndim == 2 and credit_aware_dice_pc.shape[0] == 1:
            credit_aware_dice_pc = credit_aware_dice_pc[0]
        self.credit_aware_dice_per_channel.append(credit_aware_dice_pc)
        self.credit_aware_dice_reduced.append(float(np.nanmean(credit_aware_dice_pc)))

    # =======================================================
    #                   EXPORT METRICS
    # =======================================================
    def get_metrics(self):
        return {
            "fname": self.fnames,

            # ---- segmentation ----
            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),
            "credit_aware_dice_pc": np.array(self.credit_aware_dice_per_channel),

            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.where(np.isfinite(self.msd_reduced), self.msd_reduced, np.nan),
            "ravd": np.array(self.ravd_reduced),
            "credit_aware_dice": np.array(self.credit_aware_dice_reduced),

            # ---- confusion ----
            "tp_pc": np.array(self.tp_per_channel),
            "tn_pc": np.array(self.tn_per_channel),
            "fp_pc": np.array(self.fp_per_channel),
            "fn_pc": np.array(self.fn_per_channel),

            "tp": np.array(self.tp_reduced),
            "tn": np.array(self.tn_reduced),
            "fp": np.array(self.fp_reduced),
            "fn": np.array(self.fn_reduced),

            # ---- derived metrics ----
            "precision_pc": np.array(self.precision_per_channel),
            "recall_pc": np.array(self.recall_per_channel),
            "f1_pc": np.array(self.f1_per_channel),

            "precision": np.array(self.precision_reduced),
            "recall": np.array(self.recall_reduced),
            "f1": np.array(self.f1_reduced),
        }


from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity, BinaryConfusionMatrix
)


class ClassificationMetrics:
    def __init__(self, num_classes=2, score_values=None):
        self.num_classes = int(num_classes)
        self.score_values = list(score_values) if score_values is not None else list(range(self.num_classes))
        self.fnames = []
        self.keys = []
        self.preds = []
        self.gts = []

    def update_metrics(self, pred, gt, fname, key, pred_is_label=False):
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()

        if (not pred_is_label) and pred.ndim > 1:
            pred = torch.argmax(pred, dim=1)

        pred = pred.view(-1).long()
        gt = gt.view(-1).long()
        if isinstance(fname, (str, bytes)):
            fname = [fname] * len(pred)
        else:
            fname = list(fname)
        if isinstance(key, (str, bytes)):
            key = [key] * len(pred)
        else:
            key = list(key)

        if not (len(fname) == len(key) == len(pred) == len(gt)):
            raise ValueError("fname, key, pred, and gt must have the same batch length.")

        self.fnames.extend(fname)
        self.keys.extend(key)
        self.preds.extend([p.view(1) for p in pred])
        self.gts.extend([g.view(1) for g in gt])

    def get_metrics(self):
        preds = torch.cat(self.preds, dim=0)
        gts = torch.cat(self.gts, dim=0)
        raw_preds = np.asarray([self.score_values[int(idx)] for idx in preds.tolist()], dtype=float)
        raw_gts = np.asarray([self.score_values[int(idx)] for idx in gts.tolist()], dtype=float)

        if self.num_classes == 2:
            accuracy_metric = BinaryAccuracy()
            precision_metric = BinaryPrecision()
            recall_metric = BinaryRecall()
            f1_metric = BinaryF1Score()
            specificity_metric = BinarySpecificity()
            confusion_matrix_metric = BinaryConfusionMatrix()

            overall_accuracy = accuracy_metric(preds, gts).item()
            overall_precision = precision_metric(preds, gts).item()
            overall_recall = recall_metric(preds, gts).item()
            overall_f1 = f1_metric(preds, gts).item()
            overall_specificity = specificity_metric(preds, gts).item()
            overall_balanced_accuracy = (overall_recall + overall_specificity) / 2

            overall_cm = confusion_matrix_metric(preds, gts)
            tn, fp, fn, tp = overall_cm.flatten().numpy()
            if (fp * fn) > 0:
                overall_dor = (tp * tn) / (fp * fn)
            else:
                overall_dor = float('nan')
        else:
            accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
            precision_metric = MulticlassPrecision(num_classes=self.num_classes, average="macro")
            recall_metric = MulticlassRecall(num_classes=self.num_classes, average="macro")
            f1_metric = MulticlassF1Score(num_classes=self.num_classes, average="macro")
            balanced_accuracy_metric = MulticlassRecall(num_classes=self.num_classes, average="macro")
            confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=self.num_classes)

            overall_accuracy = accuracy_metric(preds, gts).item()
            overall_precision = precision_metric(preds, gts).item()
            overall_recall = recall_metric(preds, gts).item()
            overall_f1 = f1_metric(preds, gts).item()
            overall_specificity = float("nan")
            overall_balanced_accuracy = balanced_accuracy_metric(preds, gts).item()
            overall_cm = confusion_matrix_metric(preds, gts)
            overall_dor = float('nan')

        overall_mae = float(np.mean(np.abs(raw_preds - raw_gts)))
        overall_within_1 = float(np.mean(np.abs(raw_preds - raw_gts) <= 1))
        overall_qwk = float(cohen_kappa_score(raw_gts, raw_preds, weights="quadratic"))

        joint_preds = {}
        joint_gts = {}

        for fname, joint, pred, gt in zip(self.fnames, self.keys, self.preds, self.gts):
            if joint not in joint_preds:
                joint_preds[joint] = []
                joint_gts[joint] = []

            joint_preds[joint].append(pred)
            joint_gts[joint].append(gt)

        joint_metrics = {}
        for joint in joint_preds.keys():
            joint_pred = torch.cat(joint_preds[joint], dim=0)
            joint_gt = torch.cat(joint_gts[joint], dim=0)
            joint_raw_pred = np.asarray([self.score_values[int(idx)] for idx in joint_pred.tolist()], dtype=float)
            joint_raw_gt = np.asarray([self.score_values[int(idx)] for idx in joint_gt.tolist()], dtype=float)

            acc = accuracy_metric(joint_pred, joint_gt).item()
            prec = precision_metric(joint_pred, joint_gt).item()
            rec = recall_metric(joint_pred, joint_gt).item()
            f1 = f1_metric(joint_pred, joint_gt).item()
            cm = confusion_matrix_metric(joint_pred, joint_gt)

            if self.num_classes == 2:
                specificity = specificity_metric(joint_pred, joint_gt).item()
                balanced_accuracy = (rec + specificity) / 2
                tn, fp, fn, tp = cm.flatten().tolist()
                if (fp * fn) > 0:
                    dor = (tp * tn) / (fp * fn)
                else:
                    dor = float('nan')
                confusion_values = [tn, fp, fn, tp]
            else:
                specificity = float("nan")
                balanced_accuracy = MulticlassRecall(
                    num_classes=self.num_classes,
                    average="macro",
                )(joint_pred, joint_gt).item()
                dor = float('nan')
                confusion_values = cm.tolist()

            joint_metrics[joint] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1score": f1,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "dor": dor,
                "mae": float(np.mean(np.abs(joint_raw_pred - joint_raw_gt))),
                "within_1": float(np.mean(np.abs(joint_raw_pred - joint_raw_gt) <= 1)),
                "qwk": float(cohen_kappa_score(joint_raw_gt, joint_raw_pred, weights="quadratic")),
                "confusion_matrix": confusion_values,
            }

        return {
            "num_classes": self.num_classes,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_specificity": overall_specificity,
            "overall_balanced_accuracy": overall_balanced_accuracy,
            "overall_dor": overall_dor,
            "overall_mae": overall_mae,
            "overall_within_1": overall_within_1,
            "overall_qwk": overall_qwk,
            "overall_confusion_matrix": overall_cm.flatten().tolist(),
            "joint_metrics": joint_metrics,
            "fname": self.fnames,
            "joint": self.keys,
        }



if __name__ == '__main__':
    import torch
    ravd = RAVDMetric(True)
    gt = np.zeros((1, 1,100, 100), dtype=np.uint8)
    gt[..., 20:80, 20:80] = 1  # 真值区域：60x60
    gt = torch.tensor(gt)
    pred = np.zeros((1, 1,100, 100), dtype=np.uint8)
    pred[..., 25:75, 25:75] = 1  # 预测区域：50x50
    pred = torch.tensor(pred)
    value = ravd(pred, gt)
    print("RAVD =", value)
