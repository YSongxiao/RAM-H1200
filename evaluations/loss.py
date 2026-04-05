import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from monai.losses import DiceCELoss, DiceLoss


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + MONAI DiceLoss (multi-label).

    Args:
        pos_weight: torch.Tensor or None, shape [C], per-class weight for BCE
        bce_weight: float, weight of BCE term
        dice_weight: float, weight of Dice term
        smooth: float, smoothing for DiceLoss
    """

    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(sigmoid=True, reduction="mean", squared_pred=True, smooth_nr=smooth, smooth_dr=smooth)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) raw logits
            targets: (N, C, H, W) binary masks {0,1}
        """
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)  # MONAI 内部会做 sigmoid
        return self.bce_w * loss_bce + self.dice_w * loss_dice


class MSEBCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + MONAI DiceLoss (multi-label).

    Args:
        pos_weight: torch.Tensor or None, shape [C], per-class weight for BCE
        bce_weight: float, weight of BCE term
        dice_weight: float, weight of Dice term
        smooth: float, smoothing for DiceLoss
    """

    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(sigmoid=True, reduction="mean", squared_pred=True, smooth_nr=smooth, smooth_dr=smooth)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) raw logits
            targets: (N, C, H, W) binary masks {0,1}
        """
        loss_mse = self.mse(torch.sigmoid(logits), targets)
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)  # MONAI 内部会做 sigmoid
        return loss_mse + self.bce_w * loss_bce + self.dice_w * loss_dice


class MSECEDiceLoss(nn.Module):
    """
    MSE + MONAI DiceCELoss for multi-class segmentation.
    """

    def __init__(self, class_weight=None, mse_weight=1.0, dicece_weight=1.0, smooth=1e-5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.dicece = DiceCELoss(
            softmax=True,
            reduction="mean",
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
            weight=class_weight,
        )
        self.mse_w = mse_weight
        self.dicece_w = dicece_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        loss_mse = self.mse(probs, targets)
        loss_dicece = self.dicece(logits, targets)
        return self.mse_w * loss_mse + self.dicece_w * loss_dicece


class CostAwareCELoss(nn.Module):
    def __init__(self, cost_matrix, include_background=True):
        super().__init__()
        self.cost_matrix = cost_matrix
        self.include_background = include_background

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W)
        """
        probs = torch.softmax(logits, dim=1)

        B, C, H, W = probs.shape

        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_onehot = target_onehot[:, 1:]
            cost_matrix = self.cost_matrix[1:, 1:]
        else:
            cost_matrix = self.cost_matrix

        # vectorized expected cost
        probs_exp = probs.unsqueeze(1)        # (B, 1, C, H, W)
        target_exp = target_onehot.unsqueeze(2)  # (B, C, 1, H, W)

        cost = cost_matrix.view(1, C, C, 1, 1)

        loss = (cost * target_exp * probs_exp).sum()

        return loss / (B * H * W)


class CreditAwareDiceLoss(nn.Module):
    """
    Multi-class credit-aware Dice loss for mutually exclusive segmentation.

    credit_matrix[gt_class, pred_class] indicates how much partial overlap credit
    is awarded when the GT class is gt_class and the prediction is pred_class.
    """

    def __init__(
        self,
        credit_matrix,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act=None,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ):
        super().__init__()
        credit_matrix = torch.as_tensor(credit_matrix, dtype=torch.float32)
        if credit_matrix.ndim != 2 or credit_matrix.shape[0] != credit_matrix.shape[1]:
            raise ValueError("credit_matrix must be a square matrix.")

        self.register_buffer("credit_matrix", credit_matrix)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.batch = batch

        act_count = int(sigmoid) + int(softmax) + int(other_act is not None)
        if act_count > 1:
            raise ValueError("At most one activation allowed.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # =========================
        # activation
        # =========================
        if self.sigmoid:
            probs = torch.sigmoid(logits)
        elif self.softmax:
            if logits.shape[1] == 1:
                raise ValueError("softmax=True requires at least 2 channels.")
            probs = torch.softmax(logits, dim=1)
        elif self.other_act is not None:
            probs = self.other_act(logits)
        else:
            probs = logits

        # =========================
        # one-hot GT
        # =========================
        if self.to_onehot_y:
            if probs.shape[1] == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                targets = F.one_hot(targets.long(), num_classes=probs.shape[1])
                dims = list(range(targets.ndim))
                targets = targets.permute(0, dims[-1], *dims[1:-1]).contiguous()

        if targets.shape != probs.shape:
            raise ValueError("targets must have the same shape as input after processing.")

        pred = probs.float()
        target = targets.float()

        # =========================
        # remove background
        # =========================
        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:, ...]
            target = target[:, 1:, ...]
            credit_matrix = self.credit_matrix[1:, 1:]
        else:
            credit_matrix = self.credit_matrix

        # =========================
        # credit-aware prediction
        # =========================
        credited_pred = torch.einsum("bd...,cd->bc...", pred, credit_matrix)

        reduce_dims = tuple(range(2, target.ndim))

        intersection = (target * credited_pred).sum(dim=reduce_dims)

        target_mass = target.sum(dim=reduce_dims)

        # 🔥🔥 关键修正：用原始 pred mass（不是 credited）
        pred_mass = pred.sum(dim=reduce_dims)

        # =========================
        # batch aggregation（保持你原来的逻辑）
        # =========================
        if self.batch:
            intersection = intersection.sum(dim=0, keepdim=True)
            target_mass = target_mass.sum(dim=0, keepdim=True)
            pred_mass = pred_mass.sum(dim=0, keepdim=True)

        # =========================
        # Dice (with empty-class masking)
        # =========================
        denominator = target_mass + pred_mass

        score = (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr
        )

        # 🔥 关键：mask 掉 empty class（GT=0 且 Pred=0）
        valid_mask = (target_mass > 0) | (pred_mass > 0)

        # 防止全部为空（极端情况）
        if not valid_mask.any():
            return torch.tensor(0.0, device=score.device, requires_grad=True)

        score = score[valid_mask]

        loss = 1.0 - score

        # =========================
        # reduction
        # =========================
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class CostAwareDiceLoss(nn.Module):
    """
    Multi-class symmetric cost-aware Dice loss for mutually exclusive segmentation.

    This version extends the user's original implementation by applying the cost
    matrix to both FP-like and FN-like mismatch terms.

    cost_matrix[gt_class, pred_class] should be interpreted as:
        the penalty incurred when the ground-truth class is gt_class but the model
        predicts pred_class.

    Notes:
        1. This is still a Dice-style overlap loss, but with class-confusion-aware
           penalties added on both sides.
        2. For mutually exclusive multiclass segmentation, softmax=True is usually
           the correct setting.
        3. In lesion-oriented tasks, include_background=False is usually preferred.
    """

    def __init__(
        self,
        cost_matrix,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act=None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ):
        super().__init__()
        cost_matrix = torch.as_tensor(cost_matrix, dtype=torch.float32)
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError("cost_matrix must be a square matrix.")

        self.register_buffer("cost_matrix", cost_matrix)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.batch = batch

        act_count = int(sigmoid) + int(softmax) + int(other_act is not None)
        if act_count > 1:
            raise ValueError(
                "At most one of sigmoid=True, softmax=True, or other_act is not None is allowed."
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            probs = torch.sigmoid(logits)
        elif self.softmax:
            if logits.shape[1] == 1:
                raise ValueError("softmax=True requires at least 2 channels.")
            else:
                probs = torch.softmax(logits, dim=1)
        elif self.other_act is not None:
            probs = self.other_act(logits)
        else:
            probs = logits

        if self.to_onehot_y:
            if probs.shape[1] == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                targets = F.one_hot(targets.long(), num_classes=probs.shape[1])
                dims = list(range(targets.ndim))
                targets = targets.permute(0, dims[-1], *dims[1:-1]).contiguous()

        if targets.shape != probs.shape:
            raise ValueError("targets must have the same shape as input after activation/one-hot conversion.")

        pred = probs.float()
        target = targets.float()

        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:, ...]
            target = target[:, 1:, ...]
            cost_matrix = self.cost_matrix[1:, 1:]
        else:
            cost_matrix = self.cost_matrix

        if self.squared_pred:
            pred_for_vol = pred.pow(2)
            target_for_vol = target.pow(2)
        else:
            pred_for_vol = pred
            target_for_vol = target

        reduce_dims = tuple(range(2, pred.ndim))
        if self.batch:
            reduce_dims = (0,) + reduce_dims

        # Standard exact-match TP
        tp = (pred * target).sum(dim=reduce_dims)

        # Standard per-class volumes
        pred_volume = pred_for_vol.sum(dim=reduce_dims)
        gt_volume = target_for_vol.sum(dim=reduce_dims)

        # Confusion-aware mismatch terms
        # fp_cost_for_class_j: predictions into class j coming from wrong GT classes
        # fn_cost_for_class_i: GT class i being assigned to wrong predicted classes
        num_classes = pred.shape[1]
        fp_cost = torch.zeros_like(tp)
        fn_cost = torch.zeros_like(tp)

        if self.batch:
            for pred_class in range(num_classes):
                pred_map = pred_for_vol[:, pred_class:pred_class + 1, ...]
                total_fp_cost = 0.0
                for gt_class in range(num_classes):
                    if gt_class == pred_class:
                        continue
                    gt_map = target[:, gt_class:gt_class + 1, ...]
                    penalty = cost_matrix[gt_class, pred_class]
                    total_fp_cost = total_fp_cost + (penalty * pred_map * gt_map).sum()
                fp_cost[pred_class] = total_fp_cost

            for gt_class in range(num_classes):
                gt_map = target_for_vol[:, gt_class:gt_class + 1, ...]
                total_fn_cost = 0.0
                for pred_class in range(num_classes):
                    if pred_class == gt_class:
                        continue
                    pred_map = pred[:, pred_class:pred_class + 1, ...]
                    penalty = cost_matrix[gt_class, pred_class]
                    total_fn_cost = total_fn_cost + (penalty * gt_map * pred_map).sum()
                fn_cost[gt_class] = total_fn_cost
        else:
            spatial_dims = tuple(range(2, pred.ndim))
            for b in range(pred.shape[0]):
                for pred_class in range(num_classes):
                    pred_map = pred_for_vol[b:b + 1, pred_class:pred_class + 1, ...]
                    total_fp_cost = 0.0
                    for gt_class in range(num_classes):
                        if gt_class == pred_class:
                            continue
                        gt_map = target[b:b + 1, gt_class:gt_class + 1, ...]
                        penalty = cost_matrix[gt_class, pred_class]
                        total_fp_cost = total_fp_cost + (penalty * pred_map * gt_map).sum(dim=spatial_dims)
                    fp_cost[b, pred_class] = total_fp_cost

                for gt_class in range(num_classes):
                    gt_map = target_for_vol[b:b + 1, gt_class:gt_class + 1, ...]
                    total_fn_cost = 0.0
                    for pred_class in range(num_classes):
                        if pred_class == gt_class:
                            continue
                        pred_map = pred[b:b + 1, pred_class:pred_class + 1, ...]
                        penalty = cost_matrix[gt_class, pred_class]
                        total_fn_cost = total_fn_cost + (penalty * gt_map * pred_map).sum(dim=spatial_dims)
                    fn_cost[b, gt_class] = total_fn_cost

        denominator = 2.0 * tp + fp_cost + fn_cost

        if self.jaccard:
            score = (tp + self.smooth_nr) / (tp + fp_cost + fn_cost + self.smooth_dr)
        else:
            score = (2.0 * tp + self.smooth_nr) / (denominator + self.smooth_dr)

        loss = 1.0 - score

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class CreditAwareDiceCELoss(nn.Module):
    """
    CrossEntropyLoss + CreditAwareDiceLoss for multi-class segmentation.
    """

    def __init__(
        self,
        credit_matrix,
        ce_weight=None,
        ce_loss_weight: float = 1.0,
        credit_dice_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.credit_dice = CreditAwareDiceLoss(
            credit_matrix=credit_matrix,
            include_background=include_background,
            softmax=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
        )
        self.ce_w = ce_loss_weight
        self.credit_dice_w = credit_dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_indices = targets.argmax(dim=1)
        loss_ce = self.ce(logits, target_indices)
        loss_credit_dice = self.credit_dice(logits, targets)
        return self.ce_w * loss_ce + self.credit_dice_w * loss_credit_dice


class MSECECostAwareDiceLoss(nn.Module):
    """
    MSE + DiceCELoss + cost-aware Dice loss for multi-class segmentation.
    """

    def __init__(self, cost_matrix, class_weight=None, mse_weight=1.0, dicece_weight=1.0, cadice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.dicece = DiceCELoss(
            softmax=True,
            reduction="mean",
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
            weight=class_weight,
        )
        self.cost_aware_dice = CostAwareDiceLoss(
            cost_matrix=cost_matrix,
            softmax=True,
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
        )
        self.mse_w = mse_weight
        self.dicece_w = dicece_weight
        self.cadice_w = cadice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        loss_mse = self.mse(probs, targets)
        loss_dicece = self.dicece(logits, targets)
        loss_cadice = self.cost_aware_dice(logits, targets)
        return self.mse_w * loss_mse + self.dicece_w * loss_dicece + self.cadice_w * loss_cadice


class BCEDiceWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss + MONAI DiceLoss (multi-label).

    Args:
        pos_weight: torch.Tensor or None, shape [C], per-class weight for BCE
        bce_weight: float, weight of BCE term
        dice_weight: float, weight of Dice term
        smooth: float, smoothing for DiceLoss
    """

    def __init__(self, pos_weight=5.0, neg_weight=1.0, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        # self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce = WeightedBCELoss(weight_pos=pos_weight, weight_neg=neg_weight)
        self.dice = DiceLoss(sigmoid=True, reduction="mean", squared_pred=True, smooth_nr=smooth, smooth_dr=smooth)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) raw logits
            targets: (N, C, H, W) binary masks {0,1}
        """
        loss_bce = self.bce(torch.sigmoid(logits), targets)
        loss_dice = self.dice(logits, targets)  # MONAI 内部会做 sigmoid
        return self.bce_w * loss_bce + self.dice_w * loss_dice


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss (with sigmoid probability input)

    pred:   (N, 1, H, W)   -> sigmoid 概率
    target: (N, 1, H, W)   -> 二值 mask
    weight_pos: 前景像素权重
    weight_neg: 背景像素权重
    """

    def __init__(self, weight_pos: float = 5.0, weight_neg: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        loss = -(
            self.weight_pos * target * torch.log(pred) +
            self.weight_neg * (1 - target) * torch.log(1 - pred)
        )

        return loss.mean()
