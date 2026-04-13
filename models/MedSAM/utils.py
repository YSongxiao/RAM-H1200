import random, os
import numpy as np
import torch
import torch.nn as nn
# from albumentations import Compose, HorizontalFlip, RandomScale, VerticalFlip, Rotate, Resize, ShiftScaleRotate, RandomBrightnessContrast, CenterCrop
# from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
import abc

import torch
import torch.nn as nn
import math
from dataclasses import dataclass

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# def get_transform(split, image_size):
#     if split == "train":
#         transform = Compose([
#             Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
#             RandomScale(scale_limit=(1.0, 1.1), p=0.5),
#             CenterCrop(image_size, image_size),
#             ShiftScaleRotate(rotate_limit=10),
#             RandomBrightnessContrast(p=0.5),
#             ToTensorV2()
#         ], additional_targets={'mask': 'gt'})
#     elif split == "val" or split == "test":
#         transform = Compose([
#             Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
#             ToTensorV2()
#         ], additional_targets={'mask': 'gt'})
#     else:
#         raise NotImplementedError(f"{split} is not implemented.")
#     return transform


# def get_cls_transform(split, image_size):
#     if split == "train":
#         transform = Compose([
#             Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
#             # RandomScale(scale_limit=(1.0, 1.1), p=0.5),
#             # CenterCrop(image_size, image_size),
#             ShiftScaleRotate(rotate_limit=10),
#             RandomBrightnessContrast(p=0.5),
#             ToTensorV2()
#         ])
#     elif split == "val" or split == "test":
#         transform = Compose([
#             Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
#             ToTensorV2()
#         ])
#     else:
#         raise NotImplementedError(f"{split} is not implemented.")
#     return transform


class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super(NoiseInjection, self).__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x


class NoiseMultiplicativeInjection(nn.Module):
    def __init__(self, p: float = 0.05, alpha: float = 0.05, betta: float = 0.01):
        super(NoiseMultiplicativeInjection, self).__init__()
        self.p = p
        self.alpha = alpha
        self.betta = betta

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def get_m_noise(self, x):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * self.betta + 1
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            mask_m = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask_m = (mask_m < self.p).float() * 1
            x = x + x * mask_m * self.get_m_noise(x) + self.alpha * mask * self.get_noise(x)
            return x
        return x


class WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)



class FocalRegressionLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        Focal Loss for regression tasks.

        Args:
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalRegressionLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        error = torch.abs(pred - target)  # element-wise absolute error
        loss = error ** self.gamma        # apply focal weighting

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: same as input


class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Multiclass Focal Loss.

        Args:
            gamma (float): focusing parameter
            alpha (Tensor, optional): class-wise weight tensor [num_classes], e.g., tensor([1.0, 2.0, ...])
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape [num_classes] or None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] integer class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)        # [B, C]
        probs = torch.exp(log_probs)                    # [B, C]

        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()  # [B, C]
        pt = (probs * targets_onehot).sum(1)            # [B]
        log_pt = (log_probs * targets_onehot).sum(1)    # [B]

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)     # [B]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CDW_CELoss(nn.Module):
    def __init__(self, alpha=1.0, class_weights=None):
        """
        alpha: float, 控制距离惩罚的程度
        class_weights: Tensor of shape (C,), 每个类别的权重（可选）
        """
        super(CDW_CELoss, self).__init__()
        self.alpha = alpha
        self.class_weights = class_weights  # torch.tensor 类型或 None

    def forward(self, logits, target):
        """
        logits: Tensor of shape (B, C)
        target: Tensor of shape (B,) with class indices
        """
        B, C = logits.shape
        probs = F.softmax(logits, dim=1).clamp(min=1e-8, max=1.0 - 1e-8)  # 防止 log(0)
        target_onehot = F.one_hot(target, num_classes=C).float()

        # 构造 (B, C) 的 ground-truth class index 矩阵
        target_idx = target.view(-1, 1).expand(-1, C)  # shape: (B, C)
        class_idx = torch.arange(C, device=logits.device).view(1, -1).expand(B, -1)  # shape: (B, C)
        distance_weight = (class_idx - target_idx).abs().float() ** self.alpha  # shape: (B, C)

        # mask 真实类，防止其参与 loss
        mask_non_true = 1.0 - target_onehot  # (B, C), 真实类为0，非真实类为1

        # 如果给定 class_weights，则将其广播到 (B, C)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device).view(1, -1).expand(B, -1)
        else:
            class_weights = torch.ones_like(probs)

        # 计算 loss
        log_loss = -torch.log(1.0 - probs)
        final_loss = log_loss * distance_weight * mask_non_true * class_weights
        loss = final_loss.sum(dim=1).mean()  # 先对每个样本 sum，再对 batch 平均

        return loss


@dataclass
class AdaptiveLossBalancer:
    """
    Dynamically adjust (seg_w, adv_w, d_w) to stabilise GAN training.

    Attributes
    ----------
    beta : float
        EMA 衰减系数。β 越大，历史记忆越长。
    min_w, max_w : float
        权重裁剪范围，防止过大过小导致梯度爆炸 / 消失。
    target_ratio : float
        希望保持的 `seg_grad_norm / adv_grad_norm` 目标比例。
    """
    beta: float = 0.99
    min_w: float = 0.1
    max_w: float = 30.0
    target_ratio: float = 1.0  # 想要两支梯度幅度大致相等

    # 以下字段运行时自动维护
    ema_seg: float = 0.0      # ⟨‖∇seg_loss‖⟩_EMA
    ema_adv: float = 0.0      # ⟨‖∇adv_loss‖⟩_EMA
    ema_d:   float = 0.0      # ⟨d_loss⟩_EMA，用于判别器调度

    def update(
        self,
        seg_loss: torch.Tensor,
        adv_loss: torch.Tensor,
        d_loss:   torch.Tensor,
        anchor:   torch.Tensor,   # ← 新增：通常传 final_pred
    ) -> tuple[float, float, float]:

        # 1) 梯度 w.r.t. anchor (retain_graph 保留后向图，稍后还要 .backward())
        seg_grad = torch.autograd.grad(
            seg_loss, anchor, retain_graph=True, create_graph=False
        )[0]
        adv_grad = torch.autograd.grad(
            adv_loss, anchor, retain_graph=True, create_graph=False
        )[0]

        # 2) L2-norm（可换成 mean / max，按需）
        seg_gn = seg_grad.float().norm().item()
        adv_gn = adv_grad.float().norm().item()

        # 2) 更新 EMA
        self.ema_seg = self.beta * self.ema_seg + (1 - self.beta) * seg_gn
        self.ema_adv = self.beta * self.ema_adv + (1 - self.beta) * adv_gn
        self.ema_d   = self.beta * self.ema_d   + (1 - self.beta) * d_loss.item()

        # 3) 计算 Generator 内部两项的权重
        #    想让  seg_w * ema_seg ≈ adv_w * ema_adv * target_ratio
        #    固定 seg_w = 1，求 adv_w.
        if self.ema_adv > 0:
            adv_w = (self.ema_seg / self.ema_adv) * self.target_ratio
        else:
            adv_w = 1.0
        adv_w = float(min(max(adv_w, self.min_w), self.max_w))
        seg_w = 1.0

        # 4) 计算 Discriminator 整体权重（可用于 lr 或 loss 放大因子）
        #    判别器过强 → 减小 d_w；过弱 → 增大 d_w
        #    这里用对数函数确保变化缓和
        #        目标:  <d_loss> ≈ 0.5
        strength = math.exp(self.ema_d - 0.5)   # 0.5 左右最平衡
        d_w = float(min(max(strength, self.min_w), self.max_w))
        # print("[DEBUG]", seg_gn, adv_gn, self.ema_seg, self.ema_adv)
        return seg_w, adv_w, d_w


class LossBalancer:
    def __init__(self, beta=0.99, min_w=0.1, max_w=30.0):
        self.beta, self.min_w, self.max_w = beta, min_w, max_w
        self.es, self.ea = 0., 0.

    def update(self, seg_loss, adv_loss):
        self.es = self.beta * self.es + (1 - self.beta) * seg_loss.detach().item()
        self.ea = self.beta * self.ea + (1 - self.beta) * adv_loss.detach().item()
        ratio   = self.es / (self.ea + 1e-8)      # >1 → seg 比 adv 大
        adv_w   = float(np.clip(ratio, self.min_w, self.max_w))
        seg_w   = 1.0
        return seg_w, adv_w
