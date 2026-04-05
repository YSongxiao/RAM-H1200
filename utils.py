import random, os
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, HorizontalFlip, RandomScale, VerticalFlip, Rotate, Resize, ShiftScaleRotate, RandomBrightnessContrast, CenterCrop
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
import abc
from monai.metrics import SurfaceDiceMetric, DiceMetric


def compute_dsc_monai(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
):
    """
    pred_mask, gt_mask:
        binary torch.Tensor {0,1}
        allowed shapes:
            (H, W)
            (1, H, W)
            (1, 1, H, W)

    return:
        float Dice score (foreground only)
    """

    # ---------------------------------------------------------
    # 1. Normalize shape to (B, 1, H, W)
    # ---------------------------------------------------------
    if pred_mask.dim() == 2:          # (H, W)
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.dim() == 3:        # (1, H, W)
        pred_mask = pred_mask.unsqueeze(1)

    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    elif gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    assert pred_mask.dim() == 4, f"pred_mask shape invalid: {pred_mask.shape}"
    assert gt_mask.dim() == 4, f"gt_mask shape invalid: {gt_mask.shape}"

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    # ---------------------------------------------------------
    # 2. MONAI Dice (binary, foreground only)
    # ---------------------------------------------------------
    metric = DiceMetric(
        include_background=True,  # foreground only
        reduction="mean",
        get_not_nans=False,
    )

    metric(y_pred=pred_mask, y=gt_mask)

    dsc = metric.aggregate().item()
    metric.reset()

    return dsc


def compute_nsd_monai(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    tolerance=2.0,
):
    """
    pred_mask, gt_mask:
        binary torch.Tensor {0,1}
        allowed shapes:
            (H, W)
            (1, H, W)
            (1, 1, H, W)

    tolerance:
        surface tolerance in pixels

    return:
        float NSD (foreground only)
    """

    # ---------------------------------------------------------
    # 1. Normalize shape to (B, 1, H, W)
    # ---------------------------------------------------------
    if pred_mask.dim() == 2:          # (H, W)
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.dim() == 3:        # (1, H, W)
        pred_mask = pred_mask.unsqueeze(1)

    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    elif gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    assert pred_mask.dim() == 4, f"pred_mask shape invalid: {pred_mask.shape}"
    assert gt_mask.dim() == 4, f"gt_mask shape invalid: {gt_mask.shape}"

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    # ---------------------------------------------------------
    # 2. MONAI Surface Dice (binary, no one-hot)
    # ---------------------------------------------------------
    metric = SurfaceDiceMetric(
        class_thresholds=[tolerance],
        include_background=True,  # ❗ binary foreground only
        reduction="mean",
    )

    metric(y_pred=pred_mask, y=gt_mask)

    nsd = metric.aggregate().item()
    metric.reset()

    return nsd


# class PatchStitcher:
#     """
#     Manage patch-level predictions and stitch them into full-image outputs.
#
#     Features:
#         - prob_acc: sum of predicted probabilities
#         - count_acc: number of times each pixel is covered
#         - Supports overlapping patches by averaging
#     """
#
#     def __init__(self, num_classes: int, H: int, W: int, fname: str):
#         self.num_classes = num_classes
#         self.H = H
#         self.W = W
#         self.fname = fname
#
#         # (C,H,W)
#         self.prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
#         # (1,H,W) for coverage count
#         self.count_acc = np.zeros((1, H, W), dtype=np.float32)
#
#     def add_patch(self, prob_patch: torch.Tensor, y0: int, x0: int) -> None:
#         """
#         Add a predicted patch into the accumulators.
#
#         Args:
#             prob_patch: Tensor [C,ph,pw] of probabilities (after sigmoid/softmax)
#             y0, x0: top-left coordinates in the full image
#         """
#         ph, pw = prob_patch.shape[1], prob_patch.shape[2]
#         prob_np = prob_patch.detach().cpu().numpy()
#
#         self.prob_acc[:, y0:y0+ph, x0:x0+pw] += prob_np
#         self.count_acc[:, y0:y0+ph, x0:x0+pw] += 1
#
#     def fuse(self) -> np.ndarray:
#         """
#         Fuse patches into a final probability map.
#         Returns:
#             fused_prob: np.ndarray [C,H,W] in [0,1]
#         """
#         fused = self.prob_acc / np.maximum(self.count_acc, 1e-6)
#         return fused
#
#     def get_binary_mask(self, threshold: float = 0.5) -> np.ndarray:
#         """
#         Get hard binary segmentation mask.
#
#         Returns:
#             np.ndarray [C,H,W] in {0,1}
#         """
#         fused = self.fuse()
#         return (fused > threshold).astype(np.uint8)


# class PatchStitcher:
#     """
#     Manage patch-level predictions and stitch them into full-image outputs.
#
#     Supports both:
#         - test mode (no GT)
#         - validate/train mode (with GT)
#
#     If gt_patch is provided, GT accumulators are used.
#     Otherwise, GT-related fields are ignored.
#     """
#
#     def __init__(self, num_classes: int, H: int, W: int, fname: str):
#         self.num_classes = num_classes
#         self.H = H
#         self.W = W
#         self.fname = fname
#
#         # ---- Prediction accumulators ----
#         self.prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
#         self.count_acc = np.zeros((1, H, W), dtype=np.float32)
#
#         # ---- GT accumulators (only used if add_patch receives GT) ----
#         self.has_gt = False
#         self.gt_acc = None
#         self.gt_count = None
#
#     def add_patch(self, prob_patch: torch.Tensor, y0: int, x0: int,
#                   gt_patch: torch.Tensor = None) -> None:
#         """
#         Add predicted patch. GT is optional.
#
#         Args:
#             prob_patch: [C,ph,pw]
#             y0,x0     : top-left position
#             gt_patch  : [C,ph,pw] or None
#         """
#         prob_np = prob_patch.detach().cpu().numpy()
#         ph, pw = prob_np.shape[1], prob_np.shape[2]
#
#         # Prediction accumulate
#         self.prob_acc[:, y0:y0+ph, x0:x0+pw] += prob_np
#         self.count_acc[:, y0:y0+ph, x0:x0+pw] += 1
#
#         # Handle GT only in train/val
#         if gt_patch is not None:
#             if not self.has_gt:
#                 # Lazy initialization (only when GT is needed)
#                 self.gt_acc = np.zeros((self.num_classes, self.H, self.W), dtype=np.float32)
#                 self.gt_count = np.zeros((1, self.H, self.W), dtype=np.float32)
#                 self.has_gt = True
#
#             gt_np = gt_patch.detach().cpu().numpy()
#             self.gt_acc[:, y0:y0+ph, x0:x0+pw] += gt_np
#             self.gt_count[:, y0:y0+ph, x0:x0+pw] += 1
#
#     # --------- Prediction output ---------
#     def fuse(self) -> np.ndarray:
#         return self.prob_acc / np.maximum(self.count_acc, 1e-6)
#
#     def get_binary_pred(self, threshold: float = 0.5) -> np.ndarray:
#         return (self.fuse() > threshold).astype(np.uint8)
#
#     # --------- GT output (only if available) ---------
#     def fuse_gt(self) -> np.ndarray:
#         if not self.has_gt:
#             raise RuntimeError("GT was never provided for this image.")
#         return self.gt_acc / np.maximum(self.gt_count, 1e-6)
#
#     def get_binary_gt(self, threshold: float = 0.5) -> np.ndarray:
#         if not self.has_gt:
#             raise RuntimeError("GT was never provided for this image.")
#         return (self.fuse_gt() > threshold).astype(np.uint8)


class PatchStitcher:
    """
    Manage patch-level predictions and stitch them into full-image outputs,
    using CPU-only torch tensors to avoid numpy memory fragmentation.
    """

    def __init__(self, num_classes: int, H: int, W: int, fname: str):
        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.fname = fname

        # ---- Prediction accumulators (CPU tensors) ----
        self.prob_acc = torch.zeros((num_classes, H, W), dtype=torch.float32, device="cpu")
        self.count_acc = torch.zeros((1, H, W), dtype=torch.float32, device="cpu")

        # ---- GT accumulators (lazy init) ----
        self.has_gt = False
        self.gt_acc = None
        self.gt_count = None

        # ---- Bone accumulators (lazy init) ----
        self.has_bone = False
        self.bone_acc = None     # (Cb, H, W)
        self.bone_count = None   # (1, H, W)

    def add_patch(
        self,
        prob_patch: torch.Tensor,
        y0: int,
        x0: int,
        gt_patch: torch.Tensor = None,
        bone_patch: torch.Tensor = None,
    ) -> None:
        """
        Args:
            prob_patch: (C, ph, pw) sigmoid output (or probability)
            gt_patch:   (C, ph, pw) optional
            bone_patch: (ph, pw) or (Cb, ph, pw) optional bone semantic mask/prob
                        Usually Cb=1 (bone vs background). Can be soft (0~1).
        """

        # Move patch to CPU (does NOT increase GPU memory)
        prob_patch_cpu = prob_patch.detach().cpu()
        C, ph, pw = prob_patch_cpu.shape

        # Accumulate prediction
        self.prob_acc[:, y0:y0 + ph, x0:x0 + pw] += prob_patch_cpu
        self.count_acc[:, y0:y0 + ph, x0:x0 + pw] += 1

        # GT accumulation
        if gt_patch is not None:
            if not self.has_gt:
                self.gt_acc = torch.zeros(
                    (self.num_classes, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu",
                )
                self.gt_count = torch.zeros(
                    (1, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu",
                )
                self.has_gt = True

            gt_patch_cpu = gt_patch.detach().cpu()
            self.gt_acc[:, y0:y0 + ph, x0:x0 + pw] += gt_patch_cpu
            self.gt_count[:, y0:y0 + ph, x0:x0 + pw] += 1

        # Bone accumulation (fuse like GT)
        if bone_patch is not None:
            bone_patch_cpu = bone_patch.detach().cpu()
            if bone_patch_cpu.dim() == 2:
                bone_patch_cpu = bone_patch_cpu.unsqueeze(0)  # (1, ph, pw)

            Cb, bph, bpw = bone_patch_cpu.shape
            assert (bph, bpw) == (ph, pw), \
                f"bone_patch size mismatch: bone ({bph},{bpw}) vs prob ({ph},{pw})"

            if not self.has_bone:
                self.bone_acc = torch.zeros(
                    (Cb, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu",
                )
                self.bone_count = torch.zeros(
                    (1, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu",
                )
                self.has_bone = True
            else:
                assert self.bone_acc.shape[0] == Cb, \
                    f"bone channels mismatch: existing {self.bone_acc.shape[0]} vs new {Cb}"

            self.bone_acc[:, y0:y0 + ph, x0:x0 + pw] += bone_patch_cpu
            self.bone_count[:, y0:y0 + ph, x0:x0 + pw] += 1

    # --------- Prediction output ---------
    def fuse(self) -> torch.Tensor:
        return self.prob_acc / torch.clamp(self.count_acc, min=1e-6)

    def fuse_bone(self) -> torch.Tensor:
        if not self.has_bone:
            raise RuntimeError("Bone mask was never provided for this image.")
        return self.bone_acc / torch.clamp(self.bone_count, min=1e-6)

    def get_binary_bone(self, threshold: float = 0.5, channel: int = 0) -> torch.Tensor:
        """
        Returns:
            (1, H, W) uint8 bone mask
        """
        bone = self.fuse_bone()
        assert 0 <= channel < bone.shape[0], f"bone channel out of range: {channel}/{bone.shape[0]}"
        return (bone[channel:channel + 1] > threshold).to(torch.uint8)

    def get_binary_pred(
        self,
        threshold: float = 0.5,
        apply_bone_mask: bool = False,
        bone_threshold: float = 0.5,
        bone_channel: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            threshold: BE prediction binarization threshold
            apply_bone_mask: if True, enforce BE ⊂ Bone
            bone_threshold: threshold to binarize fused bone mask
            bone_channel: which channel to use if bone has multiple channels
        """
        pred_bin = (self.fuse() > threshold).to(torch.uint8)

        if apply_bone_mask:
            if not self.has_bone:
                raise RuntimeError("apply_bone_mask=True but bone_patch was never provided.")
            bone_bin = self.get_binary_bone(threshold=bone_threshold, channel=bone_channel)  # (1,H,W)
            pred_bin = pred_bin & bone_bin  # broadcast over C

        return pred_bin

    # --------- GT output (only if available) ---------
    def fuse_gt(self) -> torch.Tensor:
        if not self.has_gt:
            raise RuntimeError("GT was never provided for this image.")
        return self.gt_acc / torch.clamp(self.gt_count, min=1e-6)

    def get_binary_gt(
            self,
            threshold: float = 0.5,
            apply_bone_mask: bool = False,
            bone_threshold: float = 0.5,
            bone_channel: int = 0,
    ) -> torch.Tensor:
        gt_bin = (self.fuse_gt() > threshold).to(torch.uint8)

        if apply_bone_mask:
            if not self.has_bone:
                raise RuntimeError("apply_bone_mask=True but bone mask not provided.")
            bone_bin = self.get_binary_bone(
                threshold=bone_threshold,
                channel=bone_channel
            )
            gt_bin = gt_bin & bone_bin

        return gt_bin


class BEPatchStitcher:
    """
    Manage patch-level predictions and stitch them into full-image outputs,
    using CPU-only torch tensors to avoid numpy memory fragmentation.
    """

    def __init__(self, num_classes: int, H: int, W: int, fname: str):
        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.fname = fname

        # ---- Prediction accumulators (CPU tensors) ----
        self.prob_acc = torch.zeros((num_classes, H, W), dtype=torch.float32, device="cpu")
        self.count_acc = torch.zeros((1, H, W), dtype=torch.float32, device="cpu")

        # ---- Logits accumulators (optional, lazy init) ----
        self.has_logits = False
        self.logits_acc = None

        # ---- GT accumulators (lazy init) ----
        self.has_gt = False
        self.gt_acc = None
        self.gt_count = None

    def add_patch(
        self,
        prob_patch: torch.Tensor,
        y0: int,
        x0: int,
        gt_patch: torch.Tensor = None,
        logits_patch: torch.Tensor = None,
    ) -> None:
        """
        Args:
            prob_patch:   (C, ph, pw) sigmoid output
            logits_patch:(C, ph, pw) raw logits (optional, for temperature scaling)
        """

        # ---- prob accumulation ----
        prob_patch_cpu = prob_patch.detach().cpu()
        C, ph, pw = prob_patch_cpu.shape

        self.prob_acc[:, y0:y0+ph, x0:x0+pw] += prob_patch_cpu
        self.count_acc[:, y0:y0+ph, x0:x0+pw] += 1

        # ---- logits accumulation (only if provided) ----
        if logits_patch is not None:
            if not self.has_logits:
                self.logits_acc = torch.zeros(
                    (self.num_classes, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu"
                )
                self.has_logits = True

            logits_patch_cpu = logits_patch.detach().cpu()
            self.logits_acc[:, y0:y0+ph, x0:x0+pw] += logits_patch_cpu

        # ---- GT accumulation ----
        if gt_patch is not None:
            if not self.has_gt:
                self.gt_acc = torch.zeros(
                    (self.num_classes, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu"
                )
                self.gt_count = torch.zeros(
                    (1, self.H, self.W),
                    dtype=torch.float32,
                    device="cpu"
                )
                self.has_gt = True

            gt_patch_cpu = gt_patch.detach().cpu()
            self.gt_acc[:, y0:y0+ph, x0:x0+pw] += gt_patch_cpu
            self.gt_count[:, y0:y0+ph, x0:x0+pw] += 1

    # --------- Prediction output ---------
    def fuse(self) -> torch.Tensor:
        return self.prob_acc / torch.clamp(self.count_acc, min=1e-6)

    def get_binary_pred(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.fuse() > threshold).to(torch.uint8)

    # --------- Logits output (optional) ---------
    def fuse_logits(self) -> torch.Tensor:
        if not self.has_logits:
            raise RuntimeError("Logits were never provided for this image.")
        return self.logits_acc / torch.clamp(self.count_acc, min=1e-6)

    # --------- GT output (only if available) ---------
    def fuse_gt(self) -> torch.Tensor:
        if not self.has_gt:
            raise RuntimeError("GT was never provided for this image.")
        return self.gt_acc / torch.clamp(self.gt_count, min=1e-6)

    def get_binary_gt(self, threshold: float = 0.5) -> torch.Tensor:
        if not self.has_gt:
            raise RuntimeError("GT was never provided for this image.")
        return (self.fuse_gt() > threshold).to(torch.uint8)


def compute_cortex_distance_map(bone_mask: np.ndarray) -> np.ndarray:
    """
    计算骨皮质 distance map。
    输入:
        bone_mask: (H, W) 二值骨mask，骨区域=1，背景=0

    输出:
        dist_map: (H, W) float32，范围缩放在 [0,1]
    """

    # bone_mask 必须为 uint8
    mask = (bone_mask > 0).astype(np.uint8)

    # ---------------------------
    # 1. 计算骨内部距离 (distance to background)
    # ---------------------------
    # bone interior → 1, background → 0
    # 距离越大表示越靠骨中心
    dist_in = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)

    # ---------------------------
    # 2. 计算骨外部距离 (distance to bone boundary)
    # ---------------------------
    # background → 1, bone → 0
    dist_out = cv2.distanceTransform(1 - mask, distanceType=cv2.DIST_L2, maskSize=5)

    # ---------------------------
    # 3. 总距离 = 内部距离 + 外部距离
    # ---------------------------
    # 表达的是一个“关于骨皮质边界的连续距离”
    dist = dist_in + dist_out

    # ---------------------------
    # 4. 归一化到 [0,1] 范围
    # ---------------------------
    dist_norm = dist.astype(np.float32)
    if dist_norm.max() > 0:
        dist_norm /= dist_norm.max()

    return dist_norm


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def get_transform(split, image_size):
    if split == "train":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            RandomScale(scale_limit=(1.0, 1.1), p=0.5),
            CenterCrop(image_size, image_size),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ], additional_targets={'mask': 'gt', "overlap": "gt"})
    elif split == "val" or split == "test":
        transform = Compose([
            ToTensorV2()
        ], additional_targets={'mask': 'gt', "overlap": "gt"})
    else:
        raise NotImplementedError(f"{split} is not implemented.")
    return transform


def get_cls_transform(split, image_size):
    if split == "train":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            # RandomScale(scale_limit=(1.0, 1.1), p=0.5),
            # CenterCrop(image_size, image_size),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ])
    elif split == "val" or split == "test":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2()
        ])
    else:
        raise NotImplementedError(f"{split} is not implemented.")
    return transform


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
