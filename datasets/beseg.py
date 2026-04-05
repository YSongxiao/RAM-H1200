from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import pandas as pd
import json
import cv2
import torch
import torch.distributed as dist
import random
from typing import List, Tuple, Optional, Dict, Any, Sequence
from utils import compute_cortex_distance_map


class BEPatchDataset(Dataset):
    """
        Patch dataset for full-hand X-ray segmentation (COCO annotations, JPG/BMP 8-bit).

        Modes:
            - "train": use COCO, random patches with foreground preference
            - "val"/"test": use COCO, sliding-window patches
            - "infer": no COCO, only images in data_root, sliding-window patches (mask=None)
        """

    def __init__(
            self,
            args=None,
            data_root: str | Path = "",
            annotation_path: Optional[str | Path] = None,
            patch_size: Tuple[int, int] = (384, 384),
            stride: Tuple[int, int] = (64, 64),
            transform: Optional[Any] = None,
            mode: str = "train",
            foreground_ratio: float = 0.7,
            max_tries: int = 20,
            use_coords: bool = False,
            flip_left_by_name: bool = False,
            normalize: str = "fixed",
            dataset_mean: Optional[float] = None,
            dataset_std: Optional[float] = None,
            train_patches_per_image: int = 24,
            center_region_half_size: int = 96,
            category_names: Optional[Sequence[str]] = None,
            add_background_channel: bool = False,
            background_name: str = "Background",
            expected_num_classes: Optional[int] = None,
    ) -> None:

        self.args = args
        self.data_root = Path(data_root)
        self.transform = transform
        self.mode = mode.lower()
        assert self.mode in ("train", "val", "test", "infer")
        self.patch_h, self.patch_w = patch_size
        self.stride_h, self.stride_w = stride
        self.foreground_ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
        self.max_tries = max_tries
        self.use_coords = use_coords
        self.flip_left_by_name = flip_left_by_name
        self.train_patches_per_image = max(1, int(train_patches_per_image))
        self.center_region_half_size = max(0, int(center_region_half_size))

        self.normalize = normalize.lower()
        assert self.normalize in ("fixed", "zscore", "minmax")
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.category_names = list(category_names) if category_names is not None else ["BE"]
        self.add_background_channel = add_background_channel
        self.background_name = background_name
        self.class_names = ([self.background_name] + self.category_names) if self.add_background_channel else list(self.category_names)
        self.channel_to_name = {i: name for i, name in enumerate(self.class_names)}
        self.name_to_channel = {name: i for i, name in self.channel_to_name.items()}

        self.filenames: List[str] = []
        self.masks: List[Optional[np.ndarray]] = []
        self.img_hw: List[Tuple[int, int]] = []

        if self.mode == "infer":
            exts = [".bmp", ".jpg", ".jpeg", ".png"]
            files = [f for f in sorted(self.data_root.iterdir()) if f.suffix.lower() in exts]

            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                H, W = img.shape
                self.filenames.append(f.name)
                self.masks.append(None)
                self.img_hw.append((H, W))

        else:
            coco = COCO(annotation_file=str(annotation_path))
            cat_name_to_id = {cat["name"]: cat["id"] for cat in coco.dataset.get("categories", [])}
            missing_cat_names = [name for name in self.category_names if name not in cat_name_to_id]
            if missing_cat_names:
                raise ValueError(f"Categories not found in COCO annotation file: {missing_cat_names}")
            self.cat_ids = [cat_name_to_id[name] for name in self.category_names]

            final_num_classes = len(self.class_names)
            if expected_num_classes is not None and final_num_classes != expected_num_classes:
                print(
                    f"[WARN] expected_num_classes={expected_num_classes}, "
                    f"but constructed classes={self.class_names} from lesion ids={self.cat_ids}."
                )

            for img_id in coco.getImgIds():
                info = coco.loadImgs([img_id])[0]
                H, W = int(info["height"]), int(info["width"])
                fname = info["file_name"]

                per_class_masks = []
                for cat_id in self.cat_ids:
                    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])

                    if len(ann_ids) == 0:
                        per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                        continue

                    anns = coco.loadAnns(ann_ids)
                    rles_all = []
                    for ann in anns:
                        seg = ann["segmentation"]
                        if isinstance(seg, list):
                            rles = maskUtils.frPyObjects(seg, H, W)
                            rles_all.extend(rles if isinstance(rles, list) else [rles])
                        else:
                            rles_all.append(seg)

                    merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                    m = maskUtils.decode(merged).astype(np.uint8)
                    per_class_masks.append(m)

                lesion_mask_stack = np.stack(per_class_masks, axis=0)
                be_mask_stack = self._make_exclusive_mask_stack(lesion_mask_stack)

                self.filenames.append(fname)
                self.masks.append(be_mask_stack)
                self.img_hw.append((H, W))

        self.index_map: List[Tuple[int, int, int, Tuple[int, int]]] = []
        self.grid_shapes: List[Tuple[int, int]] = []

        if self.mode in ("val", "test", "infer"):
            for img_idx, (H, W) in enumerate(self.img_hw):
                coords = self._build_grid(
                    H, W, self.patch_h, self.patch_w,
                    self.stride_h, self.stride_w
                )
                gy = len({y for y, _ in coords})
                gx = len({x for _, x in coords})
                self.grid_shapes.append((gy, gx))
                for (y0, x0) in coords:
                    self.index_map.append((img_idx, y0, x0, (gy, gx)))

    def _normalize_image(self, img_u8: np.ndarray) -> np.ndarray:
        img = img_u8.astype(np.float32)
        if self.normalize == "fixed":
            img = img / 255.0
        elif self.normalize == "zscore":
            img = (img - float(self.dataset_mean)) / float(self.dataset_std)
        elif self.normalize == "minmax":
            vmin = float(img.min())
            vmax = float(img.max())
            img = np.zeros_like(img, dtype=np.float32) if vmax <= vmin else (img - vmin) / (vmax - vmin)
        return img[..., np.newaxis]

    def update_sampling_ratio(self, epoch):
        if epoch < 20:
            self.foreground_ratio = 0.7
        elif epoch < 40:
            self.foreground_ratio = 0.55
        else:
            self.foreground_ratio = 0.4

    @staticmethod
    def _build_grid(H: int, W: int, ph: int, pw: int, sh: int, sw: int) -> List[Tuple[int, int]]:
        ys = list(range(0, max(1, H - ph + 1), sh))
        xs = list(range(0, max(1, W - pw + 1), sw))
        if len(ys) == 0:
            ys = [0]
        if len(xs) == 0:
            xs = [0]
        if ys[-1] != H - ph:
            ys.append(max(0, H - ph))
        if xs[-1] != W - pw:
            xs.append(max(0, W - pw))
        return [(y, x) for y in ys for x in xs]

    @staticmethod
    def _coord_channels(hw: Tuple[int, int]) -> np.ndarray:
        H, W = hw
        yy, xx = np.meshgrid(
            (np.arange(H, dtype=np.float32) + 0.5) / H,
            (np.arange(W, dtype=np.float32) + 0.5) / W,
            indexing="ij",
        )
        return np.stack([xx, yy], axis=-1)

    @staticmethod
    def _coord_channels_global(
            full_hw: Tuple[int, int],
            y0: int,
            x0: int,
            patch_hw: Tuple[int, int]
    ) -> np.ndarray:
        H_full, W_full = full_hw
        ph, pw = patch_hw

        yy_patch, xx_patch = np.meshgrid(
            np.arange(ph, dtype=np.float32),
            np.arange(pw, dtype=np.float32),
            indexing="ij",
        )

        yy_full = (yy_patch + y0 + 0.5) / float(H_full)
        xx_full = (xx_patch + x0 + 0.5) / float(W_full)

        return np.stack([xx_full, yy_full], axis=-1)

    @staticmethod
    def _global_coord_channels(
            full_hw: Tuple[int, int],
            y0: int,
            x0: int,
            patch_hw: Tuple[int, int]
    ) -> np.ndarray:
        return BEPatchDataset._coord_channels_global(
            full_hw=full_hw,
            y0=y0,
            x0=x0,
            patch_hw=patch_hw,
        )

    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        path = self.data_root / self.filenames[idx]
        img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            raise FileNotFoundError(f"Image not found: {path}")

        flip = self.flip_left_by_name and Path(self.filenames[idx]).stem.endswith("_L")
        if flip:
            img_u8 = cv2.flip(img_u8, 1)

        mask_stack = self.masks[idx]
        if mask_stack is not None and flip:
            mask_stack = mask_stack[..., ::-1].copy()

        img_hwc = self._normalize_image(img_u8)
        return img_hwc, mask_stack

    def _sample_train_patch_coords(self, mask_stack: np.ndarray, H: int, W: int) -> Tuple[int, int]:
        ph, pw = self.patch_h, self.patch_w
        grid = self._build_grid(H, W, ph, pw, self.stride_h, self.stride_w)

        lesion_stack = mask_stack[1:] if self.add_background_channel else mask_stack
        fg = (lesion_stack.sum(axis=0) > 0).astype(np.uint8)
        ys, xs = np.where(fg == 1)

        if len(ys) == 0:
            return random.choice(grid)

        if random.random() < self.foreground_ratio:
            idx = random.randint(0, len(ys) - 1)
            cy, cx = ys[idx], xs[idx]

            center_y = ph // 2
            center_x = pw // 2
            half_h = min(self.center_region_half_size, ph // 2)
            half_w = min(self.center_region_half_size, pw // 2)

            center_y_min = max(0, center_y - half_h)
            center_y_max = min(ph - 1, center_y + half_h)
            center_x_min = max(0, center_x - half_w)
            center_x_max = min(pw - 1, center_x + half_w)

            y0 = cy - random.randint(center_y_min, center_y_max)
            x0 = cx - random.randint(center_x_min, center_x_max)

            y0 = max(0, min(y0, H - ph))
            x0 = max(0, min(x0, W - pw))

            return y0, x0

        return random.choice(grid)

    def _make_exclusive_mask_stack(self, lesion_mask_stack: np.ndarray) -> np.ndarray:
        """
        Convert lesion channels into mutually exclusive labels using category_names order as priority.
        Earlier classes have higher priority on overlapping pixels.
        """
        lesion_mask_stack = (lesion_mask_stack > 0).astype(np.uint8)
        H, W = lesion_mask_stack.shape[1:]

        if lesion_mask_stack.shape[0] == 0:
            if self.add_background_channel:
                bg = np.ones((1, H, W), dtype=np.uint8)
                return bg
            return lesion_mask_stack

        label_map = np.zeros((H, W), dtype=np.uint8)
        for class_idx in reversed(range(lesion_mask_stack.shape[0])):
            label_map[lesion_mask_stack[class_idx] > 0] = class_idx + 1

        if self.add_background_channel:
            num_classes = lesion_mask_stack.shape[0] + 1
            one_hot = np.eye(num_classes, dtype=np.uint8)[label_map]
            return np.transpose(one_hot, (2, 0, 1))

        one_hot = np.eye(lesion_mask_stack.shape[0] + 1, dtype=np.uint8)[label_map]
        return np.transpose(one_hot[..., 1:], (2, 0, 1))

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.filenames) * self.train_patches_per_image
        else:
            return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode in ("val", "test", "infer"):
            img_idx = index

            if self.mode == "infer":
                img, mask_stack = self._load_image_and_mask(img_idx)
            else:
                img, mask_stack = self._load_image_and_mask(img_idx)

            if self.transform is not None:
                if mask_stack is not None:
                    m_hwc = np.transpose(mask_stack, (1, 2, 0))
                    out = self.transform(image=img, mask=m_hwc)
                    img = out["image"]
                    img = img.permute(1, 2, 0)
                    mask_stack = out["mask"]
                else:
                    out = self.transform(image=img)
                    img = out["image"]
                    img = img.permute(1, 2, 0)

            if self.use_coords:
                H, W = img.shape[:2]
                coords = self._coord_channels((H, W))
                img = np.concatenate([img, coords], axis=-1)

            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

            if mask_stack is not None:
                if isinstance(mask_stack, np.ndarray):
                    be_t = torch.from_numpy(np.ascontiguousarray(mask_stack)).permute(2, 0, 1).float()
                else:
                    be_t = mask_stack.permute(2, 0, 1).float()

                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "gt": be_t,
                    "img_idx": img_idx,
                }

            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "img_idx": img_idx,
            }

        elif self.mode == "train":
            n_imgs = len(self.filenames)
            img_idx = index % n_imgs

            img, be_stack = self._load_image_and_mask(img_idx)
            H, W = img.shape[:2]
            ph, pw = self.patch_h, self.patch_w

            y0, x0 = self._sample_train_patch_coords(be_stack, H, W)

            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            be_patch = be_stack[:, y0:y0 + ph, x0:x0 + pw]

            if self.transform is not None:
                m_hwc = np.transpose(be_patch, (1, 2, 0))
                out = self.transform(image=img_patch, mask=m_hwc)
                img_patch = out["image"]
                img_patch = img_patch.permute(1, 2, 0)
                be_patch = out["mask"]
            else:
                be_patch = np.transpose(be_patch, (1, 2, 0))

            if self.use_coords:
                coords = self._coord_channels_global(
                    full_hw=img.shape[:2],
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            be_t = torch.from_numpy(np.ascontiguousarray(be_patch)).permute(2, 0, 1).float()

            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "gt": be_t,
                "img_idx": img_idx,
                "y0": y0,
                "x0": x0,
            }

        else:
            raise RuntimeError(f"Unsupported mode: {self.mode}")


class NewBEBonePatchDataset(Dataset):
    """
        Patch dataset for full-hand X-ray segmentation (COCO annotations, JPG/BMP 8-bit).

        Modes:
            - "train": use COCO, random patches with foreground preference
            - "val"/"test": use COCO, sliding-window patches
            - "infer": no COCO, only images in data_root, sliding-window patches (mask=None)
        """

    def __init__(
            self,
            args,
            data_root: str | Path,
            be_annotation_path: Optional[str | Path] = None,
            bone_annotation_path: Optional[str | Path] = None,
            patch_size: Tuple[int, int] = (384, 384),
            stride: Tuple[int, int] = (64, 64),
            transform: Optional[Any] = None,
            mode: str = "train",
            foreground_ratio: float = 0.7,
            max_tries: int = 20,
            use_bone_mask: bool = True,
            use_coords: bool = False,
            flip_left_by_name: bool = False,
            normalize: str = "fixed",
            dataset_mean: Optional[float] = None,
            dataset_std: Optional[float] = None,
            train_patches_per_image: int = 24,
            expected_num_classes: Optional[int] = None,
    ) -> None:

        self.args = args
        self.data_root = Path(data_root)
        self.transform = transform
        self.mode = mode.lower()
        assert self.mode in ("train", "val", "test", "infer")
        self.patch_h, self.patch_w = patch_size
        self.stride_h, self.stride_w = stride
        self.foreground_ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
        self.max_tries = max_tries
        self.use_bone_mask = use_bone_mask
        self.use_coords = use_coords
        self.flip_left_by_name = flip_left_by_name
        self.train_patches_per_image = max(1, int(train_patches_per_image))

        self.normalize = normalize.lower()
        assert self.normalize in ("fixed", "zscore", "minmax")
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.filenames: List[str] = []
        self.masks: List[Optional[np.ndarray]] = []
        self.img_hw: List[Tuple[int, int]] = []
        self.bone_masks: List[np.ndarray] = []

        if self.mode == "infer":

            # === 加载图像 + bone COCO annotation（不加载 BE annotation） ===

            exts = [".bmp", ".jpg", ".jpeg", ".png"]
            files = [f for f in sorted(self.data_root.iterdir()) if f.suffix.lower() in exts]

            bone_valid_fnames = set()
            bone_masks_dict = {}

            if self.use_bone_mask or (self.mode == "infer" and self.args.post_proc):

                bone_coco = COCO(annotation_file=str(bone_annotation_path))

                all_cat_ids = bone_coco.getCatIds()
                all_cats = bone_coco.loadCats(all_cat_ids)

                skip_names = {
                    "Ring", "Metal Implant",
                    "bone-6U7D-RuyD", "bone-6U7D",
                    "Sesamoid", "SoftTissue"
                }

                bone_cats = [c for c in all_cats if c["name"] not in skip_names and c["id"] != 1]
                bone_cats_sorted = sorted(bone_cats, key=lambda x: x["id"])
                bone_cat_ids = [c["id"] for c in bone_cats_sorted]

                for img_id in bone_coco.getImgIds():

                    info = bone_coco.loadImgs([img_id])[0]
                    H, W = int(info["height"]), int(info["width"])
                    fname = info["file_name"]

                    per_class_masks = []

                    for cat_id in bone_cat_ids:

                        ann_ids = bone_coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])

                        if len(ann_ids) == 0:
                            per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                            continue

                        anns = bone_coco.loadAnns(ann_ids)

                        rles_all = []

                        for ann in anns:

                            seg = ann["segmentation"]

                            if isinstance(seg, list):
                                rles = maskUtils.frPyObjects(seg, H, W)
                                rles_all.extend(rles if isinstance(rles, list) else [rles])
                            else:
                                rles_all.append(seg)

                        merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                        m = maskUtils.decode(merged).astype(np.uint8)

                        per_class_masks.append(m)

                    bone_mask_stack = np.stack(per_class_masks, axis=0)

                    bone_valid_fnames.add(fname)
                    bone_masks_dict[fname] = bone_mask_stack

            # === 加载图像 ===
            for f in files:

                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                H, W = img.shape
                fname = f.name

                if fname not in bone_valid_fnames:
                    continue

                self.filenames.append(fname)
                self.masks.append(None)
                self.img_hw.append((H, W))

                if fname in bone_masks_dict:
                    self.bone_masks.append(bone_masks_dict[fname])

        else:
            # 先准备容器
            self.filenames = []
            self.masks = []
            self.img_hw = []
            self.bone_masks = []

            # ----------------------------------------------------------------------
            # ① 如果 use_bone_mask=True：先读取 bone mask，并记录哪些图片出现过
            # ----------------------------------------------------------------------
            bone_valid_fnames = set()
            bone_masks_dict = {}  # fname -> (mask_stack, H, W)

            if self.use_bone_mask:
                bone_coco = COCO(annotation_file=str(bone_annotation_path))
                all_cat_ids = bone_coco.getCatIds()
                all_cats = bone_coco.loadCats(all_cat_ids)

                skip_names = {"Ring", "Metal Implant", "bone-6U7D-RuyD",
                              "bone-6U7D", "Sesamoid", "SoftTissue"}

                bone_cats = [c for c in all_cats if c["name"] not in skip_names and c["id"] != 1]
                bone_cats_sorted = sorted(bone_cats, key=lambda x: x["id"])
                bone_cat_ids = [c["id"] for c in bone_cats_sorted]

                img_ids = bone_coco.getImgIds()
                for img_id in img_ids:
                    info = bone_coco.loadImgs([img_id])[0]
                    H, W = int(info["height"]), int(info["width"])

                    # ★ 统一出 fname 规则
                    fname = info["file_name"]
                    # fname = info["file_name"].split("_bmp")[0] + ".bmp"

                    # ---- 构建 per-class bone mask ----
                    per_class_masks = []
                    for cat_id in bone_cat_ids:
                        ann_ids = bone_coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                        if len(ann_ids) == 0:
                            per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                            continue
                        anns = bone_coco.loadAnns(ann_ids)

                        rles_all = []
                        for ann in anns:
                            seg = ann["segmentation"]
                            if isinstance(seg, list):
                                rles = maskUtils.frPyObjects(seg, H, W)
                                rles_all.extend(rles if isinstance(rles, list) else [rles])
                            else:
                                rles_all.append(seg)

                        merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                        m = maskUtils.decode(merged).astype(np.uint8)
                        per_class_masks.append(m)

                    bone_mask_stack = np.stack(per_class_masks, axis=0)

                    # 保存
                    bone_valid_fnames.add(fname)
                    bone_masks_dict[fname] = (bone_mask_stack, H, W)

            # ----------------------------------------------------------------------
            # ② 读取 BE mask（若 use_bone_mask=True，则必须判断是否在 bone_valid_fnames）
            # ----------------------------------------------------------------------
            coco = COCO(annotation_file=str(be_annotation_path))
            be_cat_ids = coco.getCatIds(catNms=["BE"])
            if not be_cat_ids:
                raise ValueError("Category 'BE' not found in COCO annotation file.")
            self.cat_ids = be_cat_ids

            img_ids = coco.getImgIds()
            for img_id in img_ids:
                info = coco.loadImgs([img_id])[0]
                H, W = int(info["height"]), int(info["width"])
                fname = info["file_name"]

                # ★ 若使用 bone mask，则 BE mask 的 fname 必须出现在 bone_valid_fnames 中
                if self.use_bone_mask and fname not in bone_valid_fnames:
                    continue

                # ---- 构建 per-class BE mask ----
                per_class_masks = []
                for cat_id in self.cat_ids:
                    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                    if len(ann_ids) == 0:
                        per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                        continue

                    anns = coco.loadAnns(ann_ids)
                    rles_all = []
                    for ann in anns:
                        seg = ann["segmentation"]
                        if isinstance(seg, list):
                            rles = maskUtils.frPyObjects(seg, H, W)
                            rles_all.extend(rles if isinstance(rles, list) else [rles])
                        else:
                            rles_all.append(seg)

                    merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                    m = maskUtils.decode(merged).astype(np.uint8)
                    per_class_masks.append(m)

                be_mask_stack = np.stack(per_class_masks, axis=0)

                # ---- 保存 ----
                self.filenames.append(fname)
                self.masks.append(be_mask_stack)
                self.img_hw.append((H, W))

                # 保存 bone mask（若开启）
                if self.use_bone_mask:
                    bone_mask_stack, _, _ = bone_masks_dict[fname]
                    self.bone_masks.append(bone_mask_stack)

        # ---- Precompute sliding indices for val/test/infer ----
        self.index_map: List[Tuple[int, int, int, Tuple[int, int]]] = []
        self.grid_shapes: List[Tuple[int, int]] = []

        if self.mode in ("val", "test", "infer"):
            for img_idx, (H, W) in enumerate(self.img_hw):
                coords = self._build_grid(H, W, self.patch_h, self.patch_w,
                                          self.stride_h, self.stride_w)
                gy = len({y for y, _ in coords})
                gx = len({x for _, x in coords})
                self.grid_shapes.append((gy, gx))
                for (y0, x0) in coords:
                    self.index_map.append((img_idx, y0, x0, (gy, gx)))

    # -------------------------- Normalization helpers -------------------------- #
    def _normalize_image(self, img_u8: np.ndarray) -> np.ndarray:
        """Normalize raw 8-bit grayscale image (H,W) to float HWC with 1 channel."""
        img = img_u8.astype(np.float32)
        if self.normalize == "fixed":
            img = img / 255.0
        elif self.normalize == "zscore":
            img = (img - float(self.dataset_mean)) / float(self.dataset_std)
        elif self.normalize == "minmax":
            vmin = float(img.min())
            vmax = float(img.max())
            img = np.zeros_like(img, dtype=np.float32) if vmax <= vmin else (img - vmin) / (vmax - vmin)
        return img[..., np.newaxis]  # (H,W,1)

    def update_sampling_ratio(self, epoch):
        if epoch < 20:
            self.foreground_ratio = 0.7
        elif epoch < 40:
            self.foreground_ratio = 0.5
        elif epoch < 60:
            self.foreground_ratio = 0.3
        else:
            self.foreground_ratio = 0.1

    # -------------------------- Grid & coord utils -------------------------- #
    @staticmethod
    def _build_grid(H: int, W: int, ph: int, pw: int, sh: int, sw: int) -> List[Tuple[int, int]]:
        """Generate top-left coordinates for a sliding window that fully covers the image."""
        ys = list(range(0, max(1, H - ph + 1), sh))
        xs = list(range(0, max(1, W - pw + 1), sw))
        if len(ys) == 0: ys = [0]
        if len(xs) == 0: xs = [0]
        if ys[-1] != H - ph: ys.append(max(0, H - ph))
        if xs[-1] != W - pw: xs.append(max(0, W - pw))
        return [(y, x) for y in ys for x in xs]

    @staticmethod
    def _coord_channels(hw: Tuple[int, int]) -> np.ndarray:
        """Return (H,W,2) array with channels [x_norm, y_norm] in [0,1]."""
        H, W = hw
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, H, dtype=np.float32),
            np.linspace(0.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        return np.stack([xx, yy], axis=-1)

    @staticmethod
    def _global_coord_channels(
            full_hw: Tuple[int, int],
            y0: int,
            x0: int,
            patch_hw: Tuple[int, int]
    ) -> np.ndarray:
        """
        Return (ph, pw, 2) array with absolute [x_norm, y_norm] in [0,1]
        representing the location of this patch inside the full image.

        Args:
            full_hw: (H_full, W_full)
            y0, x0: top-left corner of patch in the full image
            patch_hw: (ph, pw)
        """
        H_full, W_full = full_hw
        ph, pw = patch_hw

        # generate grid inside patch
        yy_patch, xx_patch = np.meshgrid(
            np.arange(ph, dtype=np.float32),
            np.arange(pw, dtype=np.float32),
            indexing="ij",
        )

        # convert to global coordinates
        yy_full = yy_patch + y0
        xx_full = xx_patch + x0

        # normalize to [0,1]
        y_norm = yy_full / (H_full - 1)
        x_norm = xx_full / (W_full - 1)

        return np.stack([x_norm, y_norm], axis=-1)

    # -------------------------- I/O -------------------------- #
    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load raw grayscale image (H,W) and per-class mask stack (K,H,W).
        Apply filename-based horizontal flip if requested.
        """
        path = self.data_root / self.filenames[idx]
        img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # NOTE: adjust this rule to your filename convention if needed
        flip = self.flip_left_by_name and (self.filenames[idx][-4] == "L")
        if flip:
            img_u8 = cv2.flip(img_u8, 1)

        mask_stack = self.masks[idx]
        if mask_stack is not None and flip:
            mask_stack = mask_stack[..., ::-1].copy()

        img_hwc = self._normalize_image(img_u8)  # (H,W,1) float32

        if self.use_bone_mask or (self.mode in ("test", "infer") and self.args.post_proc):
            bone_mask_stack = self.bone_masks[idx]
            if flip:
                bone_mask_stack = bone_mask_stack[..., ::-1].copy()
            return img_hwc, mask_stack, bone_mask_stack
        return img_hwc, mask_stack, None  # (H,W,1), (K,H,W) uint8

    def _sample_train_patch_coords(self, mask_stack: np.ndarray, H: int, W: int) -> Tuple[int, int]:
        """
        方案 D：从前景像素中直接采样，使前景位于 patch 中心。
        若没有前景，则 fallback 到 uniform 采样（减少背景干扰）。
        """
        ph, pw = self.patch_h, self.patch_w
        grid = self._build_grid(H, W, ph, pw, self.stride_h, self.stride_w)

        # foreground map（任何类别即可）
        fg = (mask_stack.sum(axis=0) > 0).astype(np.uint8)

        ys, xs = np.where(fg == 1)  # 所有前景坐标

        # 如果没有前景像素 → fallback uniform random
        if len(ys) == 0:
            return random.choice(grid)

        # 以 foreground_ratio 的概率采样前景中心
        if random.random() < self.foreground_ratio:
            # 1. 随机选一个前景坐标
            idx = random.randint(0, len(ys) - 1)
            cy, cx = ys[idx], xs[idx]

            # 2. 计算让 (cy,cx) 处于 patch 中心的左上角 y0,x0
            y0 = cy - ph // 2
            x0 = cx - pw // 2

            # 3. 保证 patch 在图像内部
            y0 = max(0, min(y0, H - ph))
            x0 = max(0, min(x0, W - pw))

            return (y0, x0)

        # 其余概率采样 uniform random patch
        return random.choice(grid)

    # -------------------------- Dataset API -------------------------- #
    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.filenames) * self.train_patches_per_image
        else:
            return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode == "infer":
            img_idx, y0, x0, grid_shape = self.index_map[index]
            img, _, bone_mask_stack = self._load_image_and_mask(img_idx)

            ph, pw = self.patch_h, self.patch_w
            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]

            bone_patch = None
            bone_mask_patch = None

            if self.use_bone_mask:
                bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]
                bone_patch = compute_cortex_distance_map(bone_mask_patch[0])[np.newaxis, ...]

                if self.transform is not None:
                    bone_list = [bone_patch[i] for i in range(bone_patch.shape[0])]
                    bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                    mask_list = bone_list + bone_mask_list

                    out = self.transform(image=img_patch, masks=mask_list)
                    img_patch = out["image"].permute(1, 2, 0)

                    m = out["masks"]
                    num_bone = bone_patch.shape[0]
                    num_bone_mask = bone_mask_patch.shape[0]

                    bone_patch = np.stack(m[:num_bone], axis=-1)
                    bone_mask_patch = np.stack(m[num_bone:num_bone + num_bone_mask], axis=-1)
                else:
                    bone_patch = np.transpose(bone_patch, (1, 2, 0))
                    bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))

            else:
                if self.args.post_proc:
                    bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                    bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]

                    if self.transform is not None:
                        bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                        out = self.transform(image=img_patch, masks=bone_mask_list)
                        img_patch = out["image"].permute(1, 2, 0)

                        m = out["masks"]
                        num_bone_mask = bone_mask_patch.shape[0]
                        bone_mask_patch = np.stack(m[:num_bone_mask], axis=-1)
                    else:
                        bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))
                else:
                    if self.transform is not None:
                        out = self.transform(image=img_patch)
                        img_patch = out["image"]
                        img_patch = img_patch.permute(1, 2, 0)

            if self.use_coords:
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],  # H_full, W_full
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]  # ph, pw
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            if isinstance(img_patch, np.ndarray):
                img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            else:
                img_t = img_patch.permute(2, 0, 1).float()

            if self.args.post_proc:
                if isinstance(bone_mask_patch, np.ndarray):
                    bm_t = torch.from_numpy(np.ascontiguousarray(bone_mask_patch)).permute(2, 0, 1).float()
                else:
                    bm_t = bone_mask_patch.permute(2, 0, 1).float()

            if self.use_bone_mask:
                if isinstance(bone_patch, np.ndarray):
                    bone_t = torch.from_numpy(np.ascontiguousarray(bone_patch)).permute(2, 0, 1).float()
                else:
                    bone_t = bone_patch.permute(2, 0, 1).float()
                img_t = torch.cat([img_t, bone_t], dim=0)

            # 判断是不是这个图像的最后一个 patch
            gy, gx = grid_shape
            total_patches = gy * gx
            # 这个图像的所有 index 范围
            base = sum(gy * gx for (gy, gx) in self.grid_shapes[:img_idx])
            rel_idx = index - base
            is_last = (rel_idx == total_patches - 1)

            if self.args.post_proc:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "bone_semantic_mask": bm_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }
            else:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }

        elif self.mode == "train":
            n_imgs = len(self.filenames)
            img_idx = index % n_imgs

            img, be_stack, bone_mask_stack = self._load_image_and_mask(img_idx)  # (H,W,1), (K,H,W)
            H, W = img.shape[:2]
            ph, pw = self.patch_h, self.patch_w

            # foreground-aware sampling
            y0, x0 = self._sample_train_patch_coords(be_stack, H, W)

            # crop patches
            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            be_patch = be_stack[:, y0:y0 + ph, x0:x0 + pw]
            if self.use_bone_mask:
                # 取出 bone patch (C, H, W)
                bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                bone_patch = np.any(bone_patch, axis=0).astype(np.uint8)
                bone_patch = compute_cortex_distance_map(bone_patch)[np.newaxis, ...]

                if self.transform is not None:

                    # --- Albumentations 要求 masks 是 list，每个 mask shape=(H,W) ---
                    be_list = [be_patch[i] for i in range(be_patch.shape[0])]  # C_be masks
                    bone_list = [bone_patch[i] for i in range(bone_patch.shape[0])]  # C_bone masks

                    mask_list = be_list + bone_list

                    # --- 执行变换 ---
                    out = self.transform(image=img_patch, masks=mask_list)

                    # unpack img
                    img_patch = out["image"].permute(1, 2, 0)

                    # --- 拆回 BE 和 BONE ---
                    m = out["masks"]
                    num_be = be_patch.shape[0]
                    num_bone = bone_patch.shape[0]

                    # stack back to original (H, W, C)
                    be_patch = np.stack(m[:num_be], axis=-1)
                    bone_patch = np.stack(m[num_be:num_be + num_bone], axis=-1)

                # no transform
                else:
                    be_patch = be_patch
                    bone_patch = bone_patch

            else:
                if self.transform is not None:
                    m_hwc = np.transpose(be_patch, (1, 2, 0))
                    out = self.transform(image=img_patch, mask=m_hwc)
                    img_patch = out["image"]  # CHW
                    img_patch = img_patch.permute(1, 2, 0)
                    be_patch = out["mask"]  # HWC
                else:
                    be_patch = np.transpose(be_patch, (1, 2, 0))  # HWC

            # === 在 transform 之后再拼接 coords ===
            if self.use_coords:
                # coords = self._coord_channels(img_patch.shape[:2])  # (H,W,2)
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],  # H_full, W_full
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]  # ph, pw
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            # 转 Tensor
            if isinstance(img_patch, np.ndarray):
                img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            else:
                img_t = img_patch.permute(2, 0, 1).float()

            if isinstance(be_patch, np.ndarray):
                be_t = torch.from_numpy(np.ascontiguousarray(be_patch)).permute(2, 0, 1).float()
            else:
                be_t = be_patch.permute(2, 0, 1).float()

            if self.use_bone_mask:
                if isinstance(bone_patch, np.ndarray):
                    bone_t = torch.from_numpy(np.ascontiguousarray(bone_patch)).permute(2, 0, 1).float()
                else:
                    bone_t = bone_patch.permute(2, 0, 1).float()
                img_t = torch.cat([img_t, bone_t], dim=0)  # new channels added

            # print("positive pixels:", mask_patch.sum())
            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "gt": be_t,
                "img_idx": img_idx,
                "y0": y0,
                "x0": x0,
            }

        # ======================================================================
        # VAL / TEST 模式（滑窗）
        # ======================================================================
        else:
            img_idx, y0, x0, grid_shape = self.index_map[index]

            img, be_stack, bone_mask_stack = self._load_image_and_mask(img_idx)
            ph, pw = self.patch_h, self.patch_w

            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            be_patch = be_stack[:, y0:y0 + ph, x0:x0 + pw]

            if self.use_bone_mask:
                bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]
                bone_patch = compute_cortex_distance_map(bone_mask_patch[0])[np.newaxis, ...]

                if self.transform is not None:
                    be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                    bone_list = [bone_patch[i] for i in range(bone_patch.shape[0])]
                    bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                    mask_list = be_list + bone_list + bone_mask_list  # all single-channel masks

                    out = self.transform(image=img_patch, masks=mask_list)

                    # unpack image
                    img_patch = out["image"].permute(1, 2, 0)

                    # unpack masks back to stacked form
                    m = out["masks"]
                    num_be = be_patch.shape[0]
                    num_bone = bone_patch.shape[0]
                    num_bone_mask = bone_mask_patch.shape[0]

                    be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                    bone_patch = np.stack(m[num_be:num_be + num_bone], axis=-1)  # (H,W,C)
                    bone_mask_patch = np.stack(m[num_be + num_bone:num_be + num_bone + num_bone_mask], axis=-1)  # (H,W,1)

                else:
                    be_patch = np.transpose(be_patch, (1, 2, 0))
                    bone_patch = np.transpose(bone_patch, (1, 2, 0))
                    bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))
            else:
                if self.args.post_proc or self.use_bone_mask:
                    bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                    bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]
                    if self.transform is not None:
                        be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                        bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                        mask_list = be_list + bone_mask_list  # all single-channel masks
                        # m_hwc = np.transpose(be_patch, (1, 2, 0))
                        out = self.transform(image=img_patch, mask=mask_list)
                        img_patch = out["image"]
                        img_patch = img_patch.permute(1, 2, 0)
                        # be_patch = out["mask"]
                        m = out["masks"]
                        num_be = be_patch.shape[0]
                        num_bone_mask = bone_mask_patch.shape[0]

                        be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                        bone_mask_patch = np.stack(m[num_be:num_be + num_bone_mask], axis=-1)  # (H,W,1)

                    else:
                        be_patch = np.transpose(be_patch, (1, 2, 0))
                        bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))
                else:
                    if self.transform is not None:
                        # be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                        # mask_list = be_list  # all single-channel masks
                        m_hwc = np.transpose(be_patch, (1, 2, 0))
                        out = self.transform(image=img_patch, mask=m_hwc)
                        img_patch = out["image"]
                        img_patch = img_patch.permute(1, 2, 0)
                        be_patch = out["mask"]
                        # m = out["masks"]
                        # num_be = be_patch.shape[0]

                        # be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                    else:
                        be_patch = np.transpose(be_patch, (1, 2, 0))

            # === transform 之后再拼 coords ===
            if self.use_coords:
                # coords = self._coord_channels(img_patch.shape[:2])
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],  # H_full, W_full
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]  # ph, pw
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            # 转 Tensor
            if isinstance(img_patch, np.ndarray):
                img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            else:
                img_t = img_patch.permute(2, 0, 1).float()

            if isinstance(be_patch, np.ndarray):
                be_t = torch.from_numpy(np.ascontiguousarray(be_patch)).permute(2, 0, 1).float()
            else:
                be_t = be_patch.permute(2, 0, 1).float()

            if self.args.post_proc:
                if isinstance(bone_mask_patch, np.ndarray):
                    bm_t = torch.from_numpy(np.ascontiguousarray(bone_mask_patch)).permute(2, 0, 1).float()
                else:
                    bm_t = bone_mask_patch.permute(2, 0, 1).float()

            if self.use_bone_mask:
                if isinstance(bone_patch, np.ndarray):
                    bone_t = torch.from_numpy(np.ascontiguousarray(bone_patch)).permute(2, 0, 1).float()
                else:
                    bone_t = bone_patch.permute(2, 0, 1).float()
                img_t = torch.cat([img_t, bone_t], dim=0)  # new channels added


            # 判断是不是这个图像的最后一个 patch
            gy, gx = grid_shape
            total_patches = gy * gx
            # 这个图像的所有 index 范围
            base = sum(gy * gx for (gy, gx) in self.grid_shapes[:img_idx])
            rel_idx = index - base
            is_last = (rel_idx == total_patches - 1)

            if self.args.post_proc:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "bone_semantic_mask": bm_t,
                    "gt": be_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }
            else:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "gt": be_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }


class ProBEBonePatchDataset(Dataset):
    """
        Patch dataset for full-hand X-ray segmentation (COCO annotations, JPG/BMP 8-bit).

        Modes:
            - "train": use COCO, random patches with foreground preference
            - "val"/"test": use COCO, sliding-window patches
            - "infer": no COCO, only images in data_root, sliding-window patches (mask=None)
        """

    def __init__(
            self,
            args,
            data_root: str | Path,
            be_annotation_path: Optional[str | Path] = None,
            bone_annotation_path: Optional[str | Path] = None,
            patch_size: Tuple[int, int] = (384, 384),
            stride: Tuple[int, int] = (64, 64),
            transform: Optional[Any] = None,
            mode: str = "train",
            foreground_ratio: float = 0.5,
            max_tries: int = 20,
            use_bone_mask: bool = True,
            use_coords: bool = False,
            flip_left_by_name: bool = False,
            normalize: str = "fixed",
            dataset_mean: Optional[float] = None,
            dataset_std: Optional[float] = None,
            train_patches_per_image: int = 24,
            CENTER_RADIUS=16,
            CORTEX_D=6,
            expected_num_classes: Optional[int] = None,
    ) -> None:

        self.args = args
        self.data_root = Path(data_root)
        self.transform = transform
        self.mode = mode.lower()
        assert self.mode in ("train", "val", "test", "infer")
        self.patch_h, self.patch_w = patch_size
        self.stride_h, self.stride_w = stride
        self.foreground_ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
        self.max_tries = max_tries
        self.use_bone_mask = use_bone_mask
        self.use_coords = use_coords
        self.flip_left_by_name = flip_left_by_name
        self.train_patches_per_image = max(1, int(train_patches_per_image))

        self.normalize = normalize.lower()
        assert self.normalize in ("fixed", "zscore", "minmax")
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.filenames: List[str] = []
        self.masks: List[Optional[np.ndarray]] = []
        self.img_hw: List[Tuple[int, int]] = []
        self.CENTER_RADIUS = CENTER_RADIUS
        self.CORTEX_D = CORTEX_D

        if self.mode == "infer":
            # === 只加载图像文件，不加载 annotation ===
            exts = [".bmp", ".jpg", ".jpeg", ".png"]
            files = [f for f in sorted(self.data_root.iterdir()) if f.suffix.lower() in exts]
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                H, W = img.shape
                self.filenames.append(f.name)
                self.img_hw.append((H, W))

        else:
            # 先准备容器
            self.filenames = []
            self.masks = []
            self.img_hw = []
            self.bone_masks = []

            # ----------------------------------------------------------------------
            # ① 如果 use_bone_mask=True：先读取 bone mask，并记录哪些图片出现过
            # ----------------------------------------------------------------------
            bone_valid_fnames = set()
            bone_masks_dict = {}  # fname -> (mask_stack, H, W)

            if self.use_bone_mask:
                bone_coco = COCO(annotation_file=str(bone_annotation_path))
                all_cat_ids = bone_coco.getCatIds()
                all_cats = bone_coco.loadCats(all_cat_ids)

                skip_names = {"Ring", "Metal Implant", "bone-6U7D-RuyD",
                              "bone-6U7D", "Sesamoid", "SoftTissue"}

                bone_cats = [c for c in all_cats if c["name"] not in skip_names and c["id"] != 1]
                bone_cats_sorted = sorted(bone_cats, key=lambda x: x["id"])
                bone_cat_ids = [c["id"] for c in bone_cats_sorted]

                img_ids = bone_coco.getImgIds()
                for img_id in img_ids:
                    info = bone_coco.loadImgs([img_id])[0]
                    H, W = int(info["height"]), int(info["width"])

                    # ★ 统一出 fname 规则
                    fname = info["file_name"]
                    # fname = info["file_name"].split("_bmp")[0] + ".bmp"

                    # ---- 构建 per-class bone mask ----
                    per_class_masks = []
                    for cat_id in bone_cat_ids:
                        ann_ids = bone_coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                        if len(ann_ids) == 0:
                            per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                            continue
                        anns = bone_coco.loadAnns(ann_ids)

                        rles_all = []
                        for ann in anns:
                            seg = ann["segmentation"]
                            if isinstance(seg, list):
                                rles = maskUtils.frPyObjects(seg, H, W)
                                rles_all.extend(rles if isinstance(rles, list) else [rles])
                            else:
                                rles_all.append(seg)

                        merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                        m = maskUtils.decode(merged).astype(np.uint8)
                        per_class_masks.append(m)

                    bone_mask_stack = np.stack(per_class_masks, axis=0)

                    # 保存
                    bone_valid_fnames.add(fname)
                    bone_masks_dict[fname] = (bone_mask_stack, H, W)

            # ----------------------------------------------------------------------
            # ② 读取 BE mask（若 use_bone_mask=True，则必须判断是否在 bone_valid_fnames）
            # ----------------------------------------------------------------------
            coco = COCO(annotation_file=str(be_annotation_path))
            be_cat_ids = coco.getCatIds(catNms=["BE"])
            if not be_cat_ids:
                raise ValueError("Category 'BE' not found in COCO annotation file.")
            self.cat_ids = be_cat_ids

            img_ids = coco.getImgIds()
            for img_id in img_ids:
                info = coco.loadImgs([img_id])[0]
                H, W = int(info["height"]), int(info["width"])
                fname = info["file_name"]

                # ★ 若使用 bone mask，则 BE mask 的 fname 必须出现在 bone_valid_fnames 中
                if self.use_bone_mask and fname not in bone_valid_fnames:
                    continue

                # ---- 构建 per-class BE mask ----
                per_class_masks = []
                for cat_id in self.cat_ids:
                    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                    if len(ann_ids) == 0:
                        per_class_masks.append(np.zeros((H, W), dtype=np.uint8))
                        continue

                    anns = coco.loadAnns(ann_ids)
                    rles_all = []
                    for ann in anns:
                        seg = ann["segmentation"]
                        if isinstance(seg, list):
                            rles = maskUtils.frPyObjects(seg, H, W)
                            rles_all.extend(rles if isinstance(rles, list) else [rles])
                        else:
                            rles_all.append(seg)

                    merged = rles_all[0] if len(rles_all) == 1 else maskUtils.merge(rles_all)
                    m = maskUtils.decode(merged).astype(np.uint8)
                    per_class_masks.append(m)

                be_mask_stack = np.stack(per_class_masks, axis=0)

                # ---- 保存 ----
                self.filenames.append(fname)
                self.masks.append(be_mask_stack)
                self.img_hw.append((H, W))

                # 保存 bone mask（若开启）
                if self.use_bone_mask:
                    bone_mask_stack, _, _ = bone_masks_dict[fname]
                    self.bone_masks.append(bone_mask_stack)

        # ---- Precompute sliding indices for val/test/infer ----
        self.index_map: List[Tuple[int, int, int, Tuple[int, int]]] = []
        self.grid_shapes: List[Tuple[int, int]] = []

        if self.mode in ("val", "test", "infer"):
            for img_idx, (H, W) in enumerate(self.img_hw):
                coords = self._build_grid(H, W, self.patch_h, self.patch_w,
                                          self.stride_h, self.stride_w)
                gy = len({y for y, _ in coords})
                gx = len({x for _, x in coords})
                self.grid_shapes.append((gy, gx))
                for (y0, x0) in coords:
                    self.index_map.append((img_idx, y0, x0, (gy, gx)))

    # -------------------------- Normalization helpers -------------------------- #
    def _normalize_image(self, img_u8: np.ndarray) -> np.ndarray:
        """Normalize raw 8-bit grayscale image (H,W) to float HWC with 1 channel."""
        img = img_u8.astype(np.float32)
        if self.normalize == "fixed":
            img = img / 255.0
        elif self.normalize == "zscore":
            img = (img - float(self.dataset_mean)) / float(self.dataset_std)
        elif self.normalize == "minmax":
            vmin = float(img.min())
            vmax = float(img.max())
            img = np.zeros_like(img, dtype=np.float32) if vmax <= vmin else (img - vmin) / (vmax - vmin)
        return img[..., np.newaxis]  # (H,W,1)

    def update_sampling_ratio(self, epoch):
        if epoch < 20:
            self.foreground_ratio = 0.7
        elif epoch < 40:
            self.foreground_ratio = 0.5
        elif epoch < 60:
            self.foreground_ratio = 0.3
        else:
            self.foreground_ratio = 0.1

    # -------------------------- Grid & coord utils -------------------------- #
    @staticmethod
    def _build_grid(H: int, W: int, ph: int, pw: int, sh: int, sw: int) -> List[Tuple[int, int]]:
        """Generate top-left coordinates for a sliding window that fully covers the image."""
        ys = list(range(0, max(1, H - ph + 1), sh))
        xs = list(range(0, max(1, W - pw + 1), sw))
        if len(ys) == 0: ys = [0]
        if len(xs) == 0: xs = [0]
        if ys[-1] != H - ph: ys.append(max(0, H - ph))
        if xs[-1] != W - pw: xs.append(max(0, W - pw))
        return [(y, x) for y in ys for x in xs]

    @staticmethod
    def _coord_channels(hw: Tuple[int, int]) -> np.ndarray:
        """Return (H,W,2) array with channels [x_norm, y_norm] in [0,1]."""
        H, W = hw
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, H, dtype=np.float32),
            np.linspace(0.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        return np.stack([xx, yy], axis=-1)

    @staticmethod
    def _global_coord_channels(
            full_hw: Tuple[int, int],
            y0: int,
            x0: int,
            patch_hw: Tuple[int, int]
    ) -> np.ndarray:
        """
        Return (ph, pw, 2) array with absolute [x_norm, y_norm] in [0,1]
        representing the location of this patch inside the full image.

        Args:
            full_hw: (H_full, W_full)
            y0, x0: top-left corner of patch in the full image
            patch_hw: (ph, pw)
        """
        H_full, W_full = full_hw
        ph, pw = patch_hw

        # generate grid inside patch
        yy_patch, xx_patch = np.meshgrid(
            np.arange(ph, dtype=np.float32),
            np.arange(pw, dtype=np.float32),
            indexing="ij",
        )

        # convert to global coordinates
        yy_full = yy_patch + y0
        xx_full = xx_patch + x0

        # normalize to [0,1]
        y_norm = yy_full / (H_full - 1)
        x_norm = xx_full / (W_full - 1)

        return np.stack([x_norm, y_norm], axis=-1)

    # -------------------------- I/O -------------------------- #
    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load raw grayscale image (H,W) and per-class mask stack (K,H,W).
        Apply filename-based horizontal flip if requested.
        """
        path = self.data_root / self.filenames[idx]
        img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # NOTE: adjust this rule to your filename convention if needed
        flip = self.flip_left_by_name and (self.filenames[idx][-4] == "L")
        if flip:
            img_u8 = cv2.flip(img_u8, 1)

        mask_stack = self.masks[idx]
        if flip:
            mask_stack = mask_stack[..., ::-1].copy()

        img_hwc = self._normalize_image(img_u8)  # (H,W,1) float32

        if self.use_bone_mask or (self.mode == "test" and self.args.post_proc):
            bone_mask_stack = self.bone_masks[idx]
            if flip:
                bone_mask_stack = bone_mask_stack[..., ::-1].copy()
            return img_hwc, mask_stack, bone_mask_stack
        return img_hwc, mask_stack, None  # (H,W,1), (K,H,W) uint8

    def _sample_train_patch_coords(
            self,
            be_stack: np.ndarray,
            bone_mask_stack: np.ndarray,
            H: int,
            W: int,
    ) -> Tuple[int, int]:
        """
        Sampling strategy:
          - Positive: BE exists in center window
          - Negative:
              50% hard negative (cortex_dist < d, no BE in center)
              50% random negative (no BE in center)
        """
        ph, pw = self.patch_h, self.patch_w
        grid = self._build_grid(H, W, ph, pw, self.stride_h, self.stride_w)

        # --------- prepare maps ---------
        be_map = (be_stack.sum(axis=0) > 0).astype(np.uint8)  # (H,W)
        bone_binary = np.any(bone_mask_stack, axis=0).astype(np.uint8)
        cortex_dist = compute_cortex_distance_map(bone_binary)  # (H,W)

        # --------- helper: center BE check ---------
        def center_has_be(y0, x0):
            cy = y0 + ph // 2
            cx = x0 + pw // 2
            y1 = max(0, cy - self.CENTER_RADIUS)
            y2 = min(H, cy + self.CENTER_RADIUS)
            x1 = max(0, cx - self.CENTER_RADIUS)
            x2 = min(W, cx + self.CENTER_RADIUS)
            return be_map[y1:y2, x1:x2].any()

        # --------- split candidates ---------
        pos_candidates = []
        hard_neg_candidates = []
        rand_neg_candidates = []

        for (y0, x0) in grid:
            has_be = center_has_be(y0, x0)

            if has_be:
                pos_candidates.append((y0, x0))
            else:
                # center position
                cy = y0 + ph // 2
                cx = x0 + pw // 2

                # hard negative: near cortex
                if cortex_dist[cy, cx] <= self.CORTEX_D:
                    hard_neg_candidates.append((y0, x0))
                else:
                    rand_neg_candidates.append((y0, x0))

        # --------- sampling logic ---------
        # positive
        if len(pos_candidates) > 0 and random.random() < self.foreground_ratio:
            return random.choice(pos_candidates)

        # negative
        # 1:1 hard neg : random neg
        if len(hard_neg_candidates) > 0 and len(rand_neg_candidates) > 0:
            if random.random() < 0.5:
                return random.choice(hard_neg_candidates)
            else:
                return random.choice(rand_neg_candidates)

        # fallback
        if len(hard_neg_candidates) > 0:
            return random.choice(hard_neg_candidates)
        if len(rand_neg_candidates) > 0:
            return random.choice(rand_neg_candidates)

        # ultimate fallback
        return random.choice(grid)

    # -------------------------- Dataset API -------------------------- #
    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.filenames) * self.train_patches_per_image
        else:
            return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode == "infer":
            img_idx, y0, x0, grid_shape = self.index_map[index]
            path = self.data_root / self.filenames[img_idx]
            img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = self._normalize_image(img_u8)

            ph, pw = self.patch_h, self.patch_w
            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]

            if self.transform is not None:
                out = self.transform(image=img_patch)
                img_patch = out["image"]
                img_patch = img_patch.permute(1, 2, 0)
            if self.use_coords:
                # coords = self._coord_channels(img_patch.shape[:2])
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],  # H_full, W_full
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]  # ph, pw
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            # 判断是不是这个图像的最后一个 patch
            gy, gx = grid_shape
            total_patches = gy * gx
            # 这个图像的所有 index 范围
            base = sum(gy * gx for (gy, gx) in self.grid_shapes[:img_idx])
            rel_idx = index - base
            is_last = (rel_idx == total_patches - 1)
            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "img_idx": img_idx,
                "y0": y0,
                "x0": x0,
                "grid_shape": grid_shape,
                "is_last": is_last,
            }

        elif self.mode == "train":
            n_imgs = len(self.filenames)
            img_idx = index % n_imgs
            # ---- load ----
            img, be_stack, bone_mask_stack = self._load_image_and_mask(img_idx)
            H, W = img.shape[:2]
            ph, pw = self.patch_h, self.patch_w
            y0, x0 = self._sample_train_patch_coords(be_stack, bone_mask_stack, H, W)

            # ---- crop image & BE ----

            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            be_patch = be_stack[:, y0:y0 + ph, x0:x0 + pw]

            # ==========================================================
            # ★ 新逻辑：始终计算 cortex distance map（与 use_bone_mask 无关）
            # ==========================================================
            bone_patch_raw = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
            bone_binary = np.any(bone_patch_raw, axis=0).astype(np.uint8)
            cortex_patch = compute_cortex_distance_map(bone_binary)[np.newaxis, ...]  # (1,H,W)

            # ---- Albumentations ----

            if self.transform is not None:
                # Albumentations 要求 masks 是 list[(H,W)]
                be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                cortex_list = [cortex_patch[0]]
                mask_list = be_list + cortex_list
                out = self.transform(image=img_patch, masks=mask_list)
                # unpack image
                img_patch = out["image"].permute(1, 2, 0)
                # unpack masks
                m = out["masks"]
                num_be = be_patch.shape[0]
                be_patch = np.stack(m[:num_be], axis=-1)
                cortex_patch = np.stack(m[num_be:num_be + 1], axis=-1)
            else:
                # no transform
                be_patch = np.transpose(be_patch, (1, 2, 0))
                cortex_patch = np.transpose(cortex_patch, (1, 2, 0))

            # ---- coords（不改）----

            if self.use_coords:
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            # ---- to tensor ----
            img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            be_t = torch.from_numpy(np.ascontiguousarray(be_patch)).permute(2, 0, 1).float()
            cortex_t = torch.from_numpy(np.ascontiguousarray(cortex_patch)).permute(2, 0, 1).float()

            if self.use_bone_mask:
                img_t = torch.cat([img_t, cortex_t], dim=0)

            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "gt": be_t,
                "cortex": cortex_t,  # ★ 始终提供（train 专用）
                "img_idx": img_idx,
                "y0": y0,
                "x0": x0,
            }

        # ======================================================================
        # VAL / TEST 模式（滑窗）
        # ======================================================================
        else:
            img_idx, y0, x0, grid_shape = self.index_map[index]

            img, be_stack, bone_mask_stack = self._load_image_and_mask(img_idx)
            ph, pw = self.patch_h, self.patch_w

            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            be_patch = be_stack[:, y0:y0 + ph, x0:x0 + pw]

            if self.use_bone_mask:
                bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]
                bone_patch = compute_cortex_distance_map(bone_mask_patch[0])[np.newaxis, ...]

                if self.transform is not None:
                    be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                    bone_list = [bone_patch[i] for i in range(bone_patch.shape[0])]
                    bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                    mask_list = be_list + bone_list + bone_mask_list  # all single-channel masks

                    out = self.transform(image=img_patch, masks=mask_list)

                    # unpack image
                    img_patch = out["image"].permute(1, 2, 0)

                    # unpack masks back to stacked form
                    m = out["masks"]
                    num_be = be_patch.shape[0]
                    num_bone = bone_patch.shape[0]
                    num_bone_mask = bone_mask_patch.shape[0]

                    be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                    bone_patch = np.stack(m[num_be:num_be + num_bone], axis=-1)  # (H,W,C)
                    bone_mask_patch = np.stack(m[num_be + num_bone:num_be + num_bone + num_bone_mask], axis=-1)  # (H,W,1)

                else:
                    be_patch = np.transpose(be_patch, (1, 2, 0))
                    bone_patch = np.transpose(bone_patch, (1, 2, 0))
                    bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))
            else:
                if self.args.post_proc or self.use_bone_mask:
                    bone_patch = bone_mask_stack[:, y0:y0 + ph, x0:x0 + pw]
                    bone_mask_patch = np.any(bone_patch, axis=0).astype(np.uint8)[np.newaxis, ...]
                    if self.transform is not None:
                        be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                        bone_mask_list = [bone_mask_patch[i] for i in range(bone_mask_patch.shape[0])]
                        mask_list = be_list + bone_mask_list  # all single-channel masks
                        # m_hwc = np.transpose(be_patch, (1, 2, 0))
                        out = self.transform(image=img_patch, mask=mask_list)
                        img_patch = out["image"]
                        img_patch = img_patch.permute(1, 2, 0)
                        # be_patch = out["mask"]
                        m = out["masks"]
                        num_be = be_patch.shape[0]
                        num_bone_mask = bone_mask_patch.shape[0]

                        be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                        bone_mask_patch = np.stack(m[num_be:num_be + num_bone_mask], axis=-1)  # (H,W,1)

                    else:
                        be_patch = np.transpose(be_patch, (1, 2, 0))
                        bone_mask_patch = np.transpose(bone_mask_patch, (1, 2, 0))
                else:
                    if self.transform is not None:
                        # be_list = [be_patch[i] for i in range(be_patch.shape[0])]
                        # mask_list = be_list  # all single-channel masks
                        m_hwc = np.transpose(be_patch, (1, 2, 0))
                        out = self.transform(image=img_patch, mask=m_hwc)
                        img_patch = out["image"]
                        img_patch = img_patch.permute(1, 2, 0)
                        be_patch = out["mask"]
                        # m = out["masks"]
                        # num_be = be_patch.shape[0]

                        # be_patch = np.stack(m[:num_be], axis=-1)  # (H,W,1)
                    else:
                        be_patch = np.transpose(be_patch, (1, 2, 0))

            # === transform 之后再拼 coords ===
            if self.use_coords:
                # coords = self._coord_channels(img_patch.shape[:2])
                coords = self._global_coord_channels(
                    full_hw=img.shape[:2],  # H_full, W_full
                    y0=y0,
                    x0=x0,
                    patch_hw=img_patch.shape[:2]  # ph, pw
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            # 转 Tensor
            if isinstance(img_patch, np.ndarray):
                img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            else:
                img_t = img_patch.permute(2, 0, 1).float()

            if isinstance(be_patch, np.ndarray):
                be_t = torch.from_numpy(np.ascontiguousarray(be_patch)).permute(2, 0, 1).float()
            else:
                be_t = be_patch.permute(2, 0, 1).float()

            if self.args.post_proc:
                if isinstance(bone_mask_patch, np.ndarray):
                    bm_t = torch.from_numpy(np.ascontiguousarray(bone_mask_patch)).permute(2, 0, 1).float()
                else:
                    bm_t = be_patch.permute(2, 0, 1).float()

            if self.use_bone_mask:
                if isinstance(bone_patch, np.ndarray):
                    bone_t = torch.from_numpy(np.ascontiguousarray(bone_patch)).permute(2, 0, 1).float()
                else:
                    bone_t = bone_patch.permute(2, 0, 1).float()
                img_t = torch.cat([img_t, bone_t], dim=0)  # new channels added


            # 判断是不是这个图像的最后一个 patch
            gy, gx = grid_shape
            total_patches = gy * gx
            # 这个图像的所有 index 范围
            base = sum(gy * gx for (gy, gx) in self.grid_shapes[:img_idx])
            rel_idx = index - base
            is_last = (rel_idx == total_patches - 1)

            if self.args.post_proc:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "bone_semantic_mask": bm_t,
                    "gt": be_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }
            else:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "gt": be_t,
                    "img_idx": img_idx,
                    "y0": y0,
                    "x0": x0,
                    "grid_shape": grid_shape,
                    "is_last": is_last,
                }


def get_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    distributed: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    if distributed is None:
        distributed = dist.is_available() and dist.is_initialized()

    is_train = getattr(dataset, "mode", None) == "train"
    drop_last = is_train

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory and torch.cuda.is_available()),
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    return DataLoader(**loader_kwargs)
