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


bone_name_dict = {0: "Capitate", 1: "Radius", 2: "Ulna", 3: "Hamate", 4: "Lunate", 5: "Pisifrom_Triquetrum",
                  6: "Scaphoid", 7: "Trapzium", 8: "Trapzoid", 9: "MC1", 10: "MC2",
                  11: "MC3", 12: "MC4", 13: "MC5"}

overlap_pairs = [(1, 6), (1, 4), (6, 7), (0, 6), (7, 9), (0, 11), (3, 12), (4, 6), (7, 8), (0, 8),
                 (3, 13), (7, 10), (8, 10), (10, 11)]

overlap_name_pairs = [
    (bone_name_dict[i], bone_name_dict[j])
    for i, j in overlap_pairs
]


class FullHandPatchDataset(Dataset):
    """
    Patch dataset for full-hand X-ray segmentation (COCO annotations, JPG/BMP 8-bit).

    Modes:
        - "train": use COCO, random patches with foreground preference
        - "val"/"test": use COCO, sliding-window patches
        - "infer": no COCO, only images in data_root, sliding-window patches (mask=None)
    """

    def __init__(
        self,
        data_root: str | Path,
        annotation_path: Optional[str | Path] = None,
        patch_size: Tuple[int, int] = (512, 512),
        stride: Tuple[int, int] = (384, 384),
        transform: Optional[Any] = None,
        mode: str = "train",
        foreground_ratio: float = 0.7,
        max_tries: int = 20,
        use_coords: bool = False,
        flip_left_by_name: bool = False,
        normalize: str = "fixed",
        dataset_mean: Optional[float] = None,
        dataset_std: Optional[float] = None,
        train_patches_per_image: int = 16,
        expected_num_classes: Optional[int] = None,
    ) -> None:

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

        self.normalize = normalize.lower()
        assert self.normalize in ("fixed", "zscore", "minmax")
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.filenames: List[str] = []
        self.masks: List[Optional[np.ndarray]] = []
        self.img_hw: List[Tuple[int, int]] = []
        self.channel_to_name: Dict[int, str] = {}
        self.name_to_channel: Dict[str, int] = {}

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
            # === 原始的 COCO 读取逻辑 ===
            coco = COCO(annotation_file=str(annotation_path))

            # === 排除 Ring / Metal Implant / Background 类 ===
            all_cat_ids = coco.getCatIds()
            all_cats = coco.loadCats(all_cat_ids)

            skip_names = {"Ring", "Metal Implant", "Background", "background", "Intravenous cannula"}
            valid_cats = sorted(
                [c for c in all_cats if c["name"] not in skip_names and c["id"] != 0],
                key=lambda c: c["id"]
            )

            # cat_id 列表（用于后面 mask 构建）
            self.cat_ids = [c["id"] for c in valid_cats]

            # ===== 核心新增：channel ↔ bone name =====
            self.channel_to_name = {
                ch: c["name"] for ch, c in enumerate(valid_cats)
            }
            self.name_to_channel = {
                c["name"]: ch for ch, c in enumerate(valid_cats)
            }

            # 打印一下过滤后的类别情况
            print(f"[INFO] Loaded {len(self.cat_ids)} classes (excluding {skip_names})")
            print(f"[INFO] Remaining class IDs: {self.cat_ids}")

            # 如果需要，检查和预期类别数是否一致
            if expected_num_classes is not None and len(self.cat_ids) != expected_num_classes:
                print(
                    f"[WARN] expected_num_classes={expected_num_classes}, "
                    f"but after filtering COCO has {len(self.cat_ids)} categories (ids={self.cat_ids})."
                )

            img_ids = coco.getImgIds()
            for img_id in img_ids:
                info = coco.loadImgs([img_id])[0]
                H, W = int(info["height"]), int(info["width"])
                fname = info["file_name"]
                # fname = info["file_name"].split("_bmp")[0] + ".bmp"

                # per-class mask union
                per_class_masks: List[np.ndarray] = []
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

                mask_stack = np.stack(per_class_masks, axis=0)
                self.filenames.append(fname)
                self.masks.append(mask_stack)
                self.img_hw.append((H, W))

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
        H, W = hw

        yy, xx = np.meshgrid(
            (np.arange(H, dtype=np.float32) + 0.5) / H,
            (np.arange(W, dtype=np.float32) + 0.5) / W,
            indexing="ij",
        )

        return np.stack([xx, yy], axis=-1)

    def _coord_channels_global(
            self,
            full_hw: Tuple[int, int],
            y0: int,
            x0: int,
            patch_hw: Tuple[int, int],
    ) -> np.ndarray:
        """
        Return (ph, pw, 2) global position encoding for a patch.

        Args:
            full_hw: (H_full, W_full)
            y0, x0: top-left corner of patch in full image
            patch_hw: (ph, pw)

        Returns:
            coords: (ph, pw, 2) with [x_global, y_global] in [0,1]
        """
        H_full, W_full = full_hw
        ph, pw = patch_hw

        # patch内的像素坐标（注意不是0~1，而是pixel index）
        yy, xx = np.meshgrid(
            np.arange(ph, dtype=np.float32),
            np.arange(pw, dtype=np.float32),
            indexing="ij",
        )

        # 转成全图坐标
        yy = (yy + y0 + 0.5) / float(H_full)
        xx = (xx + x0 + 0.5) / float(W_full)

        return np.stack([xx, yy], axis=-1)  # (ph, pw, 2)

    # -------------------------- I/O -------------------------- #
    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
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
        return img_hwc, mask_stack  # (H,W,1), (K,H,W) uint8

    # -------------------------- Training sampling -------------------------- #
    def _sample_train_patch_coords(
            self,
            mask_stack: np.ndarray,
            H: int,
            W: int
    ) -> Tuple[int, int]:

        ph, pw = self.patch_h, self.patch_w

        # ---- 随机全图采样 ----
        def random_uniform():
            y0 = random.randint(0, max(0, H - ph))
            x0 = random.randint(0, max(0, W - pw))
            return y0, x0

        # ---- 以 foreground 像素为中心采样 ----
        fg_map = (mask_stack.sum(axis=0) > 0)

        if random.random() < self.foreground_ratio and fg_map.any():
            # 找所有前景像素
            ys, xs = np.where(fg_map)

            idx = random.randint(0, len(ys) - 1)
            cy, cx = ys[idx], xs[idx]

            # 以该像素为中心随机偏移
            y0 = cy - random.randint(0, ph - 1)
            x0 = cx - random.randint(0, pw - 1)

            # clamp
            y0 = max(0, min(y0, H - ph))
            x0 = max(0, min(x0, W - pw))

            return y0, x0

        else:
            return random_uniform()

    # -------------------------- Dataset API -------------------------- #
    def __len__(self):
        if self.mode == "train":
            return len(self.filenames) * self.train_patches_per_image
        else:
            return len(self.filenames)  # ❗不是 index_map

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode in ("val", "test", "infer"):

            img_idx = index

            # ---- load ----
            if self.mode == "infer":
                path = self.data_root / self.filenames[img_idx]
                img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                img = self._normalize_image(img_u8)
                mask_stack = None
            else:
                img, mask_stack = self._load_image_and_mask(img_idx)

            # ---- transform ----
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

            # ---- coord（关键！！！）----
            if self.use_coords:
                H, W = img.shape[:2]
                coords = self._coord_channels((H, W))
                img = np.concatenate([img, coords], axis=-1)

            # ---- tensor ----
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

            if mask_stack is not None:
                if isinstance(mask_stack, np.ndarray):
                    mask_t = torch.from_numpy(np.ascontiguousarray(mask_stack)).permute(2, 0, 1).float()
                else:
                    mask_t = mask_stack.permute(2, 0, 1).float()

                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "gt": mask_t,
                    "img_idx": img_idx,
                }

            else:
                return {
                    "fname": self.filenames[img_idx],
                    "img": img_t,
                    "img_idx": img_idx,
                }

        elif self.mode == "train":
            n_imgs = len(self.filenames)
            img_idx = index % n_imgs

            img, mask_stack = self._load_image_and_mask(img_idx)  # (H,W,1), (K,H,W)
            H, W = img.shape[:2]
            ph, pw = self.patch_h, self.patch_w

            y0, x0 = self._sample_train_patch_coords(mask_stack, H, W)
            img_patch = img[y0:y0 + ph, x0:x0 + pw, :]
            mask_patch = mask_stack[:, y0:y0 + ph, x0:x0 + pw]

            if self.transform is not None:
                m_hwc = np.transpose(mask_patch, (1, 2, 0))
                out = self.transform(image=img_patch, mask=m_hwc)
                img_patch = out["image"]  # CHW
                img_patch = img_patch.permute(1, 2, 0)
                mask_patch = out["mask"]  # HWC
            else:
                mask_patch = np.transpose(mask_patch, (1, 2, 0))  # HWC

            # === 在 transform 之后再拼接 coords ===
            if self.use_coords:
                coords = self._coord_channels_global(
                    full_hw=self.img_hw[img_idx],
                    y0=y0,
                    x0=x0,
                    patch_hw=(self.patch_h, self.patch_w),
                )
                img_patch = np.concatenate([img_patch, coords], axis=-1)

            # 转 Tensor
            if isinstance(img_patch, np.ndarray):
                img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).permute(2, 0, 1).float()
            else:
                img_t = img_patch.permute(2, 0, 1).float()

            if isinstance(mask_patch, np.ndarray):
                mask_t = torch.from_numpy(np.ascontiguousarray(mask_patch)).permute(2, 0, 1).float()
            else:
                mask_t = mask_patch.permute(2, 0, 1).float()

            if mask_t.shape[0] != 30:
                print(self.filenames[img_idx])

            return {
                "fname": self.filenames[img_idx],
                "img": img_t,
                "gt": mask_t,
                "img_idx": img_idx,
                "y0": y0,
                "x0": x0,
            }
        else:
            raise NotImplementedError(f"{self.mode} is not implemented!")


class CarpalClassificationDataset_(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation = pd.read_excel(annotation_path)
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filedirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.joint_fnames = []
        self.joint_names = []
        self.img_links = ["Metacarpal1st", "Trapzium", "Scaphoid", "Lunate", "DistalRadius", "DistalUlna"]
        self.gts = []

        for dir_path in self.filedirs:
            filename = dir_path.name
            matched_row = self.annotation[self.annotation["Image Link"] == filename + ".bmp"]
            for key in matched_row.keys()[1:]:
                self.gts.append(matched_row[key].values[0] if matched_row[key].values[0] != 5 else 3)
                self.joint_names.append(key)
                self.joint_fnames.append(dir_path / (key + ".bmp"))

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        img = cv2.imread(str(self.joint_fnames[idx]), cv2.IMREAD_GRAYSCALE)
        gt = self.gts[idx] if self.gts[idx] == 0 else 1
        img = normalization(img[..., np.newaxis]).astype(np.float32)

        if self.joint_fnames[idx].stem[-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
        if self.transform:
            img = self.transform(image=img)["image"]

        data = {
            "fname": str(self.joint_fnames[idx]),
            "key": self.joint_names[idx],
            "img": img,
            "gt": torch.tensor(gt, dtype=torch.int64),
        }
        return data

    def __len__(self):
        return len(self.gts)


class CarpalClassificationDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform

        # 读取JSON
        with open(annotation_path, 'r') as f:
            self.annotation = json.load(f)

        self.joint_fnames = []
        self.joint_names = []
        self.gts = []

        joint_list = ["Metacarpal1st", "Trapzium", "Scaphoid", "Lunate", "DistalRadius", "DistalUlna"]

        for item in self.annotation:
            identifier = item["identifier"]  # 如 0172_0003_L
            joints = item["joints"]          # 字典，包含上述每个关节名及其评分
            dir_path = self.data_root / identifier

            for joint_name in joint_list:
                joint_path = dir_path / f"{joint_name}.bmp"
                if not joint_path.exists():
                    continue
                score = joints[joint_name]
                self.gts.append(score if score == 0 else 1)
                self.joint_names.append(joint_name)
                self.joint_fnames.append(joint_path)

    def __getitem__(self, idx):
        def normalization(data):
            range_val = np.max(data) - np.min(data)
            return (data - np.min(data)) / range_val if range_val != 0 else data

        img_path = self.joint_fnames[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = normalization(img[..., np.newaxis]).astype(np.float32)

        # 左手图像水平翻转
        if img_path.stem.endswith("L"):
            img = cv2.flip(img, 1)

        if self.transform:
            img = self.transform(image=img)["image"]

        data = {
            "fname": str(img_path),
            "key": self.joint_names[idx],
            "img": img,
            "gt": torch.tensor(self.gts[idx], dtype=torch.int64),
        }
        return data

    def __len__(self):
        return len(self.joint_fnames)


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


def get_dataloader_sampler(dataset, batch_size, shuffle=False):
    unique_labels, counts = np.unique(dataset.gts, return_counts=True)
    label_freq = dict(zip(unique_labels, counts))

    # Step 2: 给每个样本分配采样权重（用频率的倒数）
    weights = np.array([1.0 / label_freq[label] for label in dataset.gts])

    # Step 3: 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),  # 每轮epoch抽多少样本，常设为 len(weights)
        replacement=True  # 有放回采样
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


def get_dataloader_sampler_reg(dataset, batch_size, shuffle=False):
    # Step 1: 获取所有标签
    labels = np.array(dataset.total_gts)
    # Step 2: 分桶 共4个区间：[0~1), [1~2.5), [2.5~4), [4~5]
    custom_bins = [0.0, 1.0, 3.0, 10.0]
    digitized = np.digitize(labels, custom_bins)  # 每个样本的桶编号

    # Step 3: 计算每个bin的权重 = 1 / 样本数
    bin_counts = np.bincount(digitized)
    bin_weights = 1. / (bin_counts + 1e-6)  # 避免除0
    sample_weights = bin_weights[digitized]

    # Step 4: 构建WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
