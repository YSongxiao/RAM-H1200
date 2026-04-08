from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


IMAGE_STEM_TO_SCORE_KEY_BE = {
    "R": "BE_R",
    "U": "BE_U",
    "IP": "BE_IP",
    "L": "BE_L",
    "MCP-T": "BE_MCP-T",
    "MCP-I": "BE_MCP-I",
    "MCP-M": "BE_MCP-M",
    "MCP-R": "BE_MCP-R",
    "MCP-S": "BE_MCP-S",
    "CMC-T": "BE_CMC-T",
    "PIP-I": "BE_PIP-I",
    "PIP-M": "BE_PIP-M",
    "PIP-R": "BE_PIP-R",
    "PIP-S": "BE_PIP-S",
    "S": "BE_S",
    "Tm": "BE_Tm",
}

IMAGE_STEM_TO_DISPLAY_NAME = {
    "R": "Radius",
    "U": "Ulna",
    "IP": "IP",
    "L": "Lunate",
    "MCP-T": "MCP1",
    "MCP-I": "MCP2",
    "MCP-M": "MCP3",
    "MCP-R": "MCP4",
    "MCP-S": "MCP5",
    "CMC-T": "CMC1",
    "PIP-I": "PIP2",
    "PIP-M": "PIP3",
    "PIP-R": "PIP4",
    "PIP-S": "PIP5",
    "S": "Scaphoid",
    "Tm": "Trapezium",
}

def resolve_score_file(split_root, score_type):
    split_root = Path(split_root)
    score_type = str(score_type).upper()
    if score_type == "BE":
        score_path = split_root / "_annotation_be_scores.json"
    elif score_type == "JSN":
        score_path = split_root / "_annotation_jsn_scores.json"
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")

    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file: {score_path}")
    return score_path


class BEScoreDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        score_type="BE",
        image_size=224,
        transform=None,
        to_rgb=True,
        score_values=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.split_root = self.data_root / split
        self.score_type = str(score_type).upper()
        self.image_size = int(image_size)
        self.transform = transform
        self.to_rgb = to_rgb

        score_path = resolve_score_file(self.split_root, self.score_type)

        score_data = json.loads(score_path.read_text())

        self.samples = []
        self.raw_scores = []
        self.image_stem_to_score_key = {
            image_stem: score_key.replace("BE_", f"{self.score_type}_", 1)
            for image_stem, score_key in IMAGE_STEM_TO_SCORE_KEY_BE.items()
        }
        for hand_image_name, hand_scores in score_data.items():
            case_name = Path(hand_image_name).stem
            case_dir = self.split_root / case_name
            if not case_dir.exists():
                continue

            joint_scores = hand_scores
            for image_stem, score_key in self.image_stem_to_score_key.items():
                img_path = case_dir / f"{image_stem}.bmp"
                if not img_path.exists():
                    continue

                if score_key not in joint_scores:
                    continue

                raw_score = int(joint_scores[score_key])
                self.samples.append(
                    {
                        "img_path": img_path,
                        "case_name": case_name,
                        "joint_name": IMAGE_STEM_TO_DISPLAY_NAME.get(image_stem, image_stem),
                        "score_key": score_key,
                        "raw_score": raw_score,
                    }
                )
                self.raw_scores.append(raw_score)

        inferred_score_values = sorted(set(self.raw_scores))
        if score_values is None:
            self.score_values = inferred_score_values
        else:
            self.score_values = sorted({int(v) for v in score_values})
            missing_scores = sorted(set(inferred_score_values) - set(self.score_values))
            if missing_scores:
                raise RuntimeError(
                    f"Provided score_values {self.score_values} do not cover samples with scores {missing_scores} "
                    f"under {self.split_root}"
                )
        self.score_to_class = {score: idx for idx, score in enumerate(self.score_values)}
        self.class_to_score = {idx: score for score, idx in self.score_to_class.items()}
        self.labels = [self.score_to_class[sample["raw_score"]] for sample in self.samples]
        self.class_counts = Counter(self.labels)
        self.num_classes = len(self.score_values)

        if not self.samples:
            raise RuntimeError(
                f"No scoring samples found under {self.split_root} for score_type={self.score_type}. "
                f"Expected score file: {score_path}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        with Image.open(img_path) as img:
            return np.array(img.convert("L"))

    def _resize_and_tensorize(self, img):
        img = np.array(
            Image.fromarray(img).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        )
        img = img.astype(np.float32) / 255.0
        if self.to_rgb:
            img = np.repeat(img[:, :, None], 3, axis=2)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        else:
            img = torch.from_numpy(img[None, ...]).float()
        return img

    def get_sample_weights(self, power=1.0):
        class_weights = {
            cls: (1.0 / count) ** float(power)
            for cls, count in self.class_counts.items()
        }
        return [class_weights[label] for label in self.labels]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._load_image(sample["img_path"])

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1)
            img = img.float()
            if img.max() > 1:
                img = img / 255.0
            if self.to_rgb and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif (not self.to_rgb) and img.shape[0] > 1:
                img = img[:1]
        else:
            img = self._resize_and_tensorize(img)

        return {
            "img": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "ordinal_target": torch.tensor(
                [1.0 if self.labels[idx] > threshold else 0.0 for threshold in range(self.num_classes - 1)],
                dtype=torch.float32,
            ),
            "raw_score": torch.tensor(sample["raw_score"], dtype=torch.long),
            "case_name": sample["case_name"],
            "joint_name": sample["joint_name"],
            "score_key": sample["score_key"],
            "img_path": str(sample["img_path"]),
        }


def build_be_score_sampler(dataset, power=1.0, generator=None):
    weights = torch.as_tensor(dataset.get_sample_weights(power=power), dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def seed_scorecls_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def collect_score_values(data_root, split, score_type="BE"):
    split_root = Path(data_root) / split
    score_path = resolve_score_file(split_root, score_type)

    score_data = json.loads(score_path.read_text())
    score_type = str(score_type).upper()
    score_values = set()
    for _, hand_scores in score_data.items():
        joint_scores = hand_scores
        for score in joint_scores.values():
            score_values.add(int(score))
    return sorted(score_values)


def get_be_score_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    oversample=False,
    oversample_power=1.0,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    seed=None,
    distributed=False,
):
    sampler = None
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    if distributed and not dist.is_available():
        raise RuntimeError("Distributed dataloader requested but torch.distributed is unavailable.")
    if distributed and not dist.is_initialized():
        raise RuntimeError("Distributed dataloader requested before process group initialization.")
    if oversample:
        if distributed:
            raise ValueError("Oversampling is not supported with distributed training for score classification.")
        sampler = build_be_score_sampler(dataset, power=oversample_power, generator=generator)
        shuffle = False
    elif distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(seed) if seed is not None else 0,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_scorecls_worker if num_workers > 0 else None,
        generator=generator,
    )
