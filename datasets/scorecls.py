from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


BE_JOINT_TO_SCORE_KEY = {
    "DistalRadius": "BE_R",
    "DistalUlna": "BE_U",
    "IP": "BE_IP",
    "Lunate": "BE_L",
    "MCP1": "BE_MCP-T",
    "MCP2": "BE_MCP-I",
    "MCP3": "BE_MCP-M",
    "MCP4": "BE_MCP-R",
    "MCP5": "BE_MCP-S",
    "Metacarpal1st": "BE_CMC-T",
    "PIP2": "BE_PIP-I",
    "PIP3": "BE_PIP-M",
    "PIP4": "BE_PIP-R",
    "PIP5": "BE_PIP-S",
    "Scaphoid": "BE_S",
    "Trapezium": "BE_Tm",
}


class BEScoreDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        image_size=224,
        transform=None,
        to_rgb=True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.split_root = self.data_root / split
        self.image_size = int(image_size)
        self.transform = transform
        self.to_rgb = to_rgb

        score_path = self.split_root / "_be_jsn_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        score_data = json.loads(score_path.read_text())

        self.samples = []
        self.raw_scores = []
        for hand_image_name, hand_scores in score_data.items():
            case_name = hand_image_name.replace(".bmp", "")
            case_dir = self.split_root / case_name
            if not case_dir.exists():
                continue

            be_scores = hand_scores["BE"]
            for joint_name, score_key in BE_JOINT_TO_SCORE_KEY.items():
                img_path = case_dir / f"{joint_name}.bmp"
                if not img_path.exists():
                    continue

                raw_score = int(be_scores[score_key])
                self.samples.append(
                    {
                        "img_path": img_path,
                        "case_name": case_name,
                        "joint_name": joint_name,
                        "score_key": score_key,
                        "raw_score": raw_score,
                    }
                )
                self.raw_scores.append(raw_score)

        self.score_values = sorted(set(self.raw_scores))
        self.score_to_class = {score: idx for idx, score in enumerate(self.score_values)}
        self.class_to_score = {idx: score for score, idx in self.score_to_class.items()}
        self.labels = [self.score_to_class[sample["raw_score"]] for sample in self.samples]
        self.class_counts = Counter(self.labels)

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
        else:
            img = self._resize_and_tensorize(img)

        return {
            "img": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_score": torch.tensor(sample["raw_score"], dtype=torch.long),
            "case_name": sample["case_name"],
            "joint_name": sample["joint_name"],
            "score_key": sample["score_key"],
            "img_path": str(sample["img_path"]),
        }


def build_be_score_sampler(dataset, power=1.0):
    weights = torch.as_tensor(dataset.get_sample_weights(power=power), dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def get_be_score_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    oversample=False,
    oversample_power=1.0,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    sampler = None
    if oversample:
        sampler = build_be_score_sampler(dataset, power=oversample_power)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
