#!/usr/bin/env python3
"""Train and run inference for a two-class image dataset.

Expected dataset layout:

    mydataset/
      0/
        image_a.png
      1/
        image_b.png
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes, name="resnet18"):
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(image_size, augment=False):
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def warn_duplicate_images(dataset):
    seen = {}
    for path, label_idx in dataset.samples:
        image_path = Path(path)
        with Image.open(image_path) as image:
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if digest in seen:
            other_path, other_label_idx = seen[digest]
            if other_label_idx != label_idx:
                print(
                    "warning: duplicate image pixels found in different classes: "
                    f"{other_path} (class={dataset.classes[other_label_idx]}) and "
                    f"{image_path} (class={dataset.classes[label_idx]}). "
                    "These samples cannot be separated by image content."
                )
        else:
            seen[digest] = (image_path, label_idx)


def stratified_split(dataset, val_ratio, seed):
    if val_ratio <= 0:
        return list(range(len(dataset))), []

    by_class = {}
    for idx, label in enumerate(dataset.targets):
        by_class.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []
    rng = random.Random(seed)
    for indices in by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if val_count == 0 and len(shuffled) > 1:
            val_count = 1
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    if not train_indices:
        raise ValueError("No training samples remain after splitting. Lower --val-ratio.")
    return train_indices, val_indices


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def train(args):
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset_dir = Path(args.data_dir)
    dataset = datasets.ImageFolder(dataset_dir, transform=build_transform(args.image_size, args.augment))
    if len(dataset.classes) < 2:
        raise ValueError(f"Need at least two class folders under {dataset_dir}. Found: {dataset.classes}")
    warn_duplicate_images(dataset)

    train_indices, val_indices = stratified_split(dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_indices:
        val_dataset = datasets.ImageFolder(dataset_dir, transform=build_transform(args.image_size, augment=False))
        val_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = build_model(num_classes=len(dataset.classes), name=args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"classes: {dataset.class_to_idx}")
    print(f"samples: train={len(train_indices)}, val={len(val_indices)}, device={device}")

    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        message = f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}"

        score = train_acc
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            message += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            score = val_acc
        else:
            eval_train_loss, eval_train_acc = evaluate(model, train_loader, criterion, device)
            message += f" eval_train_loss={eval_train_loss:.4f} eval_train_acc={eval_train_acc:.4f}"
            score = eval_train_acc
        print(message)

        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "image_size": args.image_size,
            "model": args.model,
        },
        output_path,
    )
    print(f"saved checkpoint: {output_path}")


def iter_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


@torch.no_grad()
def predict(args):
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    image_size = checkpoint.get("image_size", args.image_size)

    model_name = checkpoint.get("model", "small_cnn")
    model = build_model(num_classes=len(class_to_idx), name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = build_transform(image_size, augment=False)

    rows = []
    for image_path in iter_image_paths(args.input):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(probs.argmax().item())
        pred_label = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx].item())
        rows.append(
            {
                "image": str(image_path),
                "pred_label": pred_label,
                "confidence": f"{confidence:.6f}",
                "probabilities": json.dumps(
                    {idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))},
                    ensure_ascii=False,
                ),
            }
        )
        print(f"{image_path}\tpred={pred_label}\tconfidence={confidence:.4f}")

    if not rows:
        raise ValueError(f"No images found in {args.input}")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "pred_label", "confidence", "probabilities"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved predictions: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference for class folders such as mydataset/0 and mydataset/1.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a classifier from class folders.")
    train_parser.add_argument("--data-dir", default="mydataset")
    train_parser.add_argument("--output", default="checkpoints/mydataset_classifier.pt")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--model", choices=["resnet18", "small_cnn"], default="resnet18")
    train_parser.add_argument("--val-ratio", type=float, default=0.0)
    train_parser.add_argument("--augment", action="store_true")
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Predict a single image or a directory of images.")
    predict_parser.add_argument("--input", default="mydataset")
    predict_parser.add_argument("--checkpoint", default="checkpoints/mydataset_classifier.pt")
    predict_parser.add_argument("--output-csv", default="predictions.csv")
    predict_parser.add_argument("--image-size", type=int, default=224)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.set_defaults(func=predict)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
