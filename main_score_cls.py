import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import monai
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.scorecls import BEScoreDataset, collect_score_values, get_be_score_dataloader
from trainer import ScoreClsTester, ScoreClsTrainer
from utils import get_cls_transform, seed_everything


MONITOR_MODE_BY_METRIC = {
    "macro_f1": "max",
    "accuracy": "max",
    "qwk": "max",
    "mae": "min",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026, help="Seed.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode.")
    parser.add_argument("--score_type", type=str, default="BE", choices=["BE", "JSN"], help="Scoring target type.")
    parser.add_argument("--image_size", type=int, default=224, help="Input size.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation/test batch size.")
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet34",
        choices=[
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "DenseNet",
            "MobileNet",
            "EfficientFormer",
            "MobileViT",
            "LeViT",
            "ConvNeXtV2",
            "EfficientNetV2",
            "MedMamba",
            "MambaVisionT",
            "MambaVisionT2",
            "MambaVisionS",
        ],
        help="Classifier backbone.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealing",
        choices=["CosineAnnealing", "Plateau", "None"],
        help="Scheduler type.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=False, help="Enable AMP.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.set_defaults(amp=False)
    parser.add_argument("--grad_clip", type=Union[None, float], default=None, help="Gradient clipping.")
    parser.add_argument("--to_rgb", dest="to_rgb", action="store_true", help="Repeat grayscale to 3 channels.")
    parser.add_argument("--no-to_rgb", dest="to_rgb", action="store_false", help="Use single-channel input.")
    parser.set_defaults(to_rgb=False)
    parser.add_argument("--oversample", action="store_true", default=False, help="Use weighted oversampling for train split.")
    parser.add_argument("--oversample_power", type=float, default=1.0, help="Oversampling weight exponent.")
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        default=False,
        help="Use threshold-wise positive weighting in ordinal BCE loss.",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="qwk",
        choices=["macro_f1", "accuracy", "qwk", "mae"],
        help="Validation metric used for early stopping and best checkpoint.",
    )
    parser.add_argument("--monitor_mode", type=str, default="max", choices=["min", "max"], help="Monitor mode.")
    parser.add_argument("--data_path", type=str, default="/mnt/data2/datasx/FullHand/NIPS26/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring", help="Dataset root.")
    parser.add_argument("--checkpoint", type=str, default="./ckpts", help="Checkpoint root or experiment directory.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint.")
    parser.add_argument("--resume_from", type=str, default="", help="Explicit checkpoint file or experiment directory.")
    parser.add_argument("--trial_name", type=str, default="Benchmark_JSNScoring", help="Trial name.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Max epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--earlystop", action="store_true", default=False, help="Enable early stopping.")
    parser.add_argument("--earlystop_patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetched batches per worker.")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", help="Enable pin memory.")
    parser.add_argument("--no-pin_memory", dest="pin_memory", action="store_false", help="Disable pin memory.")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", help="Keep workers alive.")
    parser.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false", help="Disable persistent workers.")
    parser.set_defaults(pin_memory=True, persistent_workers=False)
    parser.add_argument("--save_csv", action="store_true", default=False, help="Save csv summaries in test mode.")
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "ddp"], help="Launch mode.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP.")
    return parser.parse_args()


def init_distributed(args):
    args.is_ddp = args.launcher == "ddp"

    if not args.is_ddp:
        args.rank = 0
        args.world_size = 1
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("DDP launch requires CUDA.")

    if not dist.is_initialized():
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        if args.local_rank < 0:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(args.local_rank)

    return torch.device(f"cuda:{args.local_rank}")


def cleanup_distributed(args):
    if getattr(args, "is_ddp", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def resolve_checkpoint_artifact(path: Path) -> Tuple[Path, Path]:
    if path.is_file():
        return path, path.parent

    latest_file = path / "model_latest.pth"
    if latest_file.exists():
        return latest_file, path

    raise FileNotFoundError(f"Could not find model_latest.pth under {path}")


def get_model_tag(args) -> str:
    return args.model.lower()


def find_latest_resume_dir(checkpoint_root: Path, trial_name: str, model_tag: str, score_type: str) -> Optional[Path]:
    if not checkpoint_root.exists() or not checkpoint_root.is_dir():
        return None

    prefix = f"{trial_name}_{score_type.lower()}_{model_tag}_"
    candidates = []
    for child in checkpoint_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix) and (child / "model_latest.pth").exists():
            candidates.append((child.stat().st_mtime, child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def prepare_train_run(args):
    checkpoint_path = Path(args.checkpoint)
    resume_file = None
    run_dir = None
    model_tag = get_model_tag(args)

    if args.resume_from:
        resume_file, run_dir = resolve_checkpoint_artifact(Path(args.resume_from))
    elif args.resume:
        if checkpoint_path.exists() and checkpoint_path.is_file():
            resume_file, run_dir = resolve_checkpoint_artifact(checkpoint_path)
        elif checkpoint_path.exists() and (checkpoint_path / "model_latest.pth").exists():
            resume_file, run_dir = resolve_checkpoint_artifact(checkpoint_path)
        else:
            run_dir = find_latest_resume_dir(checkpoint_path, args.trial_name, model_tag, args.score_type)
            if run_dir is not None:
                resume_file = run_dir / "model_latest.pth"

    if resume_file is not None:
        args.model_save_path = str(run_dir)
        args.checkpoint = str(run_dir)
        print(f"Resuming training from {resume_file}")
        return torch.load(resume_file, map_location="cpu")

    checkpoint_root = checkpoint_path
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = checkpoint_root / f"{args.trial_name}_{args.score_type.lower()}_{model_tag}_{time_str}"
    run_dir.mkdir(parents=True, exist_ok=False)
    args.model_save_path = str(run_dir)
    args.checkpoint = str(run_dir)
    print(f"Starting new training run at {run_dir}")
    return None

def build_model(model_name, in_chans, num_classes, image_size=224):
    out_dims = max(num_classes - 1, 1)
    if model_name in {"MambaVisionT", "MambaVisionT2", "MambaVisionS"}:
        try:
            from mambavision import create_model as create_mambavision_model
        except ImportError as exc:
            raise ImportError(
                "MambaVision is selected but the 'mambavision' package is not available. "
                "Please install it in the current training environment first."
            ) from exc

        mambavision_models = {
            "MambaVisionT": "mamba_vision_T",
            "MambaVisionT2": "mamba_vision_T2",
            "MambaVisionS": "mamba_vision_S",
        }
        return create_mambavision_model(
            mambavision_models[model_name],
            pretrained=False,
            in_chans=in_chans,
            num_classes=out_dims,
            resolution=image_size,
        )

    if model_name in {"ResNet18", "ResNet34", "ResNet50"}:
        import timm

        resnet_models = {
            "ResNet18": "resnet18",
            "ResNet34": "resnet34",
            "ResNet50": "resnet50",
        }
        return timm.create_model(
            resnet_models[model_name],
            pretrained=False,
            in_chans=in_chans,
            num_classes=out_dims,
        )
    if model_name == "DenseNet":
        return monai.networks.nets.DenseNet121(
            spatial_dims=2,
            in_channels=in_chans,
            out_channels=out_dims,
        )
    if model_name == "MedMamba":
        from models.MedMamba import VSSM as MedMamba

        return MedMamba(in_chans=in_chans, num_classes=out_dims)

    timm_models = {
        "MobileNet": "mobilenetv2_050",
        "EfficientFormer": "efficientformerv2_s0",
        "EfficientNetV2": "tf_efficientnetv2_s.in21k_ft_in1k",
        "MobileViT": "mobilevit_s",
        "LeViT": "levit_128s",
        "ConvNeXtV2": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    }
    import timm
    return timm.create_model(
        timm_models[model_name],
        pretrained=False,
        in_chans=in_chans,
        num_classes=out_dims,
    )


def build_criterion(args, train_dataset, device):
    num_thresholds = max(len(train_dataset.score_values) - 1, 1)
    if not args.use_class_weight:
        return nn.BCEWithLogitsLoss()

    labels = np.asarray(train_dataset.labels, dtype=int)
    pos_weights = []
    for threshold in range(num_thresholds):
        pos_count = int(np.sum(labels > threshold))
        neg_count = int(len(labels) - pos_count)
        pos_weights.append(float(neg_count / max(pos_count, 1)))
    pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_loader(dataset, batch_size, shuffle, args, oversample=False, distributed=False, drop_last=False):
    return get_be_score_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        oversample=oversample,
        oversample_power=args.oversample_power,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        seed=args.seed,
        distributed=distributed,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )


def main():
    args = get_args()
    expected_monitor_mode = MONITOR_MODE_BY_METRIC[args.monitor_metric]
    if args.monitor_mode != expected_monitor_mode:
        args.monitor_mode = expected_monitor_mode
    device = init_distributed(args)
    seed_everything(args.seed + getattr(args, "rank", 0))

    train_transform = get_cls_transform(split="train", image_size=args.image_size)
    eval_transform = get_cls_transform(split="val", image_size=args.image_size)

    try:
        if args.mode == "train":
            resume_state = prepare_train_run(args) if is_main_process(args) else None
            if getattr(args, "is_ddp", False):
                payload = [resume_state]
                dist.broadcast_object_list(payload, src=0)
                resume_state = payload[0]
                dist.barrier()
            unified_score_values = sorted(
                set(collect_score_values(args.data_path, "train", args.score_type)) |
                set(collect_score_values(args.data_path, "val", args.score_type))
            )

            train_dataset = BEScoreDataset(
                data_root=args.data_path,
                split="train",
                score_type=args.score_type,
                image_size=args.image_size,
                transform=train_transform,
                to_rgb=args.to_rgb,
                score_values=unified_score_values,
            )
            val_dataset = BEScoreDataset(
                data_root=args.data_path,
                split="val",
                score_type=args.score_type,
                image_size=args.image_size,
                transform=eval_transform,
                to_rgb=args.to_rgb,
                score_values=unified_score_values,
            )

            num_classes = len(train_dataset.score_values)
            in_chans = 3 if args.to_rgb else 1
            net = build_model(args.model, in_chans, num_classes, image_size=args.image_size).to(device)
            if is_main_process(args):
                n_params = sum(p.numel() for p in net.parameters())
                print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")
            if resume_state is not None:
                net.load_state_dict(resume_state["model"])
            if getattr(args, "is_ddp", False):
                net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

            train_loader = build_loader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=not args.oversample,
                args=args,
                oversample=args.oversample,
                distributed=getattr(args, "is_ddp", False),
                drop_last=True,
            )
            val_loader = build_loader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                args=args,
                oversample=False,
                distributed=False,
                drop_last=False,
            )

            optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
            criterion = build_criterion(args, train_dataset, device)
            trainer = ScoreClsTrainer(
                args,
                net,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                num_classes=num_classes,
                score_values=train_dataset.score_values,
                device=device,
            )
            if resume_state is not None:
                trainer.load_training_state(resume_state)
            trainer.fit(args)
            return

        checkpoint_dir = Path(args.checkpoint)
        if checkpoint_dir.is_file():
            checkpoint_dir = checkpoint_dir.parent
        args.checkpoint = str(checkpoint_dir)
        checkpoint_state = torch.load(checkpoint_dir / "model_best.pth", map_location="cpu")
        checkpoint_score_values = checkpoint_state.get("score_values")

        test_dataset = BEScoreDataset(
            data_root=args.data_path,
            split="test",
            score_type=args.score_type,
            image_size=args.image_size,
            transform=eval_transform,
            to_rgb=args.to_rgb,
            score_values=checkpoint_score_values,
        )
        num_classes = len(test_dataset.score_values)
        in_chans = 3 if args.to_rgb else 1
        net = build_model(args.model, in_chans, num_classes, image_size=args.image_size).to(device)
        if is_main_process(args):
            n_params = sum(p.numel() for p in net.parameters())
            print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")
        test_loader = build_loader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            args=args,
            oversample=False,
            distributed=False,
            drop_last=False,
        )
        tester = ScoreClsTester(
            args,
            net,
            test_loader,
            num_classes=num_classes,
            score_values=test_dataset.score_values,
            device=device,
        )
        tester.test()
    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
