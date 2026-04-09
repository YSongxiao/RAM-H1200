import argparse
import inspect
import os

import monai
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.optim as optim
from monai.losses import DiceCELoss
from datetime import datetime
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Tuple, Union

from datasets.beseg import BEPatchDataset, get_dataloader
from evaluations.loss import CreditAwareDiceCELoss
from models.ACC_UNet.ACC_UNet import ACC_UNet
from models.Seg_UKAN.archs import UKAN
from models.MambaVisionSeg import get_MambaVisionSeg
from models.SwinUMamba import get_DPMSwinUMamba, get_RefinedSwinUMamba, get_SwinUMamba
from models.TransUNet.transUnet import get_TransUnet_Custom
from models.UMamba import get_UMambaBot, get_UMambaEnc
from models.UnetPlusPlus import UnetPlusPlus
from trainer import BEPatchSegInferencer, BEPatchSegTester, BESegTrainer
from utils import get_transform, seed_everything

BE_LESION_CATEGORY_NAMES = ["SvdH-BE-90", "SvdH-BE-50", "Non-SvdH-BE"]
BE_CLASS_NAMES = ["Background"] + BE_LESION_CATEGORY_NAMES
BE_CREDIT_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2026, help="Seed.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "infer"], help="Mode.")
    parser.add_argument("--image_size", type=int, default=256, help="Validation/inference roi size.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for validating.")
    parser.add_argument(
        "--model",
        type=str,
        default="MambaVisionT",
        choices=[
            "Unet", "SegResNet", "Unet++", "TransUnet", "UKAN", "DeepLabV3", "DeepLabV3+",
            "PSPNet", "PAN", "DPT", "SegFormer", "FPN", "UMambaBot", "UMambaEnc",
            "SwinUMamba", "DPMSwinUMamba", "RefinedSwinUMamba", "ACC_UNet", "SwinUNETR",
            "MambaVisionT", "MambaVisionT2", "MambaVisionS",
        ],
        help="The name of the model.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealing",
        choices=["CosineAnnealing", "Plateau"],
        help="Scheduler type.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dicece",
        choices=["dicece", "creditaware"],
        help="Training loss type.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Enable AMP.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.set_defaults(amp=True)
    parser.add_argument("--grad_clip", type=Union[None, float], default=None, help="Gradient clipping.")
    parser.add_argument("--use_coords", dest="use_coords", action="store_true", help="Enable positional channels.")
    parser.add_argument("--no-use_coords", dest="use_coords", action="store_false", help="Disable positional channels.")
    parser.set_defaults(use_coords=True)
    parser.add_argument("--tr_patches_per_img", type=int, default=24, help="Number of train patches per image.")
    parser.add_argument(
        "--center_region_half_size",
        type=int,
        default=96,
        help="Foreground sampling center-region half size in pixels.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data2/datasx/FullHand/NIPS26/RAM-H1200/Segmentation",
        help="Dataset root.",
    )
    parser.add_argument("--pretrained_weights", type=str, default="", help="Optional pretrained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpts",
        help="Checkpoint root for train mode, or experiment directory/file for resume.",
    )
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint.")
    parser.add_argument("--resume_from", type=str, default="", help="Explicit checkpoint file or directory.")
    parser.add_argument("--trial_name", type=str, default="benchmark_beseg", help="Trial name.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Max epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial lr.")
    parser.add_argument("--monitor_mode", type=str, default="max", help='Trainer mode. ("min" or "max")')
    parser.add_argument("--earlystop_patience", type=int, default=15, help="Patience for early stopping.")
    parser.add_argument(
        "--save_uncertainty_overlay",
        action="store_true",
        default=False,
        help="Whether to save uncertainty overlays in test/infer mode.",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        default=False,
        help="Whether to save overlay pdfs in test/infer mode.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=False,
        help="Whether to save prediction bundles as npz (pred/image/gt when available) in test/infer mode.",
    )
    parser.add_argument(
        "--save_npy",
        dest="save_npz",
        action="store_true",
        help="Deprecated alias of --save_npz.",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        default=False,
        help="Whether to save test metrics to csv.",
    )
    parser.add_argument(
        "--save_pred",
        action="store_true",
        default=False,
        help="Whether to save separated pred/gt overlay pdfs in test mode.",
    )
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "ddp"], help="Launch mode.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per process. Use -1 to auto-select.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetched batches per worker.")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", help="Enable pinned host memory.")
    parser.add_argument("--no-pin_memory", dest="pin_memory", action="store_false", help="Disable pinned host memory.")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", help="Keep workers alive.")
    parser.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false", help="Disable persistent workers.")
    parser.set_defaults(pin_memory=False, persistent_workers=False)
    return parser.parse_args()


def init_distributed(args):
    args.is_ddp = args.launcher == "ddp"

    if not args.is_ddp:
        args.rank = 0
        args.world_size = 1
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def configure_dataloader_args(args):
    cpu_count = os.cpu_count() or 1
    world_size = max(1, getattr(args, "world_size", 1))

    if args.num_workers < 0:
        workers_per_process = max(1, cpu_count // world_size)
        if args.mode == "train":
            args.num_workers = min(8, workers_per_process)
        else:
            args.num_workers = min(4, workers_per_process)

    args.num_workers = max(0, args.num_workers)
    args.prefetch_factor = max(1, args.prefetch_factor)
    args.pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    if args.num_workers == 0:
        args.persistent_workers = False

    if is_main_process(args):
        print(
            "DataLoader config: "
            f"num_workers={args.num_workers}, "
            f"pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"prefetch_factor={args.prefetch_factor}"
        )


def resolve_checkpoint_artifact(path: Path) -> Tuple[Path, Path]:
    if path.is_file():
        return path, path.parent

    latest_file = path / "model_latest.pth"
    if latest_file.exists():
        return latest_file, path

    raise FileNotFoundError(f"Could not find model_latest.pth under {path}")


def find_latest_resume_dir(checkpoint_root: Path, trial_name: str, model_name: str) -> Optional[Path]:
    if not checkpoint_root.exists() or not checkpoint_root.is_dir():
        return None

    model_token = model_name.lower()
    exact_candidates = []
    legacy_candidates = []
    compatible_candidates = []

    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        latest_file = child / "model_latest.pth"
        if not latest_file.exists():
            continue

        mtime = latest_file.stat().st_mtime
        if child.name == trial_name:
            exact_candidates.append((mtime, child))
            continue

        legacy_prefix = f"{trial_name}_{model_token}_"
        if child.name.startswith(legacy_prefix):
            legacy_candidates.append((mtime, child))
            continue

        compatible_token = f"_{trial_name}_{model_token}_"
        if compatible_token in child.name:
            compatible_candidates.append((mtime, child))

    candidates = exact_candidates or legacy_candidates or compatible_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def prepare_train_run(args) -> Optional[dict]:
    checkpoint_path = Path(args.checkpoint)
    resume_file = None
    run_dir = None

    if args.resume_from:
        resume_file, run_dir = resolve_checkpoint_artifact(Path(args.resume_from))
    elif args.resume:
        if checkpoint_path.exists() and checkpoint_path.is_file():
            resume_file, run_dir = resolve_checkpoint_artifact(checkpoint_path)
        elif checkpoint_path.exists() and (checkpoint_path / "model_latest.pth").exists():
            resume_file, run_dir = resolve_checkpoint_artifact(checkpoint_path)
        else:
            run_dir = find_latest_resume_dir(checkpoint_path, args.trial_name, args.model)
            if run_dir is not None:
                resume_file = run_dir / "model_latest.pth"

    if resume_file is not None:
        args.model_save_path = str(run_dir)
        args.checkpoint = str(run_dir)
        if is_main_process(args):
            print(f"Resuming training from {resume_file}")
        return torch.load(resume_file, map_location="cpu")

    if args.resume and is_main_process(args):
        print("No resumable checkpoint found. Starting a new training run.")

    checkpoint_root = checkpoint_path
    if checkpoint_root.suffix and not checkpoint_root.exists():
        checkpoint_root = checkpoint_root.parent
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = checkpoint_root / f"{args.trial_name}_{args.model.lower()}_{time_str}"
    if is_main_process(args):
        run_dir.mkdir(parents=True, exist_ok=False)
    if args.is_ddp:
        dist.barrier()

    args.model_save_path = str(run_dir)
    args.checkpoint = str(run_dir)
    if is_main_process(args):
        print(f"Starting new training run at {run_dir}")
    return None


def build_model(args, in_chans):
    out_chans = 4

    if args.model == "Unet":
        return monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=[3, 3, 3, 3, 3],
            norm_name="batch",
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2, 2],
            filters=[32, 64, 128, 256, 512],
            res_block=True,
        )
    elif args.model == "SegResNet":
        return monai.networks.nets.SegResNet(
            spatial_dims=2,
            in_channels=in_chans,
            out_channels=out_chans,
            upsample_mode="deconv",
            act="LeakyReLU",
            norm="batch",
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif args.model == "Unet++":
        model = UnetPlusPlus(spatial_dims=2, in_channels=in_chans, out_channels=out_chans, features=(32, 32, 64, 128, 256, 32))
        backbone = getattr(model, "model", None)
        if backbone is not None and not getattr(backbone, "deep_supervision", False):
            for head_name in ("final_conv_0_1", "final_conv_0_2", "final_conv_0_3"):
                head = getattr(backbone, head_name, None)
                if head is None:
                    continue
                for param in head.parameters():
                    param.requires_grad = False
        return model
    elif args.model == "UKAN":
        return UKAN(num_classes=out_chans, input_channels=in_chans, embed_dims=[128, 160, 256])
    elif args.model == "TransUnet":
        return get_TransUnet_Custom(img_size=args.image_size, n_classes=out_chans)
    elif args.model == "DeepLabV3+":
        return smp.DeepLabV3Plus(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "DeepLabV3":
        return smp.DeepLabV3(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "PSPNet":
        return smp.PSPNet(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "PAN":
        return smp.PAN(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "DPT":
        return smp.DPT(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "SegFormer":
        return smp.Segformer(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "FPN":
        return smp.FPN(in_channels=in_chans, encoder_weights=None, classes=out_chans)
    elif args.model == "UMambaBot":
        return get_UMambaBot(in_channels=in_chans, num_classes=out_chans)
    elif args.model == "UMambaEnc":
        return get_UMambaEnc(in_channels=in_chans, num_classes=out_chans)
    elif args.model == "SwinUMamba":
        return get_SwinUMamba(in_channels=in_chans, num_classes=out_chans)
    elif args.model == "DPMSwinUMamba":
        return get_DPMSwinUMamba(in_channels=in_chans, num_overlap_classes=1, num_classes=out_chans)
    elif args.model == "RefinedSwinUMamba":
        raise NotImplementedError("RefinedSwinUMamba for BE segmentation requires an explicit base checkpoint.")
    elif args.model in {"MambaVisionT", "MambaVisionT2", "MambaVisionS"}:
        variant_map = {
            "MambaVisionT": "mamba_vision_T",
            "MambaVisionT2": "mamba_vision_T2",
            "MambaVisionS": "mamba_vision_S",
        }
        return get_MambaVisionSeg(
            variant=variant_map[args.model],
            in_chans=in_chans,
            num_classes=out_chans,
            image_size=args.image_size,
            pretrained=False,
        )
    elif args.model == "ACC_UNet":
        return ACC_UNet(n_channels=in_chans, n_classes=out_chans - 1)
    elif args.model == "SwinUNETR":
        swinunetr_kwargs = dict(
            in_channels=in_chans,
            out_channels=out_chans,
            img_size=(args.image_size, args.image_size),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=2,
            downsample="merging",
            use_v2=False,
        )
        swinunetr_signature = inspect.signature(monai.networks.nets.SwinUNETR)
        compatible_kwargs = {
            key: value
            for key, value in swinunetr_kwargs.items()
            if key in swinunetr_signature.parameters
        }
        return monai.networks.nets.SwinUNETR(**compatible_kwargs)

    raise ValueError(f"Unsupported model: {args.model}")


def main():
    args = get_args()
    device = init_distributed(args)
    configure_dataloader_args(args)

    try:
        seed_everything(args.seed + getattr(args, "rank", 0))

        in_chans = 1 + (2 if args.use_coords else 0)
        num_classes = len(BE_CLASS_NAMES)
        net = build_model(args, in_chans)

        n_params = sum(p.numel() for p in net.parameters())
        if is_main_process(args):
            print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")

        if args.mode == "train":
            resume_state = prepare_train_run(args)

            if resume_state is not None:
                net.load_state_dict(resume_state["model"])
            elif args.pretrained_weights and Path(args.pretrained_weights).exists():
                if hasattr(net, "load_pretrained_weights"):
                    net.load_pretrained_weights(Path(args.pretrained_weights))
                else:
                    state = torch.load(Path(args.pretrained_weights), map_location="cpu")
                    net.load_state_dict(state["model"])

            net = net.to(device)
            if args.is_ddp:
                net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

            transform_tr = get_transform(split="train", image_size=args.image_size)
            transform_val = get_transform(split="val", image_size=args.image_size)

            train_root = Path(args.data_path) / "train"
            val_root = Path(args.data_path) / "val"

            train_dataset = BEPatchDataset(
                data_root=train_root,
                annotation_path=train_root / "_annotations_be_rle.coco.json",
                mode="train",
                transform=transform_tr,
                use_coords=args.use_coords,
                train_patches_per_image=args.tr_patches_per_img,
                center_region_half_size=args.center_region_half_size,
                category_names=BE_LESION_CATEGORY_NAMES,
                add_background_channel=True,
                expected_num_classes=num_classes,
            )
            val_dataset = BEPatchDataset(
                data_root=val_root,
                annotation_path=val_root / "_annotations_be_rle.coco.json",
                mode="val",
                transform=transform_val,
                use_coords=args.use_coords,
                train_patches_per_image=args.tr_patches_per_img,
                center_region_half_size=args.center_region_half_size,
                category_names=BE_LESION_CATEGORY_NAMES,
                add_background_channel=True,
                expected_num_classes=num_classes,
            )

            train_loader = get_dataloader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                distributed=args.is_ddp,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )
            val_loader = get_dataloader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                distributed=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

            optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-3)
            if args.loss == "dicece":
                criterion = DiceCELoss(
                    softmax=True,
                    reduction="mean",
                    squared_pred=True,
                    include_background=False,
                    smooth_nr=1e-5,
                    smooth_dr=1e-5,
                )
            else:
                criterion = CreditAwareDiceCELoss(
                    credit_matrix=BE_CREDIT_MATRIX,
                    ce_weight=None,
                    ce_loss_weight=1.0,
                    credit_dice_weight=1.0,
                    include_background=False,
                )

            trainer = BESegTrainer(args, net, train_loader, val_loader, criterion, optimizer, num_classes=num_classes, device=device)
            if resume_state is not None:
                trainer.load_training_state(resume_state)
            trainer.fit(args)

        elif args.mode == "test":
            net = net.to(device)
            transform_test = get_transform(split="test", image_size=args.image_size)
            test_root = Path(args.data_path) / "test"
            test_dataset = BEPatchDataset(
                data_root=test_root,
                annotation_path=test_root / "_annotations_be_rle.coco.json",
                mode="test",
                transform=transform_test,
                use_coords=args.use_coords,
                train_patches_per_image=args.tr_patches_per_img,
                center_region_half_size=args.center_region_half_size,
                category_names=BE_LESION_CATEGORY_NAMES,
                add_background_channel=True,
                expected_num_classes=num_classes,
            )
            test_loader = get_dataloader(
                test_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                distributed=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            tester = BEPatchSegTester(args, net, test_loader, device=device)
            tester.test()

        elif args.mode == "infer":
            net = net.to(device)
            transform_infer = get_transform(split="test", image_size=args.image_size)
            infer_root = Path(args.data_path)
            infer_dataset = BEPatchDataset(
                data_root=infer_root,
                annotation_path=None,
                mode="infer",
                transform=transform_infer,
                use_coords=args.use_coords,
                train_patches_per_image=args.tr_patches_per_img,
                center_region_half_size=args.center_region_half_size,
                category_names=BE_LESION_CATEGORY_NAMES,
                add_background_channel=True,
                expected_num_classes=num_classes,
            )
            infer_loader = get_dataloader(
                infer_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                distributed=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            inferencer = BEPatchSegInferencer(args, net, infer_loader, device=device)
            inferencer.test()

    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
