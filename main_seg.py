import argparse
import inspect
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *
from trainer import FullHandBoneSegInferencer, FullHandBoneSegTester, FullHandBoneSegTrainer
from evaluations.loss import BCEDiceLoss, MSEBCEDiceLoss
import monai
from pathlib import Path
from datasets.boneseg import FullHandPatchDataset, get_dataloader
import torch.optim as optim
from datetime import datetime
from typing import Optional, Tuple, Union
from models.UnetPlusPlus import UnetPlusPlus
from models.swin_unet.swintrans import SwinUnet
from models.swin_unet.swin_unet import get_SwinUnet_Custom
from models.Seg_UKAN.archs import UKAN
from models.MambaVisionSeg import get_MambaVisionSeg
from models.TransUNet.transUnet import get_TransUnet_Custom
from models.UMamba import get_UMambaBot, get_UMambaEnc
from models.SwinUMamba import get_SwinUMamba, get_DPMSwinUMamba, get_RefinedSwinUMamba
from models.ACC_UNet.ACC_UNet import ACC_UNet
import segmentation_models_pytorch as smp


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=2026,  # 3407 4096 1024 5214
        help='Seed',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default="train",
        choices=["train", "test", "infer"],
        help='Mode',
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='The size of the images.',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=8,
        help='Batch size for training.',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=1,
        help='Batch size for validating.',
    )

    parser.add_argument(
        '--model',
        type=str,
        default="SwinUMamba",
        choices=["Unet", "SwinUnet", "SegResNet", "Unet++", "TransUnet", "UKAN", "DeepLabV3", "DeepLabV3+", "PSPNet",
                 "PAN", "DPT", "SegFormer", "FPN", "UMambaBot", "UMambaEnc", "SwinUMamba", "DPMSwinUMamba", "RefinedSwinUMamba",
                 "ACC_UNet", "SwinUNETR", "MambaVisionT", "MambaVisionT2", "MambaVisionS"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        default="CosineAnnealing",
        choices=["CosineAnnealing", "Plateau"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--amp',
        dest='amp',
        action='store_true',
        default=True,
        help='Enable AMP.',
    )
    parser.add_argument(
        '--no-amp',
        dest='amp',
        action='store_false',
        help='Disable AMP.',
    )
    parser.set_defaults(amp=True)

    parser.add_argument(
        '--grad_clip',
        type=Union[None, float],
        default=None,
        help='Whether use grad_clip.',
    )

    parser.add_argument(
        '--use_coords',
        action="store_true",
        default=True,
        help='Whether use positional information.',
    )

    parser.add_argument(
        '--tr_patches_per_img',
        type=int,
        default=16, # 12
        help="Number of train patches per image."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help='DataLoader workers per process. Use -1 to auto-select.',
    )
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=2,
        help='Number of prefetched batches per worker.',
    )
    parser.add_argument(
        '--pin_memory',
        dest='pin_memory',
        action='store_true',
        default=True,
        help='Enable pinned host memory for DataLoader.',
    )
    parser.add_argument(
        '--no-pin_memory',
        dest='pin_memory',
        action='store_false',
        help='Disable pinned host memory for DataLoader.',
    )
    parser.add_argument(
        '--persistent_workers',
        dest='persistent_workers',
        action='store_true',
        default=False,
        help='Keep DataLoader workers alive across epochs.',
    )
    parser.add_argument(
        '--no-persistent_workers',
        dest='persistent_workers',
        action='store_false',
        help='Disable persistent DataLoader workers.',
    )
    parser.set_defaults(pin_memory=True, persistent_workers=False)

    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/data2/datasx/FullHand/NIPS26/RAM-H1200/Segmentation",
        help='The path to data.',
    )

    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default="",
        help='The pretrained weight of the model.',
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default="./ckpts",
        help='Checkpoint root for train mode, or experiment directory/file for test/infer/resume.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training from the latest checkpoint if available.',
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default="",
        help='Explicit checkpoint file or experiment directory to resume from.',
    )

    parser.add_argument(
        '--trial_name',
        type=str,
        default="benchmark_boneseg",
        help='The name of the trial.',
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='Number of epochs.',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Initial lr',
    )

    parser.add_argument(
        '--monitor_mode',
        type=str,
        default="max",
        help='The mode of trainer. ("min" or "max")'
    )

    parser.add_argument(
        '--earlystop_patience',
        type=int,
        default=15,
        help='Patience for early stopping.',
    )

    parser.add_argument(
        '--save_uncertainty_overlay',
        action='store_true',
        default=False,
        help='Whether to save the uncertainty (Test mode only).'
    )

    parser.add_argument(
        '--save_overlay',
        action='store_true',
        default=False,
        help='Whether to save the overlay (Test mode only).'
    )

    parser.add_argument(
        '--save_npz',
        action='store_true',
        default=False,
        help='Whether to save prediction bundles as npz (pred/image/gt when available; Test mode only).'
    )

    parser.add_argument(
        '--save_npy',
        dest='save_npz',
        action='store_true',
        help='Deprecated alias of --save_npz.'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=False,
        help='Whether save csv (Test mode only).',
    )

    parser.add_argument(
        '--save_pred',
        action='store_true',
        default=False,
        help='Whether save pred (Test mode only).',
    )

    parser.add_argument(
        '--save_mask',
        action='store_true',
        default=False,
        help='Whether save mask (Test mode only).',
    )
    parser.add_argument(
        '--launcher',
        type=str,
        default="none",
        choices=["none", "ddp"],
        help='Launch mode. Use "ddp" with torchrun for distributed training.',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for DDP. Usually provided by torchrun.',
    )
    args = parser.parse_args()
    return args


def init_distributed(args):
    args.is_ddp = args.launcher == "ddp"

    if not args.is_ddp:
        args.rank = 0
        args.world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        args.rank = rank
        args.world_size = world_size
        args.local_rank = local_rank

        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    else:
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        if args.local_rank < 0:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(device)

    return device


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

    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = checkpoint_root / f"{args.trial_name}_{args.model.lower()}_{time_str}"
    if is_main_process(args):
        run_dir.mkdir(parents=True, exist_ok=False)
    if args.is_ddp:
        dist.barrier(device_ids=[args.local_rank])

    args.model_save_path = str(run_dir)
    args.checkpoint = str(run_dir)
    if is_main_process(args):
        print(f"Starting new training run at {run_dir}")
    return None

def build_model(args, in_chans):
    if args.model == "Unet":
        return monai.networks.nets.DynUNet(spatial_dims=2, in_channels=in_chans, out_channels=30, kernel_size=[3, 3, 3, 3, 3],
                                          norm_name="batch", strides=[1, 2, 2, 2, 2], upsample_kernel_size=[2, 2, 2, 2, 2],
                                          filters=[32, 64, 128, 256, 512], res_block=True)

    elif args.model == "SegResNet":
        return monai.networks.nets.SegResNet(spatial_dims=2, in_channels=in_chans, out_channels=30, upsample_mode="deconv",
                                             act="LeakyReLU", norm="batch", init_filters=16, blocks_down=(1, 2, 2, 4),
                                             blocks_up=(1, 1, 1),)
    elif args.model == "Unet++":
        model = UnetPlusPlus(spatial_dims=2, in_channels=in_chans, out_channels=30, features=(32, 32, 64, 128, 256, 32))
        backbone = getattr(model, "model", None)
        if backbone is not None and not getattr(backbone, "deep_supervision", False):
            # Non-deep-supervision training only uses final_conv_0_4 in forward.
            # Freeze auxiliary heads to avoid DDP unused-parameter reduction errors.
            for head_name in ("final_conv_0_1", "final_conv_0_2", "final_conv_0_3"):
                head = getattr(backbone, head_name, None)
                if head is None:
                    continue
                for param in head.parameters():
                    param.requires_grad = False
        return model
    elif args.model == "UKAN":
        return UKAN(num_classes=30,  input_channels=3, embed_dims=[128, 160, 256])
    elif args.model == "TransUnet":
        return get_TransUnet_Custom(img_size=args.image_size, n_classes=30)
    elif args.model == "DeepLabV3+":
        return smp.DeepLabV3Plus(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "DeepLabV3":
        return smp.DeepLabV3(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "PSPNet":
        return smp.PSPNet(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "PAN":
        return smp.PAN(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "DPT":
        return smp.DPT(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "SegFormer":
        return smp.Segformer(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "FPN":
        return smp.FPN(in_channels=in_chans, encoder_weights=None, classes=30)
    elif args.model == "UMambaBot":
        return get_UMambaBot(in_channels=in_chans, num_classes=30)
    elif args.model == "UMambaEnc":
        return get_UMambaEnc(in_channels=in_chans, num_classes=30)
    elif args.model == "SwinUMamba":
        return get_SwinUMamba(in_channels=in_chans, num_classes=30)
    elif args.model == "DPMSwinUMamba":
        return get_DPMSwinUMamba(in_channels=in_chans, num_overlap_classes=14, num_classes=30)
    elif args.model == "RefinedSwinUMamba":
        return get_RefinedSwinUMamba(
            in_channels=in_chans,
            num_classes=30,
            base_ckpt="./ckpts/Annotation_swinumamba_202602151753/model_best_nsd.pth",
        )
    elif args.model in {"MambaVisionT", "MambaVisionT2", "MambaVisionS"}:
        variant_map = {
            "MambaVisionT": "mamba_vision_T",
            "MambaVisionT2": "mamba_vision_T2",
            "MambaVisionS": "mamba_vision_S",
        }
        return get_MambaVisionSeg(
            variant=variant_map[args.model],
            in_chans=in_chans,
            num_classes=30,
            image_size=args.image_size,
            pretrained=False,
        )
    elif args.model == "ACC_UNet":
        # This ACC_UNet implementation outputs n_classes + 1 channels when n_classes != 1, we modified the source code to fix it.
        return ACC_UNet(n_channels=in_chans, n_classes=30)
    elif args.model == "SwinUNETR":
        swinunetr_kwargs = dict(
            in_channels=in_chans,
            out_channels=30,
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
        # Seed everything
        seed_everything(args.seed + getattr(args, "rank", 0))

        in_chans = 1
        if args.use_coords:
            in_chans += 2

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
            train_dataset = FullHandPatchDataset(data_root=Path(args.data_path) / "train", annotation_path=Path(args.data_path) / "train" / "_annotations_bone_rle.coco.json",
                                                            mode="train", transform=transform_tr, use_coords=args.use_coords, train_patches_per_image=args.tr_patches_per_img)

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
            val_dataset = FullHandPatchDataset(data_root=Path(args.data_path) / "val", annotation_path=Path(args.data_path) / "val" / "_annotations_bone_rle.coco.json",
                                                      mode="val", transform=transform_val, use_coords=args.use_coords, train_patches_per_image=args.tr_patches_per_img)

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

            criterion = MSEBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
            trainer = FullHandBoneSegTrainer(args, net, train_loader, val_loader, criterion, optimizer, device=device)
            if resume_state is not None:
                trainer.load_training_state(resume_state)
            trainer.fit(args)

        elif args.mode == "test":
            if args.is_ddp:
                raise NotImplementedError("DDP is currently supported only for train mode in main_seg.py.")

            net = net.to(device)
            transform_test = get_transform(split="test", image_size=args.image_size)
            test_dataset = FullHandPatchDataset(data_root=Path(args.data_path) / "test", annotation_path=Path(args.data_path) / "test" / "_annotations_bone_rle.coco.json",
                                              mode="test", transform=transform_test, use_coords=args.use_coords)
            test_loader = get_dataloader(
                test_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                distributed=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            if not (Path(args.checkpoint) / "model_best.pth").exists() and not (Path(args.checkpoint) / "model_best_dice.pth").exists():
                raise KeyError("Test mode is set but checkpoint does not exist.")
            tester = FullHandBoneSegTester(args, net, test_loader, device=device)
            tester.test()

        elif args.mode == "infer":
            if args.is_ddp:
                raise NotImplementedError("DDP is currently supported only for train mode in main_seg.py.")

            net = net.to(device)
            transform_infer = get_transform(split="test", image_size=args.image_size)
            infer_dataset = FullHandPatchDataset(data_root=Path(args.data_path), mode="infer", flip_left_by_name=False,
                                                 transform=transform_infer, use_coords=args.use_coords)
            infer_loader = get_dataloader(
                infer_dataset,
                batch_size=1,
                shuffle=False,
                distributed=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            if not (Path(args.checkpoint) / "model_best_nsd.pth").exists() and not (Path(args.checkpoint) / "model_best_dice.pth").exists():
                raise KeyError("Test mode is set but checkpoint does not exist.")
            inferer = FullHandBoneSegInferencer(args, net, infer_loader, device=device)
            inferer.test()

    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
