from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import show_mask, PatchStitcher
import monai
import pandas as pd
from evaluations.metrics import (
    BESemanticSegmentationMetrics,
    CostAwareDiceMetric,
    FullHandSegmentationMetrics,
    SegmentationMetrics,
    ClassificationMetrics,
    bone_name_dict,
)
from monai.inferers import sliding_window_inference
from sklearn.metrics import ConfusionMatrixDisplay
import time
import gc
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassConfusionMatrix,
)


def _parse_cost_matrix_arg(cost_matrix_str: str, num_classes: int):
    if not cost_matrix_str:
        return [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]

    rows = []
    for row_str in cost_matrix_str.split(";"):
        row = [float(v.strip()) for v in row_str.split(",") if v.strip() != ""]
        rows.append(row)

    if len(rows) != num_classes or any(len(row) != num_classes for row in rows):
        raise ValueError(
            f"cost_matrix must be {num_classes}x{num_classes}, got "
            f"{len(rows)}x{len(rows[0]) if rows else 0}."
        )
    return rows


BE_CREDIT_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _safe_nanmean(arr, axis):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    sums = np.where(valid, arr, 0.0).sum(axis=axis, dtype=float)
    counts = valid.sum(axis=axis)
    out = np.full(np.shape(sums), np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _save_prediction_npz(output_path: Path, pred, image=None, gt=None):
    payload = {"pred": np.asarray(pred)}
    if image is not None:
        payload["image"] = np.asarray(image)
    if gt is not None:
        payload["gt"] = np.asarray(gt)
    np.savez_compressed(output_path, **payload)


def _synchronize_device(device):
    if torch.cuda.is_available():
        device_obj = torch.device(device)
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)


def _time_model_forward(forward_fn, device):
    _synchronize_device(device)
    start_time = time.perf_counter()
    output = forward_fn()
    _synchronize_device(device)
    return output, time.perf_counter() - start_time


def _update_case_inference_times(case_time_records, case_names, batch_elapsed_seconds):
    if not case_names:
        return
    per_item_ms = (batch_elapsed_seconds / len(case_names)) * 1000.0
    for case_name in case_names:
        case_time_records.setdefault(case_name, []).append(per_item_ms)


def _build_case_inference_time_df(case_time_records):
    rows = []
    for case_name, times_ms in case_time_records.items():
        if not times_ms:
            continue
        rows.append(
            {
                "Case": case_name,
                "Mean Inference Time (ms)": float(np.mean(times_ms)),
                "Std Inference Time (ms)": float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else np.nan,
                "Num Samples": int(len(times_ms)),
            }
        )
    rows.sort(key=lambda item: item["Case"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    average_row = pd.DataFrame(
        [
            {
                "Case": "Average",
                "Mean Inference Time (ms)": float(df["Mean Inference Time (ms)"].mean()),
                "Std Inference Time (ms)": float(df["Mean Inference Time (ms)"].std(ddof=1)) if len(df) > 1 else np.nan,
                "Num Samples": int(df["Num Samples"].sum()),
            }
        ]
    )
    return pd.concat([df, average_row], ignore_index=True)


def _print_case_inference_time_summary(case_time_records):
    inference_time_df = _build_case_inference_time_df(case_time_records)
    if inference_time_df.empty:
        return
    avg_time_ms = float(inference_time_df.iloc[-1]["Mean Inference Time (ms)"])
    print(f"Average inference time per case: {avg_time_ms:.2f} ms")


def _save_case_inference_time_summary(output_dir, case_time_records, filename="inference_time.txt"):
    inference_time_df = _build_case_inference_time_df(case_time_records)
    if inference_time_df.empty:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as f:
        avg_time_ms = float(inference_time_df.iloc[-1]["Mean Inference Time (ms)"])
        f.write(f"Average inference time per case: {avg_time_ms:.4f} ms\n")


class FullHandBoneSegTrainer:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, num_classes=30, device="cuda:0"):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.monitor_mode = args.monitor_mode  # "min" 或 "max"
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.device = device
        self.criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
        self.max_epoch = args.max_epoch
        self.num_classes = num_classes
        self.class_thresholds = [2] * self.num_classes
        self.dice_metric = monai.metrics.DiceMetric(reduction="mean")
        self.nsd_metric = monai.metrics.SurfaceDiceMetric(class_thresholds=self.class_thresholds, include_background=True)
        self.scaler = GradScaler() if self.amp else None
        self.earlystop = TwoStageEarlyStopping(patience=args.earlystop_patience, mode="max")
        self.use_nsd = False
        self.switch_threshold = 1.0  # Only use Dice to select model
        self.sigmoid = torch.nn.Sigmoid()
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.is_main_process = self.rank == 0

        if args.scheduler == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01)
        elif args.scheduler == "Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, cooldown=2)
        else:
            self.scheduler = None

        self.start_epoch = 0
        self.best_dice = -np.inf
        self.best_nsd = -np.inf
        self.train_loss_history = []
        self.val_metric_history = []

    def _unwrap_model(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _move_optimizer_state_to_device(self):
        target_device = torch.device(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target_device, non_blocking=True)

    def load_training_state(self, state: Dict):
        self.start_epoch = int(state.get("epoch", -1)) + 1
        metric_type = state.get("metric_type")
        val_metric = state.get("val_metric")

        self.best_dice = float(state.get("best_dice", -np.inf))
        self.best_nsd = float(state.get("best_nsd", -np.inf))
        if np.isneginf(self.best_dice) and metric_type == "dice" and val_metric is not None:
            self.best_dice = float(val_metric)
        if np.isneginf(self.best_nsd) and metric_type == "nsd" and val_metric is not None:
            self.best_nsd = float(val_metric)
        self.use_nsd = bool(state.get("use_nsd", False))
        self.train_loss_history = list(state.get("train_loss_history", []))
        self.val_metric_history = list(state.get("val_metric_history", []))

        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler")
        if self.scaler is not None and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        earlystop_state = state.get("earlystop")
        if earlystop_state is not None:
            self.earlystop.load_state_dict(earlystop_state)

        if self.is_main_process:
            best_dice_str = f"{self.best_dice:.4f}" if np.isfinite(self.best_dice) else str(self.best_dice)
            best_nsd_str = f"{self.best_nsd:.4f}" if np.isfinite(self.best_nsd) else str(self.best_nsd)
            print(
                f"Resumed training state from epoch {self.start_epoch}. "
                f"best_dice={best_dice_str}, best_nsd={best_nsd_str}"
            )
            print(
                "Resumed earlystop state: "
                f"dice={self.earlystop.counter_dice}/{self.earlystop.patience}, "
                f"nsd={self.earlystop.counter_nsd}/{self.earlystop.patience}"
            )

    def _reduce_scalar(self, value: float, average: bool = True) -> float:
        if not self.is_ddp:
            return float(value)

        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= self.world_size
        return float(tensor.item())

    def _set_sampler_epoch(self, loader, epoch: int):
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def fit(self, args):
        train_loss = list(self.train_loss_history)
        val_loss = list(self.val_metric_history)

        if self.start_epoch >= self.max_epoch:
            if self.is_main_process:
                print(
                    f"Checkpoint is already at epoch {self.start_epoch}, "
                    f"which is not less than max_epoch={self.max_epoch}. Nothing to do."
                )
            return

        for epoch in range(self.start_epoch, self.max_epoch):
            self._set_sampler_epoch(self.train_loader, epoch)
            self._set_sampler_epoch(self.val_loader, epoch)

            # device 不一致时移动网络
            if next(self.net.parameters()).device != self.device:
                self.net.to(self.device)

            # ---------------- TRAIN ----------------
            if self.amp:
                epoch_train_loss = self.train_one_epoch_amp(epoch)
            else:
                epoch_train_loss = self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss)

            # ---------------- VALIDATE ----------------
            epoch_val_metric, metric_type = self.validate(epoch)
            val_loss.append(epoch_val_metric)

            # ---------------- EARLY STOP ----------------
            should_stop = self.earlystop(epoch_val_metric, metric_type) if self.is_main_process else False
            if self.is_ddp:
                stop_tensor = torch.tensor(int(should_stop), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            # ---------------- LR SCHEDULER ----------------
            if not should_stop:
                if args.scheduler == "CosineAnnealing":
                    self.scheduler.step()
                elif args.scheduler == "Plateau":
                    self.scheduler.step(epoch_val_metric)

            # ---------------- SAVE BEST CHECKPOINT ----------------
            save_best_dice = False
            save_best_nsd = False
            if metric_type == "dice":
                if epoch_val_metric > self.best_dice:
                    if self.is_main_process:
                        print(f"New best Dice: {self.best_dice:.4f} -> {epoch_val_metric:.4f}")
                    self.best_dice = epoch_val_metric
                    save_best_dice = True
                else:
                    if self.is_main_process:
                        print(f"No Dice improvement: {epoch_val_metric:.4f} (best {self.best_dice:.4f})")

            else:  # NSD
                if epoch_val_metric > self.best_nsd:
                    if self.is_main_process:
                        print(f"New best NSD: {self.best_nsd:.4f} -> {epoch_val_metric:.4f}")
                    self.best_nsd = epoch_val_metric
                    save_best_nsd = True
                else:
                    if self.is_main_process:
                        print(f"No NSD improvement: {epoch_val_metric:.4f} (best {self.best_nsd:.4f})")

            # ---------------- SAVE LATEST ----------------
            ckpt = {
                "model": self._unwrap_model().state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "train_loss": epoch_train_loss,
                "val_metric": epoch_val_metric,
                "metric_type": metric_type,
                "train_loss_history": train_loss,
                "val_metric_history": val_loss,
                "best_dice": self.best_dice,
                "best_nsd": self.best_nsd,
                "use_nsd": self.use_nsd,
                "earlystop": self.earlystop.state_dict(),
            }
            if self.is_main_process:
                torch.save(ckpt, Path(args.model_save_path) / "model_latest.pth")
                if save_best_dice:
                    torch.save(ckpt, Path(args.model_save_path) / "model_best_dice.pth")
                if save_best_nsd:
                    torch.save(ckpt, Path(args.model_save_path) / "model_best_nsd.pth")

            # ---------------- PLOT ----------------
            if self.is_main_process:
                self.plot(args, train_loss, val_loss)

            if should_stop:
                if self.is_main_process:
                    print(f"Early stopping triggered on {metric_type.upper()} metric.")
                break
            gc.collect()

    def train_one_epoch_amp(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader, disable=not self.is_main_process)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            # Avoid non-binary value caused by resize
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            with autocast():
                pred = self.net(img)
                loss = self.criterion(pred, gt)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            if self.is_main_process:
                pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return self._reduce_scalar(avg_loss, average=True)

    def train_one_epoch(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader, disable=not self.is_main_process)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            # Avoid non-binary value caused by resize
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            pred = self.net(img)
            loss = self.criterion(pred, gt)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.grad_clip))
            self.optimizer.step()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            if self.is_main_process:
                pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return self._reduce_scalar(avg_loss, average=True)

    def validate(self, epoch):
        self.net.eval()
        model_for_eval = self._unwrap_model()
        model_for_eval.eval()
        if self.is_ddp and not self.is_main_process:
            metric_tensor = torch.zeros(2, device=self.device)
            dist.broadcast(metric_tensor, src=0)
            self.use_nsd = bool(metric_tensor[1].item())
            metric_type = "nsd" if self.use_nsd else "dice"
            return float(metric_tensor[0].item()), metric_type

        pbar = tqdm(self.val_loader, disable=not self.is_main_process)

        avg_loss = 0.0
        dice_scores = []
        nsd_scores = []

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"].to(self.device)  # (B,C,H,W)
                gt = (batch["gt"].to(self.device) > 0.5).float()

                # -------- sliding window inference --------
                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=(self.args.image_size, self.args.image_size),
                    sw_batch_size=4,
                    predictor=model_for_eval,
                    overlap=0.5,
                    mode="gaussian"  # IMPORTANT
                )

                # -------- loss --------
                loss = self.criterion(logits, gt)
                avg_loss += loss.item()

                # -------- prediction --------
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()

                # -------- Dice --------
                dice_tensor = monai.metrics.compute_dice(
                    pred,
                    gt,
                    include_background=True
                )
                dice_val = float(torch.nanmean(dice_tensor).item())
                dice_scores.append(dice_val)

                # -------- NSD --------
                if self.use_nsd:
                    nsd_tensor = monai.metrics.compute_surface_dice(
                        pred,
                        gt,
                        class_thresholds=self.class_thresholds,
                        include_background=True
                    )
                    nsd_val = float(torch.nanmean(nsd_tensor).item())
                    nsd_scores.append(nsd_val)

                # -------- log --------
                if not self.use_nsd:
                    metric_display = f"avg DSC: {np.mean(dice_scores):.4f}"
                else:
                    metric_display = f"avg NSD: {np.mean(nsd_scores):.4f}"

                if self.is_main_process:
                    pbar.set_description(
                        f"Epoch {epoch} Validating loss: {loss.item():.4f}, {metric_display}"
                    )

        avg_loss /= len(self.val_loader)
        epoch_dice = float(np.mean(dice_scores))

        # -------- metric switch --------
        if not self.use_nsd:
            if self.is_main_process:
                print(f"[Validate] Epoch {epoch} Loss={avg_loss:.4f}, Dice={epoch_dice:.4f}")

            if epoch_dice >= self.switch_threshold:
                if self.is_main_process:
                    print(
                        f"[INFO] Epoch {epoch}: Dice {epoch_dice:.4f} >= {self.switch_threshold}, switching to NSD"
                    )
                self.use_nsd = True

            if self.is_ddp:
                metric_tensor = torch.tensor([epoch_dice, int(self.use_nsd)], device=self.device, dtype=torch.float32)
                dist.broadcast(metric_tensor, src=0)

            return epoch_dice, "dice"

        else:
            epoch_nsd = float(np.mean(nsd_scores))
            if self.is_main_process:
                print(f"[Validate] Epoch {epoch} Loss={avg_loss:.4f}, NSD={epoch_nsd:.4f}")
            if self.is_ddp:
                metric_tensor = torch.tensor([epoch_nsd, 1], device=self.device, dtype=torch.float32)
                dist.broadcast(metric_tensor, src=0)
            return epoch_nsd, "nsd"

    def plot(self, args, train_loss, val_loss):
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Metric')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class BESegTrainer(FullHandBoneSegTrainer):
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, num_classes=1, device="cuda:0"):
        super().__init__(
            args=args,
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_classes=num_classes,
            device=device,
        )
        self.earlystop = EarlyStopping(patience=args.earlystop_patience, mode="max")
        self.best_metric = -np.inf
        self.loss_name = getattr(args, "loss", "creditaware")
        self.metric_name = "positive_dice"
        self.use_nsd = False

    def load_training_state(self, state: Dict):
        self.start_epoch = int(state.get("epoch", -1)) + 1
        val_metric = state.get("val_metric")

        self.best_metric = float(state.get("best_metric", -np.inf))
        if np.isneginf(self.best_metric) and val_metric is not None:
            self.best_metric = float(val_metric)

        self.train_loss_history = list(state.get("train_loss_history", []))
        self.val_metric_history = list(state.get("val_metric_history", []))

        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler")
        if self.scaler is not None and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        earlystop_state = state.get("earlystop")
        if earlystop_state is not None:
            self.earlystop.load_state_dict(earlystop_state)

        if self.is_main_process:
            best_metric_str = f"{self.best_metric:.4f}" if np.isfinite(self.best_metric) else str(self.best_metric)
            print(
                f"Resumed training state from epoch {self.start_epoch}. "
                f"best_{self.metric_name}={best_metric_str}"
            )
            print(
                "Resumed earlystop state: "
                f"counter={self.earlystop.counter}/{self.earlystop.patience}"
            )

    def validate(self, epoch):
        self.net.eval()
        model_for_eval = self._unwrap_model()
        model_for_eval.eval()

        def _safe_nanmean(values, default=0.0):
            arr = np.asarray(values, dtype=float)
            finite = np.isfinite(arr)
            if finite.any():
                return float(arr[finite].mean())
            return float(default)

        if self.is_ddp and not self.is_main_process:
            metric_tensor = torch.zeros(1, device=self.device)
            dist.broadcast(metric_tensor, src=0)
            return float(metric_tensor[0].item()), self.metric_name

        pbar = tqdm(self.val_loader, disable=not self.is_main_process)
        avg_loss = 0.0
        dice_scores = []
        positive_dice_scores = []

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"].to(self.device)
                gt = (batch["gt"].to(self.device) > 0.5).float()

                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=(self.args.image_size, self.args.image_size),
                    sw_batch_size=4,
                    predictor=model_for_eval,
                    overlap=0.5,
                    mode="gaussian",
                )

                loss = self.criterion(logits, gt)
                avg_loss += loss.item()
                probs = torch.softmax(logits, dim=1) if self.num_classes > 1 else torch.sigmoid(logits)
                if self.num_classes > 1:
                    pred_idx = torch.argmax(probs, dim=1)
                    pred = torch.nn.functional.one_hot(
                        pred_idx, num_classes=probs.shape[1]
                    ).permute(0, 3, 1, 2).float()
                    pred_fg_mass = pred[:, 1:, ...].sum(dim=(1, 2, 3))
                    gt_fg_mass = gt[:, 1:, ...].sum(dim=(1, 2, 3))
                else:
                    pred = (probs > 0.5).float()
                    pred_fg_mass = pred.sum(dim=(1, 2, 3))
                    gt_fg_mass = gt.sum(dim=(1, 2, 3))

                dice_tensor = monai.metrics.compute_dice(pred, gt, include_background=False)
                if dice_tensor.ndim == 1:
                    dice_tensor = dice_tensor.unsqueeze(1)

                finite_mask = torch.isfinite(dice_tensor)
                for sample_idx in range(dice_tensor.shape[0]):
                    sample_finite_mask = finite_mask[sample_idx]
                    if sample_finite_mask.any():
                        sample_dice = float(dice_tensor[sample_idx][sample_finite_mask].mean().item())
                    else:
                        pred_fg = float(pred_fg_mass[sample_idx].item())
                        gt_fg = float(gt_fg_mass[sample_idx].item())
                        sample_dice = 1.0 if pred_fg == 0.0 and gt_fg == 0.0 else 0.0

                    dice_scores.append(sample_dice)
                    if gt_fg_mass[sample_idx].item() > 0:
                        positive_dice_scores.append(sample_dice)

                if self.is_main_process:
                    pbar.set_description(
                        f"Epoch {epoch} Validating loss: {loss.item():.4f}, "
                        f"avg Dice: {_safe_nanmean(dice_scores):.4f}, "
                        f"avg PositiveDice: {_safe_nanmean(positive_dice_scores):.4f}"
                    )

        avg_loss /= len(self.val_loader)
        epoch_dice = _safe_nanmean(dice_scores)
        epoch_positive_dice = _safe_nanmean(positive_dice_scores)
        epoch_metric = epoch_positive_dice

        if self.is_main_process:
            print(
                f"[Validate] Epoch {epoch} Loss={avg_loss:.4f}, "
                f"Dice={epoch_dice:.4f}, "
                f"PositiveDice={epoch_positive_dice:.4f}, "
                f"Monitor={self.metric_name}:{epoch_metric:.4f}"
            )

        if self.is_ddp:
            metric_tensor = torch.tensor([epoch_metric], device=self.device, dtype=torch.float32)
            dist.broadcast(metric_tensor, src=0)

        return epoch_metric, self.metric_name

    def fit(self, args):
        train_loss = list(self.train_loss_history)
        val_loss = list(self.val_metric_history)

        if self.start_epoch >= self.max_epoch:
            if self.is_main_process:
                print(
                    f"Checkpoint is already at epoch {self.start_epoch}, "
                    f"which is not less than max_epoch={self.max_epoch}. Nothing to do."
                )
            return

        for epoch in range(self.start_epoch, self.max_epoch):
            self._set_sampler_epoch(self.train_loader, epoch)
            self._set_sampler_epoch(self.val_loader, epoch)

            if next(self.net.parameters()).device != self.device:
                self.net.to(self.device)

            if self.amp:
                epoch_train_loss = self.train_one_epoch_amp(epoch)
            else:
                epoch_train_loss = self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss)

            epoch_val_metric, metric_type = self.validate(epoch)
            val_loss.append(epoch_val_metric)

            should_stop = self.earlystop(epoch_val_metric) if self.is_main_process else False
            if self.is_ddp:
                stop_tensor = torch.tensor(int(should_stop), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if not should_stop:
                if args.scheduler == "CosineAnnealing":
                    self.scheduler.step()
                elif args.scheduler == "Plateau":
                    self.scheduler.step(epoch_val_metric)

            save_best_metric = False
            if epoch_val_metric > self.best_metric:
                if self.is_main_process:
                    print(f"New best {metric_type.upper()}: {self.best_metric:.4f} -> {epoch_val_metric:.4f}")
                self.best_metric = epoch_val_metric
                save_best_metric = True
            else:
                if self.is_main_process:
                    print(f"No {metric_type.upper()} improvement: {epoch_val_metric:.4f} (best {self.best_metric:.4f})")

            ckpt = {
                "model": self._unwrap_model().state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "train_loss": epoch_train_loss,
                "val_metric": epoch_val_metric,
                "metric_type": metric_type,
                "train_loss_history": train_loss,
                "val_metric_history": val_loss,
                "best_metric": self.best_metric,
                "earlystop": self.earlystop.state_dict(),
            }
            if self.is_main_process:
                torch.save(ckpt, Path(args.model_save_path) / "model_latest.pth")
                if save_best_metric:
                    torch.save(ckpt, Path(args.model_save_path) / "model_best.pth")

            if self.is_main_process:
                self.plot(args, train_loss, val_loss)

            if should_stop:
                if self.is_main_process:
                    print(f"Early stopping triggered on {metric_type.upper()} metric.")
                break
            gc.collect()


class LegacyBoneSegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.save_mask = False
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        self.metrics = SegmentationMetrics(num_classes=14)  # TODO: modify according to the latest version
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        total_infer_time = 0
        total_items = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                start_time = time.time()  # ⏱️ Start timing
                pred = self.net(img)
                end_time = time.time()  # ⏱️ End timing
                infer_time = end_time - start_time
                total_infer_time += infer_time
                total_items += img.shape[0]  # 批量大小
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                self.metrics.update_metrics(pred_bin, gt, batch["fname"][0])
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.save_overlay:
                    self.create_overlay(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
                if self.save_pred:
                    self.create_pred(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
                    self.create_overlay_single(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
                if self.save_mask:
                    self.create_mask(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
        if self.save_csv:
            self.create_csv(self.args)
        metrics_dict = self.metrics.get_metrics()
        dsc_reduced = metrics_dict["dsc"].mean()
        print("Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict["nsd"].mean()
        print("Mean NSD: ", nsd_reduced)

        avg_infer_time = total_infer_time / total_items
        print(f"Average inference time per item: {avg_infer_time * 1000:.2f} ms")

    def create_overlay(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = self.sigmoid(pred.detach())
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            # color = np.random.rand(3)
            # seg = torch.sigmoid(pred_mask[0][i]).cpu().numpy()
            # seg[seg > 0.5] = 1
            # seg[seg <= 0.5] = 0
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(self.colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '.pdf'), dpi=600)
        plt.close()

    def create_pred(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "pred"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_overlay_single(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay_single"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_mask(self, args, image, pred, mask, fname):
        for num, name in enumerate(fname):
            save_path = Path(args.checkpoint) / "mask_single" / name
            if not save_path.exists():
                save_path.mkdir(parents=True)

            pred_mask_bin = pred.detach()
            pred_mask_bin[pred_mask_bin > 0.5] = 1
            pred_mask_bin[pred_mask_bin <= 0.5] = 0

            plt.imsave(save_path / f"img.pdf", image[num][0].cpu().numpy(), cmap="gray")
            for i in range(14):
                plt.imsave(save_path / f"pred_{i}.pdf", pred_mask_bin[num][i].cpu().numpy(), cmap="gray")
                plt.imsave(save_path / f"gt_{i}.pdf", mask[num][i].cpu().numpy(), cmap="gray")


    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        overlap_dsc_mean_df = pd.DataFrame(metrics_dict["overlap_dsc"], columns=["Mean Overlap DSC"])
        overlap_dsc_df = pd.DataFrame(metrics_dict["overlap_dsc_per_pair"], columns=[f"Overlap DSC {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_nsd_mean_df = pd.DataFrame(metrics_dict["overlap_nsd"], columns=["Mean Overlap NSD"])
        overlap_nsd_df = pd.DataFrame(metrics_dict["overlap_nsd_per_pair"], columns=[f"Overlap NSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_voe_mean_df = pd.DataFrame(metrics_dict["overlap_voe"], columns=["Mean Overlap VOE"])
        overlap_voe_df = pd.DataFrame(metrics_dict["overlap_voe_per_pair"], columns=[f"Overlap VOE {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_msd_mean_df = pd.DataFrame(metrics_dict["overlap_msd"], columns=["Mean Overlap MSD"])
        overlap_msd_df = pd.DataFrame(metrics_dict["overlap_msd_per_pair"], columns=[f"Overlap MSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_ravd_mean_df = pd.DataFrame(metrics_dict["overlap_ravd"], columns=["Mean Overlap RAVD"])
        overlap_ravd_df = pd.DataFrame(metrics_dict["overlap_ravd_per_pair"], columns=[f"Overlap RAVD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])

        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])

        # acc_df = pd.DataFrame(metrics_dict["accuracy_pc"], columns=[f"Accuracy {bone_name_dict[i]}" for i in range(num_classes)])
        # acc_mean_df = pd.DataFrame(metrics_dict["accuracy"], columns=["Mean Accuracy"])
        # precision_df = pd.DataFrame(metrics_dict["precision_pc"], columns=[f"Precision {bone_name_dict[i]}" for i in range(num_classes)])
        # precision_mean_df = pd.DataFrame(metrics_dict["precision"], columns=["Mean Precision"])
        # recall_df = pd.DataFrame(metrics_dict["recall_pc"], columns=[f"Recall {bone_name_dict[i]}" for i in range(num_classes)])
        # recall_mean_df = pd.DataFrame(metrics_dict["recall"], columns=["Mean Recall"])
        # f1_df = pd.DataFrame(metrics_dict["f1score_pc"], columns=[f"F1-score {bone_name_dict[i]}" for i in range(num_classes)])
        # f1_mean_df = pd.DataFrame(metrics_dict["f1score"], columns=["Mean F1-score"])
        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
            [fname_df, overlap_dsc_df, overlap_dsc_mean_df, overlap_nsd_df, overlap_nsd_mean_df, overlap_voe_df,
             overlap_voe_mean_df, overlap_msd_df, overlap_msd_mean_df, overlap_ravd_df, overlap_ravd_mean_df, dsc_df,
             dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df, ravd_df, ravd_mean_df], axis=1)
        # metric_df = pd.concat(
        #     [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, hd95_df, hd95_mean_df, acc_df, acc_mean_df,
        #      precision_df, precision_mean_df, recall_df, recall_mean_df, f1_df,f1_mean_df], axis=1)
        # 仅在有限值上求均值：跳过 inf 和 NaN
        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        finite_means = _safe_nanmean(vals, axis=0)
        column_means = pd.Series(finite_means, index=metric_df.columns[1:])
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class FullHandBoneSegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        if (Path(args.checkpoint) / "model_best_nsd.pth").exists():
            self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best_nsd.pth"))["model"])
        else:
            self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best_dice.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_npz = getattr(args, "save_npz", getattr(args, "save_npy", False))
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.sigmoid = torch.nn.Sigmoid()
        self.save_mask = False
        self.metrics = FullHandSegmentationMetrics(num_classes=30)
        self.bone_name_dict = self.test_loader.dataset.channel_to_name
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145],
            # ---- 下面是我补的 10 个 ----
            [0.6523, 0.2197, 0.9011],
            [0.2457, 0.8125, 0.3928],
            [0.9124, 0.4632, 0.7821],
            [0.4912, 0.6399, 0.1223],
            [0.2105, 0.9311, 0.8420],
            [0.9734, 0.5843, 0.1028],
            [0.4188, 0.2911, 0.7555],
            [0.6876, 0.8352, 0.2764],
            [0.5579, 0.1447, 0.5348],
            [0.8662, 0.3569, 0.2287],
        ]
        # self.metrics = SegmentationMetrics(num_classes=args.num_classes)
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def _overlay_style(self, channel_idx):
        if self.bone_name_dict[channel_idx] == "SoftTissue":
            return np.array([0.98, 0.84, 0.60]), 0.14
        return np.array(self.colors[channel_idx]), 0.5

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        case_inference_times = {}

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"].to(self.device)  # (B,C,H,W)
                gt = batch["gt"]  # (B,C,H,W)
                fname = batch["fname"]

                # -------- sliding window inference --------
                logits, infer_time = _time_model_forward(
                    lambda: sliding_window_inference(
                        inputs=img,
                        roi_size=(self.args.image_size, self.args.image_size),
                        sw_batch_size=4,
                        predictor=self.net,
                        overlap=0.5,
                        mode="gaussian"
                    ),
                    self.device,
                )
                _update_case_inference_times(case_inference_times, list(fname), infer_time)

                probs = self.sigmoid(logits)
                pred = (probs > 0.5).float().cpu().numpy()
                gt = gt.numpy()

                for b in range(img.shape[0]):
                    fused_pred = pred[b]  # (C,H,W)
                    fused_gt = gt[b]
                    name = fname[b]

                    # ---- load raw image ----
                    img_path = self.test_loader.dataset.data_root / name
                    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    flip = self.test_loader.dataset.flip_left_by_name and (name[-4] == "L")
                    if flip:
                        raw_img = cv2.flip(raw_img, 1)
                        fused_pred = fused_pred[:, :, ::-1]
                        fused_gt = fused_gt[:, :, ::-1]

                    # ---- save ----
                    if self.save_overlay:
                        self.create_overlay(self.args, raw_img, fused_pred, fused_gt, name)

                    if self.save_pred:
                        self.create_overlay_separate(self.args, raw_img, fused_pred, fused_gt, name)

                    if self.save_npz:
                        self.create_npz(self.args, raw_img, fused_pred, fused_gt, name)

                    if self.save_csv:
                        self.metrics.update_metrics(
                            fused_pred[np.newaxis, ...],
                            fused_gt[np.newaxis, ...],
                            name
                        )

        _print_case_inference_time_summary(case_inference_times)
        _save_case_inference_time_summary(self.args.checkpoint, case_inference_times)

        if self.save_csv:
            self.create_csv(self.args)

    def create_npz(self, args, image, pred, gt, fname):
        save_path = Path(args.checkpoint) / "npz"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        _save_prediction_npz(
            save_path / (fname[:-4] + '.npz'),
            pred_mask_bin.astype(np.uint8),
            image=image,
            gt=gt.astype(np.uint8),
        )

    def create_overlay(self, args, image, pred, gt, fname):
        save_path = Path(args.checkpoint) / "overlay"
        save_path.mkdir(parents=True, exist_ok=True)

        # binarize
        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(22, 6))

        titles = ["Image", "Prediction", "Ground Truth"]
        for i in range(3):
            ax[i].imshow(image, cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        # --- Prediction ---
        for c in range(pred.shape[0]):
            color, alpha = self._overlay_style(c)
            show_mask(pred[c], ax[1], mask_color=color, alpha=alpha)

        # --- Ground Truth ---
        for c in range(gt.shape[0]):
            color, alpha = self._overlay_style(c)
            show_mask(gt[c], ax[2], mask_color=color, alpha=alpha)

        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + ".pdf"), dpi=600)
        plt.close()

    def create_overlay_separate(self, args, image, pred, gt, fname):
        save_path = Path(args.checkpoint) / "overlay_single"
        save_path.mkdir(parents=True, exist_ok=True)

        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
        ax_pred.imshow(image, cmap="gray")
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        for c in range(pred.shape[0]):
            color, alpha = self._overlay_style(c)
            show_mask(pred[c], ax_pred, mask_color=color, alpha=alpha)
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + "_pred.pdf"), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
        ax_gt.imshow(image, cmap="gray")
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        for c in range(gt.shape[0]):
            color, alpha = self._overlay_style(c)
            show_mask(gt[c], ax_gt, mask_color=color, alpha=alpha)
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + "_gt.pdf"), dpi=600)
        plt.close()

    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        overlap_pairs = metrics_dict["overlap_pairs"]
        overlap_pair_names = [
            f"{self.bone_name_dict[i]}&{self.bone_name_dict[j]}"
            for i, j in overlap_pairs
        ]

        dsc_pc = np.asarray(metrics_dict["dsc_pc"], dtype=float)
        nsd_pc = np.asarray(metrics_dict["nsd_pc"], dtype=float)
        voe_pc = np.asarray(metrics_dict["voe_pc"], dtype=float)
        msd_pc = np.asarray(metrics_dict["msd_pc"], dtype=float)
        ravd_pc = np.asarray(metrics_dict["ravd_pc"], dtype=float)
        overlap_dsc_pc = np.asarray(metrics_dict["overlap_dsc_per_pair"], dtype=float)
        overlap_nsd_pc = np.asarray(metrics_dict["overlap_nsd_per_pair"], dtype=float)
        overlap_voe_pc = np.asarray(metrics_dict["overlap_voe_per_pair"], dtype=float)
        overlap_msd_pc = np.asarray(metrics_dict["overlap_msd_per_pair"], dtype=float)
        overlap_ravd_pc = np.asarray(metrics_dict["overlap_ravd_per_pair"], dtype=float)

        dsc_df = pd.DataFrame(dsc_pc, columns=[f"DSC {self.bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(_safe_nanmean(dsc_pc, axis=1), columns=["Mean DSC"])
        nsd_df = pd.DataFrame(nsd_pc, columns=[f"NSD {self.bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(_safe_nanmean(nsd_pc, axis=1), columns=["Mean NSD"])
        voe_df = pd.DataFrame(voe_pc, columns=[f"VOE {self.bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(_safe_nanmean(voe_pc, axis=1), columns=["Mean VOE"])
        msd_df = pd.DataFrame(msd_pc, columns=[f"MSD {self.bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(_safe_nanmean(msd_pc, axis=1), columns=["Mean MSD"])
        ravd_df = pd.DataFrame(ravd_pc, columns=[f"RAVD {self.bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(_safe_nanmean(ravd_pc, axis=1), columns=["Mean RAVD"])
        overlap_dsc_df = pd.DataFrame(overlap_dsc_pc, columns=[f"Overlap DSC {name}" for name in overlap_pair_names])
        overlap_dsc_mean_df = pd.DataFrame(_safe_nanmean(overlap_dsc_pc, axis=1), columns=["Mean Overlap DSC"])
        overlap_nsd_df = pd.DataFrame(overlap_nsd_pc, columns=[f"Overlap NSD {name}" for name in overlap_pair_names])
        overlap_nsd_mean_df = pd.DataFrame(_safe_nanmean(overlap_nsd_pc, axis=1), columns=["Mean Overlap NSD"])
        overlap_voe_df = pd.DataFrame(overlap_voe_pc, columns=[f"Overlap VOE {name}" for name in overlap_pair_names])
        overlap_voe_mean_df = pd.DataFrame(_safe_nanmean(overlap_voe_pc, axis=1), columns=["Mean Overlap VOE"])
        overlap_msd_df = pd.DataFrame(overlap_msd_pc, columns=[f"Overlap MSD {name}" for name in overlap_pair_names])
        overlap_msd_mean_df = pd.DataFrame(_safe_nanmean(overlap_msd_pc, axis=1), columns=["Mean Overlap MSD"])
        overlap_ravd_df = pd.DataFrame(overlap_ravd_pc, columns=[f"Overlap RAVD {name}" for name in overlap_pair_names])
        overlap_ravd_mean_df = pd.DataFrame(_safe_nanmean(overlap_ravd_pc, axis=1), columns=["Mean Overlap RAVD"])

        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
            [
                fname_df,
                overlap_dsc_df,
                overlap_dsc_mean_df,
                overlap_nsd_df,
                overlap_nsd_mean_df,
                overlap_voe_df,
                overlap_voe_mean_df,
                overlap_msd_df,
                overlap_msd_mean_df,
                overlap_ravd_df,
                overlap_ravd_mean_df,
                dsc_df,
                dsc_mean_df,
                nsd_df,
                nsd_mean_df,
                voe_df,
                voe_mean_df,
                msd_df,
                msd_mean_df,
                ravd_df,
                ravd_mean_df,
            ],
            axis=1,
        )
        # 仅在有限值上求均值：跳过 inf 和 NaN
        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        finite_means = _safe_nanmean(vals, axis=0)
        column_means = pd.Series(finite_means, index=metric_df.columns[1:])
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class FullHandBoneSegInferencer:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        if (Path(args.checkpoint) / "model_best_nsd.pth").exists():
            self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best_nsd.pth"))["model"])
        elif (Path(args.checkpoint) / "model_best_dice.pth").exists():
            self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best_dice.pth"))["model"])
        else:
            self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_uncertainty_overlay = args.save_uncertainty_overlay
        self.save_overlay = args.save_overlay
        self.save_npz = getattr(args, "save_npz", getattr(args, "save_npy", False))
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.sigmoid = torch.nn.Sigmoid()
        self.save_mask = False
        self.bone_name_dict = self.test_loader.dataset.channel_to_name
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145],
            # ---- 下面是我补的 10 个 ----
            [0.6523, 0.2197, 0.9011],
            [0.2457, 0.8125, 0.3928],
            [0.9124, 0.4632, 0.7821],
            [0.4912, 0.6399, 0.1223],
            [0.2105, 0.9311, 0.8420],
            [0.9734, 0.5843, 0.1028],
            [0.4188, 0.2911, 0.7555],
            [0.6876, 0.8352, 0.2764],
            [0.5579, 0.1447, 0.5348],
            [0.8662, 0.3569, 0.2287],
        ]
        # self.metrics = SegmentationMetrics(num_classes=args.num_classes)
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def _overlay_style(self, channel_idx):
        if self.bone_name_dict[channel_idx] == "SoftTissue":
            return np.array([0.98, 0.84, 0.60]), 0.14
        return np.array(self.colors[channel_idx]), 0.5

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"].to(self.device)  # (B,C,H,W)
                fname = batch["fname"]

                # -------- sliding window --------
                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=(self.args.patch_h, self.args.patch_w),
                    sw_batch_size=4,
                    predictor=self.net,
                    overlap=0.5,
                    mode="gaussian"
                )

                probs = self.sigmoid(logits).cpu().numpy()

                for b in range(img.shape[0]):
                    fused_prob = probs[b]  # (C,H,W)
                    fused_mask = (fused_prob > 0.5).astype(np.uint8)
                    name = fname[b]

                    # ---- load raw image ----
                    img_path = self.test_loader.dataset.data_root / name
                    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    flip = self.test_loader.dataset.flip_left_by_name and (name[-4] == "L")
                    if flip:
                        raw_img = cv2.flip(raw_img, 1)
                        fused_prob = fused_prob[:, :, ::-1]
                        fused_mask = fused_mask[:, :, ::-1]

                    # ---- uncertainty ----
                    if self.save_uncertainty_overlay:
                        self.create_uncertainty_overlay(self.args, raw_img, fused_prob, name)

                    if self.save_overlay:
                        self.create_overlay(self.args, raw_img, fused_mask, name)

                    if self.save_npz:
                        self.create_npz(self.args, raw_img, fused_mask, name)

    def create_npz(self, args, image, pred, fname):
        save_path = Path(args.checkpoint) / "npz"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        _save_prediction_npz(
            save_path / (fname[:-4] + '.npz'),
            pred_mask_bin.astype(np.uint8),
            image=image,
        )

    def create_overlay(self, args, image, pred, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred[np.newaxis, ...]
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(image, 'gray')
        ax[1].imshow(image, 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i]
            color, alpha = self._overlay_style(i)
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=color, alpha=alpha)
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + '.pdf'), dpi=600)
        plt.close()

    def create_uncertainty_overlay(self, args, image, prob, fname):
        save_root = Path(args.checkpoint) / "uncertainty_overlay" / fname[:-4]
        save_root.mkdir(parents=True, exist_ok=True)

        # tensor -> numpy
        if isinstance(prob, torch.Tensor):
            prob = prob.detach().cpu().numpy()

        # distance-to-0.5 uncertainty
        uncertainty = 1.0 - 2.0 * np.abs(prob - 0.5)

        num_channels = uncertainty.shape[0]

        for i in range(num_channels):
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))

            # 原图
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Image")
            ax[0].axis("off")

            # probability
            ax[1].imshow(image, cmap="gray")
            im1 = ax[1].imshow(prob[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[1].set_title(f"Prob Channel {i}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            # uncertainty
            ax[2].imshow(image, cmap="gray")
            im2 = ax[2].imshow(uncertainty[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[2].set_title(f"Uncertainty Channel {i}")
            ax[2].axis("off")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            plt.tight_layout()

            plt.savefig(
                save_root / f"channel_{i}.png",
                dpi=300
            )

            plt.close()


class BEPatchSegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        checkpoint_candidates = [
            Path(args.checkpoint) / "model_best.pth",
            Path(args.checkpoint) / "model_latest.pth",
        ]
        checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No BE checkpoint found under {args.checkpoint}")
        self.net.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_uncertainty_overlay = getattr(args, "save_uncertainty_overlay", False)
        self.save_overlay = args.save_overlay
        self.save_npz = getattr(args, "save_npz", getattr(args, "save_npy", False))
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.softmax = torch.nn.Softmax(dim=1)
        self.metrics = BESemanticSegmentationMetrics(
            num_classes=len(self.test_loader.dataset.channel_to_name),
            credit_matrix=[row[:len(self.test_loader.dataset.channel_to_name)] for row in BE_CREDIT_MATRIX[:len(self.test_loader.dataset.channel_to_name)]],
        )
        self.channel_to_name = self.test_loader.dataset.channel_to_name
        self.colors = [
            np.array([0.931, 0.341, 0.215]),
            np.array([0.985, 0.725, 0.188]),
            np.array([0.176, 0.533, 0.855]),
        ]
        self.alpha = 0.35

        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    @staticmethod
    def _should_flip(dataset, fname):
        if not getattr(dataset, "flip_left_by_name", False):
            return False
        return Path(fname).stem.endswith("_L")

    def _foreground_channel_indices(self):
        if 0 in self.channel_to_name and self.channel_to_name[0].lower() == "background":
            return list(range(1, len(self.channel_to_name)))
        return list(range(len(self.channel_to_name)))

    def _foreground_channel_names(self):
        return [self.channel_to_name[idx] for idx in self._foreground_channel_indices()]

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        case_inference_times = {}

        with torch.no_grad():
            for _, batch in enumerate(pbar):
                img = batch["img"].to(self.device)
                gt = batch["gt"].cpu().numpy()
                fname = batch["fname"]

                logits, infer_time = _time_model_forward(
                    lambda: sliding_window_inference(
                        inputs=img,
                        roi_size=(self.args.image_size, self.args.image_size),
                        sw_batch_size=4,
                        predictor=self.net,
                        overlap=0.5,
                        mode="gaussian",
                    ),
                    self.device,
                )
                _update_case_inference_times(case_inference_times, list(fname), infer_time)

                probs = self.softmax(logits).cpu()
                pred_idx = torch.argmax(probs, dim=1)
                pred = torch.nn.functional.one_hot(
                    pred_idx, num_classes=probs.shape[1]
                ).permute(0, 3, 1, 2).numpy().astype(np.uint8)
                probs = probs.numpy()

                for b in range(img.shape[0]):
                    fused_prob = probs[b]
                    fused_pred = pred[b]
                    fused_gt = gt[b]
                    name = fname[b]

                    img_path = self.test_loader.dataset.data_root / name
                    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    if self._should_flip(self.test_loader.dataset, name):
                        raw_img = cv2.flip(raw_img, 1)
                        fused_prob = fused_prob[:, :, ::-1]
                        fused_pred = fused_pred[:, :, ::-1]
                        fused_gt = fused_gt[:, :, ::-1]

                    if self.save_uncertainty_overlay:
                        self.create_uncertainty_overlay(raw_img, fused_prob, name)
                    if self.save_overlay:
                        self.create_overlay(raw_img, fused_pred, fused_gt, name)
                    if self.save_pred:
                        self.create_overlay_separate(raw_img, fused_pred, fused_gt, name)
                    if self.save_npz:
                        self.create_npz(raw_img, fused_pred, fused_gt, name)
                    if self.save_csv:
                        self.metrics.update_metrics(
                            fused_pred[np.newaxis, ...],
                            fused_gt[np.newaxis, ...],
                            name,
                        )

        _print_case_inference_time_summary(case_inference_times)
        _save_case_inference_time_summary(self.args.checkpoint, case_inference_times)

        if self.save_csv:
            self.create_csv()

    def create_npz(self, image, pred, gt, fname):
        save_path = Path(self.args.checkpoint) / "npz"
        save_path.mkdir(parents=True, exist_ok=True)
        _save_prediction_npz(
            save_path / (fname[:-4] + ".npz"),
            pred.astype(np.uint8),
            image=image,
            gt=gt.astype(np.uint8),
        )

    def create_overlay(self, image, pred, gt, fname):
        save_path = Path(self.args.checkpoint) / "overlay"
        save_path.mkdir(parents=True, exist_ok=True)

        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(22, 6))
        titles = ["Image", "Prediction", "Ground Truth"]
        for i in range(3):
            ax[i].imshow(image, cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            show_mask(pred[channel_idx], ax[1], mask_color=self.colors[color_idx], alpha=self.alpha)
            show_mask(gt[channel_idx], ax[2], mask_color=self.colors[color_idx], alpha=self.alpha)

        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + ".pdf"), dpi=600)
        plt.close()

    def create_overlay_separate(self, image, pred, gt, fname):
        save_path = Path(self.args.checkpoint) / "overlay_single"
        save_path.mkdir(parents=True, exist_ok=True)

        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)

        fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
        ax_pred.imshow(image, cmap="gray")
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            show_mask(pred[channel_idx], ax_pred, mask_color=self.colors[color_idx], alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + "_pred.pdf"), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
        ax_gt.imshow(image, cmap="gray")
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            show_mask(gt[channel_idx], ax_gt, mask_color=self.colors[color_idx], alpha=self.alpha)
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + "_gt.pdf"), dpi=600)
        plt.close()

    def create_uncertainty_overlay(self, image, prob, fname):
        save_root = Path(self.args.checkpoint) / "uncertainty_overlay"
        save_root.mkdir(parents=True, exist_ok=True)
        uncertainty = 1.0 - 2.0 * np.abs(prob - 0.5)

        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            class_name = self.channel_to_name[channel_idx]
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Image")
            ax[0].axis("off")

            ax[1].imshow(image, cmap="gray")
            im1 = ax[1].imshow(prob[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[1].set_title(f"Probability {class_name}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            ax[2].imshow(image, cmap="gray")
            im2 = ax[2].imshow(uncertainty[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[2].set_title(f"Uncertainty {class_name}")
            ax[2].axis("off")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(save_root / f"{fname[:-4]}_{class_name}.png", dpi=300)
            plt.close()

    def create_csv(self):
        save_path = Path(self.args.checkpoint)
        metrics_dict = self.metrics.get_metrics()

        def _safe_row_nanmean(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            valid = np.isfinite(arr)
            sums = np.where(valid, arr, 0.0).sum(axis=1)
            counts = valid.sum(axis=1)
            out = np.full(arr.shape[0], np.nan, dtype=float)
            nonzero = counts > 0
            out[nonzero] = sums[nonzero] / counts[nonzero]
            return out

        dsc_pc = np.asarray(metrics_dict["dsc_pc"], dtype=float)
        nsd_pc = np.asarray(metrics_dict["nsd_pc"], dtype=float)
        voe_pc = np.asarray(metrics_dict["voe_pc"], dtype=float)
        msd_pc = np.asarray(metrics_dict["msd_pc"], dtype=float)
        ravd_pc = np.asarray(metrics_dict["ravd_pc"], dtype=float)
        precision_pc = np.asarray(metrics_dict["precision_pc"], dtype=float)
        recall_pc = np.asarray(metrics_dict["recall_pc"], dtype=float)
        f1_pc = np.asarray(metrics_dict["f1_pc"], dtype=float)
        tp_pc = np.asarray(metrics_dict["tp_pc"], dtype=float)
        tn_pc = np.asarray(metrics_dict["tn_pc"], dtype=float)
        fp_pc = np.asarray(metrics_dict["fp_pc"], dtype=float)
        fn_pc = np.asarray(metrics_dict["fn_pc"], dtype=float)

        fname_df = pd.DataFrame(metrics_dict["fname"], columns=["Case"])
        class_names = self._foreground_channel_names()
        num_fg_classes = len(class_names)
        dsc_pc = dsc_pc[:, :num_fg_classes]
        nsd_pc = nsd_pc[:, :num_fg_classes]
        voe_pc = voe_pc[:, :num_fg_classes]
        msd_pc = msd_pc[:, :num_fg_classes]
        ravd_pc = ravd_pc[:, :num_fg_classes]
        precision_pc = precision_pc[:, :num_fg_classes]
        recall_pc = recall_pc[:, :num_fg_classes]
        f1_pc = f1_pc[:, :num_fg_classes]
        tp_pc = tp_pc[:, :num_fg_classes]
        tn_pc = tn_pc[:, :num_fg_classes]
        fp_pc = fp_pc[:, :num_fg_classes]
        fn_pc = fn_pc[:, :num_fg_classes]
        dsc_df = pd.DataFrame(dsc_pc, columns=[f"DSC {name}" for name in class_names])
        dsc_mean_df = pd.DataFrame(_safe_row_nanmean(dsc_pc), columns=["Mean DSC"])
        nsd_df = pd.DataFrame(nsd_pc, columns=[f"NSD {name}" for name in class_names])
        nsd_mean_df = pd.DataFrame(_safe_row_nanmean(nsd_pc), columns=["Mean NSD"])
        voe_df = pd.DataFrame(voe_pc, columns=[f"VOE {name}" for name in class_names])
        voe_mean_df = pd.DataFrame(_safe_row_nanmean(voe_pc), columns=["Mean VOE"])
        msd_df = pd.DataFrame(msd_pc, columns=[f"MSD {name}" for name in class_names])
        msd_mean_df = pd.DataFrame(
            _safe_row_nanmean(msd_pc),
            columns=["Mean MSD"],
        )
        ravd_df = pd.DataFrame(ravd_pc, columns=[f"RAVD {name}" for name in class_names])
        ravd_mean_df = pd.DataFrame(_safe_row_nanmean(ravd_pc), columns=["Mean RAVD"])
        precision_df = pd.DataFrame(precision_pc, columns=[f"Precision {name}" for name in class_names])
        precision_mean_df = pd.DataFrame(_safe_row_nanmean(precision_pc), columns=["Mean Precision"])
        recall_df = pd.DataFrame(recall_pc, columns=[f"Recall {name}" for name in class_names])
        recall_mean_df = pd.DataFrame(_safe_row_nanmean(recall_pc), columns=["Mean Recall"])
        f1_df = pd.DataFrame(f1_pc, columns=[f"F1 {name}" for name in class_names])
        f1_mean_df = pd.DataFrame(_safe_row_nanmean(f1_pc), columns=["Mean F1"])
        tp_df = pd.DataFrame(tp_pc, columns=[f"TP {name}" for name in class_names])
        tn_df = pd.DataFrame(tn_pc, columns=[f"TN {name}" for name in class_names])
        fp_df = pd.DataFrame(fp_pc, columns=[f"FP {name}" for name in class_names])
        fn_df = pd.DataFrame(fn_pc, columns=[f"FN {name}" for name in class_names])

        metric_df = pd.concat(
            [
                fname_df,
                dsc_df,
                dsc_mean_df,
                nsd_df,
                nsd_mean_df,
                voe_df,
                voe_mean_df,
                msd_df,
                msd_mean_df,
                ravd_df,
                ravd_mean_df,
                precision_df,
                precision_mean_df,
                recall_df,
                recall_mean_df,
                f1_df,
                f1_mean_df,
                tp_df,
                tn_df,
                fp_df,
                fn_df,
            ],
            axis=1,
        )

        vals = metric_df.iloc[:, 1:].to_numpy(dtype=float)
        finite_means = _safe_nanmean(vals, axis=0)
        average_row = pd.DataFrame([["Average"] + finite_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv(save_path / "test_metrics.csv", index=False)


class BEPatchSegInferencer:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        checkpoint_candidates = [
            Path(args.checkpoint) / "model_best.pth",
            Path(args.checkpoint) / "model_latest.pth",
        ]
        checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No BE checkpoint found under {args.checkpoint}")
        self.net.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_uncertainty_overlay = getattr(args, "save_uncertainty_overlay", False)
        self.save_overlay = args.save_overlay
        self.save_npz = getattr(args, "save_npz", getattr(args, "save_npy", False))
        self.softmax = torch.nn.Softmax(dim=1)
        self.channel_to_name = self.test_loader.dataset.channel_to_name
        self.colors = [
            np.array([0.931, 0.341, 0.215]),
            np.array([0.985, 0.725, 0.188]),
            np.array([0.176, 0.533, 0.855]),
        ]
        self.alpha = 0.35

        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    @staticmethod
    def _should_flip(dataset, fname):
        if not getattr(dataset, "flip_left_by_name", False):
            return False
        return Path(fname).stem.endswith("_L")

    def _foreground_channel_indices(self):
        if 0 in self.channel_to_name and self.channel_to_name[0].lower() == "background":
            return list(range(1, len(self.channel_to_name)))
        return list(range(len(self.channel_to_name)))

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)

        with torch.no_grad():
            for _, batch in enumerate(pbar):
                img = batch["img"].to(self.device)
                fname = batch["fname"]

                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=(self.args.image_size, self.args.image_size),
                    sw_batch_size=4,
                    predictor=self.net,
                    overlap=0.5,
                    mode="gaussian",
                )

                probs = self.softmax(logits).cpu()
                pred_idx = torch.argmax(probs, dim=1)
                pred = torch.nn.functional.one_hot(
                    pred_idx, num_classes=probs.shape[1]
                ).permute(0, 3, 1, 2).numpy().astype(np.uint8)
                probs = probs.numpy()

                for b in range(img.shape[0]):
                    fused_prob = probs[b]
                    fused_pred = pred[b]
                    name = fname[b]

                    img_path = self.test_loader.dataset.data_root / name
                    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    if self._should_flip(self.test_loader.dataset, name):
                        raw_img = cv2.flip(raw_img, 1)
                        fused_prob = fused_prob[:, :, ::-1]
                        fused_pred = fused_pred[:, :, ::-1]

                    if self.save_uncertainty_overlay:
                        self.create_uncertainty_overlay(raw_img, fused_prob, name)
                    if self.save_overlay:
                        self.create_overlay(raw_img, fused_pred, name)
                    if self.save_npz:
                        self.create_npz(raw_img, fused_pred, name)

    def create_npz(self, image, pred, fname):
        save_path = Path(self.args.checkpoint) / "npz"
        save_path.mkdir(parents=True, exist_ok=True)
        _save_prediction_npz(
            save_path / (fname[:-4] + ".npz"),
            pred.astype(np.uint8),
            image=image,
        )

    def create_overlay(self, image, pred, fname):
        save_path = Path(self.args.checkpoint) / "overlay"
        save_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(image, cmap="gray")
        ax[1].imshow(image, cmap="gray")
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[0].axis("off")
        ax[1].axis("off")
        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            show_mask(
                pred[channel_idx].astype(np.uint8),
                ax[1],
                mask_color=self.colors[color_idx],
                alpha=self.alpha,
            )
        plt.tight_layout()
        plt.savefig(save_path / (fname[:-4] + ".pdf"), dpi=600)
        plt.close()

    def create_uncertainty_overlay(self, image, prob, fname):
        save_root = Path(self.args.checkpoint) / "uncertainty_overlay"
        save_root.mkdir(parents=True, exist_ok=True)
        uncertainty = 1.0 - 2.0 * np.abs(prob - 0.5)

        for color_idx, channel_idx in enumerate(self._foreground_channel_indices()):
            class_name = self.channel_to_name[channel_idx]
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Image")
            ax[0].axis("off")

            ax[1].imshow(image, cmap="gray")
            im1 = ax[1].imshow(prob[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[1].set_title(f"Probability {class_name}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            ax[2].imshow(image, cmap="gray")
            im2 = ax[2].imshow(uncertainty[channel_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax[2].set_title(f"Uncertainty {class_name}")
            ax[2].axis("off")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(save_root / f"{fname[:-4]}_{class_name}.png", dpi=300)
            plt.close()


SegTrainer = FullHandBoneSegTrainer
SegTester = LegacyBoneSegTester
PatchSegTester = FullHandBoneSegTester
PatchSegInferencer = FullHandBoneSegInferencer


class ScoreClsTrainer:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, num_classes, score_values=None, device="cuda:0"):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
        self.optimizer = optimizer
        self.num_classes = int(num_classes)
        self.score_values = list(score_values) if score_values is not None else list(range(self.num_classes))
        self.device = device
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.max_epoch = args.max_epoch
        self.monitor_mode = getattr(args, "monitor_mode", "max")
        self.monitor_metric = getattr(args, "monitor_metric", "qwk")
        self.scaler = GradScaler() if self.amp else None
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.is_main_process = self.rank == 0
        self.earlystop = EarlyStopping(
            patience=args.earlystop_patience,
            mode=self.monitor_mode,
        )
        if args.scheduler == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.max_epoch,
                eta_min=self.optimizer.param_groups[0]["lr"] * 0.01,
            )
        elif args.scheduler == "Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if self.monitor_mode == "max" else "min",
                factor=0.8,
                patience=5,
                cooldown=2,
            )
        else:
            self.scheduler = None

        self.start_epoch = 0
        self.best_metric = -np.inf if self.monitor_mode == "max" else np.inf
        self.train_loss_history = []
        self.val_metric_history = []

    def _unwrap_model(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _move_optimizer_state_to_device(self):
        target_device = torch.device(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target_device, non_blocking=True)

    def _reduce_scalar(self, value: float, average: bool = True) -> float:
        if not self.is_ddp:
            return float(value)

        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= self.world_size
        return float(tensor.item())

    def _set_sampler_epoch(self, loader, epoch: int):
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _forward_logits(self, img):
        pred = self.net(img)
        if not isinstance(pred, torch.Tensor):
            pred = pred[0]
        return pred

    def _decode_ordinal_logits(self, pred):
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        return (torch.sigmoid(pred) > 0.5).sum(dim=1).long()

    def _move_batch_to_device(self, batch):
        return batch["img"].to(self.device), batch["ordinal_target"].to(self.device)

    def load_training_state(self, state: Dict):
        self.start_epoch = int(state.get("epoch", -1)) + 1
        self.best_metric = float(
            state.get("best_metric", -np.inf if self.monitor_mode == "max" else np.inf)
        )
        self.train_loss_history = list(state.get("train_loss_history", []))
        self.val_metric_history = list(state.get("val_metric_history", []))

        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler")
        if self.scaler is not None and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        earlystop_state = state.get("earlystop")
        if earlystop_state is not None:
            self.earlystop.load_state_dict(earlystop_state)

    def fit(self, args):
        train_loss = list(self.train_loss_history)
        val_metric = list(self.val_metric_history)

        for epoch in range(self.start_epoch, self.max_epoch):
            self._set_sampler_epoch(self.train_loader, epoch)
            if next(self.net.parameters()).device != self.device:
                self.net = self.net.to(self.device)

            epoch_train_loss = self.train_one_epoch_amp(epoch) if self.amp else self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss)

            epoch_val_metric = self.validate(epoch)
            val_metric.append(epoch_val_metric)
            self.train_loss_history = train_loss
            self.val_metric_history = val_metric

            if self.is_main_process:
                self.plot(args, train_loss, val_metric)

            should_stop = (
                self.earlystop(epoch_val_metric)
                if self.is_main_process and getattr(args, "earlystop", False)
                else False
            )
            if self.is_ddp:
                stop_tensor = torch.tensor(int(should_stop), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if self.scheduler is not None and not should_stop:
                if args.scheduler == "Plateau":
                    self.scheduler.step(epoch_val_metric)
                else:
                    self.scheduler.step()

            is_better = (
                epoch_val_metric > self.best_metric
                if self.monitor_mode == "max"
                else epoch_val_metric < self.best_metric
            )
            if is_better:
                if self.is_main_process:
                    print(
                        f"New best {self.monitor_metric}: "
                        f"{self.best_metric:.4f} -> {epoch_val_metric:.4f}"
                    )
                self.best_metric = epoch_val_metric
            else:
                if self.is_main_process:
                    print(
                        f"No {self.monitor_metric} improvement: "
                        f"{epoch_val_metric:.4f} (best {self.best_metric:.4f})"
                    )

            ckpt = {
                "model": self._unwrap_model().state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "earlystop": self.earlystop.state_dict(),
                "train_loss_history": train_loss,
                "val_metric_history": val_metric,
                "val_metric": epoch_val_metric,
                "best_metric": self.best_metric,
                "monitor_metric": self.monitor_metric,
                "num_classes": self.num_classes,
                "score_values": self.score_values,
                "ordinal_method": getattr(args, "ordinal_method", "independent"),
            }
            if self.is_main_process:
                torch.save(ckpt, Path(args.model_save_path) / "model_latest.pth")
                if is_better:
                    torch.save(ckpt, Path(args.model_save_path) / "model_best.pth")

            if should_stop:
                if self.is_main_process:
                    print(f"Early stopping triggered on {self.monitor_metric}.")
                break

    def train_one_epoch_amp(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader, disable=not self.is_main_process)
        avg_loss = 0.0
        for batch in pbar:
            img, gt = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = self._forward_logits(img)
                loss = self.criterion(pred, gt)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            avg_loss += loss.item()
            if self.is_main_process:
                pbar.set_description(
                    f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                )
        avg_loss /= max(len(self.train_loader), 1)
        return self._reduce_scalar(avg_loss, average=True)

    def train_one_epoch(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader, disable=not self.is_main_process)
        avg_loss = 0.0
        for batch in pbar:
            img, gt = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self._forward_logits(img)
            loss = self.criterion(pred, gt)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.optimizer.step()

            avg_loss += loss.item()
            if self.is_main_process:
                pbar.set_description(
                    f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                )
        avg_loss /= max(len(self.train_loader), 1)
        return self._reduce_scalar(avg_loss, average=True)

    def validate(self, epoch):
        self.net.eval()
        if self.is_ddp and not self.is_main_process:
            metric_tensor = torch.zeros(1, device=self.device)
            dist.broadcast(metric_tensor, src=0)
            return float(metric_tensor[0].item())

        pbar = tqdm(self.val_loader, disable=not self.is_main_process)
        metrics = ClassificationMetrics(num_classes=self.num_classes, score_values=self.score_values)
        avg_loss = 0.0

        with torch.no_grad():
            for batch in pbar:
                img, gt = self._move_batch_to_device(batch)
                pred = self._forward_logits(img)
                loss = self.criterion(pred, gt)
                avg_loss += loss.item()
                pred_class = self._decode_ordinal_logits(pred)
                metrics.update_metrics(
                    pred_class,
                    batch["label"],
                    batch["case_name"],
                    batch["joint_name"],
                    pred_is_label=True,
                )
                if self.is_main_process:
                    pbar.set_description(
                        f"Epoch {epoch} validating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                    )

        metric_dict = metrics.get_metrics()
        metric_map = {
            "macro_f1": float(metric_dict["overall_f1"]),
            "accuracy": float(metric_dict["overall_accuracy"]),
            "qwk": float(metric_dict["overall_qwk"]),
            "mae": float(metric_dict["overall_mae"]),
        }
        metric_value = metric_map[self.monitor_metric]
        mean_loss = avg_loss / max(len(self.val_loader), 1)
        if self.is_main_process:
            print(
                f"Validation loss: {mean_loss:.4f}, "
                f"accuracy: {metric_dict['overall_accuracy']:.4f}, "
                f"macro_f1: {metric_dict['overall_f1']:.4f}, "
                f"qwk: {metric_dict['overall_qwk']:.4f}, "
                f"mae: {metric_dict['overall_mae']:.4f}"
            )
        if self.is_ddp:
            metric_tensor = torch.tensor([metric_value], device=self.device, dtype=torch.float32)
            dist.broadcast(metric_tensor, src=0)
            metric_value = float(metric_tensor[0].item())
        return metric_value

    def plot(self, args, train_loss, val_metric):
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_metric, label=f"Val {self.monitor_metric}")
        plt.title("Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class ScoreClsTester:
    def __init__(self, args, net, test_loader, num_classes, score_values=None, device="cuda:0"):
        self.args = args
        self.net = net
        state = torch.load(Path(args.checkpoint) / "model_best.pth", map_location="cpu")
        self.net.load_state_dict(state["model"])
        self.test_loader = test_loader
        self.device = device
        self.num_classes = int(num_classes)
        self.score_values = list(score_values) if score_values is not None else list(range(self.num_classes))
        self.save_csv = args.save_csv
        self.metrics = ClassificationMetrics(num_classes=self.num_classes, score_values=self.score_values)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes).to(device)
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def _forward_logits(self, img):
        pred = self.net(img)
        if not isinstance(pred, torch.Tensor):
            pred = pred[0]
        return pred

    def _decode_ordinal_logits(self, pred):
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        return (torch.sigmoid(pred) > 0.5).sum(dim=1).long()

    def _move_batch_to_device(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        case_inference_times = {}
        rows = []
        contradiction_count = 0
        contradiction_total = 0

        with torch.no_grad():
            for batch in pbar:
                img, gt = self._move_batch_to_device(batch)

                pred, infer_time = _time_model_forward(lambda: self._forward_logits(img), self.device)
                _update_case_inference_times(case_inference_times, list(batch["case_name"]), infer_time)

                threshold_pred = torch.sigmoid(pred) > 0.5
                if threshold_pred.ndim == 1:
                    threshold_pred = threshold_pred.unsqueeze(0)
                if threshold_pred.shape[1] > 1:
                    contradictions = (threshold_pred[:, :-1].int() < threshold_pred[:, 1:].int()).any(dim=1)
                    contradiction_count += int(contradictions.sum().item())
                contradiction_total += int(threshold_pred.shape[0])

                pred_class = self._decode_ordinal_logits(pred)
                self.confusion_matrix.update(pred_class, gt)
                self.metrics.update_metrics(
                    pred_class,
                    gt,
                    batch["case_name"],
                    batch["joint_name"],
                    pred_is_label=True,
                )

                for idx in range(pred_class.shape[0]):
                    rows.append(
                        {
                            "Case": batch["case_name"][idx],
                            "Joint": batch["joint_name"][idx],
                            "ScoreKey": batch["score_key"][idx],
                            "GTClass": int(gt[idx].item()),
                            "PredClass": int(pred_class[idx].item()),
                            "GTRawScore": int(batch["raw_score"][idx].item()),
                            "PredRawScore": int(self.score_values[int(pred_class[idx].item())]),
                            "ImagePath": batch["img_path"][idx],
                        }
                    )
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        confmat = self.confusion_matrix.compute().cpu().numpy()
        print("Confusion Matrix:\n", confmat)
        self.print_confusion_matrix(confmat)
        self.plot_confusion_matrix(confmat)

        metric_dict = self.metrics.get_metrics()
        print(f"Accuracy: {metric_dict['overall_accuracy']:.4f}")
        print(f"Macro F1: {metric_dict['overall_f1']:.4f}")
        print(f"Macro Recall: {metric_dict['overall_recall']:.4f}")
        print(f"Macro Precision: {metric_dict['overall_precision']:.4f}")
        print(f"QWK: {metric_dict['overall_qwk']:.4f}")
        print(f"MAE: {metric_dict['overall_mae']:.4f}")
        print(f"Within-1: {metric_dict['overall_within_1']:.4f}")
        print(f"Pos/Neg ACC: {metric_dict['overall_pos_neg_acc']:.4f}")
        print(f"Binary Sensitivity: {metric_dict['overall_binary_sensitivity']:.4f}")
        print(f"Binary Specificity: {metric_dict['overall_binary_specificity']:.4f}")
        contradiction_rate = contradiction_count / max(contradiction_total, 1)
        print(
            "Ordinal contradiction count: "
            f"{contradiction_count}/{contradiction_total} ({contradiction_rate:.4f})"
        )

        _print_case_inference_time_summary(case_inference_times)
        _save_case_inference_time_summary(self.args.checkpoint, case_inference_times)

        if self.save_csv:
            self.create_csv(metric_dict, rows, confmat)

    def print_confusion_matrix(self, confmat):
        labels = [str(score) for score in self.score_values]
        confmat_df = pd.DataFrame(
            confmat,
            index=[f"GT_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels],
        )
        print("\nConfusion Matrix Table:")
        print(confmat_df.to_string())

    def plot_confusion_matrix(self, confmat):
        display_labels = [str(score) for score in self.score_values]
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=display_labels)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(Path(self.args.checkpoint) / "ConfusionMatrix.pdf")
        plt.close()

    def create_csv(self, metrics_dict, rows, confmat):
        save_path = Path(self.args.checkpoint)
        pd.DataFrame(rows).to_csv(save_path / "test_predictions.csv", index=False)

        summary_rows = [
            {
                "Scope": "Overall",
                "Name": "All",
                "Accuracy": metrics_dict["overall_accuracy"],
                "Precision": metrics_dict["overall_precision"],
                "Recall": metrics_dict["overall_recall"],
                "F1score": metrics_dict["overall_f1"],
                "Specificity": metrics_dict["overall_specificity"],
                "BalancedAccuracy": metrics_dict["overall_balanced_accuracy"],
                "DOR": metrics_dict["overall_dor"],
                "QWK": metrics_dict["overall_qwk"],
                "MAE": metrics_dict["overall_mae"],
                "Within1": metrics_dict["overall_within_1"],
                "Pos/Neg ACC": metrics_dict["overall_pos_neg_acc"],
                "Binary Sensitivity": metrics_dict["overall_binary_sensitivity"],
                "Binary Specificity": metrics_dict["overall_binary_specificity"],
            }
        ]
        for joint, joint_metric in metrics_dict["joint_metrics"].items():
            summary_rows.append(
                {
                    "Scope": "Joint",
                    "Name": joint,
                    "Accuracy": joint_metric["accuracy"],
                    "Precision": joint_metric["precision"],
                    "Recall": joint_metric["recall"],
                    "F1score": joint_metric["f1score"],
                    "Specificity": joint_metric["specificity"],
                    "BalancedAccuracy": joint_metric["balanced_accuracy"],
                    "DOR": joint_metric["dor"],
                    "QWK": joint_metric["qwk"],
                    "MAE": joint_metric["mae"],
                    "Within1": joint_metric["within_1"],
                    "Pos/Neg ACC": joint_metric["pos_neg_acc"],
                    "Binary Sensitivity": joint_metric["binary_sensitivity"],
                    "Binary Specificity": joint_metric["binary_specificity"],
                }
            )
        pd.DataFrame(summary_rows).to_csv(save_path / "test_metrics.csv", index=False)

        cm_rows = []
        for gt_idx in range(confmat.shape[0]):
            for pred_idx in range(confmat.shape[1]):
                cm_rows.append(
                    {
                        "GTClass": gt_idx,
                        "PredClass": pred_idx,
                        "GTScore": self.score_values[gt_idx],
                        "PredScore": self.score_values[pred_idx],
                        "Count": int(confmat[gt_idx, pred_idx]),
                    }
                )
        pd.DataFrame(cm_rows).to_csv(save_path / "confusion_matrix.csv", index=False)


class TwoStageEarlyStopping:
    """
    Two-stage EarlyStopping for Dice → NSD.
    Keeps separate best scores and counters for each metric type.
    """
    def __init__(self, patience, delta=0.0, mode="max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score_dice = None
        self.best_score_nsd = None

        # 独立的 counter，防止切换阶段互相影响
        self.counter_dice = 0
        self.counter_nsd = 0

    def __call__(self, val_metric, metric_type):
        """
        metric_type: 'dice' or 'nsd'
        """

        # mode handling
        score = -val_metric if self.mode == "min" else val_metric

        # ----------------------------
        # Stage 1: Dice early-stopping
        # ----------------------------
        if metric_type == "dice":
            if self.best_score_dice is None:
                self.best_score_dice = score
                return False

            # no improvement
            if score < self.best_score_dice + self.delta:
                self.counter_dice += 1
                print(f"EarlyStopping counter (Dice): {self.counter_dice}/{self.patience}")
                return self.counter_dice >= self.patience
            else:
                self.best_score_dice = score
                self.counter_dice = 0
                return False

        # ----------------------------
        # Stage 2: NSD early-stopping
        # ----------------------------
        else:  # metric_type == "nsd"
            if self.best_score_nsd is None:
                self.best_score_nsd = score
                return False

            if score < self.best_score_nsd + self.delta:
                self.counter_nsd += 1
                print(f"EarlyStopping counter (NSD): {self.counter_nsd}/{self.patience}")
                return self.counter_nsd >= self.patience
            else:
                self.best_score_nsd = score
                self.counter_nsd = 0
                return False

    def state_dict(self):
        return {
            "patience": self.patience,
            "delta": self.delta,
            "mode": self.mode,
            "best_score_dice": self.best_score_dice,
            "best_score_nsd": self.best_score_nsd,
            "counter_dice": self.counter_dice,
            "counter_nsd": self.counter_nsd,
        }

    def load_state_dict(self, state):
        if not state:
            return

        self.patience = state.get("patience", self.patience)
        self.delta = state.get("delta", self.delta)
        self.mode = state.get("mode", self.mode)
        self.best_score_dice = state.get("best_score_dice", self.best_score_dice)
        self.best_score_nsd = state.get("best_score_nsd", self.best_score_nsd)
        self.counter_dice = state.get("counter_dice", self.counter_dice)
        self.counter_nsd = state.get("counter_nsd", self.counter_nsd)


class EarlyStopping:
    """
    Basic EarlyStopping with min/max mode.

    Args:
        patience (int): how many epochs to wait without improvement
        delta (float): minimum change to qualify as improvement
        mode (str): "min" (lower is better) or "max" (higher is better)
    """
    def __init__(self, patience=10, delta=0.0, mode="min"):
        self.patience = patience
        self.delta = delta
        self.mode = mode

        # 根据 mode 设置初始化的 best_score
        if mode == "min":
            self.best_score = float("inf")
        else:  # "max"
            self.best_score = -float("inf")

        self.counter = 0

    def __call__(self, current_score):
        """
        Returns True if training should stop.
        """

        # 判定是否改善
        if self.mode == "min":
            improved = current_score < self.best_score - self.delta
        else:
            improved = current_score > self.best_score + self.delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return False  # don't stop

        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True  # stop training
            return False

    def state_dict(self):
        return {
            "patience": self.patience,
            "delta": self.delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "counter": self.counter,
        }

    def load_state_dict(self, state):
        if not state:
            return

        self.patience = state.get("patience", self.patience)
        self.delta = state.get("delta", self.delta)
        self.mode = state.get("mode", self.mode)
        self.best_score = state.get("best_score", self.best_score)
        self.counter = state.get("counter", self.counter)
