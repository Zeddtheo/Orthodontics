# module1_train.py
# 训练 iMeshSegNet（EdgeConv 版本）——显式位置输入、GDL+加权CE、AMP+clip、Cosine LR 与可重复追踪签名。

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from m0_dataset import (
    DataConfig,
    SEG_NUM_CLASSES,
    compute_label_histogram,
    get_dataloaders,
    load_split_lists,
    set_seed,
)
from imeshsegnet import iMeshSegNet


# =============================================================================
# Losses & Metrics
# =============================================================================


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)                            # (B,C,N)
        onehot = F.one_hot(targets, num_classes=probs.size(1)).permute(0,2,1)  # (B,C,N)

        support = torch.sum(onehot, dim=(0,2))                      # (C,)
        raw_w = 1.0 / (support ** 2 + self.epsilon)
        weights = torch.where(support > 0, raw_w, torch.zeros_like(raw_w))     # ← 关键
        intersection = torch.sum(probs * onehot, dim=(0,2))
        union = torch.sum(probs, dim=(0,2)) + torch.sum(onehot, dim=(0,2))
        dice = (2 * torch.sum(weights * intersection)) / (torch.sum(weights * union) + self.epsilon)
        return 1 - dice


def build_ce_class_weights(hist: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency CE weights with clipping & mean normalisation."""
    freq = hist.astype(np.float64)
    total = freq.sum()
    if total <= 0:
        return np.ones_like(freq, dtype=np.float32)

    freq = freq.copy()
    mask = freq > 0
    if not np.any(mask):
        mask[:] = True
    min_nonzero = freq[mask].min()
    freq[~mask] = min_nonzero

    prob = freq / total
    prob = np.maximum(prob, 1e-6)
    weights = 1.0 / prob
    weights = np.clip(weights, 0.1, 10.0)
    weights /= weights.mean()
    return weights.astype(np.float32)


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets.cpu().numpy().flatten()

    cm = confusion_matrix(targets_flat, preds_flat, labels=range(num_classes))
    dsc = np.zeros(num_classes)
    sen = np.zeros(num_classes)
    ppv = np.zeros(num_classes)
    support = np.sum(cm, axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        dsc[i] = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        sen[i] = tp / (tp + fn + 1e-8)
        ppv[i] = tp / (tp + fp + 1e-8)

    foreground_mask = (np.arange(num_classes) != 0) & (support > 0)
    if not np.any(foreground_mask):
        foreground_mask = np.arange(num_classes) != 0

    mean_dsc = float(np.mean(dsc[foreground_mask])) if np.any(foreground_mask) else 0.0
    mean_sen = float(np.mean(sen[foreground_mask])) if np.any(foreground_mask) else 0.0
    mean_ppv = float(np.mean(ppv[foreground_mask])) if np.any(foreground_mask) else 0.0
    return mean_dsc, mean_sen, mean_ppv


# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingLR,
        config: "TrainConfig",
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.log_path = self.config.output_dir / "train_log.csv"
        self.best_val_dsc = -1.0
        self.amp_enabled = device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self._checked_shapes = False

        self.dice_loss = GeneralizedDiceLoss().to(self.device)
        weight_tensor = None
        if self.config.ce_class_weights is not None:
            weight_tensor = torch.tensor(self.config.ce_class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor).to(self.device)

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> Tuple[float, float, float, float]:
        self.model.train(mode=is_train)
        total_loss = 0.0
        preds_all = []
        targets_all = []
        desc = "Training" if is_train else "Validation"

        for (x, pos), y in tqdm(loader, desc=desc):
            x = x.to(self.device, non_blocking=True)
            pos = pos.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if not self._checked_shapes:
                assert x.dim() == 3 and x.size(1) == 15, "x must be (B,15,N) with z-scored features"
                assert pos.dim() == 3 and pos.size(1) == 3, "pos must be (B,3,N) in arch frame"
                self._checked_shapes = True

            # **关键修改：验证阶段添加 no_grad() 优化**
            if not is_train:
                with torch.no_grad():
                    with autocast(enabled=self.amp_enabled):
                        logits = self.model(x, pos)
                        loss_dice = self.dice_loss(logits, y)
                        loss_ce = self.ce_loss(logits, y)
                        loss = loss_dice + loss_ce
            else:
                # 训练阶段保持原有逻辑
                with autocast(enabled=self.amp_enabled):
                    logits = self.model(x, pos)
                    loss_dice = self.dice_loss(logits, y)
                    loss_ce = self.ce_loss(logits, y)
                    loss = loss_dice + loss_ce

                # 训练阶段的反向传播
                self.optimizer.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            # 移动到循环内部：
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.detach().cpu())
            targets_all.append(y.detach().cpu())

        # 移动到循环外部
        avg_loss = total_loss / max(len(loader), 1)
        preds_tensor = torch.cat(preds_all, dim=0)
        targets_tensor = torch.cat(targets_all, dim=0)
        dsc, sen, ppv = calculate_metrics(preds_tensor, targets_tensor, self.config.num_classes)
        return avg_loss, dsc, sen, ppv

    def train(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 简化CSV头：
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dsc", "lr"])

        for epoch in range(1, self.config.epochs + 1):
            if self.amp_enabled:
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_start = time.time()
            train_loss, _, _, _ = self._run_epoch(self.train_loader, is_train=True)
            epoch_time = time.time() - epoch_start
            train_peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.amp_enabled
                else 0.0
            )
            val_loss, val_dsc, val_sen, val_ppv = self._run_epoch(self.val_loader, is_train=False)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val DSC: {val_dsc:.4f} | Val SEN: {val_sen:.4f} | Val PPV: {val_ppv:.4f} | "
                f"Time: {epoch_time:.1f}s | PeakMem: {train_peak_mem:.1f}MB | LR: {current_lr:.6f}"
            )

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 简化记录内容：
                writer.writerow([epoch, train_loss, val_loss, val_dsc, current_lr])

            torch.save(self.model.state_dict(), self.config.output_dir / "last.pt")
            # 简化保存最佳模型的逻辑：
            if val_dsc > self.best_val_dsc:
                self.best_val_dsc = val_dsc
                print(f"✨ New best model found! DSC: {val_dsc:.4f}. Saving to best.pt")
                torch.save(self.model.state_dict(), self.config.output_dir / "best.pt")

            self.scheduler.step()

        print(f"Training finished. Best validation DSC: {self.best_val_dsc:.4f}")


# =============================================================================
# Configuration & Entrypoint
# =============================================================================


@dataclass
class TrainConfig:
    data_config: DataConfig = DataConfig()
    output_dir: Path = Path("outputs/segmentation/module1_train")
    epochs: int = 200
    lr: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    num_classes: int = SEG_NUM_CLASSES
    ce_class_weights: Optional[Sequence[float]] = None


def main() -> None:
    config = TrainConfig()
    set_seed(getattr(config.data_config, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data pipeline (always keep online augmentation for train loader)
    config.data_config.pin_memory = device.type == "cuda"
    config.data_config.persistent_workers = True
    config.data_config.augment = True

    if config.ce_class_weights is None:
        print("Estimating class frequencies for CE weights...")
        train_files, _ = load_split_lists(config.data_config.split_path)
        class_hist = compute_label_histogram(train_files)
        ce_weights = build_ce_class_weights(class_hist)
        config.ce_class_weights = ce_weights.tolist()
        print(
            f"CrossEntropy class weights prepared: min={ce_weights.min():.3f}, "
            f"max={ce_weights.max():.3f}"
        )

    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config.data_config)

    # Model + Optimizer + Scheduler
    print("Initializing model, loss function, optimizer, and scheduler...")
    model = iMeshSegNet(
        num_classes=config.num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=True,
        dropout_p=0.5,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        amsgrad=True,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
