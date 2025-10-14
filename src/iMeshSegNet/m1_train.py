# module1_train.py
# 训练 iMeshSegNet（EdgeConv 版本）——显式位置输入、GDL+加权CE、AMP+clip、Cosine LR 与可重复追踪签名。

from __future__ import annotations

import argparse
import csv
import math
import time
from shutil import copy2
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from m0_dataset import (
    ARCH_LABEL_ORDERS,
    DataConfig,
    SEG_NUM_CLASSES,
    SINGLE_ARCH_NUM_CLASSES,
    compute_label_histogram,
    get_dataloaders,
    load_split_lists,
    load_stats,
    set_seed,
)
from imeshsegnet import iMeshSegNet, index_points, knn_graph

SEGMENTATION_ROOT = Path("outputs/segmentation")
FINAL_SEG_PT_DIR = SEGMENTATION_ROOT / "final_pt"

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


# =============================================================================
# Losses & Metrics
# =============================================================================


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5, weight_clip_min: float = 1e-3, weight_clip_max: float = 1e3):
        super().__init__()
        self.epsilon = epsilon
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)                            # (B,C,N)
        onehot = F.one_hot(targets, num_classes=probs.size(1)).permute(0,2,1)  # (B,C,N)

        support = torch.sum(onehot, dim=(0,2))                      # (C,)
        raw_w = 1.0 / (support ** 2 + self.epsilon)
        # ← 新增：对权重做限幅，避免极小 support 类别权重爆炸
        raw_w = torch.clamp(raw_w, self.weight_clip_min, self.weight_clip_max)
        weights = torch.where(support > 0, raw_w, torch.zeros_like(raw_w))
        intersection = torch.sum(probs * onehot, dim=(0,2))
        union = torch.sum(probs, dim=(0,2)) + torch.sum(onehot, dim=(0,2))
        dice = (2 * torch.sum(weights * intersection)) / (torch.sum(weights * union) + self.epsilon)
        return 1 - dice


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 2, 1).float()
        if weights is not None:
            w = weights.unsqueeze(1).float()
        else:
            w = 1.0
        tp = (w * probs * onehot).sum(dim=(0, 2))
        fp = (w * probs * (1 - onehot)).sum(dim=(0, 2))
        fn = (w * (1 - probs) * onehot).sum(dim=(0, 2))
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        focal_tversky = torch.pow(1 - tversky.mean(), self.gamma)
        return focal_tversky
def build_ce_class_weights(hist: np.ndarray, clip_min: float = 0.3, clip_max: float = 3.0) -> np.ndarray:
    """Compute inverse-frequency CE weights with clipping & mean normalisation.
    
    ========== 优化4: 更温和的类别权重 ==========
    默认 clip 范围从 [0.1, 10] 改为 [0.3, 3.0]，减少稀有类权重放大倍数。
    """
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
    # ← 使用更温和的 clip 范围（可通过参数调整）
    weights = np.clip(weights, clip_min, clip_max)
    weights /= weights.mean()
    return weights.astype(np.float32)


def calculate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float, float, np.ndarray]:
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
    return mean_dsc, mean_sen, mean_ppv, cm


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
        stats_mean: np.ndarray,
        stats_std: np.ndarray,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.config.log_dir / "train_log.csv"
        self.best_val_dsc = -1.0
        self.amp_enabled = self.config.enable_amp and device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.amp_enabled)
        self.writer: Optional[SummaryWriter] = None
        try:
            self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))
        except Exception as exc:
            print(f"[Warning] TensorBoard writer disabled: {exc}")
        self._checked_shapes = False
        self.stats_mean = np.asarray(stats_mean, dtype=np.float32)
        self.stats_std = np.asarray(stats_std, dtype=np.float32)
        self.epochs_since_best = 0
        self.best_epoch = 0

        self.dice_loss = GeneralizedDiceLoss().to(self.device)
        self.ce_weight_tensor: Optional[torch.Tensor] = None
        if self.config.ce_class_weights is not None:
            self.ce_weight_tensor = torch.tensor(
                self.config.ce_class_weights,
                dtype=torch.float32,
                device=self.device,
            )
        self.focal_tversky_loss = FocalTverskyLoss().to(self.device)
        self.boundary_lambda = float(getattr(self.config, "boundary_lambda", 0.0))
        self.boundary_knn_k = int(getattr(self.config, "boundary_knn_k", 12))

    def _build_checkpoint_payload(self) -> dict:
        mean = np.asarray(self.stats_mean, dtype=np.float32)
        std = np.clip(np.asarray(self.stats_std, dtype=np.float32), 1e-6, None)
        fstn_module = getattr(self.model, "fstn", None)
        arch_config = {
            "glm_impl": getattr(self.model, "glm_impl", "edgeconv"),
            "use_feature_stn": bool(fstn_module),
            "k_short": int(getattr(self.model, "k_short", 6)),
            "k_long": int(getattr(self.model, "k_long", 12)),
            "with_dropout": bool(getattr(self.model, "with_dropout", False)),
            "dropout_p": float(getattr(self.model, "dropout_p", 0.0)),
        }
        dataset_obj = getattr(self.train_loader, "dataset", None)
        sampler_mode = "random"
        sampler_params: Dict[str, float] = {}
        if dataset_obj is not None:
            sampler_mode = str(getattr(dataset_obj, "sampler_mode", sampler_mode))
            boundary_focus = getattr(dataset_obj, "boundary_focus_fraction", None)
            min_samples = getattr(dataset_obj, "min_samples_per_class", None)
            if boundary_focus is not None:
                sampler_params["boundary_focus_fraction"] = float(boundary_focus)
            if min_samples is not None:
                sampler_params["min_samples_per_class"] = float(min_samples)
        pipeline = {
            "zscore": {
                "apply": True,
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
            },
            "centered": True,
            "div_by_diag": True,
            "use_frame": bool(self.config.data_config.arch_frames_path),
            "sampler": sampler_mode,
            "sampler_params": sampler_params,
            "sample_cells": int(self.config.data_config.sample_cells),
            "target_cells": int(self.config.data_config.target_cells),
            "train_ids_path": None,
            "train_arrays_path": None,
            "decim_cache_vtp": None,
            "knn_k": {"to10k": 5, "tofull": 7},
            "diag_mode": "cells",
            "seed": 42,
        }
        training_meta = {
            "epochs": int(self.config.epochs),
            "optimizer": "Adam",
            "lr": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "best_val_dsc": float(self.best_val_dsc),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }
        label_book: Dict[str, Dict[int, object]] = {}
        gingiva_idx = int(getattr(self.config.data_config, "gingiva_class_id", 15))
        for jaw, order in ARCH_LABEL_ORDERS.items():
            mapping: Dict[int, object] = {0: "background"}
            for idx, fdi in enumerate(order, start=1):
                mapping[idx] = int(fdi)
            mapping[gingiva_idx] = "gingiva"
            label_book[jaw] = mapping
        payload = {
            "state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
            "in_channels": self.model.in_channels,
            "arch": arch_config,
            "pipeline": pipeline,
            "training": training_meta,
            "ce_class_weights": self.config.ce_class_weights,
            "label_book": label_book,
            "label_mode": getattr(self.config.data_config, "label_mode", "single_arch_16"),
        }
        return payload

    def _save_checkpoint(self, path: Path) -> None:
        payload = self._build_checkpoint_payload()
        torch.save(payload, path)

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        point_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = self.ce_weight_tensor
        loss_map = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
            label_smoothing=0.05,
        )
        if point_weights is None:
            return loss_map.mean()
        weights = point_weights.float()
        loss = torch.sum(loss_map * weights) / torch.clamp(weights.sum(), min=1e-6)
        return loss

    def _run_epoch(
        self,
        loader: DataLoader,
        is_train: bool,
    ) -> Tuple[float, float, float, float, float, float, float, List[Tuple[int, int, int]], Optional[List[float]]]:
        self.model.train(mode=is_train)
        total_loss = 0.0
        preds_all = []
        targets_all = []
        # ========== 快改开关5：诊断指标 ==========
        bg_ratios = []  # ← 新增：记录背景预测比例
        entropies = []  # ← 新增：记录预测熵
        desc = "Training" if is_train else "Validation"

        for (x, pos, boundary), y in tqdm(loader, desc=desc):
            x = x.to(self.device, non_blocking=True)
            pos = pos.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            boundary = boundary.to(self.device, non_blocking=True)

            if not self._checked_shapes:
                expected_in = getattr(self.model, "in_channels", x.size(1))
                assert x.dim() == 3 and x.size(1) == expected_in, f"x must be (B,{expected_in},N) with z-scored features"
                assert pos.dim() == 3 and pos.size(1) == 3, "pos must be (B,3,N) in arch frame"
                assert boundary.shape == y.shape, "boundary mask must match label shape"
                self._checked_shapes = True

            boundary_weights = None
            if is_train and self.boundary_lambda > 0:
                boundary_weights = 1.0 + boundary.float() * self.boundary_lambda

            if not is_train:
                with torch.no_grad():
                    with autocast(self.device.type, enabled=self.amp_enabled):
                        logits = self.model(x, pos)
                        loss_dice = self.dice_loss(logits, y)
                        loss_ce = self._compute_ce_loss(logits, y)
                        loss_ft = self.focal_tversky_loss(logits, y)
                        lap_reg = logits.new_tensor(0.0)
                        if logits.size(-1) > 0:
                            # 邻域平滑：Graph Laplacian 正则防止椒盐噪声
                            p = F.softmax(logits, dim=1)
                            idx = knn_graph(pos.float(), k=8)
                            nbr = index_points(p, idx)
                            lap_reg = (nbr.mean(dim=-1) - p).pow(2).mean()
                        loss = 0.3 * loss_dice + 1.0 * loss_ce + 0.2 * loss_ft + 0.03 * lap_reg
            else:
                # 训练阶段保持原有逻辑
                with autocast(self.device.type, enabled=self.amp_enabled):
                    logits = self.model(x, pos)
                    loss_dice = self.dice_loss(logits, y)
                    loss_ce = self._compute_ce_loss(logits, y, boundary_weights)
                    loss_ft = self.focal_tversky_loss(logits, y, weights=boundary_weights)
                    lap_reg = logits.new_tensor(0.0)
                    if logits.size(-1) > 0:
                        # 邻域平滑：Graph Laplacian 正则，抑制局部孤立预测
                        p = F.softmax(logits, dim=1)
                        with torch.no_grad():
                            idx = knn_graph(pos.float(), k=8)
                        nbr = index_points(p, idx)
                        lap_reg = (nbr.mean(dim=-1) - p).pow(2).mean()
                    
                    # ========== 修复2: STN 正交正则（防止越位变形）==========
                    stn_reg = 0.0
                    fstn = getattr(self.model, "fstn", None)
                    trans = getattr(self.model, "_last_fstn", None)
                    if fstn is not None and trans is not None:
                        # || T·Tᵀ - I ||_F
                        I = torch.eye(trans.size(1), device=trans.device).unsqueeze(0).expand_as(trans)
                        stn_reg = ((trans @ trans.transpose(1, 2) - I) ** 2).sum(dim=(1, 2)).mean()
                    
                    # ← 更新：加入 Focal Tversky 提升召回
                    loss = 0.3 * loss_dice + 1.0 * loss_ce + 0.2 * loss_ft + 0.03 * lap_reg + 1e-3 * stn_reg

                # ========== 修复4: NaN/Inf 哨兵（防飙车）==========
                if not torch.isfinite(loss):
                    print("[Warn] non-finite loss; skip step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                if not torch.isfinite(logits).all():
                    print("[Warn] non-finite logits; skip step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # 训练阶段的反向传播
                self.optimizer.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    # ========== 优化3: 放宽梯度裁剪阈值 ==========
                    clip_grad_norm_(self.model.parameters(), max_norm=2.0)  # ← 从 1.0 升到 2.0
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=2.0)  # ← 从 1.0 升到 2.0
                    self.optimizer.step()

            # 移动到循环内部：
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.detach().cpu())
            targets_all.append(y.detach().cpu())
            
            # ========== 诊断指标计算 ==========
            with torch.no_grad():
                # 背景比例
                bg_ratio = (preds == 0).float().mean().item()
                bg_ratios.append(bg_ratio)
                
                # ========== 修复1: 熵计算强制 FP32 + log_softmax 防止 NaN ==========
                # 用 logits.float() 保证 FP32，再用 log_softmax 防溢出
                logp = F.log_softmax(logits.float(), dim=1)
                p    = F.softmax(logits.float(), dim=1)
                # 信息熵（按 cell 平均）
                entropy = -(p * logp).sum(dim=1).mean().item()
                entropies.append(entropy)

        # 移动到循环外部
        avg_loss = total_loss / max(len(loader), 1)
        preds_tensor = torch.cat(preds_all, dim=0)
        targets_tensor = torch.cat(targets_all, dim=0)
        # ← 修复类型问题：确保 num_classes 是 int
        num_classes = getattr(self.model, 'num_classes', self.config.num_classes)
        if isinstance(num_classes, torch.Tensor):
            num_classes = num_classes.item()
        num_classes = int(num_classes)
        dsc, sen, ppv, cm = calculate_metrics(preds_tensor, targets_tensor, num_classes)

        cm_float = cm.astype(np.float64) if cm.size > 0 else np.zeros((num_classes, num_classes), dtype=np.float64)
        tp_vec = np.diag(cm_float)
        fp_vec = cm_float.sum(axis=0) - tp_vec
        fn_vec = cm_float.sum(axis=1) - tp_vec
        denom = 2 * tp_vec + fp_vec + fn_vec
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_dsc = np.where(denom > 0, (2 * tp_vec) / denom, np.nan)

        perm_dsc = float("nan")
        top_confusions: List[Tuple[int, int, int]] = []
        if not is_train and cm.size > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                denom_matrix = 2 * tp_vec[:, None] + fp_vec[None, :] + fn_vec[:, None]
                dice_matrix = np.where(denom_matrix > 0, 2 * cm_float / denom_matrix, 0.0)
            if linear_sum_assignment is not None:
                row_idx, col_idx = linear_sum_assignment(-dice_matrix)
                perm_dsc = float(np.mean(dice_matrix[row_idx, col_idx]))
            else:
                perm_dsc = dsc

            confusion_view = cm.copy()
            np.fill_diagonal(confusion_view, 0)
            flat = confusion_view.reshape(-1)
            order = np.argsort(flat)[::-1]
            width = confusion_view.shape[1]
            for flat_idx in order[:5]:
                count = int(flat[flat_idx])
                if count <= 0:
                    break
                i = flat_idx // width
                j = flat_idx % width
                top_confusions.append((i, j, count))

        # 计算诊断指标的平均值
        avg_bg_ratio = float(np.mean(bg_ratios)) if bg_ratios else 0.0
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        return avg_loss, dsc, sen, ppv, avg_bg_ratio, avg_entropy, perm_dsc, top_confusions, (
            per_class_dsc.tolist() if cm.size > 0 else None
        )

    def train(self) -> None:
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 简化CSV头：
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dsc", "val_dsc_perm", "lr"])

        for epoch in range(1, self.config.epochs + 1):
            if self.amp_enabled:
                torch.cuda.reset_peak_memory_stats(self.device)

            dataset_obj = getattr(self.train_loader, "dataset", None)
            aug_stage = None
            if dataset_obj is not None and hasattr(dataset_obj, "set_augment_stage"):
                warm = self.config.augment_warmup_epochs
                light_span = max(0, getattr(self.config, "augment_light_epochs", 0))
                aug_stage = "off"
                if epoch > warm:
                    if light_span > 0 and epoch <= warm + light_span:
                        aug_stage = "light"
                    else:
                        aug_stage = "full"
                changed = dataset_obj.set_augment_stage(aug_stage)
                if changed:
                    print(f"[Data] Augmentation stage -> {aug_stage} at epoch {epoch}")

            base_lambda = float(self.config.boundary_lambda)
            warm_ratio = min(1.0, max(0.0, (epoch - 1) / 8.0))
            self.boundary_lambda = 1.0 * (1.0 - warm_ratio) + base_lambda * warm_ratio

            epoch_start = time.time()
            train_metrics = self._run_epoch(self.train_loader, is_train=True)
            (
                train_loss,
                train_dsc,
                train_sen,
                train_ppv,
                train_bg,
                train_ent,
                train_perm_dsc,
                _,
                _,
            ) = train_metrics
            epoch_time = time.time() - epoch_start
            train_peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.amp_enabled
                else 0.0
            )
            val_metrics = self._run_epoch(self.val_loader, is_train=False)
            (
                val_loss,
                val_dsc,
                val_sen,
                val_ppv,
                val_bg,
                val_ent,
                val_perm_dsc,
                val_confusions,
                val_per_class,
            ) = val_metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            perm_display = "nan" if math.isnan(val_perm_dsc) else f"{val_perm_dsc:.4f}"
            aug_display = f" | Aug:{aug_stage}" if aug_stage is not None else ""

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val DSC: {val_dsc:.4f} | Val DSC_perm: {perm_display} | "
                f"Val SEN: {val_sen:.4f} | Val PPV: {val_ppv:.4f} | "
                f"Val BG%: {val_bg:.3f} | Val Ent: {val_ent:.3f} | "
                f"Time: {epoch_time:.1f}s | PeakMem: {train_peak_mem:.1f}MB | LR: {current_lr:.6f}"
                f"{aug_display}"
            )
            if val_confusions:
                conf_preview = ", ".join(
                    f"{src}->{dst}:{cnt}"
                    for src, dst, cnt in val_confusions[:3]
                )
                print(f"    Top confusions: {conf_preview}")
            if val_per_class:
                arr = np.asarray(val_per_class, dtype=np.float64)
                valid_idx = np.where(~np.isnan(arr))[0]
                if valid_idx.size > 0:
                    worst = valid_idx[np.argsort(arr[valid_idx])[:3]]
                    worst_str = ", ".join(f"{idx}:{arr[idx]:.3f}" for idx in worst)
                    print(f"    Per-class DSC (worst): {worst_str}")

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 简化记录内容：
                writer.writerow([epoch, train_loss, val_loss, val_dsc, val_perm_dsc, current_lr])

            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("metrics/dsc", val_dsc, epoch)
                if not math.isnan(val_perm_dsc):
                    self.writer.add_scalar("metrics/dsc_perm", val_perm_dsc, epoch)
                self.writer.add_scalar("metrics/sensitivity", val_sen, epoch)
                self.writer.add_scalar("metrics/ppv", val_ppv, epoch)
                # ========== 新增：诊断指标日志 ==========
                self.writer.add_scalar("diagnostics/train_bg_ratio", train_bg, epoch)
                self.writer.add_scalar("diagnostics/train_entropy", train_ent, epoch)
                self.writer.add_scalar("diagnostics/val_bg_ratio", val_bg, epoch)
                self.writer.add_scalar("diagnostics/val_entropy", val_ent, epoch)
                if val_per_class:
                    for cls_idx, value in enumerate(val_per_class):
                        if math.isnan(value):
                            continue
                        self.writer.add_scalar(f"metrics/per_class_dsc/{cls_idx}", float(value), epoch)
                if aug_stage is not None:
                    self.writer.add_text("data/augment_stage", aug_stage, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)
                self.writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
                if self.amp_enabled:
                    self.writer.add_scalar("gpu/peak_mem_mb", train_peak_mem, epoch)
                if val_confusions:
                    preview = ", ".join(
                        f"{src}->{dst}:{cnt}"
                        for src, dst, cnt in val_confusions[:5]
                    )
                    self.writer.add_text("diagnostics/top_confusions", preview, epoch)

            self._save_checkpoint(self.config.checkpoint_dir / "last.pt")
            # 简化保存最佳模型的逻辑：
            if val_dsc > self.best_val_dsc:
                self.best_val_dsc = val_dsc
                self.best_epoch = epoch
                self.epochs_since_best = 0
                print(f"✨ New best model found! DSC: {val_dsc:.4f}. Saving to best.pt")
                self._save_checkpoint(self.config.checkpoint_dir / "best.pt")
            else:
                self.epochs_since_best += 1

            hopeless = epoch >= 16 and self.best_val_dsc < 0.82
            plateau = self.epochs_since_best >= 7 and val_dsc < self.best_val_dsc + 0.008
            if hopeless or plateau:
                print(f"[PlateauGuard] early stop at epoch {epoch} | best_dsc={self.best_val_dsc:.4f}")
                break

            self.scheduler.step()

            if self.epochs_since_best >= self.config.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: no improvement for {self.epochs_since_best} epochs."
                )
                break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        print(f"Training finished. Best validation DSC: {self.best_val_dsc:.4f}")


# =============================================================================
# Configuration & Entrypoint
# =============================================================================


@dataclass
class TrainConfig:
    data_config: DataConfig = field(default_factory=DataConfig)
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_root: Path = Path("outputs/segmentation/logs")
    checkpoint_root: Path = Path("outputs/segmentation/checkpoints")
    tensorboard_root: Path = Path("outputs/segmentation/tensorboard")
    epochs: int = 200
    lr: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    num_classes: int = SEG_NUM_CLASSES
    ce_class_weights: Optional[Sequence[float]] = None
    enable_amp: bool = True
    augment_warmup_epochs: int = 20
    augment_light_epochs: int = 6
    boundary_lambda: float = 3.0
    boundary_knn_k: int = 16
    early_stop_patience: int = 18
    seed: Optional[int] = None
    deterministic: bool = False

    log_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    tensorboard_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_root) / self.run_name
        self.checkpoint_dir = Path(self.checkpoint_root) / self.run_name
        self.tensorboard_dir = Path(self.tensorboard_root) / self.run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Module1 training for iMeshSegNet")
    parser.add_argument("--run-name", type=str, help="Custom name for this run; defaults to a timestamp.")
    parser.add_argument("--log-dir", type=str, help="Directory to store CSV logs; defaults to logs/<run-name>.")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to store checkpoints; defaults to checkpoints/<run-name>.")
    parser.add_argument("--tensorboard-dir", type=str, help="Directory to store TensorBoard events; defaults to tensorboard/<run-name>.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, help="Initial learning rate.")
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate for the cosine scheduler.")
    parser.add_argument("--weight-decay", type=float, help="Optimizer weight decay.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable automatic mixed precision even when CUDA is available.")
    parser.add_argument("--augment-warmup-epochs", type=int, help="Epochs to keep augmentation disabled (Gate-1 cold start).")
    parser.add_argument("--augment-light-epochs", type=int, help="Epoch count for mirror+jitter light augmentation before full augment.")
    parser.add_argument("--boundary-lambda", type=float, help="Extra loss multiplier (λ) for boundary cells.")
    parser.add_argument("--boundary-knn", type=int, help="k used for boundary-aware neighborhood weighting.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic cuDNN (may slow training).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_name:
        config = TrainConfig(run_name=args.run_name)
    else:
        config = TrainConfig()

    if args.log_dir:
        config.log_dir = Path(args.log_dir)
    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)
    if args.tensorboard_dir:
        config.tensorboard_dir = Path(args.tensorboard_dir)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.min_lr is not None:
        config.min_lr = args.min_lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.augment_warmup_epochs is not None:
        config.augment_warmup_epochs = args.augment_warmup_epochs
    if args.augment_light_epochs is not None:
        config.augment_light_epochs = args.augment_light_epochs
    if args.boundary_lambda is not None:
        config.boundary_lambda = args.boundary_lambda
    if args.boundary_knn is not None:
        config.boundary_knn_k = args.boundary_knn
    if args.disable_amp:
        config.enable_amp = False
    if args.batch_size is not None:
        config.data_config.batch_size = max(1, args.batch_size)
    if args.num_workers is not None:
        config.data_config.num_workers = max(0, args.num_workers)
    if args.seed is not None:
        config.seed = args.seed
    if args.deterministic:
        config.deterministic = True

    print(
        "Configured run directories:\n"
        f"  run_name      : {config.run_name}\n"
        f"  logs          : {config.log_dir}\n"
        f"  checkpoints   : {config.checkpoint_dir}\n"
        f"  tensorboard   : {config.tensorboard_dir}"
    )

    seed_to_use = config.seed if config.seed is not None else getattr(config.data_config, "seed", 42)
    config.data_config.seed = seed_to_use
    set_seed(seed_to_use, deterministic=config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data pipeline
    config.data_config.pin_memory = device.type == "cuda"
    config.data_config.persistent_workers = config.data_config.num_workers > 0
    # ========== 快改开关2：关闭增强做"冷启" ==========
    config.data_config.augment = False  # ← 改3：先关闭增强，待 DSC 抬头后再逐步打开
    config.data_config.label_mode = "single_arch_16"
    config.data_config.gingiva_src_label = 0
    config.data_config.gingiva_class_id = 15
    config.data_config.keep_void_zero = False  # 0 表示牙龈时禁止保留为背景
    config.num_classes = SINGLE_ARCH_NUM_CLASSES

    # TODO 优化5：增强 Warmup（待 DSC 稳定后启用）
    # 实现方案：在 Trainer.train() 里动态调整
    # if epoch <= 10:
    #     train_loader.dataset.augment = False
    # else:
    #     train_loader.dataset.augment = True

    # ========== 快改开关3：CE 类别权重 ==========
    # 选项 A：使用均匀权重（排查是否"重加权过度"）
    # config.ce_class_weights = [1.0] * config.num_classes
    train_files, _ = load_split_lists(config.data_config.split_path)
    class_hist = compute_label_histogram(
        train_files,
        label_mode=config.data_config.label_mode,
        gingiva_src_label=config.data_config.gingiva_src_label,
        gingiva_class_id=config.data_config.gingiva_class_id,
        keep_void_zero=config.data_config.keep_void_zero,
    )

    # 选项 B：使用更温和的加权（改 build_ce_class_weights 的 clip 范围）
    if config.ce_class_weights is None:
        print("Estimating class frequencies for CE weights...")
        ce_weights = build_ce_class_weights(class_hist, clip_min=0.3, clip_max=3.0)
        config.ce_class_weights = ce_weights.tolist()
        print(
            f"CrossEntropy class weights prepared: min={ce_weights.min():.3f}, "
            f"max={ce_weights.max():.3f}"
        )

    stats_mean, stats_std = load_stats(config.data_config.stats_path)

    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config.data_config)

    # Model + Optimizer + Scheduler
    print("Initializing model, loss function, optimizer, and scheduler...")
    # ========== 快改开关1：关掉重度 Dropout，开启 STN ==========
    model = iMeshSegNet(
        num_classes=config.num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=False,        # ← 改1：先关闭 Dropout（或改成 0.1）
        dropout_p=0.1,             # ← 如果 with_dropout=True，用更轻的值
        use_feature_stn=True,      # ← 改2：开启特征 STN，抵消增强带来的旋转
    ).to(device)

    # ========== 调整 classifier bias：使用零均值 log-先验，避免偏向牙龈 ==========
    priors = np.maximum(class_hist, 1) / max(class_hist.sum(), 1)   # 平滑一下，避免 0
    logit_bias = torch.log(torch.tensor(priors, dtype=torch.float32, device=device))
    logit_bias = logit_bias - logit_bias.mean()
    with torch.no_grad():
        # classifier 的最后一层是 Sequential 的倒数第一项 Conv1d
        model.classifier[-1].bias = nn.Parameter(logit_bias.clone())
    print(
        "✨ Initialized classifier bias with zero-mean log priors: "
        f"min={logit_bias.min():.3f}, max={logit_bias.max():.3f}"
    )

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        amsgrad=True,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device,
        stats_mean,
        stats_std,
    )
    trainer.train()

    FINAL_SEG_PT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt"):
        src = config.checkpoint_dir / name
        if src.exists():
            copy2(src, FINAL_SEG_PT_DIR / name)

if __name__ == "__main__":
    main()
