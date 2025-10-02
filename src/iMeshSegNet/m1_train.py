# module1_train.py
# Stage-1 训练流程：仅使用 Generalized Dice Loss，记录论文口径的 DSC/SEN/PPV/HD（含后处理）。

from __future__ import annotations

import csv
from contextlib import nullcontext
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from m0_dataset import (
    DataConfig,
    SEG_NUM_CLASSES,
    get_dataloaders,
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
        onehot = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 2, 1).float()  # (B,C,N)

        support = torch.sum(onehot, dim=(0, 2))                     # (C,)
        device = logits.device
        weights = torch.zeros_like(support, dtype=logits.dtype, device=device)

        if support.numel() > 1:
            fg_mask = torch.ones_like(support, dtype=torch.bool)
            fg_mask[0] = False
            valid_fg = fg_mask & (support > 0)
            weights[valid_fg] = 1.0 / (support[valid_fg] ** 2 + self.epsilon)

        intersection = torch.sum(probs * onehot, dim=(0, 2))
        union = torch.sum(probs, dim=(0, 2)) + torch.sum(onehot, dim=(0, 2))

        numerator = 2 * torch.sum(weights * intersection) + self.epsilon
        denominator = torch.sum(weights * union) + self.epsilon
        dice = numerator / denominator
        return 1 - dice


def compute_diag_from_points(points: np.ndarray) -> float:
    if points.size == 0:
        return 1.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(diag, 1e-6)


def hausdorff_distance_mm(pred_points: np.ndarray, gt_points: np.ndarray, fallback: float) -> float:
    if pred_points.size == 0 or gt_points.size == 0:
        return float(fallback)
    pred_tensor = torch.from_numpy(pred_points.astype(np.float32))
    gt_tensor = torch.from_numpy(gt_points.astype(np.float32))
    if pred_tensor.numel() == 0 or gt_tensor.numel() == 0:
        return float(fallback)
    dists = torch.cdist(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), p=2).squeeze(0)
    if dists.numel() == 0:
        return float(fallback)
    forward = torch.min(dists, dim=1).values.max()
    backward = torch.min(dists, dim=0).values.max()
    return float(torch.max(forward, backward).item())


def compute_case_metrics(
    pred_labels: np.ndarray,
    target_labels: np.ndarray,
    pos_mm: np.ndarray,
    num_classes: int,
    fallback_diag: float,
) -> Dict[str, float]:
    per_class_stats = []
    hd_values = []
    for cls in range(1, num_classes):
        gt_mask = target_labels == cls
        if not np.any(gt_mask):
            continue
        pred_mask = pred_labels == cls
        tp = np.sum(np.logical_and(gt_mask, pred_mask))
        fp = np.sum(np.logical_and(~gt_mask, pred_mask))
        fn = np.sum(np.logical_and(gt_mask, ~pred_mask))

        denom = (2 * tp + fp + fn)
        dsc = (2 * tp) / denom if denom > 0 else 0.0
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_stats.append((dsc, sen, ppv))

        pred_points = pos_mm[pred_mask]
        gt_points = pos_mm[gt_mask]
        hd = hausdorff_distance_mm(pred_points, gt_points, fallback_diag)
        hd_values.append(hd)

    if not per_class_stats:
        return {
            "dsc": 0.0,
            "sen": 0.0,
            "ppv": 0.0,
            "hd": float(fallback_diag),
        }

    stats = np.asarray(per_class_stats, dtype=np.float32)
    hd_array = np.asarray(hd_values, dtype=np.float32)
    return {
        "dsc": float(stats[:, 0].mean()),
        "sen": float(stats[:, 1].mean()),
        "ppv": float(stats[:, 2].mean()),
        "hd": float(hd_array.mean()),
    }


def graphcut_refine(prob_np: np.ndarray, pos_mm_np: np.ndarray, beta: float = 30.0, k: int = 6, iterations: int = 1) -> np.ndarray:
    """
    NOTE: This is an iterative probability smoothing, an approximation of graph-cut's smoothness effect, 
    not a true energy-minimizing graph-cut. This function performs k-NN based probability smoothing 
    to mimic the spatial consistency constraint of graph-cut algorithms.
    """
    if prob_np.size == 0:
        return prob_np
    N, C = prob_np.shape
    k_eff = max(1, min(k, N))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(pos_mm_np)
    distances, indices = nbrs.kneighbors(pos_mm_np, return_distance=True)
    sigma = distances[:, 1:].mean() if distances.shape[1] > 1 else distances.mean()
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    scale = beta / (sigma + 1e-6)
    weights = np.exp(-scale * distances).astype(np.float32)

    refined = prob_np.astype(np.float32).copy()
    for _ in range(max(1, iterations)):
        neighbor_accum = np.zeros_like(refined)
        weight_sum = np.zeros((N, 1), dtype=np.float32)
        for j in range(k_eff):
            nbr_idx = indices[:, j]
            w = weights[:, j:j + 1]
            neighbor_accum += w * refined[nbr_idx]
            weight_sum += w
        refined = (refined + neighbor_accum) / (1.0 + np.clip(weight_sum, 1e-6, None))
        refined = refined / np.clip(refined.sum(axis=1, keepdims=True), 1e-6, None)
    return refined


def svm_refine(pos_mm_np: np.ndarray, labels_np: np.ndarray) -> np.ndarray:
    """
    WARNING: This function performs SVM refinement on the SAME low-resolution mesh (N_low ≈ 9000),
    NOT on the original high-resolution mesh (N_high ≈ 100k) as described in the paper.
    To properly implement the paper's approach, this function should:
    1. Train SVM on low-res predictions (N_low)
    2. Predict on high-res original mesh (N_high) - CURRENTLY NOT IMPLEMENTED
    Therefore, post-processing metrics computed using this function are INVALID for paper comparison.
    """
    unique = np.unique(labels_np)
    if unique.size <= 1:
        return labels_np
    try:
        max_samples = 5000
        if pos_mm_np.shape[0] > max_samples:
            indices = np.random.choice(pos_mm_np.shape[0], size=max_samples, replace=False)
            train_x = pos_mm_np[indices]
            train_y = labels_np[indices]
        else:
            train_x = pos_mm_np
            train_y = labels_np
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
        clf.fit(train_x.astype(np.float64), train_y)
        refined = clf.predict(pos_mm_np.astype(np.float64))
        return refined.astype(labels_np.dtype, copy=False)
    except Exception:
        return labels_np


def apply_post_processing(prob_np: np.ndarray, pos_mm_np: np.ndarray) -> np.ndarray:
    refined_prob = graphcut_refine(prob_np, pos_mm_np, beta=30.0, k=6, iterations=1)
    graphcut_pred = np.argmax(refined_prob, axis=1).astype(np.int64)
    return svm_refine(pos_mm_np, graphcut_pred)


def summarize_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not metric_list:
        for key in ("dsc", "sen", "ppv", "hd"):
            summary[key] = {"mean": 0.0, "std": 0.0}
        return summary
    for key in ("dsc", "sen", "ppv", "hd"):
        values = np.array([m[key] for m in metric_list], dtype=np.float32)
        finite = np.isfinite(values)
        if not np.any(finite):
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(values[finite].mean())
            std = float(values[finite].std(ddof=0))
        summary[key] = {"mean": mean, "std": std}
    return summary


def evaluate_cases(cases: List[Dict[str, np.ndarray]], num_classes: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    raw_metrics: List[Dict[str, float]] = []
    post_metrics: List[Dict[str, float]] = []
    for case in cases:
        diag_mm = case["diag"]
        raw_metrics.append(
            compute_case_metrics(case["pred"], case["target"], case["pos_mm"], num_classes, diag_mm)
        )
        post_pred = apply_post_processing(case["prob"], case["pos_mm"])
        post_metrics.append(
            compute_case_metrics(post_pred, case["target"], case["pos_mm"], num_classes, diag_mm)
        )
    return summarize_metrics(raw_metrics), summarize_metrics(post_metrics)



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

    def _run_epoch(
        self,
        loader: DataLoader,
        is_train: bool,
    ) -> Tuple[float, Dict[str, float], Optional[Dict[str, Dict[str, float]]], Optional[Dict[str, Dict[str, float]]]]:
        self.model.train(mode=is_train)
        total_loss = 0.0
        desc = "Training" if is_train else "Validation"
        context_manager = nullcontext() if is_train else torch.no_grad()
        case_records: List[Dict[str, np.ndarray]] = []
        bg_ratios: List[float] = []
        entropy_values: List[float] = []

        for (x, pos_norm, pos_mm, pos_scale), y in tqdm(loader, desc=desc):
            x = x.to(self.device, non_blocking=True)
            pos_norm = pos_norm.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pos_mm = pos_mm  # keep on CPU for metric computation
            pos_scale = pos_scale

            if not self._checked_shapes:
                assert x.dim() == 3 and x.size(1) == 15, "x must be (B,15,N) with z-scored features"
                assert pos_norm.dim() == 3 and pos_norm.size(1) == 3, "pos must be (B,3,N)"
                self._checked_shapes = True

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with context_manager:
                with autocast(enabled=self.amp_enabled):
                    logits = self.model(x, pos_norm)
                    loss = self.dice_loss(logits, y)
            probs_detached = torch.softmax(logits.detach(), dim=1)
            preds_detached = torch.argmax(logits.detach(), dim=1)
            bg_ratio = (preds_detached == 0).float().mean().item()
            entropy = -(probs_detached * torch.log(probs_detached.clamp(min=1e-8))).sum(dim=1).mean().item()
            bg_ratios.append(bg_ratio)
            entropy_values.append(entropy)

            if is_train:
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

            total_loss += loss.item()

            if not is_train:
                probs = probs_detached.cpu()  # (B,C,N)
                preds = preds_detached.cpu()    # (B,N)
                targets_cpu = y.cpu()
                pos_mm_cpu = pos_mm.cpu()
                pos_scale_cpu = pos_scale.cpu()
                for i in range(preds.size(0)):
                    case_records.append(
                        {
                            "prob": probs[i].transpose(0, 1).contiguous().numpy(),
                            "pred": preds[i].numpy(),
                            "target": targets_cpu[i].numpy(),
                            "pos_mm": pos_mm_cpu[i].numpy(),
                            "diag": float(pos_scale_cpu[i].item()),
                        }
                    )

        avg_loss = total_loss / max(len(loader), 1)
        diag_stats = {
            "p_bg": float(np.mean(bg_ratios)) if bg_ratios else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "dice": float(avg_loss),
        }
        if not is_train:
            raw_metrics, post_metrics = evaluate_cases(case_records, self.config.num_classes)
            return avg_loss, diag_stats, raw_metrics, post_metrics
        return avg_loss, diag_stats, None, None

    def train(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_p_bg",
                    "train_entropy",
                    "val_p_bg",
                    "val_entropy",
                    "raw_dsc",
                    "raw_dsc_std",
                    "raw_sen",
                    "raw_sen_std",
                    "raw_ppv",
                    "raw_ppv_std",
                    "raw_hd",
                    "raw_hd_std",
                    "post_dsc",
                    "post_dsc_std",
                    "post_sen",
                    "post_sen_std",
                    "post_ppv",
                    "post_ppv_std",
                    "post_hd",
                    "post_hd_std",
                    "lr",
                    "sec_per_scan",
                ]
            )

        def _fmt(value: float, precision: int = 4) -> str:
            return f"{value:.{precision}f}" if np.isfinite(value) else "nan"

        for epoch in range(1, self.config.epochs + 1):
            if self.amp_enabled:
                torch.cuda.reset_peak_memory_stats(self.device)

            epoch_start = time.time()
            train_loss, train_diag, _, _ = self._run_epoch(self.train_loader, is_train=True)
            epoch_time = time.time() - epoch_start

            train_base = getattr(self.train_loader.dataset, "base_len", len(self.train_loader.dataset))
            sec_per_scan = epoch_time / max(train_base, 1)
            train_peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.amp_enabled
                else 0.0
            )

            val_loss, val_diag, val_raw, val_post = self._run_epoch(self.val_loader, is_train=False)
            current_lr = self.optimizer.param_groups[0]["lr"]

            raw_dsc = val_raw["dsc"]["mean"]
            raw_dsc_std = val_raw["dsc"]["std"]
            raw_sen = val_raw["sen"]["mean"]
            raw_sen_std = val_raw["sen"]["std"]
            raw_ppv = val_raw["ppv"]["mean"]
            raw_ppv_std = val_raw["ppv"]["std"]
            raw_hd = val_raw["hd"]["mean"]
            raw_hd_std = val_raw["hd"]["std"]

            post_dsc = val_post["dsc"]["mean"]
            post_dsc_std = val_post["dsc"]["std"]
            post_sen = val_post["sen"]["mean"]
            post_sen_std = val_post["sen"]["std"]
            post_ppv = val_post["ppv"]["mean"]
            post_ppv_std = val_post["ppv"]["std"]
            post_hd = val_post["hd"]["mean"]
            post_hd_std = val_post["hd"]["std"]

            raw_dsc_str = _fmt(raw_dsc)
            raw_dsc_std_str = _fmt(raw_dsc_std)
            raw_sen_str = _fmt(raw_sen)
            raw_sen_std_str = _fmt(raw_sen_std)
            raw_ppv_str = _fmt(raw_ppv)
            raw_ppv_std_str = _fmt(raw_ppv_std)
            raw_hd_str = _fmt(raw_hd, 3)
            raw_hd_std_str = _fmt(raw_hd_std, 3)
            post_dsc_str = _fmt(post_dsc)
            post_dsc_std_str = _fmt(post_dsc_std)
            post_sen_str = _fmt(post_sen)
            post_sen_std_str = _fmt(post_sen_std)
            post_ppv_str = _fmt(post_ppv)
            post_ppv_std_str = _fmt(post_ppv_std)
            post_hd_str = _fmt(post_hd, 3)
            post_hd_std_str = _fmt(post_hd_std, 3)

            # NOTE: Post-processed metrics (post_dsc, post_hd, etc.) are INVALID in current implementation
            # because svm_refine does not perform true upsampling to original resolution.
            # Focus on raw_ metrics for model evaluation.
            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train BG0: {train_diag['p_bg']:.3f} | "
                f"Train Ent: {train_diag['entropy']:.3f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BG0: {val_diag['p_bg']:.3f} | "
                f"Val Ent: {val_diag['entropy']:.3f} | "
                f"Raw DSC: {raw_dsc_str}±{raw_dsc_std_str} | Raw SEN: {raw_sen_str}±{raw_sen_std_str} | "
                f"Raw PPV: {raw_ppv_str}±{raw_ppv_std_str} | Raw HD: {raw_hd_str}±{raw_hd_std_str} mm | "
                # Post-processing metrics commented out as they are invalid without proper upsampling:
                # f"Post DSC: {post_dsc_str}±{post_dsc_std_str} | Post SEN: {post_sen_str}±{post_sen_std_str} | "
                # f"Post PPV: {post_ppv_str}±{post_ppv_std_str} | Post HD: {post_hd_str}±{post_hd_std_str} mm | "
                f"Time: {epoch_time:.1f}s | Sec/scan: {sec_per_scan:.2f}s | PeakMem: {train_peak_mem:.1f}MB | LR: {current_lr:.6f}"
            )

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{train_diag.get('p_bg', 0.0):.6f}",
                        f"{train_diag.get('entropy', 0.0):.6f}",
                        f"{val_diag.get('p_bg', 0.0):.6f}",
                        f"{val_diag.get('entropy', 0.0):.6f}",
                        raw_dsc_str,
                        raw_dsc_std_str,
                        raw_sen_str,
                        raw_sen_std_str,
                        raw_ppv_str,
                        raw_ppv_std_str,
                        raw_hd_str,
                        raw_hd_std_str,
                        post_dsc_str,
                        post_dsc_std_str,
                        post_sen_str,
                        post_sen_std_str,
                        post_ppv_str,
                        post_ppv_std_str,
                        post_hd_str,
                        post_hd_std_str,
                        f"{current_lr:.8f}",
                        f"{sec_per_scan:.6f}",
                    ]
                )

            torch.save(self.model.state_dict(), self.config.output_dir / "last.pt")
            if np.isfinite(raw_dsc) and raw_dsc > self.best_val_dsc:
                self.best_val_dsc = raw_dsc
                print(f"✨ New best model found! Raw DSC: {raw_dsc:.4f}. Saving to best.pt")
                torch.save(self.model.state_dict(), self.config.output_dir / "best.pt")

            self.scheduler.step()

        print(f"Training finished. Best validation raw DSC: {self.best_val_dsc:.4f}")


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


def main() -> None:
    config = TrainConfig()
    set_seed(getattr(config.data_config, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data pipeline (always keep online augmentation for train loader)
    config.data_config.pin_memory = device.type == "cuda"
    config.data_config.persistent_workers = True
    config.data_config.augment = True

    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config.data_config)

    # Model + Optimizer + Scheduler
    print("Initializing model, loss function, optimizer, and scheduler...")
    model = iMeshSegNet(
        num_classes=config.num_classes,
        glm_impl="edgeconv",
        k_short=6,              # 论文：短距离邻域
        k_long=12,              # 论文：长距离邻域
        use_feature_stn=True,   # 论文要求：启用 64×64 特征变换
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
