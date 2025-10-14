#!/usr/bin/env python3
"""
m5_overfit.py - 单样本过拟合训练脚本
用于验证模型和训练管道的正确性，通过在单个样本上过拟合来快速诊断问题。

使用方法:
    python m5_overfit.py --sample 1_L  # 对1_L.vtp进行过拟合
    python m5_overfit.py --sample 5_U --epochs 500  # 自定义epoch数
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from sklearn.neighbors import NearestNeighbors, KDTree

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from m0_dataset import (
    DECIM_CACHE,
    FDI_LABELS,
    LABEL_REMAP,
    SEG_NUM_CLASSES,
    SINGLE_ARCH_NUM_CLASSES,
    _build_single_arch_label_maps,
    _load_or_build_decimated_mm,
    extract_features,
    trim_feature_dim,
    find_label_array,
    normalize_mesh_units,
    remap_labels_single_arch,
    remap_segmentation_labels,
)
from m1_train import GeneralizedDiceLoss, FocalTverskyLoss
from imeshsegnet import iMeshSegNet, index_points, knn_graph


def calculate_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    """
    计算 DSC, Sensitivity, PPV
    
    Args:
        preds: 预测标签 (N,)
        labels: 真实标签 (N,)
        num_classes: 类别数
        
    Returns:
        (dsc, sensitivity, ppv)
    """
    dsc_list = []
    sen_list = []
    ppv_list = []
    
    for cls in range(1, num_classes):  # 跳过背景类
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        tp = (pred_mask & label_mask).sum().item()
        fp = (pred_mask & ~label_mask).sum().item()
        fn = (~pred_mask & label_mask).sum().item()
        
        if tp + fp + fn == 0:
            continue
            
        dsc = 2 * tp / (2 * tp + fp + fn + 1e-8)
        sen = tp / (tp + fn + 1e-8)
        ppv_val = tp / (tp + fp + 1e-8)
        
        dsc_list.append(dsc)
        sen_list.append(sen)
        ppv_list.append(ppv_val)
    
    return (
        float(np.mean(dsc_list)) if dsc_list else 0.0,
        float(np.mean(sen_list)) if sen_list else 0.0,
        float(np.mean(ppv_list)) if ppv_list else 0.0
    )


def _build_neighbors(pos: np.ndarray, k: int = 12) -> np.ndarray:
    n = pos.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int32)
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(pos)
    neighbors = nn.kneighbors(pos, return_distance=False)
    idx = np.tile(np.arange(n)[:, None], (1, k_eff))
    neighbors = np.where(neighbors != idx, neighbors, -1)
    return neighbors.astype(np.int32, copy=False)


def _boundary_flags(labels: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape[0], dtype=bool)
    for idx, nbrs in enumerate(neighbors):
        for nbr in nbrs:
            if nbr < 0:
                continue
            if labels[nbr] != labels[idx]:
                boundary[idx] = True
                break
    return boundary


def compute_boundary_metrics(
    pos_mm: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    gingiva_label: int = 15,
    tau: float = 0.2,
    neighbor_k: int = 12,
) -> Tuple[float, float, float, float, float]:
    neighbors = _build_neighbors(pos_mm, neighbor_k)
    gt_boundary = _boundary_flags(gt_labels, neighbors)
    pred_boundary = _boundary_flags(pred_labels, neighbors)

    precision = recall = bf1 = 0.0
    if np.any(pred_boundary) and np.any(gt_boundary):
        tree_gt = KDTree(pos_mm[gt_boundary])
        dist_pred, _ = tree_gt.query(pos_mm[pred_boundary], k=1, return_distance=True)
        precision = float(np.mean(dist_pred <= tau)) if dist_pred.size else 0.0

        tree_pred = KDTree(pos_mm[pred_boundary])
        dist_gt, _ = tree_pred.query(pos_mm[gt_boundary], k=1, return_distance=True)
        recall = float(np.mean(dist_gt <= tau)) if dist_gt.size else 0.0

        if precision + recall > 0:
            bf1 = 2 * precision * recall / (precision + recall)

    gingival_mask = gt_boundary & (gt_labels != gingiva_label)
    ger = 0.0
    if np.any(gingival_mask):
        ger = float(np.mean(pred_labels[gingival_mask] == gingiva_label))

    leak_mask = np.zeros(gt_labels.shape[0], dtype=bool)
    leak_target = np.zeros(gt_labels.shape[0], dtype=np.int32)
    for idx, nbrs in enumerate(neighbors):
        base = gt_labels[idx]
        if base <= 0 or base == gingiva_label:
            continue
        for nbr in nbrs:
            if nbr < 0:
                continue
            nb_lbl = gt_labels[nbr]
            if nb_lbl > 0 and nb_lbl != gingiva_label and nb_lbl != base:
                leak_mask[idx] = True
                leak_target[idx] = nb_lbl
                break
    ilr = 0.0
    if np.any(leak_mask):
        ilr = float(np.mean(pred_labels[leak_mask] == leak_target[leak_mask]))

    return bf1, precision, recall, ger, ilr


DEFAULT_LABEL_MODE = "single_arch_16"
DEFAULT_GINGIVA_SRC = 0
DEFAULT_GINGIVA_CLASS_ID = 15
DEFAULT_KEEP_VOID_ZERO = False
FDI_LABEL_SET = set(FDI_LABELS)
INV_LABEL_REMAP = {v: k for k, v in LABEL_REMAP.items()}


class SingleSampleDataset(Dataset):
    """单样本数据集 - 生成一次样本后反复返回"""

    def __init__(
        self,
        sample_file: str,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        label_mode: str = DEFAULT_LABEL_MODE,
        gingiva_src_label: int = DEFAULT_GINGIVA_SRC,
        gingiva_class_id: int = DEFAULT_GINGIVA_CLASS_ID,
        keep_void_zero: bool = DEFAULT_KEEP_VOID_ZERO,
        feature_dim: int = 18,
    ):
        target_cells = 10000
        sample_cells = 6000
        sample_path = Path(sample_file)
        feat_dim = int(feature_dim)
        mean = trim_feature_dim(np.asarray(mean, dtype=np.float32), feat_dim)
        std = trim_feature_dim(np.asarray(std, dtype=np.float32), feat_dim)
        std = np.clip(std, 1e-6, None)

        decimated_mesh = _load_or_build_decimated_mm(sample_path, target_cells=target_cells)
        self.decim_cache_vtp = (DECIM_CACHE / f"{sample_path.stem}.c{target_cells}.vtp").resolve()

        mesh = decimated_mesh.copy(deep=True)
        mesh.points -= mesh.center

        dec_label_info = find_label_array(mesh)
        if dec_label_info is None:
            raise RuntimeError(f"{self.decim_cache_vtp} 缺失 Label 数组")
        _, dec_labels = dec_label_info
        dec_labels = np.asarray(dec_labels, dtype=np.int64)

        full_mesh = pv.read(str(sample_path))
        full_label_info = find_label_array(full_mesh)
        if full_label_info is None:
            raise RuntimeError(f"{sample_file} 缺失 Label 数组")
        _, full_raw_labels = full_label_info
        full_raw_labels = np.asarray(full_raw_labels, dtype=np.int64)

        orig_ids = mesh.cell_data.get("vtkOriginalCellIds")
        if orig_ids is not None and full_raw_labels.size > 0:
            orig_ids_np = np.asarray(orig_ids, dtype=np.int64)
            base_labels = full_raw_labels[orig_ids_np]
        else:
            base_labels = dec_labels

        self.label_mode = label_mode
        self._label_kwargs = {
            "gingiva_src_label": gingiva_src_label,
            "gingiva_class_id": gingiva_class_id,
            "keep_void_zero": keep_void_zero,
        }
        debug_info = {
            "raw_unique": np.unique(base_labels).astype(int).tolist(),
            "intermediate_unique": None,
            "fallback_values": [],
        }
        if label_mode == "single_arch_16":
            raw_unique = np.unique(base_labels)
            needs_inverse = any(
                (val not in FDI_LABEL_SET) and (val != gingiva_src_label) and (val != 0)
                for val in raw_unique
            )
            if needs_inverse:
                base_labels = np.asarray(
                    [INV_LABEL_REMAP.get(int(v), 0) for v in base_labels],
                    dtype=np.int64,
                )
            debug_info["intermediate_unique"] = np.unique(base_labels).astype(int).tolist()
        if label_mode == "single_arch_16":
            single_arch_maps = _build_single_arch_label_maps(
                gingiva_src_label,
                gingiva_class_id,
                keep_void_zero,
            )
            labels = remap_labels_single_arch(base_labels, sample_path, single_arch_maps)
            self.num_classes_full = SINGLE_ARCH_NUM_CLASSES
            debug_info["post_unique"] = np.unique(labels).astype(int).tolist()
        elif label_mode == "full_fdi":
            labels = remap_segmentation_labels(base_labels)
            self.num_classes_full = SEG_NUM_CLASSES
            debug_info["post_unique"] = np.unique(labels).astype(int).tolist()
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        mesh, _, _, diag_after = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()

        labels_full_unique = np.unique(labels).astype(np.int64, copy=False)
        self.labels_full_unique = labels_full_unique

        feats = extract_features(mesh).astype(np.float32, copy=False)
        feats = trim_feature_dim(feats, feat_dim).astype(np.float32, copy=False)
        pos_mm = mesh.cell_centers().points.astype(np.float32)
        pos_scale = diag_after if diag_after > 1e-6 else 1.0
        pos_norm = pos_mm / pos_scale

        rng = np.random.default_rng(42)
        if feats.shape[0] > sample_cells:
            total_cells = feats.shape[0]
            min_samples_per_class = 160
            required_indices: List[int] = []
            unique_labels = np.unique(labels)
            for cls in unique_labels:
                if cls == 0:
                    continue
                cls_indices = np.where(labels == cls)[0]
                if cls_indices.size == 0:
                    continue
                quota = min(min_samples_per_class, sample_cells)
                take = min(cls_indices.size, quota)
                chosen = rng.choice(cls_indices, size=take, replace=cls_indices.size < take)
                required_indices.extend(chosen.tolist())

            if required_indices:
                required_array = np.unique(np.asarray(required_indices, dtype=np.int64))
            else:
                required_array = np.empty(0, dtype=np.int64)

            slots_left = sample_cells - required_array.size
            if slots_left < 0:
                idx = rng.choice(required_array, size=sample_cells, replace=False)
            else:
                mask = np.ones(total_cells, dtype=bool)
                if required_array.size > 0:
                    mask[required_array] = False
                pool = np.nonzero(mask)[0]
                extra_take = min(slots_left, pool.size)
                extra = rng.choice(pool, size=extra_take, replace=False) if extra_take > 0 else np.empty(0, dtype=np.int64)
                if required_array.size > 0:
                    idx = np.concatenate([required_array, extra])
                else:
                    idx = extra
                if idx.size < sample_cells:
                    deficit = sample_cells - idx.size
                    fallback = rng.choice(total_cells, size=deficit, replace=True)
                    idx = np.concatenate([idx, fallback])
            idx = idx[:sample_cells]
            rng.shuffle(idx)
            feats = feats[idx]
            pos_mm = pos_mm[idx]
            pos_norm = pos_norm[idx]
            labels = labels[idx]
        else:
            idx = np.arange(feats.shape[0], dtype=np.int64)

        self.sample_indices = idx.astype(np.int64, copy=False)

        feats_norm = (feats - mean.reshape(1, -1)) / np.clip(std.reshape(1, -1), 1e-6, None)
        features_tensor = torch.from_numpy(feats_norm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_tensor = torch.from_numpy(pos_norm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_mm_tensor = torch.from_numpy(pos_mm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_scale_tensor = torch.tensor([pos_scale], dtype=torch.float32)
        labels_tensor = torch.from_numpy(labels.astype(np.int64))

        self.sample_data = ((features_tensor, pos_tensor, pos_mm_tensor, pos_scale_tensor), labels_tensor)
        self.feature_dim = feat_dim
        self.mean = mean
        self.std = std
        self.remap_debug = debug_info

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx):
        return self.sample_data


def setup_single_sample_training(
    sample_name: str,
    dataset_root: Path,
    feature_dim: int = 15,
) -> Tuple[DataLoader, int, np.ndarray, np.ndarray]:
    """
    设置单样本训练数据
    
    Args:
        sample_name: 样本名称，如 "1_L" 或 "5_U"
        dataset_root: 数据集根目录
        
    Returns:
        (dataloader, num_classes, mean, std): 数据加载器、类别数和标准化参数
    """
    print(f"🔍 设置单样本训练: {sample_name}")
    
    # 构建样本文件路径
    sample_file = dataset_root / f"{sample_name}.vtp"
    if not sample_file.exists():
        raise FileNotFoundError(f"样本文件不存在: {sample_file}")
    
    # 加载统计信息（优先复用缓存，缺失则就地计算）
    stats_path = Path("outputs/segmentation/overfit/_stats_sample.npz")
    need_recompute = True
    if stats_path.exists():
        with np.load(stats_path) as stats:
            mean_full = stats["mean"].astype(np.float32, copy=False)
            std_full = stats["std"].astype(np.float32, copy=False)
        if mean_full.shape[0] >= feature_dim:
            mean = trim_feature_dim(mean_full, feature_dim)
            std = np.clip(trim_feature_dim(std_full, feature_dim), 1e-6, None)
            need_recompute = False
            print(f"✅ 使用缓存统计: {stats_path} (feature_dim={feature_dim})")
        else:
            print(
                f"⚠️ 统计维度不足 ({mean_full.shape[0]} < {feature_dim})，重新计算"
            )
    if need_recompute:
        mesh_mm = _load_or_build_decimated_mm(sample_file, target_cells=10000)
        mesh = mesh_mm.copy(deep=True)
        mesh.points -= mesh.center
        mesh, *_ = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()
        feats = extract_features(mesh).astype(np.float32, copy=False)
        feats = trim_feature_dim(feats, feature_dim)
        mean = feats.mean(axis=0).astype(np.float32, copy=False)
        std = np.clip(feats.std(axis=0), 1e-6, None).astype(np.float32, copy=False)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, mean=mean, std=std)
        print(f"[overfit] built sample stats -> {stats_path} (feature_dim={feature_dim})")

    # 创建单样本数据集
    single_dataset = SingleSampleDataset(
        str(sample_file),
        mean,
        std,
        label_mode=DEFAULT_LABEL_MODE,
        gingiva_src_label=DEFAULT_GINGIVA_SRC,
        gingiva_class_id=DEFAULT_GINGIVA_CLASS_ID,
        keep_void_zero=DEFAULT_KEEP_VOID_ZERO,
        feature_dim=feature_dim,
    )
    remap_debug = getattr(single_dataset, "remap_debug", None)

    # 获取一个样本来确定类别数
    sample_data = single_dataset[0]
    # 数据格式: ((features, pos, pos_mm, pos_scale), labels)
    (features, pos, pos_mm, pos_scale), labels = sample_data
    
    # 💾 保存训练侧数组（用于与推理对比）
    # 注意：features 和 pos 格式是 (C, N)，需要转置为 (N, C) 以便对比
    train_arrays_path = Path("outputs/segmentation/overfit/_train_arrays.npz")
    train_arrays_path.parent.mkdir(parents=True, exist_ok=True)
    feats_np = features.transpose(0, 1).numpy()
    pos_np = pos.transpose(0, 1).numpy()
    np.savez(str(train_arrays_path), feats=feats_np, pos=pos_np)
    print(
        f"💾 保存训练侧数组: {train_arrays_path} "
        f"(shape: feats={features.shape}->{feats_np.shape}, pos={pos.shape}->{pos_np.shape})"
    )
    single_dataset.train_arrays_path = train_arrays_path.resolve()
    
    # 🔬 保存训练时的采样索引（用于推理端复用）
    if hasattr(single_dataset, "sample_indices"):
        train_ids_path = Path("outputs/segmentation/overfit/_train_ids.npy")
        np.save(str(train_ids_path), np.asarray(single_dataset.sample_indices, dtype=np.int64))
        print(f"💾 保存训练采样索引: {train_ids_path} (shape: {np.asarray(single_dataset.sample_indices).shape})")
        single_dataset.train_ids_path = train_ids_path.resolve()
    
    unique_labels = torch.unique(labels)
    default_classes = SINGLE_ARCH_NUM_CLASSES if single_dataset.label_mode == "single_arch_16" else SEG_NUM_CLASSES
    num_classes = int(getattr(single_dataset, "num_classes_full", default_classes))
    labels_full_unique = getattr(single_dataset, "labels_full_unique", None)

    # 创建DataLoader
    # ⚡ 优化：batch_size=2 避免 BatchNorm batch_size=1 问题，同时加速训练
    dataloader = DataLoader(
        single_dataset,
        batch_size=2,
        shuffle=False,  # 单样本不需要shuffle
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✅ 单样本数据集设置完成")
    print(f"   - 样本形状: features={features.shape}, pos={pos.shape}, labels={labels.shape}")
    print(f"   - 标签模式: {single_dataset.label_mode}")
    if remap_debug is not None:
        print(f"   - 原始标签 unique: {remap_debug.get('raw_unique')}")
        if remap_debug.get('intermediate_unique') is not None:
            print(f"   - 转换后标签 unique: {remap_debug.get('intermediate_unique')}")
    if labels_full_unique is not None:
        print(f"   - 全10k唯一标签: {labels_full_unique.tolist()}")
    print(f"   - 6k唯一标签: {unique_labels.tolist()}")
    print(f"   - 类别数: {num_classes}")
    
    return dataloader, num_classes, mean, std


class OverfitTrainer:
    """单样本过拟合训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int,
        device: torch.device,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        *,
        boundary_lambda: float = 2.0,
        boundary_knn_k: int = 16,
        interior_tv_lambda: float = 0.1,
        laplacian_lambda: float = 0.03,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device
        self.mean = mean
        self.std = std
        
        # 设置损失函数
        self.dice_loss = GeneralizedDiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss().to(device)
        self.ce_weight_tensor: torch.Tensor | None = None
        self.boundary_lambda = float(max(0.0, boundary_lambda))
        self.boundary_knn_k = max(1, int(boundary_knn_k))
        self.interior_tv_lambda = float(max(0.0, interior_tv_lambda))
        self.lap_weight = float(max(0.0, laplacian_lambda))
        self.stn_weight = 1e-3
        
        # 设置优化器 - 使用较大的学习率进行快速过拟合
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,  # 比正常训练大10倍
            weight_decay=0.0  # 移除权重衰减以便过拟合
        )
        
        # 混合精度训练
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(device_type) if device.type == 'cuda' else None
        self.use_amp = device.type == 'cuda'
        self.device_type = device_type  # 保存用于 autocast
        
        # 记录训练历史
        self.history = {
            'loss': [],
            'dice_loss': [],
            'ce_loss': [],
            'ft_loss': [],
            'dsc': [],
            'accuracy': [],
            'train_bg0': [],  # 训练集背景比例
            'train_entropy': [],  # 训练集预测熵
            'val_bg0': [],  # 验证集背景比例
            'val_entropy': [],  # 验证集预测熵
            'bf1': [],
            'ger': [],
            'ilr': [],
            'val_loss': [],
            'val_sen': [],
            'val_ppv': [],
        }

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        与主训练保持一致的逐点 CE 计算（含 label smoothing / class weights）。
        """
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
        return torch.sum(loss_map * weights) / torch.clamp(weights.sum(), min=1e-6)

    def _compute_boundary_info(self, labels: torch.Tensor, pos: torch.Tensor):
        require_boundary = self.boundary_lambda > 0 or self.interior_tv_lambda > 0
        if not require_boundary:
            return None, None, None
        with torch.no_grad():
            idx = knn_graph(pos.float(), k=self.boundary_knn_k)
            labels_float = labels.unsqueeze(1).float()
            neigh_labels = index_points(labels_float, idx).squeeze(1).long()
            boundary_mask = (neigh_labels != labels.unsqueeze(-1)).any(dim=-1).float()
        return idx, boundary_mask, neigh_labels

    def _compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos: torch.Tensor,
    ):
        dice_loss = self.dice_loss(logits, labels)
        idx_tv, boundary_mask, neigh_labels = self._compute_boundary_info(labels, pos)
        boundary_weights = None
        if boundary_mask is not None and self.boundary_lambda > 0:
            boundary_weights = 1.0 + boundary_mask * self.boundary_lambda

        ce_loss = self._compute_ce_loss(logits, labels, boundary_weights)
        focal_loss = self.focal_tversky_loss(
            logits,
            labels,
            weights=boundary_weights if boundary_weights is not None else None,
        )

        total_loss = 0.3 * dice_loss + 1.0 * ce_loss + 0.2 * focal_loss

        lap_reg = logits.new_tensor(0.0)
        if logits.size(-1) > 0:
            p = F.softmax(logits, dim=1)
            with torch.no_grad():
                idx_lap = knn_graph(pos.float(), k=8)
            nbr = index_points(p, idx_lap)
            lap_reg = (nbr.mean(dim=-1) - p).pow(2).mean()

        stn_reg = logits.new_tensor(0.0)
        fstn = getattr(self.model, "fstn", None)
        trans = getattr(self.model, "_last_fstn", None)
        if fstn is not None and trans is not None:
            I = torch.eye(trans.size(1), device=trans.device).unsqueeze(0).expand_as(trans)
            stn_reg = ((trans @ trans.transpose(1, 2) - I) ** 2).sum(dim=(1, 2)).mean()

        tv_reg = logits.new_tensor(0.0)
        if self.interior_tv_lambda > 0 and idx_tv is not None and boundary_mask is not None:
            p_tv = F.softmax(logits.float(), dim=1)
            p_nei_tv = index_points(p_tv, idx_tv)
            same_cls = (neigh_labels == labels.unsqueeze(-1)).float()
            interior_mask = (1.0 - boundary_mask).unsqueeze(-1)
            weight_mask = same_cls * interior_mask
            weight_sum = weight_mask.sum()
            if torch.isfinite(weight_sum).item() and weight_sum.item() > 0:
                diff_tv = (p_tv.unsqueeze(-1) - p_nei_tv).pow(2).sum(dim=1)
                tv_reg = (diff_tv * weight_mask).sum() / weight_sum.clamp_min(1e-6)
        total_loss = (
            total_loss
            + self.lap_weight * lap_reg
            + self.stn_weight * stn_reg
            + self.interior_tv_lambda * tv_reg
        )

        return (
            total_loss,
            dice_loss,
            ce_loss,
            focal_loss,
            lap_reg,
            stn_reg,
            tv_reg,
            boundary_mask,
        )
        
    def train_epoch(self) -> dict:
        """训练一个epoch"""
        # ⚡ batch_size=2，无需特殊处理 BatchNorm
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'dice_loss': 0.0,
            'ce_loss': 0.0,
            'ft_loss': 0.0,
            'correct': 0,
            'total': 0
        }
        
        for batch_idx, batch_data in enumerate(self.dataloader):
            # 解析batch数据: ((features, pos, pos_mm, pos_scale), labels)
            (features, pos, pos_mm, pos_scale), labels = batch_data
            
            # 移动到设备
            features = features.to(self.device, non_blocking=True)
            pos = pos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast(self.device_type):
                    logits = self.model(features, pos)
                    (
                        total_loss,
                        dice_loss_t,
                        ce_loss_t,
                        ft_loss_t,
                        lap_reg_t,
                        stn_reg_t,
                        tv_reg_t,
                        boundary_mask,
                    ) = self._compute_losses(
                        logits, labels, pos
                    )
                assert self.scaler is not None
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features, pos)
                (
                    total_loss,
                    dice_loss_t,
                    ce_loss_t,
                    ft_loss_t,
                    lap_reg_t,
                    stn_reg_t,
                    tv_reg_t,
                    boundary_mask,
                ) = self._compute_losses(
                    logits, labels, pos
                )
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.numel()
            
            # 🔍 新增诊断指标：BG0 和 Entropy
            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=1)  # (B, C, N)
                # BG0: 背景类（类别0）的预测比例
                bg_ratio = (preds == 0).float().mean().item()
                # Entropy: 预测熵（衡量不确定性）
                entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1).mean().item()
            
            if 'bg_ratio' not in epoch_metrics:
                epoch_metrics['bg_ratio'] = 0.0
                epoch_metrics['entropy'] = 0.0
            
            # 累积指标
            epoch_metrics['loss'] += total_loss.item()
            epoch_metrics['dice_loss'] += dice_loss_t.item()
            epoch_metrics['ce_loss'] += ce_loss_t.item()
            epoch_metrics['ft_loss'] += ft_loss_t.item()
            epoch_metrics['correct'] += correct
            epoch_metrics['total'] += total
            epoch_metrics['bg_ratio'] += bg_ratio
            epoch_metrics['entropy'] += entropy
        
        # 计算平均指标
        num_batches = len(self.dataloader)
        epoch_metrics['loss'] /= num_batches
        epoch_metrics['dice_loss'] /= num_batches
        epoch_metrics['ce_loss'] /= num_batches
        epoch_metrics['ft_loss'] /= num_batches
        epoch_metrics['accuracy'] = epoch_metrics['correct'] / epoch_metrics['total']
        epoch_metrics['bg_ratio'] /= num_batches
        epoch_metrics['entropy'] /= num_batches
        
        return epoch_metrics
    
    def evaluate(self) -> dict:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_bg_ratios = []
        all_entropy = []
        pos_mm_np: Optional[np.ndarray] = None
        loss_sum = 0.0
        dice_sum = 0.0
        ce_sum = 0.0
        ft_sum = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in self.dataloader:
                # 解析batch数据: ((features, pos, pos_mm, pos_scale), labels)
                (features, pos, pos_mm, pos_scale), labels = batch_data
                
                features = features.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if pos_mm_np is None:
                    pos_mm_np = pos_mm[0].transpose(0, 1).cpu().numpy().astype(np.float32, copy=False)
                
                if self.use_amp:
                    with autocast(self.device_type):
                        logits = self.model(features, pos)
                else:
                    logits = self.model(features, pos)
                
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                (
                    total_loss,
                    dice_loss_t,
                    ce_loss_t,
                    ft_loss_t,
                    _lap,
                    _stn,
                    _tv,
                    _,
                ) = self._compute_losses(logits, labels, pos)
                loss_sum += float(total_loss.item())
                dice_sum += float(dice_loss_t.item())
                ce_sum += float(ce_loss_t.item())
                ft_sum += float(ft_loss_t.item())
                batch_count += 1
                
                # 计算诊断指标
                bg_ratio = (preds == 0).float().mean().item()
                entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1).mean().item()
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_bg_ratios.append(bg_ratio)
                all_entropy.append(entropy)
        
        # 计算DSC
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        dsc, sen, ppv = calculate_metrics(preds_tensor, labels_tensor, self.num_classes)
        bf1 = precision = recall = ger = ilr = 0.0
        if pos_mm_np is not None and preds_tensor.shape[0] > 0:
            bf1, precision, recall, ger, ilr = compute_boundary_metrics(
                pos_mm_np,
                labels_tensor[0].numpy(),
                preds_tensor[0].numpy(),
                gingiva_label=DEFAULT_GINGIVA_CLASS_ID,
            )
        
        return {
            'dsc': dsc, 
            'sensitivity': sen, 
            'ppv': ppv,
            'bg_ratio': np.mean(all_bg_ratios),
            'entropy': np.mean(all_entropy),
            'bf1': bf1,
            'ger': ger,
            'ilr': ilr,
            'boundary_precision': precision,
            'boundary_recall': recall,
            'loss': loss_sum / max(batch_count, 1),
            'dice_loss': dice_sum / max(batch_count, 1),
            'ce_loss': ce_sum / max(batch_count, 1),
            'ft_loss': ft_sum / max(batch_count, 1),
        }
    
    def _save_training_evidence(self, save_dir: Path, sample_name: str):
        """
        🔬 决策树节点1：保存训练证据（logits, labels, metrics）
        
        用于 --replay-train 模式下验证推理是否完全复现训练前向
        """
        import json
        
        self.model.eval()
        all_logits = []
        all_labels = []
        all_preds = []
        
        print("   📊 收集最终 epoch 的 logits 和 labels...")
        
        with torch.no_grad():
            for batch_data in self.dataloader:
                (features, pos, pos_mm, pos_scale), labels = batch_data
                features = features.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(features, pos)  # (B, C, N)
                preds = torch.argmax(logits, dim=1)  # (B, N)
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        # 合并所有批次
        all_logits = np.concatenate(all_logits, axis=0)  # (B, C, N)
        all_labels = np.concatenate(all_labels, axis=0)  # (B, N)
        all_preds = np.concatenate(all_preds, axis=0)    # (B, N)
        
        # 取第一个样本（单样本训练）
        train_logits = all_logits[0]  # (C, N)
        train_labels = all_labels[0]  # (N,)
        train_preds = all_preds[0]    # (N,)
        
        # 保存 logits 和 labels
        logits_path = save_dir.parent / "_train_logits.npy"
        labels_path = save_dir.parent / "_train_labels.npy"
        np.save(str(logits_path), train_logits)
        np.save(str(labels_path), train_labels)
        print(f"   💾 _train_logits.npy (shape: {train_logits.shape})")
        print(f"   💾 _train_labels.npy (shape: {train_labels.shape})")
        
        # 计算并保存 metrics
        dsc, sen, ppv = calculate_metrics(
            torch.from_numpy(train_preds),
            torch.from_numpy(train_labels),
            self.num_classes
        )
        
        # 计算混淆矩阵
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for true_label in range(self.num_classes):
            for pred_label in range(self.num_classes):
                conf_matrix[true_label, pred_label] = np.sum(
                    (train_labels == true_label) & (train_preds == pred_label)
                )
        
        # 计算 margin (置信度)
        probs = np.exp(train_logits) / np.exp(train_logits).sum(axis=0, keepdims=True)  # Softmax
        sorted_probs = np.sort(probs, axis=0)
        margins = sorted_probs[-1, :] - sorted_probs[-2, :]  # p_max - p_second
        
        metrics = {
            "sample_name": sample_name,
            "num_classes": int(self.num_classes),
            "num_cells": int(train_labels.shape[0]),
            "dsc": float(dsc),
            "sensitivity": float(sen),
            "ppv": float(ppv),
            "accuracy": float((train_preds == train_labels).mean()),
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_iou": [],
            "per_class_dsc": [],
            "margin_stats": {
                "mean": float(margins.mean()),
                "std": float(margins.std()),
                "min": float(margins.min()),
                "max": float(margins.max()),
                "q25": float(np.percentile(margins, 25)),
                "q50": float(np.percentile(margins, 50)),
                "q75": float(np.percentile(margins, 75))
            }
        }
        
        # Per-class 指标
        for cls in range(1, self.num_classes):
            pred_mask = (train_preds == cls)
            label_mask = (train_labels == cls)
            tp = np.sum(pred_mask & label_mask)
            fp = np.sum(pred_mask & ~label_mask)
            fn = np.sum(~pred_mask & label_mask)
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                dsc_cls = 2 * tp / (2 * tp + fp + fn)
            else:
                iou = 0.0
                dsc_cls = 0.0
            
            metrics["per_class_iou"].append(float(iou))
            metrics["per_class_dsc"].append(float(dsc_cls))
        
        metrics_path = save_dir.parent / "_train_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   💾 _train_metrics.json (DSC={dsc:.4f}, Acc={metrics['accuracy']:.4f})")
        print(f"   📊 Margin: mean={metrics['margin_stats']['mean']:.3f}, std={metrics['margin_stats']['std']:.3f}")
        print(f"   ✅ 训练证据保存完成！")
    
    def _save_checkpoint_with_pipeline(self, ckpt_path: Path, sample_name: str, epoch: int, dsc: float):
        """
        保存包含完整 pipeline 元数据契约的 checkpoint
        
        这确保推理时能完全复现训练时的前处理流程
        """
        # 构建完整的 checkpoint
        single_dataset = getattr(self.dataloader, "dataset", None)
        train_ids_path_attr = getattr(single_dataset, "train_ids_path", None) if single_dataset is not None else None
        train_ids_path = Path(train_ids_path_attr) if train_ids_path_attr else Path("outputs/segmentation/overfit/_train_ids.npy")
        train_arrays_attr = getattr(single_dataset, "train_arrays_path", None) if single_dataset is not None else None
        train_arrays_path = Path(train_arrays_attr) if train_arrays_attr else None
        decim_hint = getattr(single_dataset, "decim_cache_vtp", None) if single_dataset is not None else None
        decim_cache_vtp = Path(decim_hint) if decim_hint else None
        if decim_cache_vtp is None:
            base_seg_dataset = getattr(single_dataset, "base_dataset", None) if single_dataset is not None else None
            if base_seg_dataset is not None and getattr(base_seg_dataset, "file_paths", None):
                sample_file = Path(base_seg_dataset.file_paths[0])
                target_cells = getattr(base_seg_dataset, "target_cells", None)
                if target_cells is not None:
                    decim_cache_vtp = DECIM_CACHE / f"{sample_file.stem}.c{int(target_cells)}.vtp"
        zscore_mean = self.mean.tolist() if self.mean is not None else None
        zscore_std = self.std.tolist() if self.std is not None else None
        knn_info = {
            "glm1": int(getattr(self.model, "k_short", 6)),
            "glm2": [int(getattr(self.model, "k_short", 6)), int(getattr(self.model, "k_long", 12))],
        }
        if decim_cache_vtp is None or not decim_cache_vtp.exists():
            raise FileNotFoundError(f"Decimated cache mesh not found for {sample_name}: {decim_cache_vtp}")
        if not train_ids_path.exists():
            raise FileNotFoundError(f"Training sample ids missing: {train_ids_path}")
        decim_cache_vtp_str = str(decim_cache_vtp.resolve())
        train_ids_path_str = str(train_ids_path.resolve())
        train_arrays_path_str = str(train_arrays_path.resolve()) if train_arrays_path and train_arrays_path.exists() else None
        fstn_module = getattr(self.model, "fstn", None)
        arch_config = {
            "glm_impl": getattr(self.model, "glm_impl", "edgeconv"),
            "use_feature_stn": bool(fstn_module),
            "k_short": int(getattr(self.model, "k_short", 6)),
            "k_long": int(getattr(self.model, "k_long", 12)),
            "with_dropout": bool(getattr(self.model, "with_dropout", False)),
            "dropout_p": float(getattr(self.model, "dropout_p", 0.0)),
        }

        feature_dim = int(getattr(self.model, "in_channels", 0) or getattr(single_dataset, "feature_dim", 18))

        checkpoint = {
            # 模型权重
            "state_dict": self.model.state_dict(),
            
            # 模型架构信息
            "num_classes": self.num_classes,
            "in_channels": feature_dim,  # feature dimension propagated to inference
            "arch": arch_config,
            "train_sample_ids_path": train_ids_path_str,
            "train_arrays_path": train_arrays_path_str,

            # 前处理 pipeline 契约
            "pipeline": {
                # Z-score 标准化
                "zscore": {
                    "mean": zscore_mean,
                    "std": zscore_std,
                    "apply": True
                },
                
                # 几何预处理
                "centered": True,          # 特征提取前减质心
                "div_by_diag": True,        # 位置按盒对角线归一
                "use_frame": False,        # overfit 不使用 arch frame
                
                # 采样策略
                "sampler": "class_balanced",  # overfit 使用分层采样
                "sample_cells": 6000,      # 采样的 cell 数量
                "target_cells": 10000,     # 抽取后的目标 cell 数量
                
                # 特征布局（用于推理时的旋转对齐）
                "feature_layout": {
                    "rotate_blocks": [
                        [0, 3],    # triangle vertex 0 relative to centroid
                        [3, 6],    # vertex 1
                        [6, 9],    # vertex 2
                        [9, 12],   # face normal
                        [12, 15]   # relative centroid
                    ],
                    "extra_blocks": [
                        [15, feature_dim]  # geometric cues (edge length, area, curvature)
                    ]
                },
                
                # 随机种子（便于复现）
                "seed": 42
            },
            
            # 训练信息
            "training": {
                "sample_name": sample_name,
                "epoch": epoch,
                "best_dsc": dsc,
                "optimizer": "Adam",
                "lr": 0.01,
                "weight_decay": 0.0
            },
            
            # 标签信息（FDI 编码 → 连续索引映射）
            "label_mapping": {
                "fdi_labels": [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
                "num_classes": self.num_classes,
                "background_class": 0
            }
        }
        
        checkpoint["pipeline"].update({
            "decim_cache_vtp": decim_cache_vtp_str,
            "train_ids_path": train_ids_path_str,
            "train_arrays_path": train_arrays_path_str,
            "diag_mode": "cells",
            "zscore_mean": zscore_mean,
            "zscore_std": zscore_std,
            "knn_k": {"to10k": 5, "tofull": 7}, 
            "train_sample_ids_path": train_ids_path_str,
            
        })
        
        checkpoint["training"].update({
            "seed": 42,
            "numpy_rng": 42,
            "torch_seed": 42,
        })
        torch.save(checkpoint, ckpt_path)
        print(f"💾 保存 checkpoint (含 pipeline 契约): {ckpt_path.name}")
    
    def train(self, epochs: int, save_dir: Path, sample_name: str):
        """执行过拟合训练"""
        print(f"\n🚀 开始单样本过拟合训练:")
        print(f"   - 样本: {sample_name}")
        print(f"   - Epochs: {epochs}")
        print(f"   - 设备: {self.device}")
        print(f"   - 学习率: {self.optimizer.param_groups[0]['lr']}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        best_dsc = 0.0
        log_path = save_dir / "train_log.csv"
        log_file = log_path.open("w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_dsc", "val_dsc_perm", "val_sen", "val_ppv", "lr"]
        )
        
        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self.train_epoch()
                should_evaluate = (epoch == 1 or epoch % 10 == 0 or epoch == epochs)

                if should_evaluate:
                    eval_metrics = self.evaluate()

                    self.history['loss'].append(train_metrics['loss'])
                    self.history['dice_loss'].append(train_metrics['dice_loss'])
                    self.history['ce_loss'].append(train_metrics['ce_loss'])
                    self.history['ft_loss'].append(train_metrics['ft_loss'])
                    self.history['dsc'].append(eval_metrics['dsc'])
                    self.history['accuracy'].append(train_metrics['accuracy'])
                    self.history['train_bg0'].append(train_metrics['bg_ratio'])
                    self.history['train_entropy'].append(train_metrics['entropy'])
                    self.history['val_bg0'].append(eval_metrics['bg_ratio'])
                    self.history['val_entropy'].append(eval_metrics['entropy'])
                    self.history['bf1'].append(eval_metrics['bf1'])
                    self.history['ger'].append(eval_metrics['ger'])
                    self.history['ilr'].append(eval_metrics['ilr'])
                    self.history['val_loss'].append(eval_metrics['loss'])
                    self.history['val_sen'].append(eval_metrics['sensitivity'])
                    self.history['val_ppv'].append(eval_metrics['ppv'])

                    if eval_metrics['dsc'] > best_dsc:
                        best_dsc = eval_metrics['dsc']
                        self._save_checkpoint_with_pipeline(
                            save_dir / f"best_overfit_{sample_name}.pt",
                            sample_name,
                            epoch,
                            eval_metrics['dsc'],
                        )

                    lr = self.optimizer.param_groups[0]['lr']
                    log_writer.writerow(
                        [
                            epoch,
                            f"{train_metrics['loss']:.6f}",
                            f"{eval_metrics['loss']:.6f}",
                            f"{eval_metrics['dsc']:.6f}",
                            f"{eval_metrics['dsc']:.6f}",
                            f"{eval_metrics['sensitivity']:.6f}",
                            f"{eval_metrics['ppv']:.6f}",
                            f"{lr:.6e}",
                        ]
                    )
                    log_file.flush()

                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Loss: {train_metrics['loss']:.6f} | "
                        f"DSC: {eval_metrics['dsc']:.4f} | "
                        f"Acc: {train_metrics['accuracy']:.4f} | "
                        f"🔍 Train BG0: {train_metrics['bg_ratio']:.3f} | "
                        f"Train Ent: {train_metrics['entropy']:.3f} | "
                        f"Val BG0: {eval_metrics['bg_ratio']:.3f} | "
                        f"Val Ent: {eval_metrics['entropy']:.3f} | "
                        f"BF1@0.2: {eval_metrics['bf1']:.4f} | "
                        f"GER: {eval_metrics['ger']*100:.2f}% | "
                        f"ILR: {eval_metrics['ilr']*100:.2f}%",
                        flush=True,
                    )
                else:
                    if epoch % 5 == 0:
                        print(
                            f"Epoch {epoch:3d}/{epochs} | "
                            f"Loss: {train_metrics['loss']:.6f} | "
                            f"Acc: {train_metrics['accuracy']:.4f} | "
                            f"🔍 BG0: {train_metrics['bg_ratio']:.3f} | "
                            f"Ent: {train_metrics['entropy']:.3f}",
                            flush=True,
                        )
        finally:
            log_file.close()

        print(f"\n✅ 过拟合训练完成! 最佳DSC: {best_dsc:.4f}")
        
        # 🔬 决策树节点1：保存最终 epoch 的 logits 和 labels
        print(f"\n🔬 保存训练证据（决策树节点1）...")
        self._save_training_evidence(save_dir, sample_name)
        
        # 保存训练历史
        self.save_training_plots(save_dir, sample_name)
        
        return best_dsc
    
    def save_training_plots(self, save_dir: Path, sample_name: str):
        """保存训练曲线图"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        plt.figure(figsize=(20, 12))
        
        # Loss curves
        plt.subplot(3, 3, 1)
        plt.plot(epochs, self.history['loss'], 'b-', label='Total Loss')
        plt.plot(epochs, self.history['dice_loss'], 'r-', label='Dice Loss')
        plt.plot(epochs, self.history['ce_loss'], 'g-', label='CE Loss')
        plt.plot(epochs, self.history['ft_loss'], 'm-', label='Focal-Tversky Loss')
        plt.plot(epochs, self.history['val_loss'], 'k--', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.yscale('log')
        
        # DSC curve
        plt.subplot(3, 3, 2)
        plt.plot(epochs, self.history['dsc'], 'b-', label='DSC')
        plt.plot(epochs, self.history['val_sen'], 'g--', label='Sensitivity')
        plt.plot(epochs, self.history['val_ppv'], 'r--', label='PPV')
        plt.xlabel('Epoch')
        plt.ylabel('DSC')
        plt.title('Dice Similarity Coefficient')
        plt.legend()
        
        # Accuracy curve
        plt.subplot(3, 3, 3)
        plt.plot(epochs, self.history['accuracy'], 'g-', label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        
        # BG0 ratio curves (key diagnostic)
        plt.subplot(3, 3, 4)
        plt.plot(epochs, self.history['train_bg0'], 'r-', label='Train BG0', linewidth=2)
        plt.plot(epochs, self.history['val_bg0'], 'b-', label='Val BG0', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('BG0 Ratio')
        plt.title('Background Prediction Ratio (should drop fast)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 🔍 Entropy 曲线（关键诊断指标）
        plt.subplot(3, 3, 5)
        plt.plot(epochs, self.history['train_entropy'], 'r-', label='Train Entropy', linewidth=2)
        plt.plot(epochs, self.history['val_entropy'], 'b-', label='Val Entropy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Prediction Entropy (should drop fast)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Boundary metrics
        ax6 = plt.subplot(3, 3, 6)
        if self.history['bf1']:
            ax6.plot(epochs, self.history['bf1'], 'b-', label='BF1@0.2')
            ax6.set_ylim(0.0, 1.0)
            ax6.set_ylabel('BF1')
            ax6.set_xlabel('Epoch')
            ax6.set_title('Boundary Metrics')
            ax6.grid(True, alpha=0.3)
            ax6_t = ax6.twinx()
            ger_percent = np.array(self.history['ger']) * 100.0
            ilr_percent = np.array(self.history['ilr']) * 100.0
            ax6_t.plot(epochs, ger_percent, 'r--', label='GER (%)')
            ax6_t.plot(epochs, ilr_percent, 'g-.', label='ILR (%)')
            ax6_t.set_ylabel('Percent (%)')
            lines, labels = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_t.get_legend_handles_labels()
            ax6_t.legend(lines + lines2, labels + labels2, loc='upper right')
        else:
            ax6.set_title('Boundary Metrics (N/A)')
        
        # Last 50 epochs loss
        if len(epochs) > 50:
            plt.subplot(3, 3, 7)
            last_50 = epochs[-50:]
            plt.plot(last_50, self.history['loss'][-50:], 'b-', label='Total Loss')
            plt.plot(last_50, self.history['dice_loss'][-50:], 'r-', label='Dice Loss')
            plt.plot(last_50, self.history['ce_loss'][-50:], 'g-', label='CE Loss')
            plt.plot(last_50, self.history['ft_loss'][-50:], 'm-', label='Focal-Tversky Loss')
            plt.plot(last_50, self.history['val_loss'][-50:], 'k--', label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Last 50 Epochs - Loss')
            plt.legend()
            plt.yscale('log')
        
        # Last 50 epochs DSC
        if len(epochs) > 50:
            plt.subplot(3, 3, 8)
            plt.plot(last_50, self.history['dsc'][-50:], 'b-', label='DSC')
            plt.plot(last_50, self.history['val_sen'][-50:], 'g--', label='Sensitivity')
            plt.plot(last_50, self.history['val_ppv'][-50:], 'r--', label='PPV')
            plt.xlabel('Epoch')
            plt.ylabel('DSC')
            plt.title('Last 50 Epochs - DSC')
            plt.legend()
            
            # Last 50 epochs BG0 (key diagnostic)
            plt.subplot(3, 3, 9)
            plt.plot(last_50, self.history['train_bg0'][-50:], 'r-', label='Train BG0')
            plt.plot(last_50, self.history['val_bg0'][-50:], 'b-', label='Val BG0')
            plt.xlabel('Epoch')
            plt.ylabel('BG0 Ratio')
            plt.title('Last 50 Epochs - BG0 Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Single Sample Overfitting - {sample_name}', fontsize=16)
        plt.tight_layout()
        
        plot_file = save_dir / f"overfit_curves_{sample_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存: {plot_file}")

def _save_overfit_checkpoint(model, ckpt_path: Path, *,
                             num_classes: int,
                             mean: np.ndarray, std: np.ndarray,
                             sample_cells: int = 6000,
                             target_cells: int = 10000,
                             train_ids_path: Path | None,
                             decim_cache_vtp: Path | None,
                             train_arrays_path: Path | None = None):
    fstn_module = getattr(model, "fstn", None)
    arch_config = {
        "glm_impl": getattr(model, "glm_impl", "edgeconv"),
        "use_feature_stn": bool(fstn_module),
        "k_short": int(getattr(model, "k_short", 6)),
        "k_long": int(getattr(model, "k_long", 12)),
        "with_dropout": bool(getattr(model, "with_dropout", False)),
        "dropout_p": float(getattr(model, "dropout_p", 0.0)),
    }
    train_ids_str = str(train_ids_path.resolve()) if train_ids_path else None
    train_arrays_str = str(train_arrays_path.resolve()) if train_arrays_path else None
    decim_cache_str = str(decim_cache_vtp.resolve()) if decim_cache_vtp else None

    payload = {
        "state_dict": model.state_dict(),
        "num_classes": int(num_classes),
        "in_channels": feature_dim,
        "arch": arch_config,
        # 兼容字段（老版本会从这些键读取）
        "train_sample_ids_path": train_ids_str,
        "train_arrays_path": train_arrays_str,
        # 统一契约（推理端优先使用这里的字段）
        "pipeline": {
            "zscore": {
                "apply": True,
                "mean": mean.astype(np.float32).tolist(),
                "std":  np.clip(std, 1e-6, None).astype(np.float32).tolist(),
            },
            "centered": True,          # 训练时 mesh.points -= center
            "div_by_diag": True,       # 训练位置 pos_norm / diag
            "use_frame": False,        # 若后续提供 arch frame 可切 True
            "sampler": "random",       # 训练 6k 随机采样（推理复用 train_ids）
            "sample_cells": int(sample_cells),
            "target_cells": int(target_cells),
            "train_ids_path": train_ids_str,
            "train_arrays_path": train_arrays_str,
            "decim_cache_vtp": decim_cache_str,
            # 记录 knn k（供后处理兜底）
            "knn_k": {"to10k": 5, "tofull": 7},
            "diag_mode": "cells",
            "seed": 42,
        },
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(ckpt_path))
    print(f"💾 保存 checkpoint (含 pipeline 契约): {ckpt_path.name}")
    
def main():
    parser = argparse.ArgumentParser(description="单样本过拟合训练")
    parser.add_argument("--sample", type=str, required=True, 
                       help="样本名称，如 1_L 或 5_U")
    parser.add_argument("--epochs", type=int, default=300,
                       help="训练epoch数 (默认: 300)")
    parser.add_argument("--dataset-root", type=str, default="datasets/segmentation_dataset",
                       help="数据集根目录")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 (cuda/cpu/auto)")
    parser.add_argument("--feature-dim", type=int, default=18,
                       help="特征维度 (推荐 18 = 论文+三几何增强；可切换 15 做 ablation)")
    parser.add_argument("--boundary-lambda", type=float, default=2.0,
                       help="边界样本的额外交叉熵权重 λ（默认 2.0，可改 1.5/2.5 做 ablation）")
    parser.add_argument("--boundary-knn", type=int, default=16,
                       help="边界/TV 正则使用的近邻数量 k")
    parser.add_argument("--interior-tv-lambda", type=float, default=0.1,
                       help="同类 interior TV 正则权重（默认 0.1）")
    parser.add_argument("--laplacian-lambda", type=float, default=0.03,
                       help="Graph Laplacian 平滑正则权重（默认 0.03）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 iMeshSegNet 单样本过拟合训练")
    print("=" * 60)
    print(f"样本: {args.sample}")
    print(f"Epochs: {args.epochs}")
    print(f"数据集: {args.dataset_root}")
    feature_dim = max(1, args.feature_dim)
    print(f"特征维度: {feature_dim}")
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    
    # 设置数据
    dataset_root = Path(args.dataset_root)
    dataloader, num_classes, mean, std = setup_single_sample_training(
        args.sample,
        dataset_root,
        feature_dim=feature_dim,
    )
    
    # 创建模型
    print(f"\n🏗️  创建模型 (类别数: {num_classes})")
    model = iMeshSegNet(
        num_classes=num_classes,
        with_dropout=False,
        dropout_p=0.1,
        use_feature_stn=False,
        k_short=6,
        k_long=12,
        in_channels=feature_dim,
    )
    
    # 创建训练器（传入 mean, std 用于保存 pipeline 契约）
    trainer = OverfitTrainer(
        model,
        dataloader,
        num_classes,
        device,
        mean,
        std,
        boundary_lambda=args.boundary_lambda,
        boundary_knn_k=max(1, args.boundary_knn),
        interior_tv_lambda=max(0.0, args.interior_tv_lambda),
        laplacian_lambda=max(0.0, args.laplacian_lambda),
    )
    
    # 输出目录
    output_dir = Path("outputs/segmentation/overfit") / args.sample
    
    # 开始训练
    start_time = time.time()
    best_dsc = trainer.train(args.epochs, output_dir, args.sample)
    end_time = time.time()
    
    print(f"\n🎉 训练完成!")
    print(f"   - 最佳DSC: {best_dsc:.4f}")
    print(f"   - 训练时间: {end_time - start_time:.1f}秒")
    print(f"   - 输出目录: {output_dir}")
    
    # 期望结果分析
    print(f"\n📈 结果分析:")
    if best_dsc > 0.95:
        print(f"   ✅ 优秀! DSC > 0.95，模型能够完美过拟合单样本")
    elif best_dsc > 0.8:
        print(f"   ✅ 良好! DSC > 0.8，模型基本正常，可能需要更多epoch")
    elif best_dsc > 0.5:
        print(f"   ⚠️  一般! DSC > 0.5，模型学习能力有限，检查数据或模型结构")
    else:
        print(f"   ❌ 异常! DSC < 0.5，存在严重问题:")
        print(f"      - 检查数据格式和标签映射")
        print(f"      - 检查模型输入输出维度")
        print(f"      - 检查损失函数配置")


if __name__ == "__main__":
    main()


