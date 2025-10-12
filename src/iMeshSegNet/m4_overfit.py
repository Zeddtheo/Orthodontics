#!/usr/bin/env python3
"""
m5_overfit.py - 单样本过拟合训练脚本
用于验证模型和训练管道的正确性，通过在单个样本上过拟合来快速诊断问题。

使用方法:
    python m5_overfit.py --sample 1_L  # 对1_L.vtp进行过拟合
    python m5_overfit.py --sample 5_U --epochs 500  # 自定义epoch数
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

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
    find_label_array,
    normalize_mesh_units,
    remap_labels_single_arch,
    remap_segmentation_labels,
)
from m1_train import GeneralizedDiceLoss
from imeshsegnet import iMeshSegNet


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


DEFAULT_LABEL_MODE = "single_arch_16"
DEFAULT_GINGIVA_SRC = 0
DEFAULT_GINGIVA_CLASS_ID = 15
DEFAULT_KEEP_VOID_ZERO = True
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
    ):
        target_cells = 10000
        sample_cells = 6000
        sample_path = Path(sample_file)

        mesh10k_mm = _load_or_build_decimated_mm(sample_path, target_cells=target_cells)
        self.decim_cache_vtp = (DECIM_CACHE / f"{sample_path.stem}.c{target_cells}.vtp").resolve()

        mesh = mesh10k_mm.copy(deep=True)
        mesh.points -= mesh.center

        label_info = find_label_array(mesh)
        if label_info is None:
            raise RuntimeError(f"{sample_file} 缺失 Label 数组")
        _, raw_labels = label_info
        labels = np.asarray(raw_labels, dtype=np.int64)

        self.label_mode = label_mode
        self._label_kwargs = {
            "gingiva_src_label": gingiva_src_label,
            "gingiva_class_id": gingiva_class_id,
            "keep_void_zero": keep_void_zero,
        }
        debug_info = {
            "raw_unique": np.unique(labels).astype(int).tolist(),
            "intermediate_unique": None,
            "fallback_values": [],
        }
        if label_mode == "single_arch_16":
            raw_unique = np.unique(labels)
            needs_inverse = any(
                (val not in FDI_LABEL_SET) and (val != gingiva_src_label) and (val != 0)
                for val in raw_unique
            )
            if needs_inverse:
                labels = np.asarray(
                    [INV_LABEL_REMAP.get(int(v), 0) for v in labels],
                    dtype=np.int64,
                )
            debug_info["intermediate_unique"] = np.unique(labels).astype(int).tolist()
        if label_mode == "single_arch_16":
            single_arch_maps = _build_single_arch_label_maps(
                gingiva_src_label,
                gingiva_class_id,
                keep_void_zero,
            )
            labels = remap_labels_single_arch(labels, sample_path, single_arch_maps)
            self.num_classes_full = SINGLE_ARCH_NUM_CLASSES
            debug_info["post_unique"] = np.unique(labels).astype(int).tolist()
        elif label_mode == "full_fdi":
            labels = remap_segmentation_labels(labels)
            self.num_classes_full = SEG_NUM_CLASSES
            debug_info["post_unique"] = np.unique(labels).astype(int).tolist()
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        mesh, _, _, diag_after = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()

        labels_full_unique = np.unique(labels).astype(np.int64, copy=False)
        self.labels_full_unique = labels_full_unique

        feats = extract_features(mesh).astype(np.float32, copy=False)
        pos_mm = mesh.cell_centers().points.astype(np.float32)
        pos_scale = diag_after if diag_after > 1e-6 else 1.0
        pos_norm = pos_mm / pos_scale

        rng = np.random.default_rng(42)
        if feats.shape[0] > sample_cells:
            idx = rng.choice(feats.shape[0], size=sample_cells, replace=False)
            feats = feats[idx]
            pos_mm = pos_mm[idx]
            pos_norm = pos_norm[idx]
            labels = labels[idx]
        else:
            idx = np.arange(feats.shape[0], dtype=np.int64)

        self.sample_indices = idx.astype(np.int64)

        feats_norm = (feats - mean.reshape(1, -1)) / np.clip(std.reshape(1, -1), 1e-6, None)
        features_tensor = torch.from_numpy(feats_norm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_tensor = torch.from_numpy(pos_norm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_mm_tensor = torch.from_numpy(pos_mm.astype(np.float32)).transpose(0, 1).contiguous()
        pos_scale_tensor = torch.tensor([pos_scale], dtype=torch.float32)
        labels_tensor = torch.from_numpy(labels.astype(np.int64))

        self.sample_data = ((features_tensor, pos_tensor, pos_mm_tensor, pos_scale_tensor), labels_tensor)
        self.remap_debug = debug_info

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx):
        return self.sample_data


def setup_single_sample_training(sample_name: str, dataset_root: Path) -> Tuple[DataLoader, int, np.ndarray, np.ndarray]:
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
    if stats_path.exists():
        with np.load(stats_path) as stats:
            mean = stats["mean"].astype(np.float32, copy=False)
            std = stats["std"].astype(np.float32, copy=False)
        std = np.clip(std, 1e-6, None)
        print(f"✅ 使用缓存统计: {stats_path}")
    else:
        mesh_mm = _load_or_build_decimated_mm(sample_file, target_cells=10000)
        mesh = mesh_mm.copy(deep=True)
        mesh.points -= mesh.center
        mesh, *_ = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()
        feats = extract_features(mesh).astype(np.float32, copy=False)
        mean = feats.mean(axis=0).astype(np.float32, copy=False)
        std = np.clip(feats.std(axis=0), 1e-6, None).astype(np.float32, copy=False)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, mean=mean, std=std)
        print(f"[overfit] built sample stats -> {stats_path}")

    # 创建单样本数据集
    single_dataset = SingleSampleDataset(
        str(sample_file),
        mean,
        std,
        label_mode=DEFAULT_LABEL_MODE,
        gingiva_src_label=DEFAULT_GINGIVA_SRC,
        gingiva_class_id=DEFAULT_GINGIVA_CLASS_ID,
        keep_void_zero=DEFAULT_KEEP_VOID_ZERO,
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
    np.savez(str(train_arrays_path),
             feats=features.transpose(0, 1).numpy(),  # (15, N) -> (N, 15)
             pos=pos.transpose(0, 1).numpy())          # (3, N) -> (N, 3)
    print(f"💾 保存训练侧数组: {train_arrays_path} (shape: feats={features.shape}->{features.transpose(0,1).shape}, pos={pos.shape}->{pos.transpose(0,1).shape})")
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
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, num_classes: int, device: torch.device, 
                 mean: np.ndarray = None, std: np.ndarray = None):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device
        self.mean = mean
        self.std = std
        
        # 设置损失函数
        self.dice_loss = GeneralizedDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
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
            'dsc': [],
            'accuracy': [],
            'train_bg0': [],  # 训练集背景比例
            'train_entropy': [],  # 训练集预测熵
            'val_bg0': [],  # 验证集背景比例
            'val_entropy': []  # 验证集预测熵
        }
        
    def train_epoch(self) -> dict:
        """训练一个epoch"""
        # ⚡ batch_size=2，无需特殊处理 BatchNorm
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'dice_loss': 0.0,
            'ce_loss': 0.0,
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
                    dice_loss = self.dice_loss(logits, labels)
                    ce_loss = self.ce_loss(logits, labels)
                    total_loss = dice_loss + ce_loss
                
                assert self.scaler is not None
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features, pos)
                dice_loss = self.dice_loss(logits, labels)
                ce_loss = self.ce_loss(logits, labels)
                total_loss = dice_loss + ce_loss
                
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            epoch_metrics['dice_loss'] += dice_loss.item()
            epoch_metrics['ce_loss'] += ce_loss.item()
            epoch_metrics['correct'] += correct
            epoch_metrics['total'] += total
            epoch_metrics['bg_ratio'] += bg_ratio
            epoch_metrics['entropy'] += entropy
        
        # 计算平均指标
        num_batches = len(self.dataloader)
        epoch_metrics['loss'] /= num_batches
        epoch_metrics['dice_loss'] /= num_batches
        epoch_metrics['ce_loss'] /= num_batches
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
        
        with torch.no_grad():
            for batch_data in self.dataloader:
                # 解析batch数据: ((features, pos, pos_mm, pos_scale), labels)
                (features, pos, pos_mm, pos_scale), labels = batch_data
                
                features = features.to(self.device, non_blocking=True)
                pos = pos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast(self.device_type):
                        logits = self.model(features, pos)
                else:
                    logits = self.model(features, pos)
                
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
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
        
        return {
            'dsc': dsc, 
            'sensitivity': sen, 
            'ppv': ppv,
            'bg_ratio': np.mean(all_bg_ratios),
            'entropy': np.mean(all_entropy)
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

        checkpoint = {
            # 模型权重
            "state_dict": self.model.state_dict(),
            
            # 模型架构信息
            "num_classes": self.num_classes,
            "in_channels": 15,  # 特征维度（9点坐标 + 3法向 + 3相对位置）
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
                "sampler": "random",       # overfit 使用随机采样
                "sample_cells": 6000,      # 采样的 cell 数量
                "target_cells": 10000,     # 抽取后的目标 cell 数量
                
                # 特征布局（用于推理时的旋转对齐）
                "feature_layout": {
                    "rotate_blocks": [
                        [0, 3],    # v0: 三角形顶点0相对质心
                        [3, 6],    # v1: 三角形顶点1相对质心
                        [6, 9],    # v2: 三角形顶点2相对质心
                        [9, 12],   # normal: 法向量
                        [12, 15]   # cent_rel: cell中心相对质心
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
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics = self.train_epoch()
            
            # ⚡ 优化：只在特定 epoch 进行评估（减少验证开销）
            should_evaluate = (epoch == 1 or epoch % 10 == 0 or epoch == epochs)
            
            if should_evaluate:
                # 评估
                eval_metrics = self.evaluate()
                
                # 记录历史
                self.history['loss'].append(train_metrics['loss'])
                self.history['dice_loss'].append(train_metrics['dice_loss'])
                self.history['ce_loss'].append(train_metrics['ce_loss'])
                self.history['dsc'].append(eval_metrics['dsc'])
                self.history['accuracy'].append(train_metrics['accuracy'])
                self.history['train_bg0'].append(train_metrics['bg_ratio'])
                self.history['train_entropy'].append(train_metrics['entropy'])
                self.history['val_bg0'].append(eval_metrics['bg_ratio'])
                self.history['val_entropy'].append(eval_metrics['entropy'])
                
                # 保存最佳模型（包含完整的 pipeline 元数据契约）
                if eval_metrics['dsc'] > best_dsc:
                    best_dsc = eval_metrics['dsc']
                    self._save_checkpoint_with_pipeline(
                        save_dir / f"best_overfit_{sample_name}.pt",
                        sample_name,
                        epoch,
                        eval_metrics['dsc']
                    )
                
                # 打印进度（包含新诊断指标）
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {train_metrics['loss']:.6f} | "
                      f"DSC: {eval_metrics['dsc']:.4f} | "
                      f"Acc: {train_metrics['accuracy']:.4f} | "
                      f"🔍 Train BG0: {train_metrics['bg_ratio']:.3f} | "
                      f"Train Ent: {train_metrics['entropy']:.3f} | "
                      f"Val BG0: {eval_metrics['bg_ratio']:.3f} | "
                      f"Val Ent: {eval_metrics['entropy']:.3f}", flush=True)
            else:
                # 快速模式：只打印训练指标，不做评估
                if epoch % 5 == 0:  # 每 5 个 epoch 打印一次
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Loss: {train_metrics['loss']:.6f} | "
                          f"Acc: {train_metrics['accuracy']:.4f} | "
                          f"🔍 BG0: {train_metrics['bg_ratio']:.3f} | "
                          f"Ent: {train_metrics['entropy']:.3f}", flush=True)
        
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
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.yscale('log')
        
        # DSC curve
        plt.subplot(3, 3, 2)
        plt.plot(epochs, self.history['dsc'], 'b-', label='DSC')
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
        
        # DSC vs Loss散点图
        plt.subplot(3, 3, 6)
        plt.scatter(self.history['loss'], self.history['dsc'], alpha=0.6)
        plt.xlabel('Total Loss')
        plt.ylabel('DSC')
        plt.title('DSC vs Loss')
        
        # Last 50 epochs loss
        if len(epochs) > 50:
            plt.subplot(3, 3, 7)
            last_50 = epochs[-50:]
            plt.plot(last_50, self.history['loss'][-50:], 'b-', label='Total Loss')
            plt.plot(last_50, self.history['dice_loss'][-50:], 'r-', label='Dice Loss')
            plt.plot(last_50, self.history['ce_loss'][-50:], 'g-', label='CE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Last 50 Epochs - Loss')
            plt.legend()
            plt.yscale('log')
        
        # Last 50 epochs DSC
        if len(epochs) > 50:
            plt.subplot(3, 3, 8)
            plt.plot(last_50, self.history['dsc'][-50:], 'b-', label='DSC')
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
        "in_channels": 15,
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 iMeshSegNet 单样本过拟合训练")
    print("=" * 60)
    print(f"样本: {args.sample}")
    print(f"Epochs: {args.epochs}")
    print(f"数据集: {args.dataset_root}")
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    
    # 设置数据
    dataset_root = Path(args.dataset_root)
    dataloader, num_classes, mean, std = setup_single_sample_training(args.sample, dataset_root)
    
    # 创建模型
    print(f"\n🏗️  创建模型 (类别数: {num_classes})")
    model = iMeshSegNet(num_classes=num_classes, with_dropout=False, use_feature_stn=False)
    
    # 创建训练器（传入 mean, std 用于保存 pipeline 契约）
    trainer = OverfitTrainer(model, dataloader, num_classes, device, mean, std)
    
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


