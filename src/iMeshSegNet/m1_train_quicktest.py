#!/usr/bin/env python3
"""
MeshSegNet 快速测试版本 - 用于验证训练流程
仅训练2个epoch，快速验证所有组件是否正常工作
"""
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

import sys
sys.path.insert(0, str(Path(__file__).parent))

from m0_dataset import (
    DataConfig,
    SEG_NUM_CLASSES,
    get_dataloaders,
    set_seed,
)
from imeshsegnet import iMeshSegNet


# 复制损失函数和指标（简化版）
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 2, 1).float()
        support = torch.sum(onehot, dim=(0, 2))
        device = logits.device
        weights = torch.zeros_like(support, dtype=logits.dtype, device=device)

        if support.numel() > 1:
            fg_mask = torch.ones_like(support, dtype=torch.bool)
            fg_mask[0] = False
            valid_fg = fg_mask & (support > 0)
            weights[valid_fg] = 1.0 / (support[valid_fg] ** 2 + self.epsilon)

        intersection = torch.sum(probs * onehot, dim=(0, 2))
        union = torch.sum(probs, dim=(0, 2)) + torch.sum(onehot, dim=(0, 2))

        numerator = 2 * torch.sum(weights * intersection)
        denominator = torch.sum(weights * union) + self.epsilon
        if numerator == 0:
            return logits.new_tensor(0.0)
        dice = numerator / denominator
        return 1 - dice


@dataclass
class TrainConfig:
    data_config: DataConfig = DataConfig()
    output_dir: Path = Path("outputs/segmentation/quick_test")
    epochs: int = 2  # 快速测试：只训练2个epoch
    lr: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    num_classes: int = SEG_NUM_CLASSES


def main() -> None:
    config = TrainConfig()
    
    # 调整数据配置以加快训练
    config.data_config.batch_size = 2
    config.data_config.num_workers = 0  # CPU模式用0
    config.data_config.augment_original_copies = 1  # 减少增强副本
    config.data_config.augment_flipped_copies = 0   # 跳过镜像增强
    
    set_seed(getattr(config.data_config, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"快速测试模式: {config.epochs} epochs")
    print(f"数据增强: {config.data_config.augment_original_copies} 原始副本 + {config.data_config.augment_flipped_copies} 镜像副本")
    
    config.data_config.pin_memory = device.type == "cuda"
    config.data_config.persistent_workers = False  # 快速测试关闭
    config.data_config.augment = True

    print("\n加载数据集...")
    train_loader, val_loader = get_dataloaders(config.data_config)
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    print("\n初始化模型...")
    model = iMeshSegNet(
        num_classes=config.num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        use_feature_stn=True,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        amsgrad=True,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    
    dice_loss = GeneralizedDiceLoss().to(device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n开始训练...")
    print("=" * 80)
    
    for epoch in range(1, config.epochs + 1):
        # 训练
        model.train()
        total_loss = 0.0
        
        for batch_idx, ((x, pos_norm, pos_mm, pos_scale), y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")):
            x = x.to(device, non_blocking=True)
            pos_norm = pos_norm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, pos_norm)
            loss = dice_loss(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 只处理前5个batch进行快速测试
            if batch_idx >= 4:
                break
        
        avg_loss = total_loss / min(5, len(train_loader))
        
        # 验证（简化版）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, ((x, pos_norm, pos_mm, pos_scale), y) in enumerate(tqdm(val_loader, desc=f"Validation")):
                x = x.to(device, non_blocking=True)
                pos_norm = pos_norm.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                logits = model(x, pos_norm)
                loss = dice_loss(logits, y)
                val_loss += loss.item()
                
                # 只处理前3个batch
                if batch_idx >= 2:
                    break
        
        avg_val_loss = val_loss / min(3, len(val_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"\nEpoch {epoch}/{config.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step()
    
    # 保存模型
    torch.save(model.state_dict(), config.output_dir / "test_model.pt")
    print("\n" + "=" * 80)
    print(f"✅ 快速测试完成！模型已保存至 {config.output_dir / 'test_model.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
