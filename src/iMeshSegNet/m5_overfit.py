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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from m0_dataset import SegmentationDataset, load_split_lists
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


class SingleSampleDataset(Dataset):
    """单样本数据集 - 重复返回同一个样本用于过拟合"""
    
    def __init__(self, sample_file: str, mean: np.ndarray, std: np.ndarray):
        """
        Args:
            sample_file: 样本文件路径
            mean: 特征标准化均值
            std: 特征标准化标准差
        """
        self.base_dataset = SegmentationDataset(
            file_paths=[sample_file],
            mean=mean,
            std=std,
            arch_frames={},
            target_cells=10000,
            sample_cells=6000,
            augment=False,  # 过拟合时不使用数据增强
            augment_original_copies=1,
            augment_flipped_copies=0
        )
        
        # 预加载样本避免重复计算
        self.sample_data = self.base_dataset[0]
        
    def __len__(self):
        return 10  # ⚡ 优化：减少虚拟长度（100→10），加速训练
    
    def __getitem__(self, idx):
        # 每次都返回同一个样本
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
    
    # 加载统计信息
    stats_path = Path("outputs/segmentation/module0/stats.npz")
    if not stats_path.exists():
        raise FileNotFoundError(f"统计文件不存在: {stats_path}\n请先运行 m0_dataset.py 生成统计信息")
    
    stats = np.load(stats_path)
    mean = stats['mean']
    std = stats['std']
    
    print(f"✅ 加载统计信息: 均值={mean[:3]}..., 标准差={std[:3]}...")
    
    # 创建单样本数据集
    single_dataset = SingleSampleDataset(str(sample_file), mean, std)
    
    # 获取一个样本来确定类别数
    sample_data = single_dataset[0]
    # 数据格式: ((features, pos, pos_mm, pos_scale), labels)
    (features, pos, pos_mm, pos_scale), labels = sample_data
    
    unique_labels = torch.unique(labels)
    # 使用标签的最大值+1作为类别数，确保所有标签都在范围内
    num_classes = int(unique_labels.max().item()) + 1
    
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
    print(f"   - 唯一标签: {unique_labels.tolist()}")
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
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.use_amp = device.type == 'cuda'
        
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
                with autocast():
                    logits = self.model(features, pos)
                    dice_loss = self.dice_loss(logits, labels)
                    ce_loss = self.ce_loss(logits, labels)
                    total_loss = dice_loss + ce_loss
                
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
                    with autocast():
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
    
    def _save_checkpoint_with_pipeline(self, ckpt_path: Path, sample_name: str, epoch: int, dsc: float):
        """
        保存包含完整 pipeline 元数据契约的 checkpoint
        
        这确保推理时能完全复现训练时的前处理流程
        """
        # 构建完整的 checkpoint
        checkpoint = {
            # 模型权重
            "state_dict": self.model.state_dict(),
            
            # 模型架构信息
            "num_classes": self.num_classes,
            "in_channels": 15,  # 特征维度（9点坐标 + 3法向 + 3相对位置）
            
            # 前处理 pipeline 契约
            "pipeline": {
                # Z-score 标准化
                "zscore": {
                    "mean": self.mean.tolist() if self.mean is not None else None,
                    "std": self.std.tolist() if self.std is not None else None,
                    "apply": True
                },
                
                # 几何预处理
                "centered": True,           # 已减去质心
                "div_by_diag": False,      # 未除以对角线
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
        
        # 保存训练历史
        self.save_training_plots(save_dir, sample_name)
        
        return best_dsc
    
    def save_training_plots(self, save_dir: Path, sample_name: str):
        """保存训练曲线图"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        plt.figure(figsize=(20, 12))
        
        # 损失曲线
        plt.subplot(3, 3, 1)
        plt.plot(epochs, self.history['loss'], 'b-', label='Total Loss')
        plt.plot(epochs, self.history['dice_loss'], 'r-', label='Dice Loss')
        plt.plot(epochs, self.history['ce_loss'], 'g-', label='CE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失')
        plt.legend()
        plt.yscale('log')
        
        # DSC曲线
        plt.subplot(3, 3, 2)
        plt.plot(epochs, self.history['dsc'], 'b-', label='DSC')
        plt.xlabel('Epoch')
        plt.ylabel('DSC')
        plt.title('Dice相似系数')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(3, 3, 3)
        plt.plot(epochs, self.history['accuracy'], 'g-', label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('训练准确率')
        plt.legend()
        
        # 🔍 BG0 比例曲线（关键诊断指标）
        plt.subplot(3, 3, 4)
        plt.plot(epochs, self.history['train_bg0'], 'r-', label='Train BG0', linewidth=2)
        plt.plot(epochs, self.history['val_bg0'], 'b-', label='Val BG0', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('BG0 Ratio')
        plt.title('🔍 背景预测比例 (应快速下降)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 🔍 Entropy 曲线（关键诊断指标）
        plt.subplot(3, 3, 5)
        plt.plot(epochs, self.history['train_entropy'], 'r-', label='Train Entropy', linewidth=2)
        plt.plot(epochs, self.history['val_entropy'], 'b-', label='Val Entropy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('🔍 预测熵 (应快速下降)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # DSC vs Loss散点图
        plt.subplot(3, 3, 6)
        plt.scatter(self.history['loss'], self.history['dsc'], alpha=0.6)
        plt.xlabel('Total Loss')
        plt.ylabel('DSC')
        plt.title('DSC vs Loss')
        
        # 最后50个epoch的损失
        if len(epochs) > 50:
            plt.subplot(3, 3, 7)
            last_50 = epochs[-50:]
            plt.plot(last_50, self.history['loss'][-50:], 'b-', label='Total Loss')
            plt.plot(last_50, self.history['dice_loss'][-50:], 'r-', label='Dice Loss')
            plt.plot(last_50, self.history['ce_loss'][-50:], 'g-', label='CE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('最后50个Epoch损失')
            plt.legend()
            plt.yscale('log')
        
        # 最后50个epoch的DSC
        if len(epochs) > 50:
            plt.subplot(3, 3, 8)
            plt.plot(last_50, self.history['dsc'][-50:], 'b-', label='DSC')
            plt.xlabel('Epoch')
            plt.ylabel('DSC')
            plt.title('最后50个Epoch DSC')
            plt.legend()
            
            # 最后50个epoch的BG0 (关键诊断)
            plt.subplot(3, 3, 9)
            plt.plot(last_50, self.history['train_bg0'][-50:], 'r-', label='Train BG0')
            plt.plot(last_50, self.history['val_bg0'][-50:], 'b-', label='Val BG0')
            plt.xlabel('Epoch')
            plt.ylabel('BG0 Ratio')
            plt.title('🔍 最后50 Epoch BG0')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'单样本过拟合训练曲线 - {sample_name}', fontsize=16)
        plt.tight_layout()
        
        plot_file = save_dir / f"overfit_curves_{sample_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存: {plot_file}")


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
    model = iMeshSegNet(num_classes=num_classes)
    
    # 创建训练器（传入 mean, std 用于保存 pipeline 契约）
    trainer = OverfitTrainer(model, dataloader, num_classes, device, mean, std)
    
    # 输出目录
    output_dir = Path("outputs/overfit") / args.sample
    
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