#!/usr/bin/env python3
"""MeshSegNet训练测试脚本"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from m1_train import TrainConfig, Trainer, build_ce_class_weights
from m0_dataset import DataConfig, get_dataloaders, load_split_lists, compute_label_histogram, set_seed, SEG_NUM_CLASSES
from imeshsegnet import iMeshSegNet
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def quick_train():
    """快速测试训练"""
    print("🚀 开始MeshSegNet快速测试训练...")
    
    # 配置
    config = TrainConfig()
    config.epochs = 5  # 快速测试
    config.output_dir = Path('../../test_runs/mesh_train_test')
    config.data_config.dataset_root = Path('../../datasets/segmentation_dataset')
    config.data_config.split_path = Path('../../outputs/segmentation/module0/dataset_split.json')
    config.data_config.stats_path = Path('../../outputs/segmentation/module0/stats.npz')
    config.data_config.batch_size = 4  # 小批次
    config.data_config.num_workers = 0  # 避免多进程问题
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据准备
    config.data_config.pin_memory = device.type == "cuda"
    config.data_config.persistent_workers = False  # 避免worker问题
    config.data_config.augment = True
    
    # 计算类权重
    if config.ce_class_weights is None:
        print("计算类频率权重...")
        train_files, _ = load_split_lists(config.data_config.split_path)
        class_hist = compute_label_histogram(train_files)
        ce_weights = build_ce_class_weights(class_hist)
        config.ce_class_weights = ce_weights.tolist()
        print(f"CrossEntropy类权重: min={ce_weights.min():.3f}, max={ce_weights.max():.3f}")
    
    print("加载数据集...")
    train_loader, val_loader = get_dataloaders(config.data_config)
    print(f"训练集: {len(train_loader)} 批次")
    print(f"验证集: {len(val_loader)} 批次")
    
    # 模型
    print("初始化模型...")
    model = iMeshSegNet(
        num_classes=config.num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=True,
        dropout_p=0.5,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} 总计, {trainable_params:,} 可训练参数")
    
    # 优化器和调度器
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        amsgrad=True,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    
    # 训练器
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, config, device)
    trainer.train()
    
    print("✅ MeshSegNet快速测试训练完成!")

if __name__ == "__main__":
    quick_train()