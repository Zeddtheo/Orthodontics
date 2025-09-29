#!/usr/bin/env python3
"""
测试修改后的p0_dataset.py新功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from p0_dataset import DatasetConfig, P0PointNetRegDataset, make_dataloader
from tooth_groups import TOOTH_GROUPS, get_group_teeth

def test_group_filtering():
    """测试分组筛选功能"""
    print("🧪 测试分组筛选功能")
    
    # 测试按组筛选
    cfg = DatasetConfig(
        root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
        group="m1",  # 测试第一磨牙组
        features="all",
        select_landmarks="active",
        ensure_constant_L=True
    )
    
    try:
        dataset = P0PointNetRegDataset(cfg)
        print(f"✅ 组筛选成功: 找到 {len(dataset)} 个 m1 组样本")
        print(f"   组内牙位: {get_group_teeth('m1')}")
        
        # 测试样本
        sample = dataset[0]
        print(f"   样本形状: x={sample['x'].shape}, y={sample['y'].shape}")
        print(f"   元信息: tooth_id={sample['meta'].get('tooth_id')}, group={sample['meta'].get('group')}")
        
    except Exception as e:
        print(f"❌ 组筛选测试失败: {e}")

def test_tooth_ids_filtering():
    """测试显式牙位列表筛选"""
    print("\n🧪 测试显式牙位列表筛选")
    
    cfg = DatasetConfig(
        root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
        tooth_ids=["t31", "t32"],  # 测试指定牙位
        features="all",
        select_landmarks="active",
        ensure_constant_L=True
    )
    
    try:
        dataset = P0PointNetRegDataset(cfg)
        print(f"✅ 牙位筛选成功: 找到 {len(dataset)} 个 t31+t32 样本")
        
        # 检查前几个样本的牙位
        tooth_counts = {}
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            tooth_id = sample['meta'].get('tooth_id', 'unknown')
            tooth_counts[tooth_id] = tooth_counts.get(tooth_id, 0) + 1
        
        print(f"   前10个样本牙位分布: {tooth_counts}")
        
    except Exception as e:
        print(f"❌ 牙位筛选测试失败: {e}")

def test_augmentation_features():
    """测试增强功能"""
    print("\n🧪 测试增强功能")
    
    cfg = DatasetConfig(
        root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
        file_patterns=("*_t31.npz",),
        features="all",
        select_landmarks="active",
        augment=True,
        arch_align=True,
        mirror_prob=0.5,
        rotz_deg=15.0,
        trans_mm=0.5,
        ensure_constant_L=True
    )
    
    try:
        dataset = P0PointNetRegDataset(cfg)
        print(f"✅ 增强数据集创建成功: {len(dataset)} 个样本")
        
        # 测试多次获取同一样本，查看增强效果  
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        x1, x2 = sample1['x'], sample2['x']
        diff = (x1 - x2).abs().mean().item()
        print(f"   同一样本两次增强的差异: {diff:.6f}")
        print(f"   增强后形状: x={x1.shape}, y={sample1['y'].shape}")
        
    except Exception as e:
        print(f"❌ 增强功能测试失败: {e}")

def test_dataloader():
    """测试DataLoader兼容性"""
    print("\n🧪 测试DataLoader兼容性")
    
    cfg = DatasetConfig(
        root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
        group="central",
        features="all",
        select_landmarks="active",
        augment=True,
        arch_align=True,
        mirror_prob=0.3,
        ensure_constant_L=True
    )
    
    try:
        dataset, loader = make_dataloader(cfg, batch_size=4, shuffle=False, num_workers=0)
        print(f"✅ DataLoader创建成功: {len(dataset)} 个样本, {len(loader)} 个批次")
        
        # 测试一个批次
        batch = next(iter(loader))
        print(f"   批次形状: x={batch['x'].shape}, y={batch['y'].shape}")
        print(f"   元信息样例: {[meta.get('tooth_id') for meta in batch['meta'][:2]]}")
        
    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")

def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理")
    
    # 测试无效组名
    try:
        cfg = DatasetConfig(
            root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
            group="invalid_group"
        )
        dataset = P0PointNetRegDataset(cfg)
        print("❌ 应该抛出无效组名错误")
    except ValueError as e:
        print(f"✅ 正确捕获无效组名错误: {e}")
    
    # 测试无效牙位ID
    try:
        cfg = DatasetConfig(
            root="../../datasets/landmarks_dataset/cooked/p0/samples_consistent",
            tooth_ids=["t99", "invalid"]
        )
        dataset = P0PointNetRegDataset(cfg)
        print("❌ 应该抛出无效牙位ID错误")
    except ValueError as e:
        print(f"✅ 正确捕获无效牙位ID错误: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试修改后的p0_dataset.py功能")
    print("=" * 60)
    
    test_group_filtering()
    test_tooth_ids_filtering()
    test_augmentation_features()
    test_dataloader()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")

if __name__ == "__main__":
    main()