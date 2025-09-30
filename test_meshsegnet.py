#!/usr/bin/env python3
"""
MeshSegNet训练快速测试 - 只跑1个epoch确认功能正常
"""

import os
import sys
sys.path.append('src/iMeshSegNet')

from m1_train import TrainConfig, Trainer

def main():
    """快速测试MeshSegNet训练流程"""
    print("🚀 MeshSegNet快速训练测试")
    print("=" * 50)
    
    # 测试配置 - 只跑1个epoch
    config = TrainConfig(
        epochs=1,          # 只跑1个epoch
        batch_size=1,      # 减小batch size
        lr=0.001,
        save_interval=1,
        output_dir="outputs/segmentation/quick_test"
    )
    
    print(f"📊 配置信息:")
    print(f"  - 训练轮数: {config.epochs} (快速测试)")
    print(f"  - 批量大小: {config.data_config.batch_size}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 输出目录: {config.output_dir}")
    
    # 创建训练器并训练
    trainer = Trainer(config)
    
    try:
        print("\n🏃 开始训练...")
        trainer.train()
        print("\n✅ MeshSegNet训练测试完成！功能正常。")
        return True
    except Exception as e:
        print(f"\n❌ 训练测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)