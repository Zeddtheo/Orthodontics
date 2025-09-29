#!/usr/bin/env python3
"""
PointnetReg 一键训练脚本
======================

选择预设配置，一键启动训练
"""

import subprocess
from pathlib import Path

# 默认数据集路径
DEFAULT_ROOT = r"c:\MISC\Deepcare\Orthodontics\datasets\landmarks_dataset\cooked\p0\samples_consistent"

def main():
    """主函数"""
    print("=" * 60)
    print("🦷 PointnetReg 一键训练")
    print("=" * 60)
    
    # 检查环境
    if not Path("p1_train.py").exists():
        print("❌ 错误: 请在 src/PointnetReg 目录下运行")
        return
    
    # 训练选项
    commands = {
        "1": {
            "name": "快速测试 (3轮)",
            "desc": "验证环境和流程，t31牙位",
            "cmd": ["python", "p1_train.py", "--mode", "per_tooth", "--tooth", "t31", 
                   "--epochs", "3", "--batch_size", "4", "--augment", "--root", DEFAULT_ROOT]
        },
        "2": {
            "name": "标准单牙位训练",
            "desc": "t31牙位，80轮标准训练",
            "cmd": ["python", "p1_train.py", "--mode", "per_tooth", "--tooth", "t31",
                   "--epochs", "80", "--batch_size", "8", "--augment", "--root", DEFAULT_ROOT]
        },
        "3": {
            "name": "标准组训练",
            "desc": "第一磨牙组，80轮训练",
            "cmd": ["python", "p1_train.py", "--mode", "per_group", "--group", "m1",
                   "--epochs", "80", "--batch_size", "12", "--augment", "--arch_align", 
                   "--mirror_prob", "0.2", "--root", DEFAULT_ROOT]
        },
        "4": {
            "name": "增强训练",
            "desc": "中切牙组，完整增强功能",
            "cmd": ["python", "p1_train.py", "--mode", "per_group", "--group", "central",
                   "--epochs", "100", "--batch_size", "16", "--augment", "--arch_align",
                   "--mirror_prob", "0.4", "--zscore", "--lr", "5e-4", "--root", DEFAULT_ROOT]
        },
        "5": {
            "name": "生产级训练",
            "desc": "尖牙组，150轮高质量训练", 
            "cmd": ["python", "p1_train.py", "--mode", "per_group", "--group", "canine",
                   "--epochs", "150", "--batch_size", "20", "--augment", "--arch_align",
                   "--mirror_prob", "0.5", "--zscore", "--lr", "3e-4", "--workers", "8", "--root", DEFAULT_ROOT]
        }
    }
    
    # 显示选项
    print("\n📋 训练选项:")
    for key, info in commands.items():
        print(f"  {key}. {info['name']}")
        print(f"     {info['desc']}")
    
    # 用户选择
    try:
        choice = input(f"\n请选择 (1-{len(commands)}): ").strip()
        
        if choice not in commands:
            print("❌ 无效选择")
            return
        
        # 执行训练
        cmd_info = commands[choice]
        print(f"\n🚀 开始训练: {cmd_info['name']}")
        print(f"命令: {' '.join(cmd_info['cmd'])}")
        
        if input("\n确认开始? (Y/n): ").strip().lower() not in ['', 'y', 'yes']:
            print("❌ 已取消")
            return
        
        # 运行命令
        result = subprocess.run(cmd_info['cmd'], check=True)
        print("\n✅ 训练完成!")
        
        # 显示评估命令
        if "--mode per_tooth" in ' '.join(cmd_info['cmd']):
            tooth = next(cmd_info['cmd'][i+1] for i, arg in enumerate(cmd_info['cmd']) if arg == '--tooth')
            print(f"\n📊 评估命令:")
            print(f"python p2_eval.py --mode per_tooth --tooth {tooth}")
        elif "--mode per_group" in ' '.join(cmd_info['cmd']):
            group = next(cmd_info['cmd'][i+1] for i, arg in enumerate(cmd_info['cmd']) if arg == '--group')
            print(f"\n📊 评估命令:")
            print(f"python p2_eval.py --mode per_group --group {group}")
        
    except KeyboardInterrupt:
        print("\n⚠️  已取消")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e.returncode}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    main()