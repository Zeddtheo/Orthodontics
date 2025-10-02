"""
对比使用 pipeline 契约前后的推理结果
验证契约系统的正确性
"""
import numpy as np
from pathlib import Path

def compare_predictions(old_npy: Path, new_npy: Path):
    """比较两个推理结果文件"""
    old_pred = np.load(str(old_npy))
    new_pred = np.load(str(new_npy))
    
    print(f"=== 对比推理结果 ===")
    print(f"旧推理 (无契约): {old_npy}")
    print(f"新推理 (有契约): {new_npy}")
    print()
    
    # 基本统计
    print(f"形状匹配: {old_pred.shape == new_pred.shape}")
    print(f"  旧: {old_pred.shape}")
    print(f"  新: {new_pred.shape}")
    print()
    
    # 完全一致性检查
    if old_pred.shape == new_pred.shape:
        exact_match = np.all(old_pred == new_pred)
        match_ratio = np.sum(old_pred == new_pred) / old_pred.size
        print(f"预测完全一致: {exact_match}")
        print(f"匹配率: {match_ratio * 100:.2f}%")
        print()
    
    # 标签分布对比
    print("标签分布对比:")
    print(f"{'Label':<8} {'旧 (cells)':<15} {'新 (cells)':<15} {'差异':<10}")
    print("-" * 50)
    
    old_unique, old_counts = np.unique(old_pred, return_counts=True)
    new_unique, new_counts = np.unique(new_pred, return_counts=True)
    
    old_dict = dict(zip(old_unique, old_counts))
    new_dict = dict(zip(new_unique, new_counts))
    
    all_labels = sorted(set(old_unique) | set(new_unique))
    
    for label in all_labels:
        old_count = old_dict.get(label, 0)
        new_count = new_dict.get(label, 0)
        diff = new_count - old_count
        diff_sign = "+" if diff > 0 else ""
        print(f"L{label:<7} {old_count:<15} {new_count:<15} {diff_sign}{diff}")
    
    print()
    print(f"旧推理总类别数: {len(old_unique)}")
    print(f"新推理总类别数: {len(new_unique)}")
    
    # 检查遗漏类别
    old_missing = set(range(15)) - set(old_unique)
    new_missing = set(range(15)) - set(new_unique)
    
    if old_missing == new_missing:
        print(f"\n✅ 遗漏类别一致: {sorted(old_missing)}")
    else:
        print(f"\n⚠️ 遗漏类别不一致!")
        print(f"  旧: {sorted(old_missing)}")
        print(f"  新: {sorted(new_missing)}")


if __name__ == "__main__":
    old = Path("outputs/overfit/infer/1_U_pred.npy")
    new = Path("outputs/overfit/1_U/test_contract_infer/1_U_pred.npy")
    
    if not old.exists():
        print(f"❌ 旧推理结果不存在: {old}")
        exit(1)
    
    if not new.exists():
        print(f"❌ 新推理结果不存在: {new}")
        exit(1)
    
    compare_predictions(old, new)
