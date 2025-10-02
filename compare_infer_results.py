"""
对比两次推理结果是否一致
"""
import numpy as np
import pyvista as pv
from collections import Counter

def compare_predictions(file1, file2):
    """对比两个推理结果文件"""
    print(f"\n📊 对比推理结果:")
    print(f"  文件1: {file1}")
    print(f"  文件2: {file2}")
    print()
    
    # 加载预测结果
    pred1 = np.load(file1.replace('_colored.vtp', '_pred.npy'))
    pred2 = np.load(file2.replace('_colored.vtp', '_pred.npy'))
    
    # 加载网格
    mesh1 = pv.read(file1)
    mesh2 = pv.read(file2)
    
    print(f"✅ 预测形状:")
    print(f"  文件1: {pred1.shape}")
    print(f"  文件2: {pred2.shape}")
    print()
    
    # 对比预测是否完全一致
    if np.array_equal(pred1, pred2):
        print("✅ 预测结果完全一致！")
    else:
        diff_count = np.sum(pred1 != pred2)
        diff_ratio = diff_count / len(pred1) * 100
        print(f"⚠️  预测结果有差异:")
        print(f"  不同的单元格数: {diff_count} / {len(pred1)} ({diff_ratio:.2f}%)")
    
    print()
    
    # 统计标签分布
    counter1 = Counter(pred1)
    counter2 = Counter(pred2)
    
    print("📈 标签分布对比:")
    print(f"{'标签':<6} {'文件1':<10} {'文件2':<10} {'差异':<10}")
    print("-" * 40)
    
    all_labels = sorted(set(counter1.keys()) | set(counter2.keys()))
    for label in all_labels:
        c1 = counter1.get(label, 0)
        c2 = counter2.get(label, 0)
        diff = c2 - c1
        label_name = f"L{label}"
        print(f"{label_name:<6} {c1:<10} {c2:<10} {diff:+<10}")
    
    print()
    
    # 网格顶点数对比
    print("🔷 网格信息:")
    print(f"  文件1: {mesh1.n_points} 点, {mesh1.n_cells} 单元格")
    print(f"  文件2: {mesh2.n_points} 点, {mesh2.n_cells} 单元格")
    
    # 检查颜色数据
    if 'Colors' in mesh1.point_data and 'Colors' in mesh2.point_data:
        colors1 = mesh1.point_data['Colors']
        colors2 = mesh2.point_data['Colors']
        if np.array_equal(colors1, colors2):
            print("  ✅ 点颜色数据完全一致")
        else:
            print("  ⚠️  点颜色数据有差异")

if __name__ == '__main__':
    # 对比两次确定性推理结果
    file1 = 'outputs/overfit/1_U/infer_deterministic_1_U.vtp/1_U_colored.vtp'
    file2 = 'outputs/overfit/1_U/infer_deterministic2_1_U.vtp/1_U_colored.vtp'
    
    print("=" * 60)
    print("验证确定性推理：两次运行应该产生完全一致的结果")
    print("=" * 60)
    compare_predictions(file1, file2)
