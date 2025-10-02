"""
详细分析 overfit 训练和推理结果
"""
import numpy as np
import pyvista as pv

print("=" * 70)
print("📊 Overfit 训练与推理结果分析")
print("=" * 70)

# 1. 加载训练数据的真实标签分布
print("\n1️⃣  训练数据（1_U.vtp）真实标签分布:")
print("-" * 70)
train_mesh = pv.read('datasets/segmentation_dataset/1_U.vtp')
train_labels = train_mesh.cell_data['Label']

# 映射 FDI 到连续索引
from src.iMeshSegNet.m0_dataset import remap_segmentation_labels
train_labels_mapped = remap_segmentation_labels(np.asarray(train_labels))

unique_train, counts_train = np.unique(train_labels_mapped, return_counts=True)
total_train = len(train_labels_mapped)

for u, c in zip(unique_train, counts_train):
    percentage = c / total_train * 100
    bar = '█' * int(percentage / 2)  # 可视化条形图
    print(f"  L{u:2d}: {c:6d} cells ({percentage:5.1f}%) {bar}")

print(f"\n  总计: {total_train:,} cells, {len(unique_train)} 个类别")

# 2. 推理结果标签分布
print("\n2️⃣  推理结果（1_U_colored.vtp）标签分布:")
print("-" * 70)
pred_labels = np.load('outputs/overfit/infer/1_U_pred.npy')
unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
total_pred = len(pred_labels)

for u, c in zip(unique_pred, counts_pred):
    percentage = c / total_pred * 100
    bar = '█' * int(percentage / 2)
    print(f"  L{u:2d}: {c:6d} cells ({percentage:5.1f}%) {bar}")

print(f"\n  总计: {total_pred:,} cells, {len(unique_pred)} 个类别")

# 3. 对比分析
print("\n3️⃣  对比分析:")
print("-" * 70)

missing = set(unique_train) - set(unique_pred)
extra = set(unique_pred) - set(unique_train)

if missing:
    print(f"❌ 遗漏的类别: {sorted(missing)}")
    print(f"   这些类别在训练数据中存在，但模型没有预测出来")
else:
    print("✅ 没有遗漏类别！")

if extra:
    print(f"⚠️  多余的类别: {sorted(extra)}")
    print(f"   这些类别在训练数据中不存在，但模型预测了")

# 4. 类别级别对比
print("\n4️⃣  各类别预测准确性（训练 vs 推理）:")
print("-" * 70)
print(f"{'类别':<6} {'训练数据%':>12} {'推理结果%':>12} {'差异':>10} {'状态':<10}")
print("-" * 70)

# 创建完整的类别列表（0-14）
all_labels = set(range(15))
train_dist = {u: c/total_train*100 for u, c in zip(unique_train, counts_train)}
pred_dist = {u: c/total_pred*100 for u, c in zip(unique_pred, counts_pred)}

for label in sorted(all_labels):
    train_pct = train_dist.get(label, 0.0)
    pred_pct = pred_dist.get(label, 0.0)
    diff = pred_pct - train_pct
    
    if label not in pred_dist:
        status = "❌ 遗漏"
    elif abs(diff) > 20:
        status = "⚠️  偏差大"
    elif abs(diff) > 10:
        status = "⚠️  偏差中"
    else:
        status = "✅ 正常"
    
    print(f"L{label:2d}     {train_pct:>10.1f}%  {pred_pct:>10.1f}%  {diff:>+9.1f}%  {status}")

# 5. 总结
print("\n5️⃣  总结:")
print("-" * 70)

# 计算背景类别的变化
bg_train = train_dist.get(0, 0)
bg_pred = pred_dist.get(0, 0)

print(f"背景类别 (L0):")
print(f"  训练数据: {bg_train:.1f}%")
print(f"  推理结果: {bg_pred:.1f}%")
print(f"  差异: {bg_pred - bg_train:+.1f}%")

if bg_pred > bg_train + 10:
    print(f"  ⚠️  背景比例过高，模型将许多牙齿预测为背景")

# 计算覆盖率
coverage = len(unique_pred) / len(unique_train) * 100
print(f"\n类别覆盖率: {coverage:.1f}% ({len(unique_pred)}/{len(unique_train)})")

if coverage < 100:
    print(f"  ❌ 模型未能学习所有类别")
    print(f"  建议:")
    print(f"    1. 增加训练轮数到 100-200 epochs")
    print(f"    2. 调整学习率或损失函数权重")
    print(f"    3. 检查类别不平衡问题")
else:
    print(f"  ✅ 模型成功学习了所有类别！")

print("\n" + "=" * 70)
print("📈 训练曲线图已保存: outputs/overfit/1_U/overfit_curves_1_U.png")
print("🎨 推理结果已保存: outputs/overfit/infer/1_U_colored.vtp")
print("=" * 70)
