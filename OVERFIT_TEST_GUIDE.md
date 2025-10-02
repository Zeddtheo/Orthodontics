# Overfit 测试 - BG0 和 Entropy 诊断指标

**日期**: 2025-10-02  
**目的**: 验证模型能否过拟合单样本，并使用新增的诊断指标快速定位问题  
**状态**: 🔄 运行中

---

## 🎯 测试目标

### 主要验证点

1. **模型基本功能**: 能否在单样本上快速过拟合（DSC > 0.95）
2. **BG0 指标**: 背景预测比例应快速下降（从 ~1.0 → <0.1）
3. **Entropy 指标**: 预测熵应快速下降（从 ~2.7 → <0.5）
4. **数据标签**: 标签映射是否正确
5. **数据增强**: 增强是否破坏了样本

---

## 🔍 新增诊断指标

### 1. BG0 (Background Ratio)

**定义**: 预测为背景类（类别0）的点的比例

```python
bg_ratio = (preds == 0).float().mean()
```

**意义**:
- **正常行为**: 应快速下降
  - Epoch 1: ~0.8-1.0 (初始时大部分预测为背景)
  - Epoch 20: ~0.3-0.5
  - Epoch 100: <0.2
  - Epoch 200: <0.1

- **异常行为**: 持续贴地（>0.7）
  - 可能原因1: 标签映射错误（背景类占主导）
  - 可能原因2: 模型结构问题
  - 可能原因3: 学习率过低

### 2. Entropy (Prediction Entropy)

**定义**: 预测分布的平均熵（不确定性）

```python
entropy = -(probs * log(probs)).sum(dim=1).mean()
```

**意义**:
- **正常行为**: 应快速下降
  - Epoch 1: ~2.0-2.7 (随机初始化，高度不确定)
  - Epoch 20: ~1.0-1.5
  - Epoch 100: ~0.5-0.8
  - Epoch 200: <0.3

- **异常行为**: 持续高值（>2.0）
  - 可能原因1: 模型无法学习
  - 可能原因2: 特征标准化问题
  - 可能原因3: 数据损坏

---

## 🚀 测试命令

```bash
cd "c:\MISC\Deepcare\Orthodontics"
.venv\Scripts\python.exe "src\iMeshSegNet\m5_overfit.py" --sample 1_U --epochs 200
```

### 参数说明

- `--sample 1_U`: 使用 1_U.vtp 样本
- `--epochs 200`: 训练 200 个 epoch
- 学习率: 0.01 (比正常训练大10倍，加速过拟合)
- 权重衰减: 0.0 (移除正则化，允许过拟合)

---

## 📊 预期结果

### 正常情况（模型工作正常）

```
Epoch   1/200 | Loss: 0.850000 | DSC: 0.1200 | Acc: 0.2000 | 
                🔍 Train BG0: 0.950 | Train Ent: 2.600 | 
                Val BG0: 0.940 | Val Ent: 2.580

Epoch  20/200 | Loss: 0.450000 | DSC: 0.5500 | Acc: 0.6500 | 
                🔍 Train BG0: 0.400 | Train Ent: 1.200 | 
                Val BG0: 0.410 | Val Ent: 1.250

Epoch 100/200 | Loss: 0.120000 | DSC: 0.8800 | Acc: 0.9200 | 
                🔍 Train BG0: 0.150 | Train Ent: 0.600 | 
                Val BG0: 0.160 | Val Ent: 0.650

Epoch 200/200 | Loss: 0.030000 | DSC: 0.9700 | Acc: 0.9800 | 
                🔍 Train BG0: 0.050 | Train Ent: 0.200 | 
                Val BG0: 0.060 | Val Ent: 0.250
```

**特征**:
- ✅ Loss 稳定下降
- ✅ DSC 快速上升 (最终 >0.95)
- ✅ BG0 快速下降 (<0.1)
- ✅ Entropy 快速下降 (<0.3)

### 异常情况1：标签映射错误

```
Epoch   1/200 | Loss: 0.900000 | DSC: 0.0100 | Acc: 0.1000 | 
                🔍 Train BG0: 0.980 | Train Ent: 2.700 | 
                Val BG0: 0.980 | Val Ent: 2.700

Epoch  20/200 | Loss: 0.880000 | DSC: 0.0200 | Acc: 0.1200 | 
                🔍 Train BG0: 0.975 | Train Ent: 2.650 | 
                Val BG0: 0.975 | Val Ent: 2.650

Epoch 200/200 | Loss: 0.850000 | DSC: 0.0500 | Acc: 0.1500 | 
                🔍 Train BG0: 0.970 | Train Ent: 2.600 | 
                Val BG0: 0.970 | Val Ent: 2.600
```

**特征**:
- ❌ Loss 几乎不下降
- ❌ DSC 极低 (<0.1)
- ❌ BG0 持续贴地 (>0.97) ⚠️ 关键指标
- ❌ Entropy 持续高值 (>2.6) ⚠️ 关键指标

**诊断**: 
- 问题: 标签映射错误，背景类占据了大部分点
- 解决: 检查 `LABEL_REMAP` 和 `remap_segmentation_labels` 函数

### 异常情况2：特征标准化问题

```
Epoch   1/200 | Loss: 0.900000 | DSC: 0.0500 | Acc: 0.1000 | 
                🔍 Train BG0: 0.900 | Train Ent: 2.700 | 
                Val BG0: 0.900 | Val Ent: 2.700

Epoch  20/200 | Loss: 0.850000 | DSC: 0.1200 | Acc: 0.2000 | 
                🔍 Train BG0: 0.750 | Train Ent: 2.400 | 
                Val BG0: 0.750 | Val Ent: 2.400

Epoch 200/200 | Loss: 0.700000 | DSC: 0.3000 | Acc: 0.4500 | 
                🔍 Train BG0: 0.600 | Train Ent: 2.000 | 
                Val BG0: 0.600 | Val Ent: 2.000
```

**特征**:
- ⚠️ Loss 下降缓慢
- ⚠️ DSC 低 (<0.5)
- ⚠️ BG0 下降慢 (>0.6) ⚠️ 关键指标
- ⚠️ Entropy 下降慢 (>2.0) ⚠️ 关键指标

**诊断**:
- 问题: 特征值范围异常，标准化可能失败
- 解决: 检查 `stats.npz` 和特征计算

---

## 📈 输出文件

训练完成后会生成：

1. **模型文件**:
   - `outputs/overfit/1_U/best_overfit_1_U.pt` - 最佳模型权重

2. **训练曲线**:
   - `outputs/overfit/1_U/overfit_curves_1_U.png` - 包含9个子图：
     * Loss 曲线（总损失、Dice损失、CE损失）
     * DSC 曲线
     * 准确率曲线
     * **🔍 BG0 曲线** (关键诊断)
     * **🔍 Entropy 曲线** (关键诊断)
     * DSC vs Loss 散点图
     * 最后50 epoch 的 Loss
     * 最后50 epoch 的 DSC
     * **🔍 最后50 epoch 的 BG0** (关键诊断)

---

## 🔧 排查步骤

### 如果 BG0 和 Entropy 持续贴地：

#### 步骤 1: 检查数据标签
```python
# 在 m5_overfit.py 的 setup_single_sample_training 中查看输出
✅ 唯一标签: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ^^^^ 应该有多个类别
```

如果只有 `[0]` 或 `[0, 65]`，说明标签映射失败。

#### 步骤 2: 检查标签分布
```python
# 添加到 setup_single_sample_training
label_counts = torch.bincount(labels)
print(f"标签分布: {label_counts}")
```

**正常**: 每个类别都有一定数量的点（不应该0占90%+）
**异常**: 背景类占据绝大多数

#### 步骤 3: 检查特征值范围
```python
# 在训练前打印
print(f"特征范围: min={features.min()}, max={features.max()}, mean={features.mean()}")
```

**正常**: z-score 标准化后应该 mean≈0, std≈1
**异常**: 极端值（如 min=-1e6 或 max=1e10）

#### 步骤 4: 检查数据增强
```python
# 在 m0_dataset.py 的 SegmentationDataset.__getitem__ 中
# 确保 augment=False 在过拟合测试中
```

---

## 📝 当前测试状态

### 测试配置

```
样本: 1_U
Epochs: 200
数据集: datasets/segmentation_dataset
设备: cpu
学习率: 0.01

样本形状: 
  features=torch.Size([15, 6000])
  pos=torch.Size([3, 6000])
  labels=torch.Size([6000])

唯一标签: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
类别数: 15
```

### 观察点

✅ **标签正常**: 15个类别都存在（包括背景）
✅ **数据形状正常**: 6000个点，15维特征
🔄 **等待第一个epoch结果**: 查看BG0和Entropy初始值

---

## 🎓 理论背景

### 为什么 BG0 和 Entropy 重要？

#### 1. BG0 作为"懒惰"指标

模型在训练初期可能会"偷懒"，预测所有点为最常见的类别（通常是背景类）。这样能快速降低损失，但不学习真正的分割。

- **健康模型**: BG0 快速下降，说明模型在学习区分不同类别
- **懒惰模型**: BG0 持续高值，说明模型停留在"全预测背景"的局部最优

#### 2. Entropy 作为"信心"指标

熵衡量预测的不确定性：
- **高熵** (>2.0): 模型对每个点的类别都不确定，接近随机猜测
- **低熵** (<0.5): 模型对预测很有信心

过拟合过程中，Entropy 应该快速下降，因为模型学会了在单个样本上做出确定性预测。

#### 3. 组合诊断

| BG0 | Entropy | 诊断 |
|-----|---------|------|
| 高 (>0.7) | 高 (>2.0) | 模型完全不学习 |
| 高 (>0.7) | 低 (<1.0) | 模型"懒惰"预测背景 |
| 低 (<0.3) | 高 (>2.0) | 模型学习但不确定 |
| 低 (<0.3) | 低 (<0.5) | ✅ 正常过拟合 |

---

## 🚦 下一步

### 等待结果

1. ⏳ 等待第一个epoch完成（约1-2分钟）
2. 📊 查看BG0和Entropy初始值
3. 🔍 分析前20个epoch的趋势
4. 📈 检查最终收敛结果（epoch 200）

### 如果正常

- ✅ 模型和数据管道都正常
- ✅ 可以继续完整训练
- ✅ BG0和Entropy指标有效

### 如果异常

- 🔍 根据BG0和Entropy的表现定位问题
- 🔧 应用上述排查步骤
- 🧪 修复后重新测试

---

**创建时间**: 2025-10-02  
**测试命令**: `python m5_overfit.py --sample 1_U --epochs 200`  
**预计时间**: 约10-20分钟（CPU上200 epoch）  
**状态**: 🔄 第一个epoch运行中
