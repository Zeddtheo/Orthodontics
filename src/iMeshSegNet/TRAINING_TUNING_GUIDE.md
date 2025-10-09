# iMeshSegNet 训练调优指南

## 当前问题诊断

### 症状
- 单样本过拟合能快速达到 DSC=1.0（说明模型结构正确）
- 真实训练在 10 个 epoch 后卡在 DSC≈0.2–0.3 的平台期
- 验证损失缓慢下降，但指标不抬头
- 典型的"欠拟合/学不到信号"现象

### 根本原因分析

1. **训练难度过高的组合拳**
   - ✅ 强几何增强（旋转、缩放、抖动）始终开启
   - ✅ 重度 Dropout（0.5）在分类头
   - ❌ 没有开启 Feature STN 去对齐增强后的特征
   - ❌ 双重重加权损失（GDL + 加权 CE）放大稀有类噪声

2. **单样本 vs 真实训练的对比**
   | 设置 | 单样本 Overfit | 真实训练 |
   |------|----------------|----------|
   | 学习率 | 0.01 | 0.001 |
   | Dropout | 关闭 | 0.5（重度）|
   | 增强 | 关闭 | 开启 |
   | Z-score | 样本自身统计 | 数据集统计 |
   | 结果 | DSC=1.0（秒上）| DSC=0.2~0.3（平台）|

---

## 已应用的修复（2×2 验证矩阵）

### 快改开关汇总

#### ✅ 改1：关闭重度 Dropout + 开启 Feature STN
```python
# m1_train.py -> main()
model = iMeshSegNet(
    num_classes=config.num_classes,
    glm_impl="edgeconv",
    k_short=6, k_long=12,
    with_dropout=False,        # ← 改1：先关闭（或设为 0.1）
    dropout_p=0.1,
    use_feature_stn=True,      # ← 改2：开启 STN 对齐旋转特征
).to(device)
```

**原理**：
- Feature STN 可以学习一个 64×64 的变换矩阵，抵消几何增强对特征空间的扰动
- 重度 Dropout 在早期会稀释信号，关闭后能让模型先"听懂人话"

#### ✅ 改2：关闭增强做"冷启"
```python
# m1_train.py -> main()
config.data_config.augment = False  # ← 先关闭，待 DSC 抬头后再逐步打开
```

**验证方案**：
- 跑 5~10 个 epoch
- 若 DSC 从 0.2~0.3 快速抬到 0.4~0.6+，说明"增强+无STN+Dropout"是主因
- 之后可以用"增强 warmup"：前 10 epoch 关闭，之后逐步打开

#### ✅ 改3：降低 Dice 损失的主导性
```python
# m1_train.py -> Trainer._run_epoch()
loss = 0.5 * loss_dice + 1.0 * loss_ce  # ← 原来是 1.0 + 1.0
```

**原理**：
- GDL 对稀有类按 1/support² 加权，小类权重增长极快
- 叠加加权 CE 后，早期会把优化推向"难例/噪声主导"
- 降低 Dice 权重让 CE 先稳定类间界限

#### ✅ 改4：CE 类别权重快速测试开关
```python
# m1_train.py -> main()
# 选项 A：使用均匀权重（排查是否"重加权过度"）
config.ce_class_weights = [1.0] * config.num_classes

# 选项 B：使用更温和的加权（修改 build_ce_class_weights）
# 把 clip 范围从 [0.1, 10] 改成 [0.3, 3]
```

#### ✅ 改5：Dice 权重限幅
```python
# m1_train.py -> GeneralizedDiceLoss
raw_w = torch.clamp(raw_w, 1e-3, 1e3)  # ← 防止极小 support 类别权重爆炸
```

#### ✅ 改6：诊断指标（背景比例 & 预测熵）
```python
# 每个 epoch 输出：
# Val BG%: 背景预测比例（高 → 模型偏向背景）
# Val Ent: 预测熵（高 → 模型犹豫不决）
```

---

## 验证矩阵（推荐顺序）

### 实验 1：基线（确认问题）
```python
# m1_train.py
with_dropout=True, dropout_p=0.5
use_feature_stn=False
config.data_config.augment=True
loss = 1.0 * loss_dice + 1.0 * loss_ce
```
**预期**：DSC 继续卡在 0.2~0.3

---

### 实验 2：最激进修复（快速验证）
```python
with_dropout=False
use_feature_stn=True
config.data_config.augment=False
loss = 0.5 * loss_dice + 1.0 * loss_ce
config.ce_class_weights = [1.0] * config.num_classes  # 均匀权重
```
**预期**：DSC 在 3~5 个 epoch 内快速抬到 0.5~0.7+

---

### 实验 3：逐步加回正则化
```python
with_dropout=True, dropout_p=0.1  # ← 轻度 Dropout
use_feature_stn=True
config.data_config.augment=False
loss = 0.5 * loss_dice + 1.0 * loss_ce
# 使用原有的加权 CE（恢复类别平衡）
```
**预期**：DSC 稳步爬升，无早停平台

---

### 实验 4：完整流程（带增强）
```python
with_dropout=True, dropout_p=0.1
use_feature_stn=True
config.data_config.augment=True  # ← 重新打开
loss = 0.5 * loss_dice + 1.0 * loss_ce
```
**预期**：收敛速度略慢，但最终泛化性更好

---

## 诊断指标解读

### 背景比例（BG%）
- **正常**：0.3~0.5（与数据集背景比例相近）
- **异常**：>0.8（模型过度偏向背景，说明前景类学不会）

### 预测熵（Entropy）
- **正常**：逐步下降，稳定在 0.5~1.5
- **异常**：长期高于 2.5（模型犹豫不决，没有形成清晰类别锚点）

---

## 进阶优化建议

### 1. 增强 Warmup
```python
# 在 Trainer.train() 里动态调整
if epoch <= 10:
    config.data_config.augment = False
else:
    config.data_config.augment = True
```

### 2. 学习率 Warmup
```python
# 前 5 个 epoch 线性从 0 升到 config.lr
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
# 在 CosineAnnealingLR 之前使用
```

### 3. 渐进式 Dropout
```python
# 前 20 个 epoch dropout=0.1，之后逐步升到 0.3
if epoch <= 20:
    model.dropout_p = 0.1
else:
    model.dropout_p = min(0.3, 0.1 + (epoch - 20) * 0.01)
```

### 4. 损失权重动态调整
```python
# 前 10 个 epoch 只用 CE，之后逐步引入 Dice
if epoch <= 10:
    loss = loss_ce
else:
    dice_weight = min(1.0, (epoch - 10) / 20)
    loss = dice_weight * loss_dice + loss_ce
```

---

## 常见问题 FAQ

### Q1: 为什么单样本能 overfit，真实训练不行？
**A**: 单样本用了"作弊级"设置（大 LR、无正则、无增强、样本内统计），只证明模型结构正确，但不代表训练流程合理。

### Q2: BatchNorm 在 batch_size=2 时会有问题吗？
**A**: 不会。你的 BN 是在 (B, C, N) 维度上做的，即使 B=2，也有 2×6000=12000 个 cell 参与统计，足够稳定。

### Q3: 什么时候可以加回 Dropout？
**A**: 等 DSC 稳定在 0.6+ 后，再从 0.1 逐步加到 0.3。不建议用 0.5（过重）。

### Q4: Feature STN 会增加多少计算量？
**A**: 几乎可以忽略（<1% 训练时间），但对旋转增强的鲁棒性提升明显。

---

## 快速命令

### 运行实验 2（最激进修复）
```powershell
# 确保已修改 m1_train.py 里的开关
cd c:\MISC\Deepcare\Orthodontics\src\iMeshSegNet
python m1_train.py --epochs 10 --run-name "fix_aggressive"
```

### 对比基线
```powershell
# 先在代码里恢复原设置，然后运行
python m1_train.py --epochs 10 --run-name "baseline"
```

### 查看 TensorBoard
```powershell
tensorboard --logdir outputs/segmentation/tensorboard
```

---

## 修改清单（代码层面）

### ✅ `imeshsegnet.py`
1. ✅ 添加 `self.k_short` 和 `self.k_long` 属性（元数据追踪）
2. ✅ **kNN 图构建强制 FP32**（防止 AMP 量化误差破坏邻接质量）
3. ✅ **GLMEdgeConv 添加 `forward_idx()` 方法**（复用预计算的 kNN 索引）
4. ✅ **iMeshSegNet.forward() 一次性计算 kNN**（减少 50% O(N²) 开销）

### ✅ `m1_train.py`
1. ✅ `GeneralizedDiceLoss`: 添加权重限幅（防止极小类权重爆炸）
2. ✅ `build_ce_class_weights`: 更温和的 clip 范围 [0.3, 3.0]（原 [0.1, 10]）
3. ✅ `Trainer._run_epoch()`: 添加诊断指标（BG%、熵）
4. ✅ `Trainer.train()`: 输出诊断指标到终端和 TensorBoard
5. ✅ 梯度裁剪阈值从 1.0 提升到 2.0（更顺畅的梯度流）
6. ✅ `main()`: 应用 5 个快改开关（STN、Dropout、增强、损失权重）

---

## 预期结果

### 修复前（基线）
```
Epoch 10/200 | Train Loss: 1.2 | Val Loss: 1.1 | Val DSC: 0.25 | Val SEN: 0.22 | Val PPV: 0.28
```

### 修复后（实验 2）
```
Epoch 5/10  | Train Loss: 0.4 | Val Loss: 0.5 | Val DSC: 0.62 | Val SEN: 0.58 | Val PPV: 0.67
Epoch 10/10 | Train Loss: 0.2 | Val Loss: 0.3 | Val DSC: 0.75 | Val SEN: 0.72 | Val PPV: 0.78
```

---

## 下一步行动

1. ✅ **立即运行**：实验 2（最激进修复），验证修复有效
2. ⏭️ **逐步回退**：实验 3、4，找到最优平衡点
3. 🔬 **长期训练**：用最优配置跑 100~200 个 epoch
4. 📊 **对比分析**：用 TensorBoard 可视化不同实验的曲线

---

生成时间：2025-10-09
基于问题：DSC 卡在 0.2~0.3 平台期（早期欠拟合）
