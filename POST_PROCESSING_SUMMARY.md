# MeshSegNet 后处理修复总结

## ✅ 已完成的修改（第一部分：短期修复）

### 1. graphcut_refine 函数 - 添加警告注释

**文件**: `src/iMeshSegNet/m1_train.py` (Line ~145)

```python
def graphcut_refine(prob_np: np.ndarray, pos_mm_np: np.ndarray, beta: float = 30.0, k: int = 6, iterations: int = 1) -> np.ndarray:
    """
    NOTE: This is an iterative probability smoothing, an approximation of graph-cut's smoothness effect, 
    not a true energy-minimizing graph-cut. This function performs k-NN based probability smoothing 
    to mimic the spatial consistency constraint of graph-cut algorithms.
    """
```

**说明**: 明确标注这不是真正的图割算法，只是基于 k-NN 的概率平滑。

---

### 2. svm_refine 函数 - 添加警告注释

**文件**: `src/iMeshSegNet/m1_train.py` (Line ~175)

```python
def svm_refine(pos_mm_np: np.ndarray, labels_np: np.ndarray) -> np.ndarray:
    """
    WARNING: This function performs SVM refinement on the SAME low-resolution mesh (N_low ≈ 9000),
    NOT on the original high-resolution mesh (N_high ≈ 100k) as described in the paper.
    To properly implement the paper's approach, this function should:
    1. Train SVM on low-res predictions (N_low)
    2. Predict on high-res original mesh (N_high) - CURRENTLY NOT IMPLEMENTED
    Therefore, post-processing metrics computed using this function are INVALID for paper comparison.
    """
```

**说明**: 明确指出缺少上采样步骤，导致 post_ 指标无效。

---

### 3. 修改打印输出 - 隐藏无效的 post_ 指标

**文件**: `src/iMeshSegNet/m1_train.py` (Line ~415)

**修改前**:
```python
print(
    f"Epoch {epoch}/{self.config.epochs} | "
    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
    f"Raw DSC: {raw_dsc_str}±{raw_dsc_std_str} | ... | "
    f"Post DSC: {post_dsc_str}±{post_dsc_std_str} | ... | "
    f"Time: {epoch_time:.1f}s | ... "
)
```

**修改后**:
```python
# NOTE: Post-processed metrics (post_dsc, post_hd, etc.) are INVALID in current implementation
# because svm_refine does not perform true upsampling to original resolution.
# Focus on raw_ metrics for model evaluation.
print(
    f"Epoch {epoch}/{self.config.epochs} | "
    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
    f"Raw DSC: {raw_dsc_str}±{raw_dsc_std_str} | Raw SEN: {raw_sen_str}±{raw_sen_std_str} | "
    f"Raw PPV: {raw_ppv_str}±{raw_ppv_std_str} | Raw HD: {raw_hd_str}±{raw_hd_std_str} mm | "
    # Post-processing metrics commented out as they are invalid without proper upsampling:
    # f"Post DSC: {post_dsc_str}±{post_dsc_std_str} | Post SEN: {post_sen_str}±{post_sen_std_str} | "
    # f"Post PPV: {post_ppv_str}±{post_ppv_std_str} | Post HD: {post_hd_str}±{post_hd_std_str} mm | "
    f"Time: {epoch_time:.1f}s | Sec/scan: {sec_per_scan:.2f}s | PeakMem: {train_peak_mem:.1f}MB | LR: {current_lr:.6f}"
)
```

**说明**: 注释掉所有 post_ 指标的打印，添加警告说明，只显示有效的 raw_ 指标。

---

## 📊 现在应该关注的指标

### ✅ 有效指标（在低分辨率网格上评估，N ≈ 9000）

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **raw_dsc** | Dice 系数（重叠度） | ⭐⭐⭐⭐⭐ 最重要 |
| **raw_sen** | 敏感度/召回率 | ⭐⭐⭐⭐ |
| **raw_ppv** | 精确度 | ⭐⭐⭐⭐ |
| **raw_hd** | Hausdorff 距离 (mm) | ⭐⭐⭐ |

**训练时只看这些指标！**

### ❌ 无效指标（已注释，不要参考）

- ~~post_dsc~~
- ~~post_sen~~
- ~~post_ppv~~
- ~~post_hd~~

**原因**: 未在原始高分辨率网格上评估，计算基准错误。

---

## 🎯 训练指导

### 当前阶段（使用短期修复版本）

1. **启动训练**:
   ```powershell
   cd "c:\MISC\Deepcare\Orthodontics"
   C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train.py"
   ```

2. **监控指标**:
   - 关注 `Raw DSC` 的变化
   - 目标: `Raw DSC > 0.85` 为良好性能
   - 论文参考: `Raw DSC ≈ 0.87-0.90`

3. **输出示例**:
   ```
   Epoch 1/200 | Train Loss: 0.2345 | Val Loss: 0.1876 | 
   Raw DSC: 0.8234±0.0567 | Raw SEN: 0.8456±0.0432 | 
   Raw PPV: 0.8123±0.0589 | Raw HD: 1.234±0.456 mm | 
   Time: 3245.1s | Sec/scan: 13.52s | PeakMem: 3456.7MB | LR: 0.001000
   ```

4. **最佳模型保存**:
   - 自动保存在: `outputs/segmentation/module1_train/best.pt`
   - 基于 `raw_dsc` 最高值
   - 每个 epoch 也保存为 `last.pt`

---

## 📝 CSV 日志说明

训练日志保存在: `outputs/segmentation/module1_train/train_log.csv`

**列含义**:
- `epoch`: 训练轮次
- `train_loss`: 训练损失
- `val_loss`: 验证损失
- `raw_dsc`, `raw_dsc_std`: Raw DSC 均值和标准差 ✅ **有效**
- `raw_sen`, `raw_sen_std`: Raw 敏感度 ✅ **有效**
- `raw_ppv`, `raw_ppv_std`: Raw 精确度 ✅ **有效**
- `raw_hd`, `raw_hd_std`: Raw Hausdorff 距离 ✅ **有效**
- `post_dsc`, ... : Post 指标 ❌ **无效，忽略这些列**
- `lr`: 学习率
- `sec_per_scan`: 每个样本训练时间

**分析建议**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv("outputs/segmentation/module1_train/train_log.csv")

# 绘制 Raw DSC 曲线
plt.plot(df['epoch'], df['raw_dsc'], label='Raw DSC')
plt.xlabel('Epoch')
plt.ylabel('Raw DSC')
plt.legend()
plt.show()

# 找到最佳 epoch
best_epoch = df.loc[df['raw_dsc'].idxmax()]
print(f"Best epoch: {best_epoch['epoch']}")
print(f"Best Raw DSC: {best_epoch['raw_dsc']:.4f}")
```

---

## 🔧 下一步：长期修复（可选）

如果需要与论文结果精确对比，需要实现完整的高分辨率评估：

详见: `POST_PROCESSING_FIX_GUIDE.md`

**核心步骤**:
1. 修改数据加载器传递原始文件路径
2. 实现 `evaluate_cases_with_upsampling` 函数
3. 在原始高分辨率网格（N ≈ 100k）上计算 post_ 指标

**优先级**: 低（除非需要发表论文或精确复现）

---

## 📚 相关文档

- `TRAINING_GUIDE.md` - 完整训练指南
- `MESHSEGNET_QUICKSTART.md` - 快速启动指南
- `POST_PROCESSING_FIX_GUIDE.md` - 后处理修复详细计划
- `README.md` - 项目概述

---

## ✨ 总结

### 修改清单

- ✅ `graphcut_refine` 函数添加警告注释
- ✅ `svm_refine` 函数添加警告注释  
- ✅ 打印输出只显示有效的 `raw_*` 指标
- ✅ 创建完整的修复文档

### 现在的状态

**可以正常训练**: ✅
**指标是否可信**: 
- `raw_*` 指标: ✅ 完全可信
- `post_*` 指标: ❌ 不可信（已隐藏）

**下一步**: 开始训练，关注 `raw_dsc` 指标！

---

**更新时间**: 2025-10-02
**版本**: v2.1 (后处理修复版)
