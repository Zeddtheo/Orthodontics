# 推理指南 (Inference Guide)

## 概述

使用 `m3_infer.py` 脚本可以对原始 STL/VTP 文件进行推理，输出带颜色标签的 VTP 文件。

## 功能特性

- ✅ 支持原始 STL 文件（无颜色信息）作为输入
- ✅ 自动进行网格预处理（归一化、抽取、三角化）
- ✅ 支持 overfit 训练和正常训练的模型
- ✅ 输出带颜色的 VTP 文件（15个类别用不同颜色标识）
- ✅ 同时输出预测标签的 NPY 文件
- ✅ 显示类别分布统计

## 颜色映射

脚本使用以下颜色映射（15个类别）：

| 类别 | 颜色 | RGB值 | 说明 |
|------|------|-------|------|
| 0 | 灰色 | [128, 128, 128] | 背景/牙龈 |
| 1 | 红色 | [255, 0, 0] | 牙齿1 |
| 2 | 橙色 | [255, 127, 0] | 牙齿2 |
| 3 | 黄色 | [255, 255, 0] | 牙齿3 |
| 4 | 绿色 | [0, 255, 0] | 牙齿4 |
| 5 | 青色 | [0, 255, 255] | 牙齿5 |
| 6 | 蓝色 | [0, 0, 255] | 牙齿6 |
| 7 | 紫色 | [127, 0, 255] | 牙齿7 |
| 8 | 品红 | [255, 0, 255] | 牙齿8 |
| 9 | 粉色 | [255, 192, 203] | 牙齿9 |
| 10 | 棕色 | [165, 42, 42] | 牙齿10 |
| 11 | 金色 | [255, 215, 0] | 牙齿11 |
| 12 | 暗青 | [0, 128, 128] | 牙齿12 |
| 13 | 暗紫 | [128, 0, 128] | 牙齿13 |
| 14 | 暗橙 | [255, 140, 0] | 牙齿14 |

## 使用方法

### 1. Overfit 模型推理（单样本测试）

```powershell
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\overfit\best_model.pth `
    --input datasets\landmarks_dataset\raw\1\1_L.stl `
    --stats outputs\segmentation\stats.npz `
    --out outputs\overfit\infer `
    --sample-cells 6000 `
    --target-cells 10000
```

**参数说明：**
- `--ckpt`: 模型权重文件路径
- `--input`: 输入 STL 文件（可以是单个文件或目录）
- `--stats`: 特征标准化统计文件（训练时生成）
- `--out`: 输出目录
- `--sample-cells`: 采样的 cell 数量（默认 8192）
- `--target-cells`: 抽取后的目标 cell 数量（默认 25000）

### 2. 正常训练模型推理

```powershell
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\1\1_L.stl `
    --stats outputs\segmentation\stats.npz `
    --out outputs\segmentation\infer
```

### 3. 批量推理

```powershell
# 推理整个目录的所有 STL 文件
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\ `
    --stats outputs\segmentation\stats.npz `
    --out outputs\segmentation\infer_batch `
    --ext .stl
```

### 4. 混合格式推理

```powershell
# 同时推理 STL 和 VTP 文件
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\ `
    --stats outputs\segmentation\stats.npz `
    --out outputs\segmentation\infer_mixed `
    --ext .stl .vtp
```

## 输出文件

推理完成后，输出目录包含：

1. **`<文件名>_colored.vtp`**: 带颜色的 VTP 文件
   - `cell_data["RGB"]`: 每个 cell 的 RGB 颜色
   - `cell_data["PredLabel"]`: 每个 cell 的预测标签 (0-14)

2. **`<文件名>_pred.npy`**: 预测标签数组
   - NumPy 数组，形状 `(N,)`，数据类型 `int32`
   - 包含每个 cell 的预测标签

## 示例输出

```
[Infer] files=1  device=cpu  target_cells=10000  sample_cells=6000
  ✓ 1_L -> 1_L_colored.vtp (cells=6000, labels=[L0:5290, L1:478, L3:42, L4:109, L5:16, L6:45, L12:16, L13:4])
[Done] outputs in: C:\MISC\Deepcare\Orthodontics\outputs\overfit\infer
```

**解读：**
- `cells=6000`: 采样后的 cell 数量
- `labels=[...]`: 各类别的 cell 数量分布
  - `L0:5290` - 背景/牙龈占 5290 个 cells
  - `L1:478` - 牙齿1占 478 个 cells
  - 依此类推

## 可视化

使用 ParaView 或其他 VTK 可视化工具打开 `*_colored.vtp` 文件：

1. 打开文件后，在 **Coloring** 选项中选择 `RGB`
2. 可以看到不同颜色标识的牙齿区域
3. 也可以选择 `PredLabel` 查看原始标签值

## 高级选项

### 使用 Arch Frame（可选）

如果有 arch frame 信息（JSON 格式），可以添加：

```powershell
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\1\1_L.stl `
    --stats outputs\segmentation\stats.npz `
    --arch-frames outputs\segmentation\arch_frames.json `
    --out outputs\segmentation\infer
```

### 使用 GPU 加速

```powershell
# 默认使用 CPU，如果有 GPU 可以指定
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\1\1_L.stl `
    --stats outputs\segmentation\stats.npz `
    --out outputs\segmentation\infer `
    --device cuda:0
```

### 自定义类别数

```powershell
# 默认 15 类，可以根据模型修改
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\segmentation\checkpoints\best.pth `
    --input datasets\landmarks_dataset\raw\1\1_L.stl `
    --stats outputs\segmentation\stats.npz `
    --out outputs\segmentation\infer `
    --num-classes 15
```

## 故障排除

### 1. 模型加载失败

**错误**: `Weights only load failed`

**解决**: 脚本已自动处理不同 PyTorch 版本的兼容性，如果仍然失败，检查：
- 模型文件是否完整
- PyTorch 版本是否兼容

### 2. 输入文件格式错误

**错误**: `Invalid file extension`

**解决**: 确保输入文件是 `.stl` 或 `.vtp` 格式

### 3. Stats 文件缺失

**错误**: `FileNotFoundError: stats.npz`

**解决**: 
```powershell
# 从训练数据生成 stats.npz
.venv\Scripts\python.exe src\iMeshSegNet\m0_dataset.py
```

### 4. 内存不足

**错误**: `RuntimeError: out of memory`

**解决**: 减小采样数量：
```powershell
--sample-cells 4000 --target-cells 8000
```

## 测试验证

### Overfit 模型测试结果

使用 overfit 模型对 `1_L.stl` 进行推理：

```
✅ 训练结果（100 epochs）:
   - DSC: 0.9229
   - Accuracy: 0.9618
   - Entropy: 0.001

✅ 推理结果:
   - 输出: 1_L_colored.vtp (6000 cells)
   - 类别分布: L0:5290, L1:478, L3:42, L4:109, L5:16, L6:45, L12:16, L13:4
   - 主要预测: 背景/牙龈占主导（正常现象）
```

## 下一步

1. **训练完整模型**: 使用 `m1_train.py` 训练完整数据集
2. **评估模型**: 使用测试集评估模型性能
3. **批量推理**: 对所有测试数据进行推理
4. **可视化分析**: 使用 ParaView 分析推理结果

## 参考

- 训练指南: `TRAINING_GUIDE.md`
- Overfit 测试指南: `OVERFIT_TEST_GUIDE.md`
- 数据集说明: `README.md`
