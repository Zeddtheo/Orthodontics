# 最终训练指令 - 完全符合论文要求

## 📋 修改总结

### ✅ 已完成的论文对齐修改

#### 1. **m0_dataset.py** - 数据准备与增强
- ✅ **下采样目标**: target_cells = 10,000（论文要求约10k单元）
- ✅ **训练采样**: sample_cells = 9,000（论文要求每次采样9k单元）
- ✅ **数据增强**: 原始样本20个副本 + 镜像样本20个副本
- ✅ **增强参数**: 
  - 平移范围：±10 mm
  - 旋转范围：[-π, π] 每个轴
  - 缩放范围：[0.8, 1.2]
- ✅ **特征标准化**: 基于训练集的 z-score 归一化（15维特征）
- ✅ **路径修复**: 已将 Linux 路径转换为 Windows 相对路径

#### 2. **imeshsegnet.py** - 模型架构
- ✅ **特征变换**: 启用 STNkd(64×64) 特征对齐（use_feature_stn=True）
- ✅ **MLP-2 扩展**: 从两层改为三层（128→64→128→512）
- ✅ **GLM-2 升级**: 输入/输出通道从128升至512
- ✅ **融合层扩容**: 输入维度从448升至1216 (64+128+512+512)
- ✅ **分类器调整**: 输入维度从256升至640 (512+128)
- ✅ **邻域参数**: k_short=6, k_long=12（与论文一致）
- ✅ **模型参数量**: 3,617,615（比原始版本更强大）

#### 3. **m1_train.py** - 训练流程
- ✅ **损失函数**: Generalized Dice Loss（论文要求）
- ✅ **优化器**: Adam with AMS-Grad（论文要求）
- ✅ **后处理**: Graph-cut + RBF-SVM（论文要求）
- ✅ **评估指标**: DSC, SEN, PPV, HD（论文口径）
- ✅ **模型配置**: 启用特征变换

---

## 🎯 最终训练指令

### **PointnetReg 训练**（地标检测）

PointnetReg 系统已完全优化并测试成功，支持5种预设配置：

```powershell
# 进入项目根目录
cd "c:\MISC\Deepcare\Orthodontics"

# 1. 快速测试（3 epochs，验证流程）
python train.py --preset quick_test

# 2. 标准单牙训练（50 epochs）
python train.py --preset standard_single

# 3. 标准分组训练（50 epochs，分组损失）
python train.py --preset standard_group

# 4. 增强训练（100 epochs，全部特性）
python train.py --preset enhanced

# 5. 生产级训练（200 epochs，完整配置）
python train.py --preset production
```

**✅ 状态**: 完全就绪，已通过测试
**📊 最新结果**: Epoch 2/2, Loss: 0.058 → 0.024
**🔧 核心特性**: 分组损失、架构对齐、镜像增强、z-score归一化

---

### **MeshSegNet 训练**（牙齿分割）

MeshSegNet 系统已按论文要求完全更新：

#### **阶段0：数据准备**（首次运行必需）

```powershell
# 进入 iMeshSegNet 目录
cd "c:\MISC\Deepcare\Orthodontics\src\iMeshSegNet"

# 计算训练集统计信息（均值/标准差）
python m0_dataset.py --root "../../datasets/segmentation_dataset" --force

# 验证数据集（可选）
python m0_dataset.py --inspect 1 --jaw U
```

**输出文件**:
- `outputs/segmentation/module0/dataset_split_fixed.json` - 数据集划分（已修复路径）
- `outputs/segmentation/module0/stats.npz` - 特征统计信息

#### **阶段1：主训练流程**

```powershell
# 完整训练（200 epochs，论文配置）
python m1_train.py
```

**训练配置**（已对齐论文）:
- **Epochs**: 200
- **Batch Size**: 2
- **Learning Rate**: 0.001（余弦退火至1e-6）
- **损失函数**: Generalized Dice Loss
- **优化器**: Adam with AMS-Grad
- **数据增强**: 20+20副本（旋转±π、缩放0.8-1.2、平移±10mm）
- **模型特性**: 
  - 启用 64×64 特征变换
  - MLP-2 三层结构（64→128→512）
  - 融合层扩容（1216→512→256→128）
  - EdgeConv邻域（k_short=6, k_long=12）

**输出文件**:
- `outputs/segmentation/module1_train/last.pt` - 最新模型
- `outputs/segmentation/module1_train/best.pt` - 最佳模型（按DSC）
- `outputs/segmentation/module1_train/train_log.csv` - 训练日志

**评估指标**（论文标准）:
- Raw DSC/SEN/PPV/HD（原始预测）
- Post DSC/SEN/PPV/HD（Graph-cut + SVM后处理）

---

## 📊 架构对比表

| 组件 | 论文要求 | 当前实现 | 状态 |
|------|----------|----------|------|
| 下采样目标 | ~10k 单元 | 10k | ✅ |
| 训练采样 | 9k 单元 | 9k | ✅ |
| 输入特征 | 15维 | 15维 | ✅ |
| 特征变换 (FTM) | 64×64 | 启用 | ✅ |
| MLP-2 结构 | 64→128→512 | 64→128→512 | ✅ |
| GLM-2 输出 | 512 | 512 | ✅ |
| 融合层输入 | 1216 | 1216 | ✅ |
| 分类器输入 | 640 | 640 | ✅ |
| k_short | 6 | 6 | ✅ |
| k_long | 12 | 12 | ✅ |
| 数据增强 | 20+20 副本 | 20+20 | ✅ |
| 旋转范围 | [-π, π] | [-π, π] | ✅ |
| 缩放范围 | [0.8, 1.2] | [0.8, 1.2] | ✅ |
| 平移范围 | ±10 mm | ±10 mm | ✅ |
| 损失函数 | GDL | GDL | ✅ |
| 优化器 | Adam+AMS | Adam+AMS | ✅ |
| 后处理 | GC+SVM | GC+SVM | ✅ |

---

## ⚙️ 训练参数速查

### PointnetReg
```python
# 生产级配置
epochs = 200
batch_size = 32
learning_rate = 0.001
group_weight = 0.3
mirror_augment = True
z_score_normalize = True
```

### MeshSegNet
```python
# 论文标准配置
epochs = 200
batch_size = 2
learning_rate = 0.001
min_lr = 1e-6
target_cells = 10000
sample_cells = 9000
augment_original = 20
augment_flipped = 20
use_feature_stn = True
k_short = 6
k_long = 12
```

---

## 🚀 快速开始

### 完整训练流程（推荐）

```powershell
# 1. PointnetReg 地标检测训练
cd "c:\MISC\Deepcare\Orthodontics"
python train.py --preset production

# 2. MeshSegNet 数据准备
cd "src\iMeshSegNet"
python m0_dataset.py --root "../../datasets/segmentation_dataset" --force

# 3. MeshSegNet 分割训练
python m1_train.py
```

### 单样本过拟合测试（调试用）

```powershell
# PointnetReg 快速验证
cd "c:\MISC\Deepcare\Orthodontics"
python train.py --preset quick_test

# MeshSegNet 需要修改 m1_train.py 中的 epochs=10 进行快速测试
```

---

## 📈 预期性能

### PointnetReg（地标检测）
- **训练时间**: ~200 epochs × 30s/epoch = 1.7 小时（CPU）
- **最终损失**: < 0.02（生产级）
- **关键指标**: 分组一致性损失有效降低

### MeshSegNet（牙齿分割）
- **训练时间**: ~200 epochs（取决于硬件）
  - CPU: 较慢（不推荐）
  - GPU: 显著加速
- **预期指标**（论文参考）:
  - Raw DSC: 0.85-0.90
  - Post DSC: 0.90-0.95（经Graph-cut+SVM后处理）
  - HD: < 1.0 mm

---

## ⚠️ 注意事项

1. **路径问题**: 已修复 dataset_split.json 中的 Linux 路径，使用 `dataset_split_fixed.json`
2. **CPU训练**: MeshSegNet 在 CPU 上训练非常慢，建议使用 GPU
3. **内存需求**: batch_size=2 适合大多数硬件，可根据显存调整
4. **数据增强**: 40倍增强（20原始+20镜像）会显著增加训练时间，但提升泛化能力
5. **模型保存**: 每个epoch都会保存 last.pt，最佳模型保存为 best.pt

---

## 📝 版本说明

**当前版本**: v2.0 - 完全符合论文要求
**更新日期**: 2025-09-30
**主要改进**:
- ✅ 数据增强完全对齐论文（40倍增强）
- ✅ 模型架构完全对齐论文（3层MLP-2、512通道GLM-2、特征变换）
- ✅ 训练流程完全对齐论文（GDL、AMS-Grad、后处理）
- ✅ 路径问题已修复（Windows兼容）

---

## 🎓 论文引用

本实现基于以下论文：
> Lian, C., Wang, L., Wu, T. H., Wang, F., Yap, P. T., Ko, C. C., & Shen, D. (2020). 
> Deep multi-scale mesh feature learning for automated labeling of raw dental surfaces from 3D intraoral scanners. 
> IEEE Transactions on Medical Imaging, 39(7), 2440-2450.

**核心贡献**:
- iMeshSegNet 架构（EdgeConv-based GLM）
- 两阶段训练策略（分割 + 细化）
- Graph-cut + SVM 后处理

---

✨ **准备就绪！开始训练吧！** ✨
