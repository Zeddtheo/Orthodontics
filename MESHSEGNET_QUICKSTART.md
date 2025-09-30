# MeshSegNet 训练启动指南

## ✅ 是的，可以直接点击运行了！

但需要注意以下几点：

### 📋 训练前检查清单

#### 1. ✅ 必需文件已准备
- [x] `outputs/segmentation/module0/dataset_split_fixed.json` - 数据集划分
- [x] `outputs/segmentation/module0/stats.npz` - 特征统计
- [x] `src/iMeshSegNet/m0_dataset.py` - 数据加载器
- [x] `src/iMeshSegNet/m1_train.py` - 训练脚本
- [x] `src/iMeshSegNet/imeshsegnet.py` - 模型定义

#### 2. ✅ 依赖已安装
- [x] PyTorch
- [x] PyVista
- [x] scikit-learn
- [x] tqdm
- [x] numpy

#### 3. ⚠️ **重要提醒：CPU训练非常慢！**

根据测试：
- **每个batch**: ~10秒（CPU）
- **每个epoch**: ~6.5小时（2400 batches × 10秒）
- **200 epochs**: ~1300小时 ≈ **54天**！

---

## 🚀 启动方式

### 方式1：在VS Code中直接运行（推荐）

1. 打开 `src\iMeshSegNet\m1_train.py`
2. 点击右上角的 ▶️ 运行按钮
3. 或按 `Ctrl+F5`

**注意**: 确保VS Code使用的是虚拟环境的Python解释器：
`C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe`

### 方式2：命令行运行

```powershell
# 从项目根目录运行
cd "c:\MISC\Deepcare\Orthodontics"
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train.py"
```

### 方式3：快速测试版本（强烈推荐先运行这个！）

```powershell
# 快速测试：只训练2个epoch，验证流程
cd "c:\MISC\Deepcare\Orthodontics"
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train_quicktest.py"
```

**快速测试的优点**：
- ✅ 只训练2个epoch
- ✅ 只使用前5个batch训练
- ✅ 只使用前3个batch验证
- ✅ 无镜像增强（1倍增强 vs 40倍）
- ✅ 约10-15分钟完成
- ✅ 验证所有组件是否正常工作

---

## 📊 训练配置对比

| 配置项 | 完整训练 (m1_train.py) | 快速测试 (m1_train_quicktest.py) |
|--------|------------------------|-----------------------------------|
| Epochs | 200 | 2 |
| 数据增强 | 20原始+20镜像 | 1原始+0镜像 |
| 每epoch批次 | 全部2400 | 前5个 |
| 验证批次 | 全部60 | 前3个 |
| CPU预计时间 | ~1300小时 | ~15分钟 |
| GPU预计时间 | ~10-20小时 | ~5分钟 |

---

## 🎯 推荐工作流程

### 第一步：快速验证（必做）
```powershell
# 运行快速测试确保一切正常
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train_quicktest.py"
```

**期待输出**：
```
使用设备: cpu (或 cuda)
快速测试模式: 2 epochs
加载数据集...
训练批次数: 2400
验证批次数: 60
初始化模型...
开始训练...
Epoch 1/2 | Train Loss: 0.xxxx | Val Loss: 0.xxxx
Epoch 2/2 | Train Loss: 0.xxxx | Val Loss: 0.xxxx
✅ 快速测试完成！
```

### 第二步：完整训练（有GPU时）
```powershell
# 如果有GPU，运行完整训练
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train.py"
```

### 第三步：监控训练进度
训练日志保存在：
- `outputs/segmentation/module1_train/train_log.csv`
- `outputs/segmentation/module1_train/last.pt` - 最新模型
- `outputs/segmentation/module1_train/best.pt` - 最佳模型

---

## ⚙️ 如果需要调整配置

### 降低训练时间的选项：

#### 选项1：减少epochs（在 m1_train.py 中修改）
```python
@dataclass
class TrainConfig:
    epochs: int = 20  # 从200改为20
```

#### 选项2：减少数据增强（在 m1_train.py 中修改）
```python
def main() -> None:
    config = TrainConfig()
    # 添加这些行
    config.data_config.augment_original_copies = 5  # 从20改为5
    config.data_config.augment_flipped_copies = 5   # 从20改为5
```

#### 选项3：增加batch size（需要更多内存）
```python
@dataclass
class TrainConfig:
    data_config: DataConfig = DataConfig()
    
# 在 DataConfig 中修改
@dataclass
class DataConfig:
    batch_size: int = 4  # 从2改为4（需要足够内存）
```

---

## ✨ 当前状态总结

| 项目 | 状态 |
|------|------|
| 数据集准备 | ✅ 完成 |
| 模型架构 | ✅ 已更新符合论文 |
| 依赖安装 | ✅ 完成 |
| 训练脚本 | ✅ 就绪 |
| 快速测试脚本 | ✅ 已创建 |
| **可以开始训练** | ✅ **是的！** |

---

## 🎬 立即开始

**推荐命令**（先运行快速测试）：
```powershell
cd "c:\MISC\Deepcare\Orthodontics"
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train_quicktest.py"
```

**成功后再运行完整训练**（如果有GPU）：
```powershell
C:/MISC/Deepcare/Orthodontics/.venv/Scripts/python.exe "src\iMeshSegNet\m1_train.py"
```

---

## ⚠️ 注意事项

1. **CPU训练**: 极其缓慢，仅推荐快速测试
2. **GPU训练**: 强烈推荐用于完整训练
3. **内存需求**: batch_size=2 需要约4-8GB RAM/VRAM
4. **磁盘空间**: 模型文件约100MB，日志文件较小

---

**答案：是的，可以直接运行！但强烈建议先运行快速测试版本验证流程。** ✨
