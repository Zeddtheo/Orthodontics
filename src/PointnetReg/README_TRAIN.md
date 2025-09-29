# PointnetReg 一键训练使用指南

## 🎯 简化说明

现在只有一个简洁的训练脚本：**`train.py`**

## 🚀 使用方法

### 启动训练
```bash
cd src/PointnetReg
python train.py
```

### 训练选项
选择对应数字即可：

1. **快速测试 (3轮)** - 验证环境和流程，t31牙位
2. **标准单牙位训练** - t31牙位，80轮标准训练  
3. **标准组训练** - 第一磨牙组，80轮训练
4. **增强训练** - 中切牙组，完整增强功能
5. **生产级训练** - 尖牙组，150轮高质量训练

## 📊 训练完成后评估

脚本会自动显示对应的评估命令，例如：
```bash
# 评估单牙位模型
python p2_eval.py --mode per_tooth --tooth t31

# 评估组模型  
python p2_eval.py --mode per_group --group central
```

## 🎛️ 预设配置详情

| 选项 | 模式 | 目标 | 轮数 | 批大小 | 增强功能 | 预计时间 |
|------|------|------|------|--------|----------|----------|
| 1 | per_tooth | t31 | 3 | 4 | 基础 | 5分钟 |  
| 2 | per_tooth | t31 | 80 | 8 | 基础 | 2-4小时 |
| 3 | per_group | m1 | 80 | 12 | 对齐+镜像 | 4-8小时 |
| 4 | per_group | central | 100 | 16 | 完整增强 | 6-12小时 |
| 5 | per_group | canine | 150 | 20 | 最强增强 | 12-24小时 |

## 🔧 自定义训练

如需自定义参数，直接使用 `p1_train.py`：
```bash  
python p1_train.py --mode per_group --group m1 --epochs 100 --batch_size 16 --augment --arch_align --mirror_prob 0.3
```

更多参数说明请参考 `TRAINING_PARAMETERS_GUIDE.md`

---

**推荐使用流程**：
1. 新手：选择 **1** (快速测试) 验证环境
2. 日常：选择 **2** 或 **3** 进行标准训练  
3. 高质量：选择 **4** 或 **5** 进行增强训练