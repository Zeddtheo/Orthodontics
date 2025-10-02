# Pipeline 契约系统实现指南

## ✅ 已完成

###  m5_overfit.py - 保存完整 Pipeline 契约

训练时自动保存：
- Z-score 参数 (mean, std)
- 几何预处理配置 (centered, div_by_diag)
- 采样策略 (sampler, sample_cells, target_cells)
- 特征布局 (rotate_blocks)
- 模型架构 (num_classes, in_channels)

### 2. m3_infer.py - Pipeline 契约加载器

添加了 `load_pipeline_meta()` 和 `_load_model_with_contract()` 函数

## 🔄 进行中

需要修改 `_infer_single_mesh` 使用 pipeline 契约中的参数

## 📝 使用示例

```powershell
# 训练（自动保存契约）
.venv\Scripts\python.exe src\iMeshSegNet\m5_overfit.py --sample 1_U --epochs 100

# 推理（自动读取契约）
.venv\Scripts\python.exe src\iMeshSegNet\m3_infer.py `
    --ckpt outputs\overfit\1_U\best_overfit_1_U.pt `
    --input datasets\landmarks_dataset\raw\1\1_U.stl `
    --out outputs\overfit\infer
```

## 🎯 优势

1. **训练推理一致**：自动使用训练时的配置
2. **多模型支持**：同一推理脚本支持 overfit 和 normal 模型
3. **契约验证**：防止参数不匹配
4. **CLI 覆盖**：需要时可手动覆盖参数

## 详细文档

见 `POST_PROCESSING_FIX_GUIDE.md`
