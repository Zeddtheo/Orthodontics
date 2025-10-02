# MeshSegNet 后处理评估修复指南

## 📋 问题诊断

### 当前实现的问题

1. **graphcut_refine 函数**
   - ❌ 不是真正的图割算法
   - ✅ 只是基于 k-NN 的概率平滑
   - 📝 现已添加注释说明

2. **svm_refine 函数的致命缺陷**
   - ❌ 在**低分辨率**网格上训练和预测（N ≈ 9000）
   - ❌ 没有上采样到**原始高分辨率**网格（N ≈ 100k）
   - 📊 导致所有 `post_*` 指标无效

3. **评估指标的错误**
   - ✅ `raw_*` 指标：有效（在低分辨率上直接预测）
   - ❌ `post_*` 指标：**无效**（未在高分辨率上评估）

---

## ✅ 第一部分：短期修复（已完成）

### 修改1：graphcut_refine 函数添加警告注释

**位置**: `m1_train.py` Line ~145

**修改内容**:
```python
def graphcut_refine(prob_np: np.ndarray, pos_mm_np: np.ndarray, beta: float = 30.0, k: int = 6, iterations: int = 1) -> np.ndarray:
    """
    NOTE: This is an iterative probability smoothing, an approximation of graph-cut's smoothness effect, 
    not a true energy-minimizing graph-cut. This function performs k-NN based probability smoothing 
    to mimic the spatial consistency constraint of graph-cut algorithms.
    """
```

### 修改2：svm_refine 函数添加警告注释

**位置**: `m1_train.py` Line ~175

**修改内容**:
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

### 修改3：注释掉打印输出中的 post_ 指标

**位置**: `m1_train.py` Line ~415

**修改内容**:
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

### 关键变化

- ✅ 所有警告注释已添加
- ✅ 打印输出已简化，只显示有效的 `raw_*` 指标
- ✅ 模型保存逻辑已正确锚定在 `raw_dsc` 上
- ⚠️ CSV 日志仍保留所有指标（便于后续分析）

---

## 🔧 第二部分：长期修复（待实现）

### 核心目标

实现论文描述的完整后处理流程：
1. 在低分辨率网格（N ≈ 9000）上进行模型预测
2. 应用 Graph-cut 平滑（在低分辨率上）
3. **使用 SVM 上采样到原始高分辨率网格（N ≈ 100k）**
4. 在高分辨率网格上计算最终的 `post_*` 指标

---

### 实现路线图

#### 步骤1：修改数据加载器，传递原始文件路径

**文件**: `m0_dataset.py`

**修改 SegmentationDataset.__getitem__**:

```python
def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, str]:
    base_idx, variant_idx = self._resolve_index(idx)
    file_path = Path(self.file_paths[base_idx])
    
    # ... 现有的数据加载和处理逻辑 ...
    
    # 返回时添加原始文件路径
    return (x, pos_norm, pos_mm_scaled, pos_scale), y_tensor, str(file_path)
```

**修改 segmentation_collate**:

```python
def segmentation_collate(batch):
    batch_data, ys, filepaths = zip(*batch)
    xs, pos_norms, pos_mms, pos_scales = zip(*batch_data)
    x = torch.stack(xs, dim=0)
    pos_norm = torch.stack(pos_norms, dim=0)
    pos_mm = torch.stack(pos_mms, dim=0)
    pos_scale = torch.stack(pos_scales, dim=0)
    y = torch.stack(ys, dim=0)
    return (x, pos_norm, pos_mm, pos_scale), y, list(filepaths)
```

---

#### 步骤2：修改训练器的验证循环

**文件**: `m1_train.py`

**修改 Trainer._run_epoch**:

```python
def _run_epoch(self, loader: DataLoader, is_train: bool):
    # ...
    for batch_data in tqdm(loader, desc=desc):
        if len(batch_data) == 3:  # 新格式：包含文件路径
            (x, pos_norm, pos_mm, pos_scale), y, filepaths = batch_data
        else:  # 兼容旧格式
            (x, pos_norm, pos_mm, pos_scale), y = batch_data
            filepaths = [None] * x.size(0)
        
        # ... 训练/验证逻辑 ...
        
        if not is_train:
            # ... 收集预测结果 ...
            for i in range(preds.size(0)):
                case_records.append({
                    "prob": probs[i].transpose(0, 1).contiguous().numpy(),
                    "pred": preds[i].numpy(),
                    "target": targets_cpu[i].numpy(),
                    "pos_mm": pos_mm_cpu[i].numpy(),
                    "diag": float(pos_scale_cpu[i].item()),
                    "filepath": filepaths[i],  # 新增：原始文件路径
                })
```

---

#### 步骤3：实现新的评估函数（支持高分辨率上采样）

**文件**: `m1_train.py`

**新增函数 evaluate_cases_with_upsampling**:

```python
def evaluate_cases_with_upsampling(
    cases: List[Dict[str, any]], 
    num_classes: int
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    完整实现论文的后处理评估流程：
    1. Raw metrics: 在低分辨率网格上直接评估
    2. Post metrics: Graph-cut (低分辨率) + SVM上采样到高分辨率 + 高分辨率评估
    """
    from m0_dataset import find_label_array, remap_segmentation_labels
    import pyvista as pv
    
    raw_metrics: List[Dict[str, float]] = []
    post_metrics: List[Dict[str, float]] = []
    
    for case in tqdm(cases, desc="Evaluating with upsampling"):
        # ===== 1. Raw Metrics（低分辨率直接预测）=====
        raw_metrics.append(
            compute_case_metrics(
                pred_labels=case["pred"],
                target_labels=case["target"],
                pos_mm=case["pos_mm"],
                num_classes=num_classes,
                fallback_diag=case["diag"]
            )
        )
        
        # ===== 2. Post Metrics（完整后处理流程）=====
        if case["filepath"] is None:
            # 如果没有文件路径，回退到旧方法
            post_pred = apply_post_processing(case["prob"], case["pos_mm"])
            post_metrics.append(
                compute_case_metrics(
                    pred_labels=post_pred,
                    target_labels=case["target"],
                    pos_mm=case["pos_mm"],
                    num_classes=num_classes,
                    fallback_diag=case["diag"]
                )
            )
            continue
        
        # 2.1 在低分辨率上应用 Graph-cut
        low_res_probs = case["prob"]  # (N_low, C), N_low ≈ 9000
        low_res_pos_mm = case["pos_mm"]  # (N_low, 3)
        
        refined_low_res_probs = graphcut_refine(
            low_res_probs, 
            low_res_pos_mm, 
            beta=30.0, 
            k=6, 
            iterations=1
        )
        low_res_pred_labels = np.argmax(refined_low_res_probs, axis=1)
        
        # 2.2 加载原始高分辨率网格
        try:
            high_res_mesh = pv.read(case["filepath"])
            
            # 提取高分辨率的真实标签
            label_result = find_label_array(high_res_mesh)
            if label_result is None:
                raise ValueError(f"No labels found in {case['filepath']}")
            
            _, high_res_true_labels_raw = label_result
            high_res_true_labels = remap_segmentation_labels(
                np.asarray(high_res_true_labels_raw)
            )
            
            # 获取高分辨率的单元中心点坐标
            high_res_mesh.compute_normals(cell_normals=True, inplace=True)
            high_res_pos_mm = high_res_mesh.cell_centers().points  # (N_high, 3), N_high ≈ 100k
            
        except Exception as e:
            print(f"Warning: Failed to load high-res mesh from {case['filepath']}: {e}")
            # 回退到低分辨率评估
            post_metrics.append(
                compute_case_metrics(
                    pred_labels=low_res_pred_labels,
                    target_labels=case["target"],
                    pos_mm=case["pos_mm"],
                    num_classes=num_classes,
                    fallback_diag=case["diag"]
                )
            )
            continue
        
        # 2.3 使用 SVM 从低分辨率上采样到高分辨率
        try:
            # 训练 SVM（在低分辨率预测结果上）
            unique_labels = np.unique(low_res_pred_labels)
            if unique_labels.size <= 1:
                # 如果只有一个类别，直接用该类别填充
                high_res_pred_labels = np.full(high_res_pos_mm.shape[0], unique_labels[0])
            else:
                # 限制训练样本数量以加速
                max_train_samples = 5000
                if low_res_pos_mm.shape[0] > max_train_samples:
                    train_indices = np.random.choice(
                        low_res_pos_mm.shape[0], 
                        size=max_train_samples, 
                        replace=False
                    )
                    train_x = low_res_pos_mm[train_indices]
                    train_y = low_res_pred_labels[train_indices]
                else:
                    train_x = low_res_pos_mm
                    train_y = low_res_pred_labels
                
                # 训练 SVM
                svm = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
                svm.fit(train_x.astype(np.float64), train_y)
                
                # 预测高分辨率标签
                high_res_pred_labels = svm.predict(high_res_pos_mm.astype(np.float64))
            
            # 2.4 在高分辨率上计算 post 指标
            high_res_diag = compute_diag_from_points(high_res_pos_mm)
            post_metrics.append(
                compute_case_metrics(
                    pred_labels=high_res_pred_labels,
                    target_labels=high_res_true_labels,
                    pos_mm=high_res_pos_mm,
                    num_classes=num_classes,
                    fallback_diag=high_res_diag
                )
            )
            
        except Exception as e:
            print(f"Warning: SVM upsampling failed for {case['filepath']}: {e}")
            # 回退到低分辨率评估
            post_metrics.append(
                compute_case_metrics(
                    pred_labels=low_res_pred_labels,
                    target_labels=case["target"],
                    pos_mm=case["pos_mm"],
                    num_classes=num_classes,
                    fallback_diag=case["diag"]
                )
            )
    
    return summarize_metrics(raw_metrics), summarize_metrics(post_metrics)
```

---

#### 步骤4：修改训练器使用新的评估函数

**文件**: `m1_train.py`

**修改 Trainer._run_epoch**:

```python
def _run_epoch(self, loader: DataLoader, is_train: bool):
    # ...
    
    avg_loss = total_loss / max(len(loader), 1)
    if not is_train:
        # 使用新的评估函数（支持高分辨率上采样）
        raw_metrics, post_metrics = evaluate_cases_with_upsampling(
            case_records, 
            self.config.num_classes
        )
        return avg_loss, raw_metrics, post_metrics
    return avg_loss, None, None
```

---

### 实现后的效果

完成长期修复后：

1. ✅ **Raw Metrics**: 在低分辨率网格上评估（N ≈ 9000）
   - 反映模型在下采样网格上的直接性能

2. ✅ **Post Metrics**: 在原始高分辨率网格上评估（N ≈ 100k）
   - Graph-cut 平滑（低分辨率）
   - SVM 上采样到高分辨率
   - 在高分辨率上计算最终指标
   - **与论文方法完全一致**

3. ✅ **可比性**: Post metrics 可以与论文结果直接比较

---

## 📊 当前状态总结

### 已完成 ✅

- [x] 为 `graphcut_refine` 添加警告注释
- [x] 为 `svm_refine` 添加警告注释
- [x] 注释掉打印输出中的无效 `post_*` 指标
- [x] 确认模型保存逻辑正确使用 `raw_dsc`
- [x] 创建完整的长期修复实现计划

### 待完成 ⏳

- [ ] 修改 `m0_dataset.py` 传递文件路径
- [ ] 修改 `segmentation_collate` 函数
- [ ] 修改 `Trainer._run_epoch` 收集文件路径
- [ ] 实现 `evaluate_cases_with_upsampling` 函数
- [ ] 测试完整的上采样评估流程

---

## 🎯 训练建议

### 当前阶段（短期修复后）

1. **关注指标**: 只看 `raw_*` 指标
   - `raw_dsc`: Dice 系数（最重要）
   - `raw_sen`: 敏感度/召回率
   - `raw_ppv`: 精确度
   - `raw_hd`: Hausdorff 距离

2. **忽略指标**: 完全忽略 `post_*` 指标
   - 这些指标在当前实现中是无效的

3. **模型选择**: 根据 `raw_dsc` 选择最佳模型
   - 训练器已正确配置为保存 `raw_dsc` 最高的模型

### 长期修复后

完成上采样评估后：
- `raw_*` 指标: 低分辨率性能参考
- `post_*` 指标: 高分辨率最终性能（可与论文比较）

---

## 📝 参考

**论文方法** (正确的后处理流程):
1. 模型在低分辨率网格预测 (N ≈ 9k)
2. Graph-cut 平滑 (低分辨率)
3. **SVM 上采样到原始高分辨率** (N ≈ 100k)
4. 在高分辨率上评估最终指标

**当前实现** (短期修复后):
1. 模型在低分辨率网格预测 (N ≈ 9k)
2. ~~Graph-cut 平滑~~ (k-NN 概率平滑)
3. ~~SVM 在低分辨率上细化~~ (无上采样)
4. ✅ 在低分辨率上评估 `raw_*` 指标（有效）
5. ❌ `post_*` 指标无效（已注释）

---

**创建日期**: 2025-10-02
**最后更新**: 2025-10-02
**状态**: 短期修复已完成，长期修复待实现
