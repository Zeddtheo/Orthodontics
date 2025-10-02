"""
诊断脚本：对训练样本进行推理，验证模型是否真的学到了
"""
import torch
import numpy as np
import pyvista as pv
from pathlib import Path
import sys

sys.path.insert(0, 'src/iMeshSegNet')
from imeshsegnet import iMeshSegNet
from m0_dataset import extract_features, normalize_mesh_units, load_stats, remap_segmentation_labels

# 1. 加载模型
device = torch.device('cpu')
model = iMeshSegNet(num_classes=15).to(device)
ckpt = torch.load('outputs/overfit/best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ 模型加载成功 (Epoch {ckpt['epoch']}, DSC={ckpt['val_dsc']:.4f})")

# 2. 加载训练样本
mesh = pv.read('datasets/segmentation_dataset/1_U.vtp')
mesh.points -= mesh.center
mesh, scale_factor, diag_before, diag_after = normalize_mesh_units(mesh)
mesh = mesh.triangulate()

# 3. 先提取真实标签（在抽取前）
true_labels_raw = remap_segmentation_labels(np.asarray(mesh.cell_data['Label']))

# 4. 抽取到 10000 cells (与推理一致)
if mesh.n_cells > 10000:
    reduction = 1.0 - (10000 / float(mesh.n_cells))
    mesh_decimated = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)
else:
    mesh_decimated = mesh
print(f"✅ 网格抽取: {mesh_decimated.n_cells} cells")

# 5. 采样到 6000 cells
N = mesh_decimated.n_cells
if N > 6000:
    ids = np.random.permutation(N)[:6000]
    submesh = mesh_decimated.extract_cells(ids.astype(np.int32)).clean()
    submesh = submesh.cast_to_unstructured_grid().extract_surface()
    # 由于抽取和采样会改变 cell 的对应关系，我们只能用采样后的网格
    # 无法精确获取真实标签，所以只能评估大致的性能
    true_labels = np.zeros(submesh.n_cells, dtype=np.int64)  # 占位符
    has_true_labels = False
else:
    submesh = mesh_decimated
    true_labels = np.zeros(submesh.n_cells, dtype=np.int64)
    has_true_labels = False

print(f"✅ 采样: {submesh.n_cells} cells")

# 5. 提取特征
feats = extract_features(submesh).astype(np.float32)
pos_raw = submesh.cell_centers().points.astype(np.float32)
scale_pos = diag_after if diag_after > 1e-6 else 1.0
pos_raw = pos_raw / scale_pos

# 6. 标准化
mean, std = load_stats(Path('outputs/segmentation/stats.npz'))
feats_t = torch.from_numpy(feats)
feats_t = (feats_t - torch.from_numpy(mean)) / torch.from_numpy(std)
feats_t = feats_t.transpose(0, 1).contiguous().unsqueeze(0)  # (1,15,N)

pos_t = torch.from_numpy(pos_raw).transpose(0, 1).contiguous().unsqueeze(0)  # (1,3,N)

feats_t = feats_t.to(device).float()
pos_t = pos_t.to(device).float()

# 7. 推理
with torch.no_grad():
    logits = model(feats_t, pos_t)  # (1,15,N)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (N,)

# 8. 只分析预测分布（无法精确计算准确率，因为抽取/采样改变了 cell 对应关系）
print(f"\n📊 推理结果:")
unique_pred, counts_pred = np.unique(pred, return_counts=True)
print("  预测标签分布:")
for u, c in zip(unique_pred, counts_pred):
    print(f"    L{u}: {c} cells ({c/len(pred)*100:.1f}%)")

# 9. 对比原始训练数据的标签分布
print(f"\n📈 原始训练数据 (1_U.vtp) 标签分布:")
unique_true, counts_true = np.unique(true_labels_raw, return_counts=True)
for u, c in zip(unique_true, counts_true):
    print(f"    L{u}: {c} cells ({c/len(true_labels_raw)*100:.1f}%)")

# 10. 检查哪些类别被遗漏
missing = set(unique_true) - set(unique_pred)
if missing:
    print(f"\n❌ 模型遗漏了这些类别: {sorted(missing)}")
    print("   这表明模型过拟合没有成功学到所有类别！")
else:
    print("\n✅ 模型输出了所有训练类别")
