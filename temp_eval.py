import numpy as np
import pyvista as pv
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, KDTree

order = [31,32,33,34,35,36,37,41,42,43,44,45,46,47]
class_to_fdi = {i+1: order[i] for i in range(len(order))}
class_to_fdi.update({0: 0, 15: 0})

def ensure_fdi(arr):
    arr = np.asarray(arr, dtype=np.int32)
    if arr.size == 0:
        return arr
    if arr.max() <= 20:
        mapped = np.array([class_to_fdi.get(int(v), int(v)) for v in arr], dtype=np.int32)
        return mapped
    return arr

def load_mesh_labels(path):
    mesh = pv.read(str(path))
    labels = None
    if 'Label' in mesh.cell_data:
        labels = mesh.cell_data['Label']
    elif 'PredLabel' in mesh.cell_data:
        labels = mesh.cell_data['PredLabel']
    else:
        raise RuntimeError(f'No Label/PredLabel in {path}')
    labels = ensure_fdi(labels)
    centers = mesh.cell_centers().points.astype(np.float32)
    return mesh, labels, centers

def dice(gt, pred, cls):
    mask_gt = gt == cls
    mask_pred = pred == cls
    inter = np.logical_and(mask_gt, mask_pred).sum(dtype=np.int64)
    tot = mask_gt.sum(dtype=np.int64) + mask_pred.sum(dtype=np.int64)
    return 2.0 * inter / (tot + 1e-8)

def accuracy(gt, pred):
    return float((gt == pred).sum(dtype=np.int64) / max(1, gt.size))

def boundary_flags(labels, pos, k=12):
    N = pos.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=bool)
    k_eff = min(max(1, k), N-1)
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm='auto')
    nn.fit(pos)
    idx = nn.kneighbors(pos, return_distance=False)
    boundary = np.zeros(N, dtype=bool)
    for i in range(N):
        if np.any(labels[idx[i]] != labels[i]):
            boundary[i] = True
    return boundary

def boundary_metrics(gt_labels, pred_labels, pos, k=12, tau=0.2, gingiva_label=15):
    gt_boundary = boundary_flags(gt_labels, pos, k)
    pred_boundary = boundary_flags(pred_labels, pos, k)

    precision = recall = bf1 = 0.0
    if np.any(pred_boundary) and np.any(gt_boundary):
        tree_gt = KDTree(pos[gt_boundary])
        dist_pred, _ = tree_gt.query(pos[pred_boundary], k=1, return_distance=True)
        precision = float(np.mean(dist_pred <= tau)) if dist_pred.size else 0.0

        tree_pred = KDTree(pos[pred_boundary])
        dist_gt, _ = tree_pred.query(pos[gt_boundary], k=1, return_distance=True)
        recall = float(np.mean(dist_gt <= tau)) if dist_gt.size else 0.0
        if precision + recall > 0:
            bf1 = 2 * precision * recall / (precision + recall)

    gingival_mask = gt_boundary & (gt_labels != gingiva_label)
    ger = float(np.mean(pred_labels[gingival_mask] == gingiva_label)) if np.any(gingival_mask) else 0.0

    leak_mask = np.zeros(gt_labels.shape[0], dtype=bool)
    leak_target = np.zeros(gt_labels.shape[0], dtype=np.int32)
    nn = NearestNeighbors(n_neighbors=min(max(1, k), pos.shape[0]-1), algorithm='auto')
    nn.fit(pos)
    idx = nn.kneighbors(pos, return_distance=False)
    for i in range(pos.shape[0]):
        base = gt_labels[i]
        if base <= 0 or base == gingiva_label:
            continue
        neigh = idx[i]
        for j in neigh:
            if gt_labels[j] > 0 and gt_labels[j] != gingiva_label and gt_labels[j] != base:
                leak_mask[i] = True
                leak_target[i] = gt_labels[j]
                break
    ilr = float(np.mean(pred_labels[leak_mask] == leak_target[leak_mask])) if np.any(leak_mask) else 0.0

    return dict(bf1=bf1, precision=precision, recall=recall, ger=ger, ilr=ilr,
                gt_boundary=int(gt_boundary.sum()), pred_boundary=int(pred_boundary.sum()))

def per_tooth_dsc(gt, pred):
    teeth = sorted([cls for cls in np.unique(gt) if cls not in (0,)])
    stats = {}
    for cls in teeth:
        stats[int(cls)] = dice(gt, pred, cls)
    return stats

GT_PATH = Path('datasets/segmentation_dataset/1_L.vtp')
PRED_EXACT = Path('outputs/segmentation/overfit/1_L/infer_exact/1_L_full_colored.vtp')
PRED_POST = Path('outputs/segmentation/overfit/1_L/infer/1_L_full_colored.vtp')

_, gt_labels, gt_pos = load_mesh_labels(GT_PATH)
_, exact_labels, exact_pos = load_mesh_labels(PRED_EXACT)
_, post_labels, post_pos = load_mesh_labels(PRED_POST)

# Align positions: predictions share same geometry as GT (same cell order). Validate size.
assert gt_labels.shape[0] == exact_labels.shape[0] == post_labels.shape[0], 'cell count mismatch'
pos = gt_pos  # assume same ordering

results = {}
for name, pred in [('exact', exact_labels), ('post', post_labels)]:
    acc = accuracy(gt_labels, pred)
    avg_dsc = np.mean([dice(gt_labels, pred, cls) for cls in range(31, 48)])
    teeth_dsc = per_tooth_dsc(gt_labels, pred)
    boundary = boundary_metrics(gt_labels, pred, pos, k=12, tau=0.2)
    confusion = {
        'false_background_cells': int(((pred == 0) & (gt_labels != 0)).sum()),
        'false_tooth_cells': int(((pred != gt_labels) & (gt_labels != 0)).sum()),
    }
    results[name] = {
        'accuracy': acc,
        'mean_dsc_31_47': float(avg_dsc),
        'per_tooth_dsc': teeth_dsc,
        'boundary': boundary,
        'confusion': confusion,
    }

import json
print(json.dumps(results, indent=2))
