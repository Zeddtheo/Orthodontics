from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

@dataclass
class PPConfig:
    knn_10k: int = 5
    knn_full: int = 7
    seed_conf_th: float = 0.80   # ↑ 从 0.70 提到 0.80
    bg_seed_th: float = 0.995    # ↑ 牙龈更保守
    gc_beta: float = 12.0        # ← 此处充当 λ，12 是经验稳妥值
    gc_k: int = 8                # ↑ 稍大一点邻域
    gc_iterations: int = 3       # ↑ ICM 迭代
    normal_gamma: float = 10.0   #（保留签名，不再参与权重）
    svm_max_train: int = 8000    # ↑ SVM 训练样本上限
    svm_c: float = 8.0           # ↑ 更硬边界
    svm_gamma: float | str = "scale"
    svm_random_state: Optional[int] = 42
    fill_radius: float = 0.55    # ↓ 半径更小，避免跨牙桥接
    fill_majority: float = 0.65
    fill_max_neighbors: int = 20
    min_component_size: int = 150      # ↓ 留出完整牙冠，仍可清小噪点
    clean_component_neighbors: int = 16


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)


# ==== NEW: helper for kNN edge list (undirected, unique) ====
from typing import List
from sklearn.neighbors import NearestNeighbors

def _knn_edges(pos_mm: np.ndarray, k: int) -> np.ndarray:
    N = int(pos_mm.shape[0])
    if N <= 1 or k <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    k_eff = min(k, N-1)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(pos_mm)
    idx = nbrs.kneighbors(pos_mm, return_distance=False)  # (N, k)
    ii = np.repeat(np.arange(N, dtype=np.int32), k_eff)
    jj = idx.reshape(-1).astype(np.int32, copy=False)
    # make undirected & unique: keep i<j
    mask = ii < jj
    edges = np.stack([ii[mask], jj[mask]], axis=1)
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    # unique
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32, copy=False)


# ==== NEW: φ_ij & θ_ij based pairwise weights (Eq.(3) in paper) ====
def _pairwise_weights_dihedral(pos_mm: np.ndarray,
                               normals: Optional[np.ndarray],
                               edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(0, dtype=np.float32)
    ci = pos_mm[edges[:, 0]]
    cj = pos_mm[edges[:, 1]]
    phi = np.linalg.norm(ci - cj, axis=1)              # φ_ij = |c_i - c_j|
    phi = np.clip(phi, 1e-6, None)

    if normals is None or normals.shape[0] != pos_mm.shape[0]:
        # 没有法向时退化为距离权（仍能抑制跨远邻的断裂）
        return (1.0 / phi).astype(np.float32)

    ni = normals[edges[:, 0]]
    nj = normals[edges[:, 1]]
    ni = ni / np.clip(np.linalg.norm(ni, axis=1, keepdims=True), 1e-6, None)
    nj = nj / np.clip(np.linalg.norm(nj, axis=1, keepdims=True), 1e-6, None)
    cos = np.clip(np.sum(ni * nj, axis=1), -1.0, 1.0)
    theta = np.arccos(np.abs(cos))                     # θ_ij ∈ [0, π]

    # 论文里的光滑项：w_ij ∝ -log(θ/π) * φ_ij
    w = -np.log(np.clip(theta / np.pi, 1e-4, 1.0)) * phi
    # 数值稳定 & 上限裁剪
    w = np.clip(w, 0.0, np.percentile(w, 99.5))
    return w.astype(np.float32, copy=False)


# ==== NEW: ICM optimizer for Potts model (no external deps) ====
def _icm_potts(unary: np.ndarray,
               edges: np.ndarray,
               w_ij: np.ndarray,
               lam: float,
               init_labels: np.ndarray,
               max_iter: int = 5) -> np.ndarray:
    """
    Energy:  E(L) = sum_i unary[i, L_i] + lam * sum_(i,j) w_ij * [L_i != L_j]
    """
    N, C = unary.shape
    labels = init_labels.astype(np.int32, copy=True)

    # adjacency lists
    nbrs: List[List[int]] = [[] for _ in range(N)]
    nbrw:  List[List[float]] = [[] for _ in range(N)]
    for (i, j), wij in zip(edges, w_ij):
        nbrs[i].append(j); nbrw[i].append(float(wij))
        nbrs[j].append(i); nbrw[j].append(float(wij))

    wsum = np.asarray([sum(nbrw[i]) if nbrw[i] else 0.0 for i in range(N)], dtype=np.float32)

    for _ in range(max(1, int(max_iter))):
        changed = 0
        for i in range(N):
            if not nbrs[i]:
                # 无邻居时只看数据项
                new_l = int(np.argmin(unary[i]))
            else:
                # 对每个候选标签 c：penalty_i(c) = Σ_j w_ij * [c != L_j]
                #        = wsum[i] - Σ_{j: L_j == c} w_ij
                nb_labels = np.asarray([labels[j] for j in nbrs[i]], dtype=np.int32)
                nb_weights = np.asarray(nbrw[i], dtype=np.float32)
                # 用加权 bincount 统计 Σ_{j: L_j == c} w_ij
                acc = np.bincount(nb_labels, weights=nb_weights, minlength=C)
                penalty = wsum[i] - acc  # shape (C,)
                energies = unary[i] + float(lam) * penalty
                new_l = int(np.argmin(energies))

            if new_l != labels[i]:
                labels[i] = new_l
                changed += 1
        if changed == 0:
            break
    return labels


# ==== REPLACEMENT: true energy-based refinement (paper Eq.(2)(3)) ====
def _graphcut_refine(
    prob_np: np.ndarray,
    pos_mm_np: np.ndarray,
    beta: float,          # 用作 λ（平滑权重）
    k: int,
    iterations: int,
    normals_np: Optional[np.ndarray] = None,
    normal_gamma: float = 10.0,  # 保留签名兼容，实际未再使用
) -> np.ndarray:
    """
    把网络 softmax 输出 prob_np (N,C) 变成离散标签，再依据论文的局部光滑项
    w_ij = -log(theta/pi) * phi 进行 Potts 模型优化（ICM 近似）。
    返回 one-hot 概率，以便下游 seed/conf 逻辑不变。
    论文公式见 Eq.(2)(3)。
    """
    if prob_np.size == 0 or pos_mm_np is None or pos_mm_np.size == 0:
        return prob_np

    N, C = prob_np.shape
    # 数据项：D_i(c) = -log( p_i(c) + eps )
    eps = 1e-6
    unary = -np.log(np.clip(prob_np.astype(np.float32), eps, 1.0))

    # kNN 边与权
    edges = _knn_edges(pos_mm_np.astype(np.float32), int(max(1, k)))
    w_ij = _pairwise_weights_dihedral(pos_mm_np.astype(np.float32), normals_np, edges)

    # 初值：最大似然
    init_labels = np.argmax(prob_np, axis=1).astype(np.int32)
    labels = _icm_potts(unary, edges, w_ij, lam=float(beta), init_labels=init_labels, max_iter=int(max(1, iterations)))

    # 回写为 one-hot 概率（与现有 seed/conf 逻辑兼容）
    refined = np.zeros_like(prob_np, dtype=np.float32)
    refined[np.arange(N), labels] = 1.0
    return refined


def knn_transfer(
    src_pos: np.ndarray,
    src_labels: np.ndarray,
    dst_pos: np.ndarray,
    k: int,
    src_conf: Optional[np.ndarray] = None,
    src_extra: Optional[np.ndarray] = None,
):
    if dst_pos is None or dst_pos.size == 0:
        return (None, None) if src_extra is None else (None, None, None)
    if src_pos is None or src_pos.size == 0:
        return (None, None) if src_extra is None else (None, None, None)
    num_src = src_pos.shape[0]
    k_eff = max(1, min(k, num_src))
    neigh = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    neigh.fit(src_pos)
    dists, idxs = neigh.kneighbors(dst_pos, return_distance=True)
    weights = 1.0 / np.clip(dists * dists, 1e-8, None)

    out_labels = np.zeros(dst_pos.shape[0], dtype=np.int32)
    out_conf: Optional[np.ndarray] = np.zeros(dst_pos.shape[0], dtype=np.float32) if src_conf is not None else None
    out_extra: Optional[np.ndarray] = (
        np.zeros(dst_pos.shape[0], dtype=np.float32) if src_extra is not None else None
    )

    for i in range(dst_pos.shape[0]):
        nbr_idx = idxs[i]
        w = weights[i]
        neigh_labels = src_labels[nbr_idx]
        if src_conf is not None:
            w_eff = w * src_conf[nbr_idx]
        else:
            w_eff = w
        uniq, inv = np.unique(neigh_labels, return_inverse=True)
        tally = np.zeros(len(uniq), dtype=np.float32)
        np.add.at(tally, inv, w_eff)
        best_idx = int(tally.argmax())
        out_labels[i] = int(uniq[best_idx])
        if out_conf is not None:
            total = float(tally.sum())
            out_conf[i] = float(tally[best_idx] / total) if total > 1e-8 else 0.0
        if out_extra is not None:
            extra_vals = src_extra[nbr_idx]
            weight_sum = float(w_eff.sum())
            if weight_sum <= 1e-8:
                out_extra[i] = float(extra_vals.mean())
            else:
                out_extra[i] = float(np.dot(extra_vals, w_eff) / weight_sum)

    if out_extra is not None:
        return out_labels, out_conf, out_extra
    return out_labels, out_conf
def _radius_majority_fill(
    train_pos: np.ndarray,
    train_labels: np.ndarray,
    query_pos: np.ndarray,
    radius: float,
    majority_thresh: float,
    max_neighbors: int,
) -> Optional[np.ndarray]:
    if query_pos is None or query_pos.size == 0:
        return None
    if train_pos is None or train_pos.size == 0:
        return None
    nn = NearestNeighbors(radius=radius, algorithm="auto")
    nn.fit(train_pos)
    distances_list, indices_list = nn.radius_neighbors(query_pos, radius=radius, return_distance=True)
    fill = np.full(query_pos.shape[0], -1, dtype=np.int32)
    for i, (dists, idxs) in enumerate(zip(distances_list, indices_list)):
        if idxs.size == 0:
            continue
        if idxs.size > max_neighbors:
            order = np.argsort(dists)[:max_neighbors]
            idxs = idxs[order]
        neigh_labs = train_labels[idxs]
        uniq, counts = np.unique(neigh_labs, return_counts=True)
        total = counts.sum()
        if total == 0:
            continue
        majority = counts.max() / total
        if majority >= majority_thresh:
            fill[i] = int(uniq[counts.argmax()])
    return fill


def svm_upsample(
    train_pos: np.ndarray,
    train_labels: np.ndarray,
    query_pos: np.ndarray,
    cfg: PPConfig,
) -> np.ndarray:
    if query_pos is None or query_pos.size == 0:
        return np.zeros(0, dtype=np.int32)
    if train_pos is None or train_pos.size == 0:
        return np.zeros(query_pos.shape[0], dtype=np.int32)
    uniq = np.unique(train_labels)
    if uniq.size == 0:
        return np.zeros(query_pos.shape[0], dtype=np.int32)
    if uniq.size == 1:
        return np.full(query_pos.shape[0], uniq[0], dtype=np.int32)

    max_samples = getattr(cfg, "svm_max_train", None)
    if max_samples is None:
        max_samples = getattr(cfg, "svm_max_samples", train_pos.shape[0])
    try:
        max_samples = max(int(max_samples), 1)
    except Exception:
        max_samples = train_pos.shape[0]

    indices = np.arange(train_pos.shape[0])
    if train_pos.shape[0] > max_samples:
        rng = np.random.default_rng(cfg.svm_random_state)
        indices = rng.choice(train_pos.shape[0], size=max_samples, replace=False)
    train_x = train_pos[indices].astype(np.float64, copy=False)
    train_y = train_labels[indices].astype(np.int32, copy=False)
    try:
        clf = SVC(kernel="rbf",
          C=cfg.svm_c,
          gamma=cfg.svm_gamma,
          class_weight="balanced",
          decision_function_shape="ovr",
          random_state=cfg.svm_random_state)
        clf.fit(train_x, train_y)
        preds = clf.predict(query_pos.astype(np.float64, copy=False))
        return preds.astype(np.int32, copy=False)
    except Exception:
        fallback_labels, _ = knn_transfer(train_pos, train_labels, query_pos, cfg.knn_full)
        return np.zeros(query_pos.shape[0], dtype=np.int32) if fallback_labels is None else fallback_labels.astype(np.int32, copy=False)


def _seed_full_labels(
    labels10: np.ndarray,
    conf10: Optional[np.ndarray],
    orig_cell_ids: np.ndarray,
    full_count: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels10 = np.asarray(labels10, dtype=np.int32)
    orig_ids = np.asarray(orig_cell_ids, dtype=np.int64).reshape(-1)
    conf_arr = None if conf10 is None else np.asarray(conf10, dtype=np.float32).reshape(-1)

    seed_labels = np.full(full_count, 0, dtype=np.int32)
    seed_mask = np.zeros(full_count, dtype=bool)
    valid_mask = (orig_ids >= 0) & (orig_ids < full_count)
    if conf_arr is not None:
        valid_mask &= conf_arr >= float(threshold)

    if not np.any(valid_mask):
        return seed_labels, seed_mask, valid_mask

    valid_ids = orig_ids[valid_mask]
    seed_labels[valid_ids] = labels10[valid_mask]
    seed_mask[valid_ids] = True
    return seed_labels, seed_mask, valid_mask


def _build_neighbors(pos_mm: np.ndarray, k: int) -> Optional[np.ndarray]:
    if pos_mm is None or pos_mm.size == 0:
        return None
    N = pos_mm.shape[0]
    if N <= 1 or k <= 0:
        return None
    k_eff = min(k, N - 1 if N > 1 else 1)
    if k_eff <= 0:
        return None
    neigh = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    neigh.fit(pos_mm)
    return neigh.kneighbors(pos_mm, return_distance=False)


def _cleanup_small_components(
    labels: np.ndarray,
    pos_mm: np.ndarray,
    min_size: int,
    neighbor_k: int,
) -> np.ndarray:
    if labels is None or labels.size == 0 or min_size <= 0:
        return labels
    neighbors = _build_neighbors(pos_mm, neighbor_k)
    if neighbors is None:
        return labels

    labels_out = labels.astype(np.int32, copy=True)
    visited = np.zeros(labels_out.shape[0], dtype=bool)

    for idx in range(labels_out.shape[0]):
        label = labels_out[idx]
        if label <= 0 or visited[idx]:
            continue
        stack = [idx]
        component = []
        visited[idx] = True
        while stack:
            current = stack.pop()
            component.append(current)
            for nbr in neighbors[current]:
                if labels_out[nbr] == label and not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        if len(component) >= min_size:
            continue
        boundary_labels = []
        for node in component:
            for nbr in neighbors[node]:
                nb_label = labels_out[nbr]
                if nb_label != label:
                    boundary_labels.append(nb_label)
        if boundary_labels:
            uniq, counts = np.unique(boundary_labels, return_counts=True)
            majority_label = int(uniq[counts.argmax()])
        else:
            majority_label = 0
        for node in component:
            labels_out[node] = majority_label
    return labels_out


def postprocess_6k_10k_full(
    *,
    prob6k: np.ndarray,            # (N6,C)
    pos_mm6: np.ndarray,           # (N6,3)
    pos_mm10: np.ndarray,          # (N10,3)
    pos_mm_full: np.ndarray,       # (Nfull,3)
    cfg: PPConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    # 1) 近似GraphCut：概率平滑（含法向一致性权重） -> 6k标签/置信度
    prob6_ref = _graphcut_refine(
        prob_np=prob6k, pos_mm_np=pos_mm6,
        beta=cfg.gc_beta, k=cfg.gc_k, iterations=cfg.gc_iterations,
        normals_np=None, normal_gamma=cfg.normal_gamma
    )
    conf6 = prob6_ref.max(axis=1).astype(np.float32)
    lab6  = prob6_ref.argmax(axis=1).astype(np.int32)

    # 2) 6k -> 10k：高置信度播种 + KNN 转移
    #    a) 先整体 KNN 转移得到粗标签与置信度
    lab10_knn, conf10_knn = knn_transfer(pos_mm6, lab6, pos_mm10, cfg.knn_10k, conf6)
    if lab10_knn is None:
        # 极端兜底：若KNN失败，全部置背景
        lab10_knn = np.zeros(pos_mm10.shape[0], np.int32)
        conf10_knn = np.zeros_like(lab10_knn, np.float32)
    #    b) 用种子覆盖高置信度区域，其余保留KNN结果
    seed_mask10 = conf10_knn >= float(cfg.seed_conf_th)
    lab10 = lab10_knn.copy()
    # seed_mask 本质已是“高置信度占优”，可直接使用
    # 3) 10k 清理：去小连通域、再半径多数填洞
    lab10 = _cleanup_small_components(
        labels=lab10, pos_mm=pos_mm10,
        min_size=int(cfg.min_component_size),
        neighbor_k=int(cfg.clean_component_neighbors),
    )
    fill = _radius_majority_fill(
        train_pos=pos_mm10, train_labels=lab10,
        query_pos=pos_mm10,
        radius=float(cfg.fill_radius),
        majority_thresh=float(cfg.fill_majority),
        max_neighbors=int(cfg.fill_max_neighbors),
    )
    if fill is not None:
        need = fill >= 0
        lab10[need] = fill[need].astype(np.int32)

    # 4) 10k -> full：SVM(RBF) 上采样（论文原始做法）
    lab_full = svm_upsample(
        train_pos=pos_mm10, train_labels=lab10,
        query_pos=pos_mm_full, cfg=cfg
    )

    return lab10.astype(np.int32, copy=False), lab_full.astype(np.int32, copy=False)

