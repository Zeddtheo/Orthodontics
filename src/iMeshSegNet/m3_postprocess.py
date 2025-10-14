from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

@dataclass
class PPConfig:
    knn_10k: int = 5
    knn_full: int = 7
    seed_conf_th: float = 0.80
    bg_seed_th: float = 0.995
    gc_beta: float = 10.0
    gc_k: int = 4
    gc_iterations: int = 2
    full_gc_lambda: float = 6.0
    full_gc_iterations: int = 2
    full_gc_enabled: bool = True
    normal_gamma: float = 10.0
    svm_max_train: int = 0
    svm_c: float = 1.0
    svm_gamma: float | str = "scale"
    svm_random_state: Optional[int] = 42
    fill_radius: float = 0.30
    fill_majority: float = 0.70
    fill_max_neighbors: int = 18
    min_component_size: int = 800
    clean_component_neighbors: int = 12
    min_component_size_full: int = 2000
    enforce_single_component: bool = True
    softmax_temp_full: float = 0.75
    low_conf_threshold: float = 0.65
    low_conf_neighbors: int = 16
    low_conf_delta_th: float = 0.15
    tiny_component_size: int = 10
    gingiva_dilate_iters: int = 2
    gingiva_dilate_thresh: float = 0.58
    gingiva_dilate_k: int = 12
    gingiva_label: int = 0
    gingiva_protect_seeds: bool = True
    gingiva_protect_conf: float = 0.96


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)


# ==== NEW: helper for kNN edge list (undirected, unique) ====
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
    theta = np.arccos(np.clip(cos, -1.0, 1.0))         # θ_ij ∈ [0, π]

    # 基础平滑项：-log(θ/π)·φ
    w_base = -np.log(np.clip(theta / np.pi, 1e-4, 1.0)) * phi

    # 凹凸判别：根据 (ni × nj) 与 (cj-ci) 的方向确定符号
    cross_vec = np.cross(ni, nj)
    dir_vec = cj - ci
    is_convex = np.sum(cross_vec * dir_vec, axis=1) > 0.0

    beta_ij = 1.0 + np.abs(cos)
    coef = np.where(is_convex, 30.0 * beta_ij, 1.0)

    w = w_base * coef
    w = np.clip(w, 0.0, np.percentile(w, 99.5))
    return w.astype(np.float32, copy=False)


def build_cell_adjacency(mesh: pv.PolyData) -> List[np.ndarray]:
    if mesh is None or mesh.n_cells == 0:
        return []
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    adjacency: List[set[int]] = [set() for _ in range(faces.shape[0])]
    edge_map: Dict[Tuple[int, int], int] = {}
    for cid, tri in enumerate(faces):
        a, b, c = map(int, tri)
        edges = ((a, b), (b, c), (c, a))
        for u, v in edges:
            key = (u, v) if u <= v else (v, u)
            if key in edge_map:
                other = edge_map[key]
                if other != cid:
                    adjacency[cid].add(other)
                    adjacency[other].add(cid)
            else:
                edge_map[key] = cid
    return [np.fromiter(neigh, dtype=np.int32) if neigh else np.empty(0, dtype=np.int32) for neigh in adjacency]


def adjacency_to_edge_list(adjacency: Optional[List[np.ndarray]]) -> np.ndarray:
    if adjacency is None:
        return np.zeros((0, 2), dtype=np.int32)
    edges: List[Tuple[int, int]] = []
    total = len(adjacency)
    for i, neigh in enumerate(adjacency):
        if neigh is None:
            continue
        neigh_arr = np.asarray(neigh, dtype=np.int64)
        if neigh_arr.size == 0:
            continue
        for j in neigh_arr:
            if j < 0 or j >= total:
                continue
            if i < int(j):
                edges.append((i, int(j)))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(edges, dtype=np.int32)


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
    edges: Optional[np.ndarray] = None,
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

    if edges is None or edges.size == 0:
        edges_used = _knn_edges(pos_mm_np.astype(np.float32), int(max(1, k)))
    else:
        edges_used = np.asarray(edges, dtype=np.int32)
        if edges_used.ndim != 2 or edges_used.shape[1] != 2:
            raise ValueError("edges must be (E,2) array")
    w_ij = _pairwise_weights_dihedral(pos_mm_np.astype(np.float32), normals_np, edges_used)

    # 初值：最大似然
    init_labels = np.argmax(prob_np, axis=1).astype(np.int32)
    labels = _icm_potts(
        unary,
        edges_used,
        w_ij,
        lam=float(beta),
        init_labels=init_labels,
        max_iter=int(max(1, iterations)),
    )

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
    extra_is_vector = False
    extra_dim = 0
    if src_extra is not None:
        src_extra = np.asarray(src_extra)
        if src_extra.ndim == 1:
            extra_dim = 1
        elif src_extra.ndim == 2:
            extra_dim = src_extra.shape[1]
            extra_is_vector = True
        else:
            raise ValueError("src_extra must be 1D or 2D array")
    num_src = src_pos.shape[0]
    k_eff = max(1, min(k, num_src))
    neigh = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    neigh.fit(src_pos)
    dists, idxs = neigh.kneighbors(dst_pos, return_distance=True)
    weights = 1.0 / np.clip(dists * dists, 1e-8, None)

    out_labels = np.zeros(dst_pos.shape[0], dtype=np.int32)
    out_conf: Optional[np.ndarray] = np.zeros(dst_pos.shape[0], dtype=np.float32) if src_conf is not None else None
    out_extra: Optional[np.ndarray] = None
    if src_extra is not None:
        if extra_is_vector:
            out_extra = np.zeros((dst_pos.shape[0], extra_dim), dtype=np.float32)
        else:
            out_extra = np.zeros(dst_pos.shape[0], dtype=np.float32)

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
                if extra_is_vector:
                    out_extra[i] = extra_vals.mean(axis=0)
                else:
                    out_extra[i] = float(extra_vals.mean())
            else:
                if extra_is_vector:
                    out_extra[i] = np.dot(w_eff, extra_vals) / weight_sum
                else:
                    out_extra[i] = float(np.dot(extra_vals, w_eff) / weight_sum)

    if out_extra is not None:
        return out_labels, out_conf, out_extra
    return out_labels, out_conf


def _soft_assign_probabilities(
    prob_src: Optional[np.ndarray],
    indices: Optional[np.ndarray],
    weights: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if prob_src is None or indices is None or weights is None:
        return None
    prob_src = np.asarray(prob_src, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float32)
    if idx.ndim != 2 or w.shape != idx.shape:
        return None
    if prob_src.ndim != 2 or prob_src.shape[0] <= np.max(idx):
        return None
    selected = prob_src[idx]  # (N_full, K, C)
    w_exp = w[..., None]       # (N_full, K, 1)
    probs = (selected * w_exp).sum(axis=1)
    norm = probs.sum(axis=1, keepdims=True)
    np.divide(
        probs,
        np.clip(norm, 1e-8, None),
        out=probs,
        where=norm > 0,
    )
    return probs


def _majority_filter_low_conf(
    pos: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    threshold: float,
    k: int,
    *,
    prob: Optional[np.ndarray] = None,
    delta_thresh: float = 0.0,
) -> np.ndarray:
    if labels.size == 0 or conf is None:
        return labels
    mask = conf < float(threshold)
    if not np.any(mask):
        return labels
    N = labels.shape[0]
    k_eff = max(2, min(int(k), N))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(pos)
    neighbors = nn.kneighbors(return_distance=False)
    out = labels.astype(np.int32, copy=True)
    max_label = int(labels.max()) if labels.size else 0
    use_prob = prob is not None and isinstance(prob, np.ndarray) and prob.shape[0] == labels.shape[0]
    for idx in np.nonzero(mask)[0]:
        if use_prob:
            vec = np.asarray(prob[idx], dtype=np.float32)
            if vec.ndim == 1 and vec.size > 0:
                top1 = float(vec.max())
                if vec.size > 1:
                    top2 = float(np.partition(vec, -2)[-2])
                else:
                    top2 = 0.0
                if delta_thresh > 0.0 and (top1 - top2) < float(delta_thresh):
                    continue
        neigh_labels = labels[neighbors[idx]]
        counts = np.bincount(neigh_labels, minlength=max_label + 1)
        out[idx] = int(counts.argmax())
    return out
def _radius_majority_fill(
    train_pos: np.ndarray,
    train_labels: np.ndarray,
    query_pos: np.ndarray,
    radius: float,
    majority_thresh: float,
    max_neighbors: int,
    *,
    current_labels: Optional[np.ndarray] = None,
    bg_label: int = 0,
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
        if current_labels is not None:
            if i < current_labels.shape[0] and current_labels[i] != bg_label:
                continue
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      seeds_full: (N_full,)  以 -1 表示未播种；其他为类别id
      seed_mask:  (N_full,)  bool，True表示该full cell为高置信度种子（后续不可覆盖）
    """
    labels10 = np.asarray(labels10, dtype=np.int32)
    assert orig_cell_ids is not None and orig_cell_ids.size == labels10.size, \
        "orig_cell_ids 与 10k 预测不等长"
    if conf10 is None or conf10.size != labels10.size:
        conf10 = np.ones(labels10.shape[0], dtype=np.float32)

    seeds_full = np.full(int(full_count), -1, dtype=np.int32)
    seed_mask  = np.zeros(int(full_count), dtype=bool)

    ok = (orig_cell_ids >= 0) & (orig_cell_ids < full_count)
    ids = orig_cell_ids[ok].astype(np.int64, copy=False)
    labs = labels10[ok]
    cfs = conf10[ok].astype(np.float32, copy=False)

    # 置信度从高到低遍历：高置信度优先写入
    order = np.argsort(-cfs)
    for idx in order:
        oid = int(ids[idx])
        if cfs[idx] >= float(threshold):
            if not seed_mask[oid]:
                seeds_full[oid] = int(labs[idx])
                seed_mask[oid] = True
    return seeds_full, seed_mask



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
    *,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if labels is None or labels.size == 0 or min_size <= 0:
        return labels
    neighbors = _build_neighbors(pos_mm, neighbor_k)
    if neighbors is None:
        return labels

    labels_out = labels.astype(np.int32, copy=True)
    protect = None
    if protect_mask is not None:
        protect_arr = np.asarray(protect_mask, dtype=bool)
        if protect_arr.shape[0] == labels_out.shape[0]:
            protect = protect_arr

    N = labels_out.shape[0]
    for cls in np.unique(labels_out):
        if cls <= 0:
            continue
        idx_cls = np.where(labels_out == cls)[0]
        if idx_cls.size == 0:
            continue
        visited = np.zeros(N, dtype=bool)
        components: List[np.ndarray] = []
        for start in idx_cls:
            if visited[start]:
                continue
            stack = [int(start)]
            comp: List[int] = []
            visited[start] = True
            while stack:
                current = stack.pop()
                comp.append(current)
                nbrs = neighbors[current]
                for nbr in nbrs:
                    if nbr < 0 or nbr >= N:
                        continue
                    if not visited[nbr] and labels_out[nbr] == cls:
                        visited[nbr] = True
                        stack.append(int(nbr))
            components.append(np.asarray(comp, dtype=np.int64))

        if not components:
            continue
        components.sort(key=lambda c: c.size, reverse=True)
        for comp_idx, comp_arr in enumerate(components):
            if comp_idx == 0:
                continue  # 保留最大连通域，即便它小于 min_size
            if comp_arr.size >= min_size:
                continue
            if protect is not None and np.any(protect[comp_arr]):
                continue
            boundary_labels: List[int] = []
            for node in comp_arr:
                nbrs = neighbors[node]
                for nbr in nbrs:
                    if nbr < 0 or nbr >= N:
                        continue
                    nb_label = labels_out[nbr]
                    if nb_label != cls:
                        boundary_labels.append(int(nb_label))
            if not boundary_labels:
                continue
            uniq, counts = np.unique(boundary_labels, return_counts=True)
            majority_label = int(uniq[counts.argmax()])
            labels_out[comp_arr] = majority_label
    return labels_out


def _cleanup_small_components_full(
    pos: np.ndarray,
    labels: np.ndarray,
    min_size: int,
    k: int = 12,
    protect_mask: Optional[np.ndarray] = None,
    adjacency: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    if labels is None or labels.size == 0 or min_size <= 0:
        return labels
    N = labels.shape[0]
    if N == 0:
        return labels
    if adjacency is not None and len(adjacency) == N:
        neighbors_list = adjacency
    else:
        k_eff = max(2, min(k, N))
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(pos)
        nbr_arr = nn.kneighbors(return_distance=False)
        neighbors_list = [nbr_arr[i] for i in range(N)]

    out = labels.astype(np.int32, copy=True)
    visited = np.zeros(N, dtype=bool)
    protect = None
    if protect_mask is not None and protect_mask.shape[0] == N:
        protect = protect_mask.astype(bool)

    for cls in np.unique(out):
        idx_cls = np.where(out == cls)[0]
        for start in idx_cls:
            if visited[start]:
                continue
            stack = [int(start)]
            comp: list[int] = []
            visited[start] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = neighbors_list[u]
                for v in neigh:
                    if v < 0 or v >= N:
                        continue
                    if not visited[v] and out[v] == cls:
                        visited[v] = True
                        stack.append(int(v))
            comp_arr = np.asarray(comp, dtype=np.int64)
            if comp_arr.size >= min_size:
                continue
            if protect is not None and np.any(protect[comp_arr]):
                continue
            boundary_labels: List[int] = []
            for node in comp_arr:
                neigh = neighbors_list[node]
                for nbr in neigh:
                    if nbr < 0 or nbr >= N:
                        continue
                    nb_label = out[nbr]
                    if nb_label != cls:
                        boundary_labels.append(int(nb_label))
            if not boundary_labels:
                continue
            maj = int(np.bincount(np.asarray(boundary_labels), minlength=int(out.max()) + 1).argmax())
            out[comp_arr] = maj
    return out


def _enforce_single_component(
    pos: np.ndarray,
    labels: np.ndarray,
    k: int = 12,
    protect_mask: Optional[np.ndarray] = None,
    adjacency: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    if labels is None or labels.size == 0:
        return labels
    N = labels.shape[0]
    if N == 0:
        return labels
    unique_classes = [int(cls) for cls in np.unique(labels) if int(cls) > 0]
    if not unique_classes:
        return labels
    if adjacency is not None and len(adjacency) == N:
        neighbors_list = adjacency
    else:
        k_eff = max(2, min(k, N))
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(pos)
        nbr_arr = nn.kneighbors(return_distance=False)
        neighbors_list = [nbr_arr[i] for i in range(N)]
    out = labels.astype(np.int32, copy=True)
    protect = None
    if protect_mask is not None and protect_mask.shape[0] == N:
        protect = protect_mask.astype(bool)

    for cls in unique_classes:
        idx_cls = np.where(out == cls)[0]
        if idx_cls.size == 0:
            continue
        visited = np.zeros(N, dtype=bool)
        components: List[np.ndarray] = []
        for start in idx_cls:
            if visited[start]:
                continue
            stack = [int(start)]
            comp = []
            visited[start] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = neighbors_list[u]
                for v in neigh:
                    if not visited[v] and out[v] == cls:
                        visited[v] = True
                        stack.append(int(v))
            components.append(np.asarray(comp, dtype=np.int64))
        if len(components) <= 1:
            continue
        components.sort(key=lambda c: c.size, reverse=True)
        for comp_arr in components[1:]:
            if comp_arr.size == 0:
                continue
            if protect is not None and np.any(protect[comp_arr]):
                continue
            neigh_labels: List[int] = []
            for node in comp_arr:
                neigh = neighbors_list[node]
                for nbr in neigh:
                    if nbr < 0 or nbr >= N:
                        continue
                    nb_label = out[nbr]
                    if nb_label != cls:
                        neigh_labels.append(int(nb_label))
            if not neigh_labels:
                majority = 0
            else:
                majority = int(np.bincount(np.asarray(neigh_labels), minlength=int(out.max()) + 1).argmax())
                if majority == cls:
                    majority = 0
            out[comp_arr] = majority
    return out


def _dilate_label(
    pos: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    *,
    k: int = 12,
    threshold: float = 0.5,
    iterations: int = 1,
    protect_mask: Optional[np.ndarray] = None,
    adjacency: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    if labels is None or labels.size == 0 or iterations <= 0:
        return labels
    if adjacency is not None and len(adjacency) == labels.shape[0]:
        neighbors_list = adjacency
    else:
        neighbors = _build_neighbors(pos, k)
        if neighbors is None:
            return labels
        neighbors_list = [neighbors[i] for i in range(neighbors.shape[0])]
    out = labels.astype(np.int32, copy=True)
    protect: Optional[np.ndarray] = None
    if protect_mask is not None and protect_mask.shape[0] == out.shape[0]:
        protect = protect_mask.astype(bool, copy=False)
    for _ in range(int(max(1, iterations))):
        mask_target = out == int(target_label)
        ratios = np.zeros(out.shape[0], dtype=np.float32)
        for idx, neigh in enumerate(neighbors_list):
            if neigh is None or len(neigh) == 0:
                continue
            neigh = np.asarray(neigh, dtype=np.int64)
            valid = (neigh >= 0) & (neigh < out.shape[0])
            if not np.any(valid):
                continue
            neigh_valid = neigh[valid]
            ratios[idx] = float(mask_target[neigh_valid].sum()) / float(neigh_valid.size)
        grow = (~mask_target) & (ratios >= float(threshold))
        if protect is not None:
            grow &= ~protect
        if not np.any(grow):
            break
        out[grow] = int(target_label)
    return out


def postprocess_6k_10k_full(
    pos6: np.ndarray,
    logits6: np.ndarray,
    pos10: np.ndarray,
    pos_full: np.ndarray,
    *,
    normals6: Optional[np.ndarray] = None,
    normals10: Optional[np.ndarray] = None,
    orig_cell_ids: Optional[np.ndarray],
    assign_ids: Optional[np.ndarray] = None,
    assign_indices: Optional[np.ndarray] = None,
    assign_weights: Optional[np.ndarray] = None,
    cfg: PPConfig,
    full_adjacency: Optional[List[np.ndarray]] = None,
    adjacency10: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    logs: Dict[str, float] = {}

    # ---- 6k 概率 -> 10k 标签/置信度（带平滑）----
    prob6 = softmax_np(logits6)
    num_classes = prob6.shape[1]
    prob6_ref = _graphcut_refine(
        prob6,
        pos6,
        beta=cfg.gc_beta,
        k=cfg.gc_k,
        iterations=cfg.gc_iterations,
        normals_np=normals6,
        normal_gamma=cfg.normal_gamma,
    )
    labels6 = np.argmax(prob6_ref, axis=1).astype(np.int32)
    conf6 = prob6_ref[np.arange(prob6_ref.shape[0]), labels6].astype(np.float32)

    labels10, conf10, prob10 = knn_transfer(
        pos6, labels6,
        pos10, cfg.knn_10k,
        src_conf=conf6,
        src_extra=prob6_ref,
    )
    if conf10 is None:
        conf10 = np.ones(labels10.shape[0], dtype=np.float32)
    if prob10 is None:
        prob10 = np.eye(num_classes, dtype=np.float32)[labels10]
    else:
        row_sum = prob10.sum(axis=1, keepdims=True)
        np.divide(prob10, np.clip(row_sum, 1e-8, None), out=prob10)
    logs["conf10_mean"] = float(np.mean(conf10))

    labels10 = _cleanup_small_components(
        labels=labels10,
        pos_mm=pos10,
        min_size=int(cfg.min_component_size),
        neighbor_k=int(cfg.clean_component_neighbors),
    )

    N_full = int(pos_full.shape[0])
    orig_ids_valid = orig_labs_valid = orig_conf_valid = None
    if orig_cell_ids is not None and orig_cell_ids.size > 0 and N_full > 0:
        mapped = np.asarray(orig_cell_ids, dtype=np.int64)
        valid = (mapped >= 0) & (mapped < N_full)
        if np.any(valid):
            idx_valid = np.nonzero(valid)[0]
            orig_ids_valid = mapped[idx_valid].astype(np.int64, copy=False)
            orig_labs_valid = labels10[idx_valid]
            orig_conf_valid = conf10[idx_valid] if conf10 is not None else np.ones(idx_valid.size, dtype=np.float32)

    # (B) 10k -> full 加权 KNN 种子
    labels_knn, conf_knn = knn_transfer(
        src_pos=pos10,
        src_labels=labels10,
        dst_pos=pos_full,
        k=cfg.knn_full,
        src_conf=conf10,
    )
    if conf_knn is None:
        conf_knn = np.ones(N_full, dtype=np.float32)

    prob_full_knn = _soft_assign_probabilities(prob10, assign_indices, assign_weights)
    used_soft = False
    if (
        assign_indices is not None
        and assign_weights is not None
        and assign_indices.shape[0] == N_full
        and assign_indices.shape == assign_weights.shape
    ):
        used_soft = True
        vote_bins = np.zeros((N_full, num_classes), dtype=np.float32)
        idx_range = np.arange(N_full, dtype=np.int32)
        for k in range(assign_indices.shape[1]):
            lbl = labels10[assign_indices[:, k]]
            np.add.at(vote_bins, (idx_range, lbl), assign_weights[:, k])
        knn_labels = np.argmax(vote_bins, axis=1).astype(np.int32)
        weight_sum = vote_bins.sum(axis=1, keepdims=True)
        knn_conf = vote_bins[np.arange(N_full), knn_labels]
        knn_conf = np.divide(knn_conf, np.clip(weight_sum.squeeze(), 1e-8, None))
        if prob_full_knn is not None and prob_full_knn.shape[0] == N_full:
            vote_bins = prob_full_knn  # 保留软概率供后续调试/日志
        thresh = float(getattr(cfg, 'low_conf_threshold', 0.0))
        if thresh > 0.0:
            knn_labels = _majority_filter_low_conf(
                pos_full, knn_labels, knn_conf, threshold=thresh, k=int(getattr(cfg, 'low_conf_neighbors', 12))
            )
            knn_conf = np.clip(knn_conf, 0.0, 1.0)
    else:
        knn_labels = labels_knn.astype(np.int32, copy=False)
        knn_conf = conf_knn.astype(np.float32, copy=False)

    seed_full = knn_labels.astype(np.int32, copy=False)
    seed_conf = knn_conf.astype(np.float32, copy=False)

    if orig_ids_valid is not None:
        for oid, lab, cf in zip(orig_ids_valid, orig_labs_valid, orig_conf_valid):
            if cf > seed_conf[oid]:
                seed_full[oid] = int(lab)
                seed_conf[oid] = float(cf)

    mask_knn = knn_conf >= float(cfg.seed_conf_th)
    if used_soft:
        logs['soft_seed_ratio'] = float(np.mean(mask_knn))

    if assign_ids is not None:
        assign_arr = np.asarray(assign_ids, dtype=np.int64)
        if assign_arr.shape[0] == N_full:
            valid_assign = (assign_arr >= 0) & (assign_arr < labels10.shape[0])
            if np.any(valid_assign):
                conf_src = conf10 if conf10 is not None else np.ones(labels10.shape[0], dtype=np.float32)
                assign_conf = np.zeros(N_full, dtype=np.float32)
                assign_labels = np.full(N_full, -1, dtype=np.int32)
                assign_conf[valid_assign] = conf_src[assign_arr[valid_assign]]
                assign_labels[valid_assign] = labels10[assign_arr[valid_assign]]
                mask_assign = (assign_conf >= float(cfg.seed_conf_th)) & (assign_conf > seed_conf)
                if np.any(mask_assign):
                    seed_full[mask_assign] = assign_labels[mask_assign]
                    seed_conf[mask_assign] = assign_conf[mask_assign]

    seed_ratio = float(np.mean(seed_conf >= float(cfg.seed_conf_th))) if N_full > 0 else 0.0
    logs['seed_ratio'] = seed_ratio

    # ---------- 仅对未标记空洞做半径多数 ----------
    unl_mask = seed_full < 0
    if np.any(unl_mask):
        fill = _radius_majority_fill(
            train_pos=pos_full[seed_full >= 0],
            train_labels=seed_full[seed_full >= 0],
            query_pos=pos_full[unl_mask],
            radius=cfg.fill_radius,
            majority_thresh=cfg.fill_majority,
            max_neighbors=cfg.fill_max_neighbors,
        )
        if fill is not None:
            take = fill >= 0
            if np.any(take):
                unl_indices = np.nonzero(unl_mask)[0]
                seed_full[unl_indices[take]] = fill[take].astype(np.int32, copy=False)
                fill_conf = float(getattr(cfg, "seed_conf_th", 0.7)) * 0.5
                seed_conf[unl_indices[take]] = np.maximum(seed_conf[unl_indices[take]], fill_conf)

    # ---------- 余量 KNN 填补（禁用 SVM，保持软传递一致） ----------
    if np.any(seed_full < 0):
        train_mask = seed_full >= 0
        if np.count_nonzero(train_mask) > 0:
            fallback = knn_transfer(
                pos_full[train_mask],
                seed_full[train_mask],
                pos_full,
                max(int(cfg.knn_full), 1),
                src_conf=seed_conf[train_mask],
            )
            if fallback is not None:
                if isinstance(fallback, tuple):
                    fallback_labels = fallback[0]
                    fallback_conf = fallback[1] if len(fallback) > 1 else None
                else:
                    fallback_labels = fallback
                    fallback_conf = None
                if fallback_labels is not None:
                    missing = seed_full < 0
                    seed_full[missing] = fallback_labels[missing].astype(np.int32, copy=False)
                    if fallback_conf is not None:
                        seed_conf[missing] = np.maximum(seed_conf[missing], fallback_conf[missing].astype(np.float32, copy=False))
        remaining = seed_full < 0
        if np.any(remaining):
            seed_full[remaining] = 0
            seed_conf[remaining] = np.maximum(seed_conf[remaining], 0.0)
    pred_full = seed_full.copy()

    # ---------- Full 级小连通域清理 ----------
    protect_general = seed_full >= 0
    pred_full = _cleanup_small_components_full(
        pos=pos_full,
        labels=pred_full,
        min_size=int(getattr(cfg, "min_component_size_full", cfg.min_component_size)),
        k=int(cfg.clean_component_neighbors),
        protect_mask=protect_general,
        adjacency=full_adjacency,
    )
    if getattr(cfg, "enforce_single_component", True):
        pred_full = _enforce_single_component(
            pos=pos_full,
            labels=pred_full,
            k=int(cfg.clean_component_neighbors),
            protect_mask=protect_general,
            adjacency=full_adjacency,
        )

    if (
        getattr(cfg, "full_gc_enabled", True)
        and N_full > 0
        and num_classes > 0
    ):
        prob_for_icm: Optional[np.ndarray]
        if (
            prob_full_knn is not None
            and prob_full_knn.shape[0] == N_full
            and prob_full_knn.shape[1] == num_classes
        ):
            prob_for_icm = prob_full_knn.astype(np.float32, copy=True)
            row_sum = prob_for_icm.sum(axis=1, keepdims=True)
            np.divide(
                prob_for_icm,
                np.clip(row_sum, 1e-8, None),
                out=prob_for_icm,
                where=row_sum > 0,
            )
        else:
            prob_for_icm = np.zeros((N_full, num_classes), dtype=np.float32)
            valid = (pred_full >= 0) & (pred_full < num_classes)
            if np.any(valid):
                idx_valid = np.nonzero(valid)[0]
                prob_for_icm[idx_valid, pred_full[idx_valid]] = 1.0
            invalid = ~valid
            if np.any(invalid):
                prob_for_icm[invalid] = 1.0 / float(num_classes)
        strong_mask = (seed_conf >= float(cfg.seed_conf_th)) & (pred_full >= 0) & (pred_full < num_classes)
        if np.any(strong_mask):
            idx_strong = np.nonzero(strong_mask)[0]
            prob_for_icm[idx_strong] = 0.0
            prob_for_icm[idx_strong, pred_full[idx_strong]] = 1.0

        edges_full = adjacency_to_edge_list(full_adjacency)
        if edges_full.size == 0:
            edges_full = _knn_edges(pos_full.astype(np.float32), max(int(cfg.knn_full), 1))
        lam_full = float(getattr(cfg, "full_gc_lambda", 8.0))
        max_iter_full = int(getattr(cfg, "full_gc_iterations", 2))
        if edges_full.size > 0 and lam_full > 0.0 and max_iter_full > 0:
            w_full = _pairwise_weights_dihedral(pos_full.astype(np.float32), None, edges_full)
            unary_full = -np.log(np.clip(prob_for_icm, 1e-6, 1.0))
            pred_full = _icm_potts(
                unary=unary_full,
                edges=edges_full,
                w_ij=w_full,
                lam=lam_full,
                init_labels=pred_full,
                max_iter=max_iter_full,
            )

    gingiva_iters = int(getattr(cfg, "gingiva_dilate_iters", 0))
    if gingiva_iters > 0:
        gingiva_label = int(getattr(cfg, "gingiva_label", 0))
        gingiva_thresh = float(getattr(cfg, "gingiva_dilate_thresh", 0.5))
        gingiva_k = int(getattr(cfg, "gingiva_dilate_k", cfg.clean_component_neighbors))
        protect_mask = None
        if getattr(cfg, "gingiva_protect_seeds", True):
            protect_thr = float(getattr(cfg, "gingiva_protect_conf", cfg.seed_conf_th))
            protect_mask = (seed_conf >= protect_thr) & (seed_full != gingiva_label)
        pred_full = _dilate_label(
            pos=pos_full,
            labels=pred_full,
            target_label=gingiva_label,
            k=gingiva_k,
            threshold=gingiva_thresh,
            iterations=gingiva_iters,
            protect_mask=protect_mask,
            adjacency=full_adjacency,
        )

    tiny_size = int(getattr(cfg, 'tiny_component_size', 0))
    if tiny_size > 0:
        pred_full = _cleanup_small_components_full(
            pos=pos_full,
            labels=pred_full,
            min_size=tiny_size,
            k=int(cfg.clean_component_neighbors),
            protect_mask=None,
            adjacency=full_adjacency,
        )

    return labels10.astype(np.int32, copy=False), pred_full.astype(np.int32, copy=False), logs
