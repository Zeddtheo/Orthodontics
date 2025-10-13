# module0_dataset.py
# 数据集与数据加载器工具，用于 iMeshSegNet 训练阶段。

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import pyvista as pv
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


SEG_ROOT = Path("outputs/segmentation")
DECIM_CACHE = SEG_ROOT / "module0" / "cache_decimated"
DECIM_CACHE.mkdir(parents=True, exist_ok=True)

# 自动单位归一控制：当包围盒对角线小于 1（推测单位为米）时放大到毫米
UNIT_SCALE_THRESHOLD = 1.0  # if bounding-box diag < 1 (likely metres), rescale to mm
UNIT_SCALE_FACTOR = 1000.0   # convert m -> mm

# =============================================================================
# Part 0.5: Label remapping（FDI -> contiguous）
# =============================================================================


# 恒牙 FDI 编码顺序（背景沿用 0）
FDI_LABELS: Tuple[int, ...] = (
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
)

LABEL_REMAP: Dict[int, int] = {0: 0}
for idx, tooth in enumerate(FDI_LABELS, start=1):
    LABEL_REMAP[tooth] = idx

# 乳牙或未定义标签：默认为背景（或后续可按需改成 ignore_index）
LABEL_REMAP[65] = 0

# 供各模块共享的类别数（全口设置）
SEG_NUM_CLASSES = 1 + len(FDI_LABELS)

# 单弓 14 牙 + 牙龈（16 类）设置
ARCH_LABEL_ORDERS: Dict[str, Tuple[int, ...]] = {
    "U": (
        21, 22, 23, 24, 25, 26, 27,
        11, 12, 13, 14, 15, 16, 17,
    ),
    "L": (
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47,
    ),
}
SINGLE_ARCH_NUM_CLASSES = 16


def _build_mirror_pairs(order: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    half = len(order) // 2
    return tuple(zip(order[:half], order[half:]))


ARCH_MIRROR_PAIRS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    jaw: _build_mirror_pairs(order)
    for jaw, order in ARCH_LABEL_ORDERS.items()
}

FULL_FDI_MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    pair for pairs in ARCH_MIRROR_PAIRS.values() for pair in pairs
)


def _remap_value(v: int) -> int:
    return LABEL_REMAP.get(int(v), 0)


_vectorized_remap = np.vectorize(_remap_value, otypes=[np.int64])


def remap_segmentation_labels(arr: np.ndarray) -> np.ndarray:
    """Map raw FDI labels to contiguous indices starting from 0."""
    if arr.size == 0:
        return arr.astype(np.int64, copy=False)
    return _vectorized_remap(arr)


def _infer_jaw_from_stem(stem: str) -> str:
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[-1].upper()
        if suffix in ARCH_LABEL_ORDERS:
            return suffix
    raise ValueError(
        f"无法从文件名推断颌侧: {stem}. 需要 '_U' 或 '_L' 后缀以确保标签映射正确"
    )


def _build_single_arch_label_maps(
    gingiva_src: int,
    gingiva_class: int,
    keep_void_zero: bool,
) -> Dict[str, Dict[int, int]]:
    if keep_void_zero and gingiva_src == 0:
        raise ValueError(
            "gingiva_src_label=0 与 keep_void_zero=True 冲突：0 不能同时代表背景和牙龈"
        )
    maps: Dict[str, Dict[int, int]] = {}
    for jaw, order in ARCH_LABEL_ORDERS.items():
        mapping: Dict[int, int] = {}
        if keep_void_zero:
            mapping[0] = 0
        for idx, tooth in enumerate(order, start=1):
            mapping[tooth] = idx
        mapping[gingiva_src] = gingiva_class
        maps[jaw] = mapping
    return maps


def _apply_label_mapping(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.int64, copy=False)
    mapped = np.empty_like(labels, dtype=np.int64)
    uniq = np.unique(labels)
    for raw in uniq:
        mapped[labels == raw] = mapping.get(int(raw), 0)
    return mapped


def remap_labels_single_arch(
    labels: np.ndarray,
    file_path: Path,
    single_arch_maps: Dict[str, Dict[int, int]],
) -> np.ndarray:
    jaw = _infer_jaw_from_stem(file_path.stem)
    mapping = single_arch_maps.get(jaw)
    if mapping is None:
        raise KeyError(f"No label mapping found for jaw: {jaw}")
    return _apply_label_mapping(labels, mapping)


def _decim_cache_path(src: Path, target_cells: int) -> Path:
    return DECIM_CACHE / f"{src.stem}.c{target_cells}.vtp"


def _assign_cache_path(src: Path, target_cells: int) -> Path:
    return DECIM_CACHE / f"{src.stem}.c{target_cells}.assign.npy"


def _assign_knn_cache_path(src: Path, target_cells: int) -> Path:
    return DECIM_CACHE / f"{src.stem}.c{target_cells}.assign_knn.npz"


def _ensure_polydata(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    if hasattr(mesh, "cast_to_polydata"):
        return mesh.cast_to_polydata()
    raise TypeError("Mesh is not convertible to PolyData")


def _build_soft_assign_cache(
    full_mesh: pv.PolyData,
    dec_mesh: pv.PolyData,
    assign_path: Path,
    assign_knn_path: Path,
    *,
    knn_k: int = 4,
) -> None:
    try:
        full_mesh = _ensure_polydata(full_mesh)
        dec_mesh = _ensure_polydata(dec_mesh)
        full_centers = full_mesh.cell_centers().points.astype(np.float32)
        dec_centers = dec_mesh.cell_centers().points.astype(np.float32)
        if full_centers.size == 0 or dec_centers.size == 0:
            return
        # 确保具备法向
        if "Normals" not in full_mesh.cell_data:
            full_mesh = full_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        if "Normals" not in dec_mesh.cell_data:
            dec_mesh = dec_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        full_normals = np.asarray(full_mesh.cell_data.get("Normals"), dtype=np.float32, copy=False)
        dec_normals = np.asarray(dec_mesh.cell_data.get("Normals"), dtype=np.float32, copy=False)
        if full_normals.shape[0] != full_centers.shape[0]:
            full_normals = np.zeros_like(full_centers)
        if dec_normals.shape[0] != dec_centers.shape[0]:
            dec_normals = np.zeros_like(dec_centers)
        # 归一化
        def _safe_normalize(arr: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            return np.divide(arr, np.clip(norm, 1e-6, None), out=np.zeros_like(arr), where=norm > 1e-6)
        full_normals = _safe_normalize(full_normals)
        dec_normals = _safe_normalize(dec_normals)

        k_eff = max(1, min(int(knn_k), dec_centers.shape[0]))
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(dec_centers)
        dists, indices = nn.kneighbors(full_centers, return_distance=True)
        if "Area" in dec_mesh.cell_data:
            areas = np.asarray(dec_mesh.cell_data["Area"], dtype=np.float32)
        else:
            dec_with_area = dec_mesh.compute_cell_sizes(length=False, area=True, volume=False)
            areas = np.asarray(dec_with_area.cell_data["Area"], dtype=np.float32)
        areas = np.clip(areas, 1e-8, None)
        weights = 1.0 / np.clip(dists, 1e-8, None)
        weights *= areas[indices]
        # 法向相似度加权（抑制跨面传播）
        selected_normals = dec_normals[indices]  # (N_full, k, 3)
        full_normals_exp = full_normals[:, None, :]  # (N_full, 1, 3)
        cos_sim = np.clip(np.abs((selected_normals * full_normals_exp).sum(axis=2)), 0.0, 1.0)
        angle_thresh = np.cos(np.radians(40.0))
        valid_mask = cos_sim >= angle_thresh
        gamma = 3.0
        angular_weights = np.exp(-gamma * (1.0 - cos_sim))
        angular_weights *= valid_mask.astype(np.float32)
        weights *= angular_weights

        weight_sum = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights,
            weight_sum,
            out=np.full_like(weights, 1.0 / float(k_eff)),
            where=weight_sum > 1e-8,
        ).astype(np.float32, copy=False)
        # 若全部权重被筛空，则退化为均匀
        zero_mask = ~np.isfinite(weights).all(axis=1) | (weights.sum(axis=1) <= 1e-6)
        if np.any(zero_mask):
            weights[zero_mask] = 1.0 / float(k_eff)

        assign_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(assign_path, indices[:, 0].astype(np.int32, copy=False), allow_pickle=False)
        np.savez(
            assign_knn_path,
            indices=indices.astype(np.int32, copy=False),
            weights=weights,
        )
    except Exception:
        pass


def _load_or_build_decimated_mm(raw_path: Path, target_cells: int) -> pv.PolyData:
    cache_path = _decim_cache_path(raw_path, target_cells)
    assign_path = _assign_cache_path(raw_path, target_cells)
    assign_knn_path = _assign_knn_cache_path(raw_path, target_cells)
    if cache_path.exists():
        mesh_mm = _ensure_polydata(pv.read(str(cache_path)))
        if not assign_knn_path.exists():
            try:
                full_mesh = _ensure_polydata(pv.read(str(raw_path))).triangulate()
                _build_soft_assign_cache(full_mesh, mesh_mm, assign_path, assign_knn_path)
            except Exception:
                pass
        return mesh_mm

    mesh_full = _ensure_polydata(pv.read(str(raw_path))).triangulate()
    mesh_mm = mesh_full.copy(deep=True)
    orig_centers = mesh_full.cell_centers().points.astype(np.float32)

    label_info = find_label_array(mesh_full)
    labels = None
    if label_info is not None:
        _, raw_labels = label_info
        labels = remap_segmentation_labels(np.asarray(raw_labels))

    orig_ids = np.arange(mesh_full.n_cells, dtype=np.int64)

    if mesh_full.n_cells > target_cells:
        reduction = 1.0 - (target_cells / float(mesh_full.n_cells))
        decimated = mesh_full.decimate_pro(
            reduction,
            feature_angle=45,
            preserve_topology=True,
        )
        mesh_mm = decimated
        vtk_ids = decimated.cell_data.get("vtkOriginalCellIds")
        if vtk_ids is not None:
            orig_ids = np.asarray(vtk_ids, dtype=np.int64)
        else:
            dec_centers = decimated.cell_centers().points.astype(np.float32)
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nn.fit(orig_centers)
            orig_ids = nn.kneighbors(dec_centers, return_distance=False).reshape(-1).astype(np.int64)
        if labels is not None and orig_ids.max() < labels.shape[0]:
            labels = labels[orig_ids]
        else:
            labels = None

    if labels is not None and labels.shape[0] == mesh_mm.n_cells:
        mesh_mm.cell_data["Label"] = labels.astype(np.int64, copy=False)

    mesh_mm.cell_data["vtkOriginalCellIds"] = orig_ids.astype(np.int64, copy=False)

    try:
        mesh_mm.save(cache_path, binary=True)
    except Exception:
        pass
    _build_soft_assign_cache(mesh_full, mesh_mm, assign_path, assign_knn_path)

    return mesh_mm.copy(deep=True)


def _estimate_diag(mesh: pv.PolyData) -> float:
    bounds = mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
    if bounds is None:
        return 0.0
    extent = np.array(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
        dtype=np.float32,
    )
    return float(np.linalg.norm(extent))


def normalize_mesh_units(mesh: pv.PolyData) -> Tuple[pv.PolyData, float, float, float]:
    """Ensure mesh coordinates are approximately in millimetres."""
    diag_before = _estimate_diag(mesh)
    scale = 1.0
    if 0.0 < diag_before < UNIT_SCALE_THRESHOLD:
        mesh.points *= UNIT_SCALE_FACTOR
        scale = UNIT_SCALE_FACTOR
        diag_after = diag_before * UNIT_SCALE_FACTOR
    else:
        diag_after = diag_before
    return mesh, scale, diag_before, diag_after

# =============================================================================
# Part 0: 通用工具
# =============================================================================


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_subject_id_from_path(file_path: Path) -> Optional[str]:
    match = re.fullmatch(r"(\d+)_([LU])\.vtp", file_path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return str(int(match.group(1)))


def find_label_array(mesh: pv.PolyData) -> Optional[Tuple[str, np.ndarray]]:
    for key in ["Label", "labels"]:
        if key in mesh.cell_data:
            return key, mesh.cell_data[key]
    return None


def extract_features(mesh: pv.PolyData) -> np.ndarray:
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    vertex_coords = mesh.points[faces.flatten()].reshape(mesh.n_cells, 9)
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    cell_normals = mesh.cell_data["Normals"].astype(np.float32, copy=False)
    centers = mesh.cell_centers().points.astype(np.float32, copy=False)
    relative_positions = centers - mesh.center

    # 三角形几何属性
    v0 = mesh.points[faces[:, 0]]
    v1 = mesh.points[faces[:, 1]]
    v2 = mesh.points[faces[:, 2]]
    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2
    edge_lengths = np.stack(
        [
            np.linalg.norm(e01, axis=1),
            np.linalg.norm(e12, axis=1),
            np.linalg.norm(e20, axis=1),
        ],
        axis=1,
    )
    mean_edge = edge_lengths.mean(axis=1, dtype=np.float32)
    cross = np.cross(e01, -e20)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    # 邻面法向差（离散曲率提示）
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for cid, tri in enumerate(faces):
        a, b, c = map(int, tri)
        edges = ((a, b), (b, c), (c, a))
        for u, v in edges:
            key = (u, v) if u <= v else (v, u)
            edge_map.setdefault(key, []).append(cid)
    normal_variation = np.zeros(mesh.n_cells, dtype=np.float32)
    counts = np.zeros(mesh.n_cells, dtype=np.int32)
    for cells in edge_map.values():
        if len(cells) < 2:
            continue
        for i_idx in range(len(cells)):
            for j_idx in range(i_idx + 1, len(cells)):
                ci = cells[i_idx]
                cj = cells[j_idx]
                ni = cell_normals[ci]
                nj = cell_normals[cj]
                dot = np.clip(np.dot(ni, nj), -1.0, 1.0)
                angle = np.arccos(dot)
                normal_variation[ci] += angle
                normal_variation[cj] += angle
                counts[ci] += 1
                counts[cj] += 1
    np.divide(
        normal_variation,
        np.clip(counts, 1, None),
        out=normal_variation,
        where=counts > 0,
    )

    extra_feats = np.stack(
        [
            mean_edge.astype(np.float32, copy=False),
            area.astype(np.float32, copy=False),
            normal_variation.astype(np.float32, copy=False),
        ],
        axis=1,
    )

    return np.hstack([vertex_coords, cell_normals, relative_positions, extra_feats]).astype(np.float32, copy=False)


def random_transform(points: np.ndarray) -> np.ndarray:
    """Apply a random similarity transform for augmentation."""
    theta = np.radians(random.uniform(-10, 10))
    phi = np.radians(random.uniform(-10, 10))
    psi = np.radians(random.uniform(-20, 20))

    def _rot(axis: str, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    rot = _rot("z", psi) @ _rot("y", phi) @ _rot("x", theta)
    scale = random.uniform(0.95, 1.05)
    jitter = np.random.normal(0.0, 0.005, size=points.shape).astype(np.float32, copy=False)
    return (points @ rot.T) * scale + jitter


def light_jitter(points: np.ndarray, sigma: float = 0.002) -> np.ndarray:
    """Apply light Gaussian jitter without rotation/scaling."""
    jitter = np.random.normal(0.0, sigma, size=points.shape).astype(np.float32, copy=False)
    return points + jitter


# =============================================================================
# Part 1: 数据划分工具（与 Module0 一致，保留以兼容旧流程）
# =============================================================================

def validate_and_split_by_subject(
    root_dir: Path,
    split_path: Path,
    test_size: float,
    random_state: int,
    *,
    force: bool = False,
) -> Tuple[List[str], List[str]]:
    if split_path.exists() and not force:
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        return split_data.get("train", []), split_data.get("val", [])

    subjects: Dict[str, List[Path]] = defaultdict(list)
    for f_path in root_dir.glob("*.vtp"):
        subject_id = get_subject_id_from_path(f_path)
        if not subject_id:
            continue
        subjects[subject_id].append(f_path)

    subject_ids = sorted(subjects.keys())
    if not subject_ids:
        raise ValueError("No valid subjects found in the data directory.")

    from sklearn.model_selection import train_test_split

    train_ids, val_ids = train_test_split(subject_ids, test_size=test_size, random_state=random_state)
    train_files = [str(f) for sid in train_ids for f in subjects[sid]]
    val_files = [str(f) for sid in val_ids for f in subjects[sid]]

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_files, "val": val_files}, f, indent=4)

    return train_files, val_files


# =============================================================================
# Part 2: 配置与辅助加载函数
# =============================================================================


def load_stats(stats_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not stats_path.exists():
        raise FileNotFoundError(f"stats file not found: {stats_path}")
    with np.load(stats_path) as data:
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def load_arch_frames(path: Optional[Path]) -> Dict[str, torch.Tensor]:
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frames: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (4, 4):
            arr = arr[:3, :3]
        if arr.shape != (3, 3):
            continue
        frames[key] = torch.from_numpy(arr)
    return frames


def segmentation_collate(
    batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
):
    xs: List[torch.Tensor] = []
    poss: List[torch.Tensor] = []
    boundaries: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for (x, pos, boundary), y in batch:
        xs.append(x)
        poss.append(pos)
        boundaries.append(boundary)
        ys.append(y)
    x = torch.stack(xs, dim=0)
    pos = torch.stack(poss, dim=0)
    boundary = torch.stack(boundaries, dim=0)
    y = torch.stack(ys, dim=0)
    return (x, pos, boundary), y


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


class DataConfig:
    split_path: Path = SEG_ROOT / "module0" / "dataset_split.json"
    stats_path: Path = SEG_ROOT / "module0" / "stats.npz"
    arch_frames_path: Optional[Path] = None
    batch_size: int = 2
    num_workers: int = field(default_factory=_default_workers)  # type: ignore[misc]
    persistent_workers: bool = True
    target_cells: int = 10000
    sample_cells: int = 6000
    augment: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    label_mode: str = "single_arch_16"  # "single_arch_16" | "full_fdi"
    gingiva_src_label: int = 0
    gingiva_class_id: int = 15
    keep_void_zero: bool = True
    seed: int = 42


# =============================================================================
# Part 2.5: 数据准备工具
# =============================================================================


def compute_feature_stats(file_paths: Sequence[str], target_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    if not file_paths:
        raise ValueError("Training file list is empty; cannot compute statistics.")

    sum_vec: Optional[np.ndarray] = None
    sum_sq_vec: Optional[np.ndarray] = None
    total_faces = 0

    for idx, path_str in enumerate(file_paths, 1):
        file_path = Path(path_str)
        try:
            mesh = pv.read(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read mesh: {file_path}") from exc

        mesh.points -= mesh.center
        mesh, _, _, _ = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()

        if mesh.n_cells > target_cells:
            reduction = 1.0 - (target_cells / float(mesh.n_cells))
            mesh = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)

        features = extract_features(mesh).astype(np.float64)
        if sum_vec is None or sum_sq_vec is None:
            feature_dim = features.shape[1]
            sum_vec = np.zeros(feature_dim, dtype=np.float64)
            sum_sq_vec = np.zeros(feature_dim, dtype=np.float64)

        sum_vec += features.sum(axis=0)
        sum_sq_vec += (features ** 2).sum(axis=0)
        total_faces += features.shape[0]

        if idx % 20 == 0 or idx == len(file_paths):
            print(f"    processed {idx}/{len(file_paths)} files", flush=True)

    if total_faces == 0:
        raise ValueError("No faces found across training meshes; cannot compute statistics.")
    if sum_vec is None or sum_sq_vec is None:
        raise ValueError("Failed to accumulate feature statistics.")

    mean = sum_vec / total_faces
    variance = np.maximum(sum_sq_vec / total_faces - mean ** 2, 1e-6)
    std = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


def load_split_lists(split_path: Path) -> Tuple[List[str], List[str]]:
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_files = [str(Path(p)) for p in split.get("train", [])]
    val_files = [str(Path(p)) for p in split.get("val", [])]
    return train_files, val_files


def compute_label_histogram(
    file_paths: Sequence[str],
    *,
    label_mode: str = "full_fdi",
    gingiva_src_label: int = 0,
    gingiva_class_id: int = 15,
    keep_void_zero: bool = True,
) -> np.ndarray:
    if label_mode == "single_arch_16":
        counts = np.zeros(SINGLE_ARCH_NUM_CLASSES, dtype=np.int64)
        single_arch_maps = _build_single_arch_label_maps(
            gingiva_src_label,
            gingiva_class_id,
            keep_void_zero,
        )
    else:
        counts = np.zeros(SEG_NUM_CLASSES, dtype=np.int64)
        single_arch_maps = None
    for path_str in file_paths:
        file_path = Path(path_str)
        try:
            mesh = pv.read(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read mesh: {file_path}") from exc

        result = find_label_array(mesh)
        if result is None:
            continue
        _, raw_labels = result
        raw_arr = np.asarray(raw_labels, dtype=np.int64)
        if label_mode == "single_arch_16":
            if single_arch_maps is None:
                raise RuntimeError("Single-arch label maps not initialised")
            remapped = remap_labels_single_arch(raw_arr, file_path, single_arch_maps)
        else:
            remapped = remap_segmentation_labels(raw_arr)
        uniq, freq = np.unique(remapped, return_counts=True)
        counts[uniq] += freq
    return counts


def prepare_module0(
    root_dir: Path,
    config: DataConfig,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    force: bool = False,
    skip_stats: bool = False,
) -> None:
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    root_dir = root_dir.resolve()
    print(f"[Module0] 数据根目录: {root_dir}")

    train_files, val_files = validate_and_split_by_subject(
        root_dir,
        config.split_path,
        test_size,
        random_state,
        force=force,
    )
    print(f"[Module0] 训练文件数量: {len(train_files)} | 验证文件数量: {len(val_files)}")
    print(f"[Module0] 划分保存至: {config.split_path.resolve()}")

    if skip_stats:
        print(f"[Module0] 跳过统计计算 (--skip-stats)")
        return

    if config.stats_path.exists() and not force:
        print(f"[Module0] 统计文件已存在: {config.stats_path.resolve()} (使用 --force 重新计算)")
        return

    if not train_files:
        raise ValueError("No training files available to compute statistics.")

    print("[Module0] 正在计算训练集特征统计 (mean/std)...")
    mean, std = compute_feature_stats(train_files, config.target_cells)
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.stats_path, mean=mean, std=std)
    print(f"[Module0] 统计文件已写入: {config.stats_path.resolve()}")


# =============================================================================
# Part 3: 数据集类
# =============================================================================


class SegmentationDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        arch_frames: Dict[str, torch.Tensor],
        target_cells: int,
        sample_cells: int,
        augment: bool,
        *,
        label_mode: str,
        gingiva_src_label: int,
        gingiva_class_id: int,
        keep_void_zero: bool,
    ):
        self.file_paths = file_paths
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.arch_frames = arch_frames
        self.target_cells = target_cells
        self.sample_cells = sample_cells
        self._augment_stage = "full" if augment else "off"
        self._unit_warned = False
        self.label_mode = label_mode
        self.gingiva_src_label = gingiva_src_label
        self.gingiva_class_id = gingiva_class_id
        self.keep_void_zero = keep_void_zero
        self._single_arch_maps: Optional[Dict[str, Dict[int, int]]] = None
        self.min_samples_per_class = 160  # 小类保底采样数量（侧重边界牙）
        self.boundary_focus_fraction = 0.4  # 采样时优先保留的边界占比
        self._mirror_pairs_single_arch: Dict[str, List[Tuple[int, int]]] = {}
        self._mirror_pairs_full: List[Tuple[int, int]] = []
        if self.label_mode == "single_arch_16":
            self._single_arch_maps = _build_single_arch_label_maps(
                self.gingiva_src_label,
                self.gingiva_class_id,
                self.keep_void_zero,
            )
            for jaw, raw_pairs in ARCH_MIRROR_PAIRS.items():
                mapping = self._single_arch_maps[jaw]
                cls_pairs: List[Tuple[int, int]] = []
                for left_raw, right_raw in raw_pairs:
                    left_cls = int(mapping.get(left_raw, 0))
                    right_cls = int(mapping.get(right_raw, 0))
                    if left_cls <= 0 or right_cls <= 0:
                        continue
                    cls_pairs.append((left_cls, right_cls))
                self._mirror_pairs_single_arch[jaw] = cls_pairs
        elif self.label_mode == "full_fdi":
            for left_raw, right_raw in FULL_FDI_MIRROR_PAIRS:
                left_cls = int(LABEL_REMAP.get(left_raw, 0))
                right_cls = int(LABEL_REMAP.get(right_raw, 0))
                if left_cls <= 0 or right_cls <= 0:
                    continue
                self._mirror_pairs_full.append((left_cls, right_cls))

    @property
    def augment(self) -> bool:
        return self._augment_stage != "off"

    @property
    def augment_stage(self) -> str:
        return self._augment_stage

    def set_augment_stage(self, stage: str) -> bool:
        stage_lc = stage.lower()
        if stage_lc not in {"off", "light", "full"}:
            raise ValueError(f"Unsupported augment stage: {stage}")
        if stage_lc == self._augment_stage:
            return False
        self._augment_stage = stage_lc
        return True

    def __len__(self) -> int:
        return len(self.file_paths)

    def _lookup_arch_frame(self, stem: str) -> Optional[torch.Tensor]:
        if stem in self.arch_frames:
            return self.arch_frames[stem]
        base = stem.split("_")[0]
        return self.arch_frames.get(base)

    def _remap_labels(self, labels: np.ndarray, file_path: Path) -> np.ndarray:
        if self.label_mode == "single_arch_16":
            if self._single_arch_maps is None:
                raise RuntimeError("Single-arch label maps not initialised.")
            return remap_labels_single_arch(labels, file_path, self._single_arch_maps)
        if self.label_mode == "full_fdi":
            return remap_segmentation_labels(labels)
        raise ValueError(f"Unsupported label_mode: {self.label_mode}")

    def _apply_mirror_label_swap(self, labels: np.ndarray, file_path: Path) -> np.ndarray:
        if labels.size == 0:
            return labels
        if self.label_mode == "single_arch_16":
            jaw = _infer_jaw_from_stem(file_path.stem)
            pairs = self._mirror_pairs_single_arch.get(jaw, [])
        elif self.label_mode == "full_fdi":
            pairs = self._mirror_pairs_full
        else:
            return labels
        if not pairs:
            return labels

        swapped = labels
        updated = False
        for left_cls, right_cls in pairs:
            if left_cls == right_cls:
                continue
            left_mask = labels == left_cls
            right_mask = labels == right_cls
            if left_mask.any():
                if not updated:
                    swapped = labels.copy()
                    updated = True
                swapped[left_mask] = right_cls
            if right_mask.any():
                if not updated:
                    swapped = labels.copy()
                    updated = True
                swapped[right_mask] = left_cls
        return swapped if updated else labels

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        file_path = Path(self.file_paths[idx])
        mesh = pv.read(str(file_path))
        mesh.points -= mesh.center
        frame_tensor = self._lookup_arch_frame(file_path.stem)

        result = find_label_array(mesh)
        if result is None:
            raise RuntimeError(f"File {file_path} is missing a label array.")
        _, raw_labels = result
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.size > 0:
            labels = self._remap_labels(labels, file_path)

        mesh, scale_factor, diag_before, diag_after = normalize_mesh_units(mesh)
        if scale_factor != 1.0 and not self._unit_warned:
            print(
                f"[SegmentationDataset] Detected small bounding box (diag={diag_before:.4f}); "
                f"scaling {file_path.name} by {scale_factor:g} to mm units.",
                flush=True,
            )
            self._unit_warned = True

        mesh = mesh.triangulate()

        if self._augment_stage != "off":
            if random.random() > 0.5:
                mesh.points[:, 0] *= -1
                mesh = mesh.flip_faces()
                labels = self._apply_mirror_label_swap(labels, file_path)
            if self._augment_stage == "light":
                mesh.points = light_jitter(mesh.points)
            else:
                mesh.points = random_transform(mesh.points)

        if mesh.n_cells > self.target_cells:
            reduction = 1.0 - (self.target_cells / mesh.n_cells)
            decimated = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)
            original_ids = decimated.cell_data.get("vtkOriginalCellIds")
            if original_ids is not None and len(original_ids) == decimated.n_cells:
                mesh = decimated
                labels = labels[original_ids]

        boundary_flags = self._compute_boundary_flags(mesh, labels)

        if frame_tensor is not None:
            frame_np = frame_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            rotated_points = mesh.points.astype(np.float32, copy=False) @ frame_np.T
            mesh.points = rotated_points

        features = extract_features(mesh).astype(np.float32)

        pos_raw = mesh.cell_centers().points.astype(np.float32)
        scale_pos = diag_after if diag_after > 1e-6 else 1.0
        pos_raw = pos_raw / scale_pos

        if features.shape[0] > self.sample_cells:
            total_cells = features.shape[0]
            required_indices: List[int] = []
            for cls in np.unique(labels):
                if cls == 0:
                    continue
                cls_indices = np.where(labels == cls)[0]
                quota = min(self.min_samples_per_class, self.sample_cells)
                if cls_indices.size == 0 or quota == 0:
                    continue
                replace = cls_indices.size < quota
                chosen = np.random.choice(cls_indices, size=quota, replace=replace)
                required_indices.extend(chosen.tolist())

            boundary_indices = np.where(boundary_flags)[0]
            if boundary_indices.size > 0:
                boundary_quota = int(self.sample_cells * self.boundary_focus_fraction)
                boundary_take = min(boundary_indices.size, boundary_quota)
                if boundary_take > 0:
                    replace_boundary = boundary_indices.size < boundary_take
                    boundary_choice = np.random.choice(
                        boundary_indices,
                        size=boundary_take,
                        replace=replace_boundary,
                    )
                    required_indices.extend(boundary_choice.tolist())

            if required_indices:
                required_array = np.unique(np.asarray(required_indices, dtype=np.int64))
            else:
                required_array = np.empty(0, dtype=np.int64)

            slots_left = self.sample_cells - required_array.size
            if slots_left < 0:
                indices = np.random.choice(required_array, size=self.sample_cells, replace=False)
            else:
                mask = np.ones(total_cells, dtype=bool)
                if required_array.size > 0:
                    mask[required_array] = False
                pool = np.nonzero(mask)[0]
                extra_take = min(slots_left, pool.size)
                extra = np.random.choice(pool, size=extra_take, replace=False) if extra_take > 0 else np.empty(0, dtype=np.int64)
                indices = np.concatenate([required_array, extra]) if required_array.size > 0 else extra
                if indices.size < self.sample_cells:
                    deficit = self.sample_cells - indices.size
                    fallback = np.random.choice(total_cells, size=deficit, replace=True)
                    indices = np.concatenate([indices, fallback])
            indices = indices[: self.sample_cells]
            np.random.shuffle(indices)
            features = features[indices]
            pos_raw = pos_raw[indices]
            labels = labels[indices]
            boundary_flags = boundary_flags[indices]
        else:
            indices = np.arange(features.shape[0], dtype=np.int64)

        features_tensor = torch.from_numpy(features)
        features_tensor = (features_tensor - self.mean) / self.std
        features_tensor = features_tensor.transpose(0, 1).contiguous()  # (15, N)

        pos_tensor = torch.from_numpy(pos_raw).transpose(0, 1).contiguous()  # (3, N)

        labels_tensor = torch.from_numpy(labels.astype(np.int64))
        boundary_tensor = torch.from_numpy(boundary_flags.astype(np.float32, copy=False))

        return (features_tensor, pos_tensor, boundary_tensor), labels_tensor

    @staticmethod
    def _compute_boundary_flags(mesh: pv.PolyData, labels: np.ndarray) -> np.ndarray:
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        n_cells = faces.shape[0]
        boundary = np.zeros(n_cells, dtype=bool)
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for cid, tri in enumerate(faces):
            v0, v1, v2 = map(int, tri)
            edges = ((v0, v1), (v1, v2), (v2, v0))
            for a, b in edges:
                key = (a, b) if a <= b else (b, a)
                edge_map.setdefault(key, []).append(cid)
        for cells in edge_map.values():
            if len(cells) == 1:
                boundary[cells[0]] = True
                continue
            base_lbl = labels[cells[0]]
            for cid in cells[1:]:
                if labels[cid] != base_lbl:
                    boundary[cells[0]] = True
                    boundary[cid] = True
        return boundary


def get_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_files, val_files = load_split_lists(config.split_path)

    mean, std = load_stats(config.stats_path)
    arch_frames = load_arch_frames(config.arch_frames_path)

    train_dataset = SegmentationDataset(
        train_files,
        mean,
        std,
        arch_frames,
        target_cells=config.target_cells,
        sample_cells=config.sample_cells,
        augment=config.augment,
        label_mode=config.label_mode,
        gingiva_src_label=config.gingiva_src_label,
        gingiva_class_id=config.gingiva_class_id,
        keep_void_zero=config.keep_void_zero,
    )

    val_dataset = SegmentationDataset(
        val_files,
        mean,
        std,
        arch_frames,
        target_cells=config.target_cells,
        sample_cells=config.sample_cells,
        augment=False,
        label_mode=config.label_mode,
        gingiva_src_label=config.gingiva_src_label,
        gingiva_class_id=config.gingiva_class_id,
        keep_void_zero=config.keep_void_zero,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        collate_fn=segmentation_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        collate_fn=segmentation_collate,
    )

    return train_loader, val_loader


# =============================================================================
# Part 4: 样本检查工具
# =============================================================================


def inspect_sample(root_dir: str, sample_id: str | int, jaw: str = "L") -> None:
    sample_id_raw = str(sample_id).strip()
    candidates: List[str] = []
    if sample_id_raw:
        candidates.append(f"{sample_id_raw}_{jaw}.vtp")

    try:
        sample_id_int = int(sample_id_raw)
    except ValueError:
        print(f"Error: invalid sample id '{sample_id}'. It must be numeric.")
        return

    padded = f"{sample_id_int:03d}_{jaw}.vtp"
    if padded not in candidates:
        candidates.append(padded)

    root_path = Path(root_dir)
    file_path: Optional[Path] = None
    file_name: Optional[str] = None
    for name in candidates:
        candidate_path = root_path / name
        if candidate_path.exists():
            file_path = candidate_path
            file_name = name
            break

    if file_path is None:
        print("Error: no matching file found. Tried candidates:")
        for name in candidates:
            print(f"  - {root_path / name}")
        try:
            existing_files = sorted(root_path.glob(f"*_{jaw}.vtp"))
            if existing_files:
                print(f"\nAvailable {jaw} files in directory:")
                for f in existing_files[:10]:
                    print(f"  - {f.name}")
        except Exception:
            pass
        return

    print(f"\n--- 正在检查样本: {file_name} ---")
    try:
        mesh = pv.read(file_path)
        result = find_label_array(mesh)
        if result is None:
            print(f"警告: 未找到标签数组。可用的cell_data键: {list(mesh.cell_data.keys())}")
            return

        label_key, labels = result
        unique_labels = np.unique(labels)
        print(f"成功加载网格，找到标签数组: '{label_key}'")
        print(f"包含的唯一标签值 (共 {len(unique_labels)} 个): {sorted(unique_labels.tolist())}")

        mesh.set_active_scalars(label_key)

        output_dir = SEG_ROOT / "module0" / "inspections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_vtp_path = output_dir / f"inspected_{file_name}"
        mesh.save(output_vtp_path, binary=True)

        print("\n[行动号召] 数据验证已准备就绪:")
        print(f"1. 已生成带原始标签的VTP文件:\n   -> {output_vtp_path.resolve()}")
        print("2. 请使用 ParaView 或 CloudCompare 打开此文件进行最终确认。")

    except Exception as exc:
        import traceback
        print(f"处理文件时发生错误: {exc}")
        traceback.print_exc()



def main() -> None:
    parser = argparse.ArgumentParser("Module0 dataset preparation")
    parser.add_argument("--root", type=str, default="datasets/segmentation_dataset", help="Path to the raw VTP directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject split")
    parser.add_argument("--force", action="store_true", help="Regenerate split/statistics even if they exist")
    parser.add_argument("--skip-stats", action="store_true", help="Skip statistics calculation (split only)")
    parser.add_argument("--inspect", type=str, help="Inspect a single case id, e.g. 136")
    parser.add_argument("--jaw", type=str, default="L", help="Jaw flag when inspecting (L or U)")
    parser.add_argument("--target-cells", type=int, help="Override DataConfig.target_cells")
    parser.add_argument("--sample-cells", type=int, help="Override DataConfig.sample_cells")
    args = parser.parse_args()

    if args.inspect is not None:
        inspect_sample(args.root, args.inspect, args.jaw.upper())
        return

    config_kwargs = {}
    if args.target_cells is not None:
        config_kwargs["target_cells"] = args.target_cells
    if args.sample_cells is not None:
        config_kwargs["sample_cells"] = args.sample_cells

    config = DataConfig(**config_kwargs)
    prepare_module0(
        root_dir=Path(args.root),
        config=config,
        test_size=args.test_size,
        random_state=args.seed,
        force=args.force,
        skip_stats=args.skip_stats,
    )
    print("[Module0] dataset preparation finished.")


if __name__ == "__main__":
    main()
