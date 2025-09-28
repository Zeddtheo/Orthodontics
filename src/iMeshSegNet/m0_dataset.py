# module0_dataset.py
# 数据集与数据加载器工具，用于 iMeshSegNet 训练阶段。

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader, Dataset


SEG_ROOT = Path("outputs/segmentation")

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

# 供各模块共享的类别数
SEG_NUM_CLASSES = 1 + len(FDI_LABELS)


def _remap_value(v: int) -> int:
    return LABEL_REMAP.get(int(v), 0)


_vectorized_remap = np.vectorize(_remap_value, otypes=[np.int64])


def remap_segmentation_labels(arr: np.ndarray) -> np.ndarray:
    """Map raw FDI labels to contiguous indices starting from 0."""
    if arr.size == 0:
        return arr.astype(np.int64, copy=False)
    return _vectorized_remap(arr)


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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    cell_normals = mesh.cell_data["Normals"]
    relative_positions = mesh.cell_centers().points - mesh.center
    return np.hstack([vertex_coords, cell_normals, relative_positions])


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
    jitter = np.random.normal(0.0, 0.005, size=points.shape)
    return (points @ rot.T) * scale + jitter


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


def segmentation_collate(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]):
    batch_data, ys = zip(*batch)
    xs, poss = zip(*batch_data)
    x = torch.stack(xs, dim=0)
    pos = torch.stack(poss, dim=0)
    y = torch.stack(ys, dim=0)
    return (x, pos), y


@dataclass
class DataConfig:
    split_path: Path = SEG_ROOT / "module0" / "dataset_split.json"
    stats_path: Path = SEG_ROOT / "module0" / "stats.npz"
    arch_frames_path: Optional[Path] = None
    batch_size: int = 2
    num_workers: int = 0
    persistent_workers: bool = True
    target_cells: int = 25000
    sample_cells: int = 8192
    augment: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True


# =============================================================================
# Part 2.5: 数据准备工具
# =============================================================================


def compute_feature_stats(file_paths: Sequence[str], target_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    if not file_paths:
        raise ValueError("Training file list is empty; cannot compute statistics.")

    sum_vec = np.zeros(15, dtype=np.float64)
    sum_sq_vec = np.zeros(15, dtype=np.float64)
    total_faces = 0

    for idx, path_str in enumerate(file_paths, 1):
        file_path = Path(path_str)
        try:
            mesh = pv.read(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read mesh: {file_path}") from exc

        if not isinstance(mesh, pv.PolyData):
            if hasattr(mesh, 'cast_to_polydata'):
                mesh = mesh.cast_to_polydata()
            else:
                continue  # Skip non-PolyData files
                
        mesh.points -= mesh.center
        mesh, _, _, _ = normalize_mesh_units(mesh)
        mesh = mesh.triangulate()

        if mesh.n_cells > target_cells:
            reduction = 1.0 - (target_cells / float(mesh.n_cells))
            mesh = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)

        features = extract_features(mesh).astype(np.float64)

        sum_vec += features.sum(axis=0)
        sum_sq_vec += (features ** 2).sum(axis=0)
        total_faces += features.shape[0]

        if idx % 20 == 0 or idx == len(file_paths):
            print(f"    processed {idx}/{len(file_paths)} files", flush=True)

    if total_faces == 0:
        raise ValueError("No faces found across training meshes; cannot compute statistics.")

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


def compute_label_histogram(file_paths: Sequence[str]) -> np.ndarray:
    counts = np.zeros(SEG_NUM_CLASSES, dtype=np.int64)
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
        remapped = remap_segmentation_labels(np.asarray(raw_labels))
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
    ):
        self.file_paths = file_paths
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.arch_frames = arch_frames
        self.target_cells = target_cells
        self.sample_cells = sample_cells
        self.augment = augment
        self._unit_warned = False

    def __len__(self) -> int:
        return len(self.file_paths)

    def _lookup_arch_frame(self, stem: str) -> Optional[torch.Tensor]:
        if stem in self.arch_frames:
            return self.arch_frames[stem]
        base = stem.split("_")[0]
        return self.arch_frames.get(base)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        file_path = Path(self.file_paths[idx])
        mesh = pv.read(str(file_path))
        
        if not isinstance(mesh, pv.PolyData):
            if hasattr(mesh, 'cast_to_polydata'):
                mesh = mesh.cast_to_polydata()
            else:
                raise RuntimeError(f"File {file_path} is not a valid PolyData mesh.")
                
        mesh.points -= mesh.center

        result = find_label_array(mesh)
        if result is None:
            raise RuntimeError(f"File {file_path} is missing a label array.")
        _, raw_labels = result
        labels = remap_segmentation_labels(np.asarray(raw_labels))

        mesh, scale_factor, diag_before, diag_after = normalize_mesh_units(mesh)
        if scale_factor != 1.0 and not self._unit_warned:
            print(
                f"[SegmentationDataset] Detected small bounding box (diag={diag_before:.4f}); "
                f"scaling {file_path.name} by {scale_factor:g} to mm units.",
                flush=True,
            )
            self._unit_warned = True

        mesh = mesh.triangulate()

        if self.augment:
            if random.random() > 0.5:
                points = np.array(mesh.points)
                points[:, 0] *= -1
                mesh.points = points
                mesh = mesh.flip_faces()
            mesh.points = random_transform(np.array(mesh.points))

        if mesh.n_cells > self.target_cells:
            reduction = 1.0 - (self.target_cells / mesh.n_cells)
            decimated = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)
            original_ids = decimated.cell_data.get("vtkOriginalCellIds")
            if original_ids is not None and len(original_ids) == decimated.n_cells:
                mesh = decimated
                labels = labels[original_ids]

        features = extract_features(mesh).astype(np.float32)

        pos_raw = mesh.cell_centers().points.astype(np.float32)
        scale_pos = diag_after if diag_after > 1e-6 else 1.0
        pos_raw = pos_raw / scale_pos

        if features.shape[0] > self.sample_cells:
            indices = np.random.permutation(features.shape[0])[: self.sample_cells]
            features = features[indices]
            pos_raw = pos_raw[indices]
            labels = labels[indices]

        features_tensor = torch.from_numpy(features)
        features_tensor = (features_tensor - self.mean) / self.std
        # 修复: 转置为 (15, N) 格式以匹配训练脚本期望
        features_tensor = features_tensor.transpose(0, 1).contiguous()  # (15, N)
        
        pos_tensor = torch.from_numpy(pos_raw)
        frame = self._lookup_arch_frame(file_path.stem)
        if frame is not None:
            pos_tensor = (frame @ pos_tensor.T).T
        # 修复: 转置为 (3, N) 格式以匹配训练脚本期望
        pos_tensor = pos_tensor.transpose(0, 1).contiguous()  # (3, N)

        labels_tensor = torch.from_numpy(labels.astype(np.int64))

        # 修复: 返回训练脚本期望的格式 (x, pos), y
        return (features_tensor, pos_tensor), labels_tensor


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
    )

    val_dataset = SegmentationDataset(
        val_files,
        mean,
        std,
        arch_frames,
        target_cells=config.target_cells,
        sample_cells=config.sample_cells,
        augment=False,
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
        if not isinstance(mesh, pv.PolyData):
            if hasattr(mesh, 'cast_to_polydata'):
                mesh = mesh.cast_to_polydata()
            else:
                print(f"警告: 文件 {file_name} 不是有效的PolyData格式")
                return
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
