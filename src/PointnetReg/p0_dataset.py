from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tooth_groups import TOOTH_GROUPS, get_group_teeth, get_tooth_group, validate_tooth_id

Tensor = torch.Tensor
Array = np.ndarray


def _to_tensor(data: Array, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.from_numpy(data).to(dtype)


def _rotz(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _apply_rigid(x_cn: Array, R: Array, t: Array) -> Array:
    xyz = R @ x_cn[0:3] + t.reshape(3, 1)
    blocks = [xyz]
    if x_cn.shape[0] >= 6:
        blocks.append(R @ x_cn[3:6])
        if x_cn.shape[0] > 6:
            blocks.append(x_cn[6:])
    elif x_cn.shape[0] > 3:
        blocks.append(x_cn[3:])
    return np.concatenate(blocks, axis=0)


def _apply_mirror(x_cn: Array) -> Array:
    """应用左右镜像变换 (x -> -x)"""
    x_mirrored = x_cn.copy()
    x_mirrored[0] *= -1  # x坐标取反
    if x_cn.shape[0] >= 6:  # 如果有法向量，也需要镜像
        x_mirrored[3] *= -1  # 法向量x分量取反
    return x_mirrored


def _extract_tooth_id_from_path(path: Path) -> Optional[str]:
    """从文件路径中提取牙位ID"""
    # 匹配模式：数字_字母_牙位.npz，如 001_L_t31.npz
    pattern = r'.*_([tT]\d{2})\.npz$'
    match = re.search(pattern, path.name)
    if match:
        return match.group(1).lower()
    return None


def _load_stats(stats_path: str) -> Tuple[Array, Array]:
    """加载统计信息文件"""
    try:
        stats = np.load(stats_path)
        mean = stats.get("mean", np.zeros(1))
        std = stats.get("std", np.ones(1))
        return mean.astype(np.float32), std.astype(np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)


@dataclass
class DatasetConfig:
    root: Union[str, Path]
    file_patterns: Sequence[str] = ("*.npz",)
    features: str = "all"
    select_landmarks: str = "active"
    augment: bool = False
    rotz_deg: float = 15.0
    trans_mm: float = 0.5
    ensure_constant_L: bool = True
    dtype: torch.dtype = torch.float32
    
    # ✅ 新增：分组/牙位筛选
    group: Optional[str] = None  # 牙位组名，与file_patterns互斥
    tooth_ids: Optional[List[str]] = None  # 显式牙位列表，优先生效
    
    # ✅ 新增：牙弓对齐
    arch_align: bool = True  # 是否进行牙弓对齐
    arch_keys: Tuple[str, str] = ("arch_R", "arch_t")  # 对齐矩阵和平移键名
    
    # ✅ 新增：左右镜像增强
    mirror_prob: float = 0.5  # 镜像增强概率
    
    # ✨ 可选：特征标准化
    zscore: bool = False  # 是否进行Z-score标准化
    stats_path: Optional[str] = None  # 统计信息文件路径


class P0PointNetRegDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)
        
        # ✅ 支持分组/牙位筛选
        files: List[Path] = []
        
        if cfg.tooth_ids is not None:
            # 显式牙位列表优先生效
            for tooth_id in cfg.tooth_ids:
                if not validate_tooth_id(tooth_id):
                    raise ValueError(f"Invalid tooth_id: {tooth_id}")
                patterns = [f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"]
                for pat in patterns:
                    files.extend(sorted(root.glob(pat)))
        elif cfg.group is not None:
            # 按组内牙位聚合样本
            group_teeth = get_group_teeth(cfg.group)
            if not group_teeth:
                raise ValueError(f"Invalid group name: {cfg.group}")
            for tooth_id in group_teeth:
                patterns = [f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"]
                for pat in patterns:
                    files.extend(sorted(root.glob(pat)))
        else:
            # 使用传统的file_patterns方式
            for pat in cfg.file_patterns:
                files.extend(sorted(root.glob(pat)))
        
        if not files:
            search_desc = f"group={cfg.group}" if cfg.group else f"tooth_ids={cfg.tooth_ids}" if cfg.tooth_ids else f"file_patterns={cfg.file_patterns}"
            raise FileNotFoundError(f"no npz files found under {root} with {search_desc}")
        
        self.files = files
        
        # 加载统计信息（如果需要）
        self.stats_mean = None
        self.stats_std = None
        if cfg.zscore and cfg.stats_path:
            self.stats_mean, self.stats_std = _load_stats(cfg.stats_path)

        x0, y0, m0, _ = self._peek(self.files[0])
        self.C = x0.shape[0]
        self.N = x0.shape[1]
        self.L_all = y0.shape[0]
        self._active_L = int(m0.sum()) if m0 is not None else self.L_all
        self._use_channels = slice(0, 3 if cfg.features == "xyz" else self.C)
        self.L = self._active_L if cfg.select_landmarks == "active" else self.L_all

        if cfg.ensure_constant_L:
            target = self._active_L if cfg.select_landmarks == "active" else self.L_all
            for path in self.files:
                _, y, m, _ = self._peek(path)
                current = int(m.sum()) if (cfg.select_landmarks == "active" and m is not None) else y.shape[0]
                if current != target:
                    group_info = f" in group '{cfg.group}'" if cfg.group else ""
                    raise ValueError(
                        f"Inconsistent landmark count in {path.name}: expected {target}, got {current}. "
                        f"This indicates annotation template inconsistency{group_info}. "
                        f"Please check landmark definitions for this tooth type."
                    )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=True) as data:
            x = data["x"]
            y = data["y"]
            mask = data.get("loss_mask", data.get("mask"))
            meta = data.get("meta", {})

        if x.ndim != 2:
            raise ValueError(f"bad x shape: {x.shape}")
        if x.shape[0] == self.N:
            x = x.T
        x = x[self._use_channels]

        if self.cfg.select_landmarks == "active" and mask is not None:
            y = y[mask.astype(bool)]
            mask_out = None
        else:
            mask_out = mask

        # ✅ 牙弓对齐（arch frame alignment）
        if self.cfg.arch_align and isinstance(meta, dict):
            arch_R_key, arch_t_key = self.cfg.arch_keys
            if arch_R_key in meta and arch_t_key in meta:
                try:
                    arch_R = np.asarray(meta[arch_R_key], dtype=np.float32).reshape(3, 3)
                    arch_t = np.asarray(meta[arch_t_key], dtype=np.float32).reshape(3)
                    x = _apply_rigid(x, arch_R, arch_t)
                except Exception:
                    # 如果对齐失败，继续使用原始数据
                    pass

        # ✅ 数据增强（在对齐后进行）
        if self.cfg.augment:
            # 旋转和平移增强
            theta = math.radians(np.random.uniform(-self.cfg.rotz_deg, self.cfg.rotz_deg))
            R = _rotz(theta)
            t = np.random.uniform(-self.cfg.trans_mm, self.cfg.trans_mm, size=3).astype(np.float32)
            x = _apply_rigid(x, R, t)
            
            # ✅ 左右镜像增强
            if np.random.random() < self.cfg.mirror_prob:
                x = _apply_mirror(x)

        # ✨ 特征标准化
        if self.cfg.zscore and self.stats_mean is not None and self.stats_std is not None:
            # 按通道标准化
            C = x.shape[0]
            if len(self.stats_mean) == C and len(self.stats_std) == C:
                x = (x - self.stats_mean.reshape(-1, 1)) / (self.stats_std.reshape(-1, 1) + 1e-8)
            else:
                # 简单的xyz标准化
                if C >= 3:
                    xyz = x[:3]
                    xyz_mean = xyz.mean(axis=1, keepdims=True)
                    xyz_std = xyz.std(axis=1, keepdims=True) + 1e-8
                    x[:3] = (xyz - xyz_mean) / xyz_std

        sample = {"x": _to_tensor(x, self.cfg.dtype), "y": _to_tensor(y, self.cfg.dtype)}
        if mask_out is not None:
            sample["mask"] = _to_tensor(mask_out.astype(np.float32), self.cfg.dtype)
        
        # 🧩 元信息透传（补充tooth_id/group字段）
        meta_dict = {"path": str(path)}
        if isinstance(meta, dict):
            meta_dict.update(meta)
        
        # 提取牙位ID
        tooth_id = _extract_tooth_id_from_path(path)
        if tooth_id:
            meta_dict["tooth_id"] = tooth_id
            meta_dict["group"] = get_tooth_group(tooth_id)
        elif self.cfg.group:
            meta_dict["group"] = self.cfg.group

        sample["meta"] = meta_dict
        return sample

    def _peek(self, path: Path) -> Tuple[Array, Array, Optional[Array], dict]:
        with np.load(path, allow_pickle=True) as data:
            x = data["x"]
            y = data["y"]
            mask = data.get("loss_mask", data.get("mask"))
            meta = data.get("meta", {})
            if isinstance(meta, np.ndarray):
                meta = meta.item()
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"bad tensor shape in {path.name}")
        if x.shape[0] == y.shape[1]:
            x = x.T
        return x, y, mask, meta


def collate_p0(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    out = {"x": xs, "y": ys, "meta": [b["meta"] for b in batch]}
    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out


def make_dataloader(
    cfg: DatasetConfig,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[P0PointNetRegDataset, DataLoader]:
    dataset = P0PointNetRegDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_p0,
        drop_last=False,
    )
    return dataset, loader
