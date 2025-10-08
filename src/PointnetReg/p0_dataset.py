from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor
Array = np.ndarray


# --------------------------- utils ---------------------------

def _to_tensor(a: Array, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.from_numpy(a).to(dtype)

def _rigid_z_rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def apply_rigid_to_x(x_cn: Array, R: Array, t: Array) -> Array:
    """
    x_cn: (C, N) with first 3 rows = xyz, next 3 rows (optional) = normals.
    Applies xyz' = R*xyz + t, normal' = R*normal (if present).
    """
    C, N = x_cn.shape
    assert C >= 3, "x must have at least 3 dims (xyz)."
    xyz = x_cn[0:3, :]
    xyz = R @ xyz + t.reshape(3, 1)
    out = [xyz]
    if C >= 6:
        nrm = x_cn[3:6, :]
        nrm = R @ nrm
        out.append(nrm)
        if C > 6:
            out.append(x_cn[6:, :])  # extras unchanged
    else:
        if C > 3:
            out.append(x_cn[3:, :])
    return np.concatenate(out, axis=0)


# ------------------------ config/dataclass ------------------------

@dataclass
class DatasetConfig:
    root: Union[str, Path]                      # 目录，含 *.npz
    file_patterns: Sequence[str] = ("*.npz",)  # 支持按牙过滤，如 "*_t31.npz"
    features: str = "pn"                       # "pn" | "xyz" （pn = pos+nrm+cent_rel）
    select_landmarks: str = "active"           # "active" 只保留 mask==1 的通道；"all" 保留全部并返回 mask
    augment: bool = False                      # 只做刚体增强（Rz + 平移）
    rotz_deg: float = 15.0                     # 增强旋转幅度（±度）
    trans_mm: float = 0.5                      # 增强平移幅度（各轴均匀±）
    ensure_constant_L: bool = True             # 同一数据集内 L 必须一致（建议“每牙一个数据集”）
    dtype: torch.dtype = torch.float32


# --------------------------- dataset ---------------------------

class P0PointNetRegDataset(Dataset):
    """
    读取预处理好的 *.npz，为 PointNet-Reg 提供样本：
      x: (C, N), y: (L, N), mask: (L,) [可选], meta: dict
    DataLoader 会把它们堆成：
      x: (B, C, N), y: (B, L, N), mask: (B, L)  (如 select_landmarks='all')
    """
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)
        files: List[Path] = []
        for pat in cfg.file_patterns:
            files += sorted(root.glob(pat))
        if not files:
            raise FileNotFoundError(f"No .npz found under {root} with {cfg.file_patterns}")
        self.files = files

        # 确认通道与 L（第一次样本为基准）
        # 约定：npz 至少包含 x(N,C)、y(Lmax,N) 与 loss_mask(Lmax,)（或 mask）
        x0, y0, m0, _ = self._peek(self.files[0])
        self.C = x0.shape[0]
        self.N = x0.shape[1]
        self.L_all = y0.shape[0]
        self._active_L = int(m0.sum()) if m0 is not None else self.L_all

        # features 选择
        if cfg.features == "xyz":
            self._use_channels = slice(0, 3)
        elif cfg.features in ("pn", "all"):
            # 期望新 npz 的 x 为 [pos(3), nrm(3), cent_rel(3)]
            self._use_channels = slice(0, min(9, self.C))
        else:
            raise ValueError("features must be 'pn', 'all', or 'xyz'.")

        # 决定实际导出的 L
        self.L = self._active_L if cfg.select_landmarks == "active" else self.L_all

        # 可选：全体文件 L 一致性检查（建议“每牙一数据集”，L 自然一致）
        if cfg.ensure_constant_L:
            for f in self.files:
                _, y, m, _ = self._peek(f)
                L_all = y.shape[0]
                L_act = int(m.sum()) if m is not None else L_all
                tgt = self._active_L if cfg.select_landmarks == "active" else self.L_all
                got = L_act if cfg.select_landmarks == "active" else L_all
                if got != tgt:
                    raise ValueError(f"Inconsistent L in {f.name}: expect {tgt}, got {got}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as Z:
            x = Z["x"]  # (N, C) or (C, N)  —— 我们统一成 (C,N)
            y = Z["y"]  # (Lmax, N)
            mask = Z.get("loss_mask", Z.get("mask", None))  # (Lmax,)
            meta = Z.get("meta", {}).item() if "meta" in Z else {}

        # 统一形状
        if x.shape == (self.N, self.C):
            x = x.T
        elif x.shape == (self.C, self.N):
            pass  # already (C,N)
        elif x.shape[0] == self.N and x.shape[1] >= 3:
            x = x.T
        elif x.shape[1] == self.N and x.shape[0] >= 3:
            pass  # treat as (C,N) even if通道数与self.C不同
        assert x.shape[0] >= 3 and x.shape[1] == self.N, f"x shape bad: {x.shape}"

        # 选择特征通道
        x = x[self._use_channels, :]  # (C_use, N)

        # 选择 landmarks 通道
        if self.cfg.select_landmarks == "active" and mask is not None:
            sel = mask.astype(bool)
            y = y[sel, :]
            mask_out = None
        else:
            mask_out = mask  # 训练时可用作 loss 掩码（BCE/MSE）

        # 可选刚体增强（不改变 y，因为欧氏距离在刚体下不变）
        if self.cfg.augment:
            theta = math.radians(np.random.uniform(-self.cfg.rotz_deg, self.cfg.rotz_deg))
            R = _rigid_z_rotation_matrix(theta)
            t = np.random.uniform(-self.cfg.trans_mm, self.cfg.trans_mm, size=(3,)).astype(np.float32)
            # 旋转 xyz、normal，并把 cent_rel 也一并旋转（若存在）
            x[:3, :] = (R @ x[:3, :] + t.reshape(3, 1)).astype(np.float32)
            if x.shape[0] >= 6:
                x[3:6, :] = (R @ x[3:6, :]).astype(np.float32)
            if x.shape[0] >= 9:
                x[6:9, :] = (R @ x[6:9, :]).astype(np.float32)

        # 转 tensor
        sample = {
            "x": _to_tensor(x, self.cfg.dtype),         # (C_use, N)
            "y": _to_tensor(y, self.cfg.dtype),         # (L, N)
        }
        if mask_out is not None:
            sample["mask"] = _to_tensor(mask_out.astype(np.float32), self.cfg.dtype)  # (L_all,)
        # 附带元信息（字符串保持 Python 对象）
        sample["meta"] = {"path": str(path), **(meta if isinstance(meta, dict) else {})}
        return sample

    def _peek(self, path: Path) -> Tuple[Array, Array, Optional[Array], dict]:
        with np.load(path, allow_pickle=True) as Z:
            x = Z["x"]
            y = Z["y"]
            mask = Z.get("loss_mask", Z.get("mask", None))
            meta = Z.get("meta", {})
            if isinstance(meta, np.ndarray):
                meta = meta.item()
        # 统一到 (C,N)
        x = x.T if x.shape[0] != x.shape[1] and x.shape[0] == y.shape[1] else x
        if x.ndim != 2:
            raise ValueError(f"bad x ndim in {path.name}: {x.shape}")
        if y.ndim != 2:
            raise ValueError(f"bad y ndim in {path.name}: {y.shape}")
        return x if x.shape[0] < x.shape[1] else x.T, y, mask, meta


# ------------------------- collate & loader -------------------------

def collate_p0(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    要求同一批次 (L, N) 一致（推荐“每牙一个 DataLoader”）。
    返回：
      x: (B,C,N), y: (B,L,N), mask: (B,L) [可选], meta: list[dict]
    """
    xs = torch.stack([b["x"] for b in batch], dim=0)              # (B,C,N)
    ys = torch.stack([b["y"] for b in batch], dim=0)              # (B,L,N)
    out = {"x": xs, "y": ys, "meta": [b["meta"] for b in batch]}
    if "mask" in batch[0]:
        ms = torch.stack([b["mask"] for b in batch], dim=0)       # (B,L_all)
        out["mask"] = ms
    return out


def make_dataloader(
    cfg: DatasetConfig,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[P0PointNetRegDataset, DataLoader]:
    ds = P0PointNetRegDataset(cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=pin_memory,
                    collate_fn=collate_p0, drop_last=False)
    return ds, dl
