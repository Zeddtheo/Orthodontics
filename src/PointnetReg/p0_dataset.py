from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


class P0PointNetRegDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)
        files: List[Path] = []
        for pat in cfg.file_patterns:
            files.extend(sorted(root.glob(pat)))
        if not files:
            raise FileNotFoundError(f"no npz under {root} with {cfg.file_patterns}")
        self.files = files

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
                    raise ValueError(f"inconsistent landmark count in {path.name}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
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

        if self.cfg.augment:
            theta = math.radians(np.random.uniform(-self.cfg.rotz_deg, self.cfg.rotz_deg))
            R = _rotz(theta)
            t = np.random.uniform(-self.cfg.trans_mm, self.cfg.trans_mm, size=3).astype(np.float32)
            x = _apply_rigid(x, R, t)

        sample = {"x": _to_tensor(x, self.cfg.dtype), "y": _to_tensor(y, self.cfg.dtype)}
        if mask_out is not None:
            sample["mask"] = _to_tensor(mask_out.astype(np.float32), self.cfg.dtype)
        meta_dict = {"path": str(path)}
        if isinstance(meta, dict):
            meta_dict.update(meta)
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


def collate_p0(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
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
