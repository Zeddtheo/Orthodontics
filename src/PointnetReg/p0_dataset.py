from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

Tensor = torch.Tensor
Array = np.ndarray

DEFAULT_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]
DEFAULT_LANDMARK_COUNTS: Dict[str, int] = {
    "t11": 7, "t12": 7, "t13": 5, "t14": 9, "t15": 9, "t16": 12, "t17": 12,
    "t21": 7, "t22": 7, "t23": 5, "t24": 9, "t25": 9, "t26": 12, "t27": 12,
    "t31": 7, "t32": 7, "t33": 5, "t34": 9, "t35": 9, "t36": 12, "t37": 12,
    "t41": 7, "t42": 7, "t43": 5, "t44": 9, "t45": 9, "t46": 12, "t47": 12,
}


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
    tooth_id: Optional[str] = None             # 牙位编号（如 "t11"），用于健康检查
    expected_points: Optional[int] = 3000      # 期望点数 N
    expected_landmarks: Optional[int] = None   # 期望 landmarks 数，默认按字典
    landmark_def_path: Optional[Union[str, Path]] = None  # 自定义 landmark 定义文件
    health_check: bool = True                  # 是否启用关卡 0 检查


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

        self._landmark_counts = self._resolve_landmark_counts(cfg.landmark_def_path, root)
        self._tooth_id = self._normalize_tooth(cfg.tooth_id)
        if cfg.expected_landmarks is not None:
            self._expected_landmarks = cfg.expected_landmarks
        elif self._tooth_id:
            self._expected_landmarks = self._landmark_counts.get(self._tooth_id)
        else:
            self._expected_landmarks = None

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

        if cfg.health_check:
            issues = self._run_health_check()
            if issues:
                message = "关卡 0｜数据与形状体检未通过：\n" + "\n".join(issues)
                raise ValueError(message)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as Z:
            x = Z["x"]  # (N, C) or (C, N)  —— 我们统一成 (C,N)
            y = Z["y"]  # (Lmax, N)
            mask = Z.get("loss_mask", Z.get("mask", None))  # (Lmax,)
            landmarks = Z.get("landmarks", None)
            pos = Z.get("pos", None)
            meta = Z.get("meta", {}).item() if "meta" in Z else {}

        # 统一形状
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.shape}")
        if x.shape[0] > x.shape[1]:
            x = x.T
        if x.shape[0] < 3:
            raise ValueError(f"x must have >=3 channels, got {x.shape}")
        if x.shape[1] != self.N:
            # Allow variable N (e.g. legacy samples) when batch_size=1.
            self.N = x.shape[1]

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
        if landmarks is not None:
            sample["landmarks"] = _to_tensor(landmarks.astype(np.float32), self.cfg.dtype)  # (L_max,3)
        if pos is not None:
            pos_arr = pos.astype(np.float32)
            if pos_arr.ndim == 2 and pos_arr.shape[1] == 3:
                pos_arr = pos_arr.T  # (3,N)
            if pos_arr.shape[0] != 3 and pos_arr.shape[1] == 3:
                pos_arr = pos_arr.T
            sample["pos"] = _to_tensor(pos_arr, self.cfg.dtype)  # (3,N)
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

    @staticmethod
    def _normalize_tooth(tooth_id: Optional[str]) -> Optional[str]:
        if tooth_id is None:
            return None
        key = str(tooth_id).strip()
        if not key:
            return None
        if key[0].lower() == "t":
            return key.lower()
        if key.isdigit():
            return f"t{key}"
        return key.lower()

    @staticmethod
    def _ensure_channels_first(x: Array) -> Array:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        return x if x.shape[0] <= x.shape[1] else x.T

    @staticmethod
    def _ensure_pos_shape(pos: Array, target_N: Optional[int]) -> Array:
        if pos.ndim != 2:
            raise ValueError(f"pos must be 2D, got shape {pos.shape}")
        if pos.shape[0] == 3:
            out = pos
        elif pos.shape[1] == 3:
            out = pos.T
        else:
            raise ValueError(f"pos must be (3,N) or (N,3), got {pos.shape}")
        if target_N is not None and out.shape[1] != target_N:
            raise ValueError(f"pos expects N={target_N}, got {out.shape[1]}")
        return out

    def _resolve_landmark_counts(
        self,
        explicit_path: Optional[Union[str, Path]],
        root: Path,
    ) -> Dict[str, int]:
        counts = dict(DEFAULT_LANDMARK_COUNTS)
        candidate: Optional[Path] = None
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                candidate = p
        if candidate is None:
            for base in [root, *root.parents]:
                test = base / "landmark_def.json"
                if test.exists():
                    candidate = test
                    break
        if candidate is None:
            return counts
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            templates = payload.get("templates", {})
            per_tooth = payload.get("per_tooth", {})
            for key, tpl_name in per_tooth.items():
                try:
                    norm_key = self._normalize_tooth(key)
                except ValueError:
                    continue
                names = templates.get(tpl_name)
                if isinstance(names, list) and names:
                    counts[norm_key] = len(names)
        except Exception as exc:
            print(f"[warn] failed to load landmark_def.json from {candidate}: {exc}")
        return counts

    def _infer_tooth_from_path(self, path: Path) -> Optional[str]:
        parts = path.stem.split("_")
        for part in reversed(parts):
            part = part.strip()
            if not part:
                continue
            try:
                return self._normalize_tooth(part)
            except ValueError:
                continue
        return self._tooth_id

    def _run_health_check(self) -> List[str]:
        issues: List[str] = []
        expected_N = self.cfg.expected_points
        expected_C = 9 if self.cfg.features == "pn" else (3 if self.cfg.features == "xyz" else None)
        for path in self.files:
            try:
                with np.load(path, allow_pickle=True) as Z:
                    tooth_key = self._infer_tooth_from_path(path)
                    expected_landmarks = self._expected_landmarks
                    if expected_landmarks is None and tooth_key is not None:
                        expected_landmarks = self._landmark_counts.get(tooth_key)
                    file_issues = self._inspect_npz(path, Z, expected_C, expected_N, expected_landmarks)
            except Exception as exc:
                issues.append(f"{path.name}: 无法读取（{exc}）")
            else:
                if file_issues:
                    issues.extend(file_issues)
        return issues

    def _inspect_npz(
        self,
        path: Path,
        Z: np.lib.npyio.NpzFile,
        expected_C: Optional[int],
        expected_N: Optional[int],
        expected_landmarks: Optional[int],
    ) -> List[str]:
        problems: List[str] = []
        required_keys = {"x", "y", "pos"}
        missing = [k for k in required_keys if k not in Z]
        if missing:
            problems.append(f"{path.name}: 缺少字段 {missing}")
            return problems

        mask_key = "loss_mask" if "loss_mask" in Z else ("mask" if "mask" in Z else None)
        if mask_key is None:
            problems.append(f"{path.name}: 缺少 loss_mask/mask 字段")
            return problems

        meta_raw = Z.get("meta", None)
        if meta_raw is None:
            problems.append(f"{path.name}: 缺少 meta 字段")
            return problems

        x = np.asarray(Z["x"])
        y = np.asarray(Z["y"])
        mask = np.asarray(Z[mask_key])
        pos = np.asarray(Z["pos"])

        try:
            x_cf = self._ensure_channels_first(x)
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
            x_cf = None
        else:
            if expected_C is not None and x_cf.shape[0] != expected_C:
                problems.append(
                    f"{path.name}: x 通道数 {x_cf.shape[0]} 与期望 {expected_C} 不符（特征='{self.cfg.features}'）"
                )
            if expected_N is not None and x_cf.shape[1] != expected_N:
                problems.append(f"{path.name}: 点数 N={x_cf.shape[1]} 与期望 {expected_N} 不符")

        if y.ndim != 2:
            problems.append(f"{path.name}: y 维度异常 {y.shape}")
        elif expected_N is not None and y.shape[1] != expected_N:
            problems.append(f"{path.name}: y 的 N={y.shape[1]} 与期望 {expected_N} 不符")

        if mask.ndim != 1:
            problems.append(f"{path.name}: mask 应为 1D，当前 {mask.shape}")
        elif y.ndim == 2 and mask.shape[0] != y.shape[0]:
            problems.append(f"{path.name}: mask 长度 {mask.shape[0]} 与 y 的 L {y.shape[0]} 不符")

        try:
            pos_cf = self._ensure_pos_shape(pos, expected_N if expected_N is not None else (x_cf.shape[1] if x_cf is not None else None))
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
        else:
            if not np.isfinite(pos_cf).all():
                problems.append(f"{path.name}: pos 含 NaN/Inf")

        if x_cf is not None and not np.isfinite(x_cf).all():
            problems.append(f"{path.name}: x 含 NaN/Inf")
        if y.ndim == 2 and not np.isfinite(y).all():
            problems.append(f"{path.name}: y 含 NaN/Inf")
        if mask.ndim == 1 and not np.isfinite(mask).all():
            problems.append(f"{path.name}: mask 含 NaN/Inf")

        if mask.ndim == 1:
            rounded = np.round(mask)
            if not np.allclose(mask, rounded, atol=1e-4):
                problems.append(f"{path.name}: mask 非 0/1 值")
            mask_sum = float(mask.sum())
            if not np.isclose(mask_sum, round(mask_sum), atol=1e-3):
                problems.append(f"{path.name}: mask.sum()={mask_sum:.3f} 不是整数")
            else:
                active_landmarks = int(round(mask_sum))
                if expected_landmarks is not None and active_landmarks != expected_landmarks:
                    problems.append(
                        f"{path.name}: mask.sum()={active_landmarks} 与期望 landmarks {expected_landmarks} 不符"
                    )
                if y.ndim == 2 and expected_landmarks is not None and y.shape[0] < expected_landmarks:
                    problems.append(
                        f"{path.name}: y 的 L={y.shape[0]} 少于期望 {expected_landmarks}"
                    )

        meta = meta_raw
        if isinstance(meta_raw, np.ndarray):
            try:
                meta = meta_raw.item()
            except ValueError:
                meta = meta_raw.tolist()
        if not isinstance(meta, dict):
            problems.append(f"{path.name}: meta 不是 dict（实际类型 {type(meta)}）")
        else:
            for key in ("center_mm", "bounds_mm"):
                if key not in meta:
                    problems.append(f"{path.name}: meta 缺少 {key}")
                    continue
                arr = np.asarray(meta[key], dtype=np.float32)
                if arr.size != 3:
                    problems.append(f"{path.name}: meta.{key} 期望长度 3，当前 {arr.shape}")
                elif not np.isfinite(arr).all():
                    problems.append(f"{path.name}: meta.{key} 含 NaN/Inf")

        return problems


# ------------------------- collate & loader -------------------------

def collate_p0(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    要求同一批次 (L, N) 一致（推荐“每牙一个 DataLoader”）。
    返回：
      x: (B,C,N), y: (B,L,N), mask: (B,L) [可选], meta: list[dict]
    """
    max_n = max(b["x"].shape[-1] for b in batch)

    def _pad_feat(t: Tensor, target_n: int) -> Tensor:
        diff = target_n - t.shape[-1]
        if diff <= 0:
            return t
        return F.pad(t, (0, diff))

    xs = torch.stack([_pad_feat(b["x"], max_n) for b in batch], dim=0)              # (B,C,N)
    ys = torch.stack([_pad_feat(b["y"], max_n) for b in batch], dim=0)              # (B,L,N)
    out = {"x": xs, "y": ys, "meta": [b["meta"] for b in batch]}
    if "mask" in batch[0]:
        ms = torch.stack([b["mask"] for b in batch], dim=0)       # (B,L_all)
        out["mask"] = ms
    if "landmarks" in batch[0]:
        lms = torch.stack([b["landmarks"] for b in batch], dim=0)  # (B,L_max,3)
        out["landmarks"] = lms
    if "pos" in batch[0]:
        poss = torch.stack([_pad_feat(b["pos"], max_n) for b in batch], dim=0)       # (B,3,N)
        out["pos"] = poss
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
