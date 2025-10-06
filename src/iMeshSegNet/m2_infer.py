#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import pyvista as pv

# --- 保证本地导入 ---
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from imeshsegnet import iMeshSegNet
from m0_dataset import (
    SEG_NUM_CLASSES,
    extract_features,
    normalize_mesh_units,
)
from m3_postprocess import PPConfig, postprocess_6k_10k_full  # 近似graphcut + SVM
# ^ 若你把 postprocess 文件名不同，请同步这里的导入

# ---------------- 色表（0..14） ----------------
LABEL_COLORS = {
    0:  [160,160,160], 1:[255,0,0], 2:[255,127,0], 3:[255,255,0], 4:[0,255,0],
    5:  [0,255,255], 6:[0,0,255], 7:[127,0,255], 8:[255,0,255], 9:[255,192,203],
    10: [165,42,42], 11:[255,215,0], 12:[0,128,128], 13:[128,0,128], 14:[255,140,0],
}

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)

def _compute_diag(points: np.ndarray) -> float:
    if points.size == 0: return 1.0
    bmin, bmax = points.min(0), points.max(0)
    d = float(np.linalg.norm(bmax - bmin))
    return max(d, 1e-6)

def _fps(xyz: np.ndarray, m: int) -> np.ndarray:
    n = int(xyz.shape[0])
    if m >= n: return np.arange(n, dtype=np.int64)
    sel = np.zeros(m, dtype=np.int64)
    dist = np.full(n, np.inf, np.float32)
    far = 0
    for i in range(m):
        sel[i] = far
        d = np.linalg.norm(xyz - xyz[far], axis=1)
        dist = np.minimum(dist, d)
        far = int(np.argmax(dist))
    return sel

def _apply_colors(mesh: pv.PolyData, labels: np.ndarray) -> pv.PolyData:
    labels = labels.astype(np.int32, copy=False)
    cell_rgb = np.zeros((mesh.n_cells, 3), np.uint8)
    for lab in np.unique(labels):
        cell_rgb[labels == lab] = np.array(LABEL_COLORS.get(int(lab), [90,90,90]), np.uint8)
    mesh.cell_data["RGB"] = cell_rgb
    mesh.cell_data["PredLabel"] = labels

    # 多数投票把 cell 标签映射到点，避免插值彩边
    pt_lab = np.zeros(mesh.n_points, np.int32)
    pt_rgb = np.zeros((mesh.n_points,3), np.uint8)
    for pid in range(mesh.n_points):
        cids = mesh.point_cell_ids(pid)
        if len(cids) == 0: continue
        labs = labels[cids]
        uniq, cnt = np.unique(labs, return_counts=True)
        lab = uniq[np.argmax(cnt)]
        pt_lab[pid] = lab
        pt_rgb[pid] = cell_rgb[cids[0]]
    mesh.point_data["RGB"] = pt_rgb
    mesh.point_data["PredLabel"] = pt_lab
    return mesh

# ---------------- 契约加载 ----------------
def _load_pipeline(ckpt_path: Path, args=None):
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    P = ckpt.get("pipeline", {}) if isinstance(ckpt, dict) else {}
    z = P.get("zscore", {}) if isinstance(P.get("zscore"), dict) else {}

    meta = {
        "zscore_apply": bool(z.get("apply", True)),
        "mean": np.array(z.get("mean"), np.float32) if z.get("mean") is not None else None,
        "std":  np.array(z.get("std"),  np.float32) if z.get("std")  is not None else None,
        "centered": bool(P.get("centered", True)),
        "div_by_diag": bool(P.get("div_by_diag", True)),
        "sampler": str(P.get("sampler", "random")),
        "sample_cells": int(P.get("sample_cells", 6000)),
        "target_cells": int(P.get("target_cells", 10000)),
        "train_ids_path": P.get("train_ids_path", ckpt.get("train_sample_ids_path")),
        "decim_cache_vtp": P.get("decim_cache_vtp", None),
        "knn_k": P.get("knn_k", {"to10k":5, "tofull":7}),
        "num_classes": int(ckpt.get("num_classes", SEG_NUM_CLASSES)),
        "in_channels": int(ckpt.get("in_channels", 15)),
    }
    arch_raw = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
    meta["arch"] = {
        "glm_impl": str(arch_raw.get("glm_impl", "edgeconv")),
        "k_short": int(arch_raw.get("k_short", 6)) if arch_raw.get("k_short") is not None else 6,
        "k_long": int(arch_raw.get("k_long", 12)) if arch_raw.get("k_long") is not None else 12,
        "with_dropout": bool(arch_raw.get("with_dropout", False)),
        "dropout_p": float(arch_raw.get("dropout_p", 0.1)) if arch_raw.get("dropout_p") is not None else 0.1,
        "use_feature_stn": bool(arch_raw.get("use_feature_stn", True)),
    }
    print("\n📋 Pipeline 契约:")
    print(f"   Z-score: {'✓' if meta['zscore_apply'] else '✗'} (mean: {None if meta['mean'] is None else meta['mean'].shape})")
    print(f"   Centered: {meta['centered']}, Div by diag: {meta['div_by_diag']}")
    print(f"   Sampler: {meta['sampler']}, Target cells: {meta['target_cells']}, Sample cells: {meta['sample_cells']}")
    print(f"   Arch: glm={meta['arch']['glm_impl']}, k={meta['arch']['k_short']}/{meta['arch']['k_long']}, fstn={meta['arch']['use_feature_stn']}")
    return ckpt, meta

@torch.no_grad()
def _build_model(ckpt_path: Path, device: torch.device, meta: dict) -> iMeshSegNet:
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    arch = meta.get("arch", {})
    model = iMeshSegNet(
        num_classes=meta["num_classes"],
        glm_impl=arch.get("glm_impl", "edgeconv"),
        k_short=arch.get("k_short", 6),
        k_long=arch.get("k_long", 12),
        with_dropout=arch.get("with_dropout", False),
        dropout_p=arch.get("dropout_p", 0.1),
        use_feature_stn=arch.get("use_feature_stn", True),
    )
    if isinstance(state, dict):
        load_state = state
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in state:
                load_state = state[key]
                break
        try:
            model.load_state_dict(load_state)
        except RuntimeError as exc:
            print(f"[Warn] load_state_dict strict=True failed: {exc}. Retrying with strict=False.")
            model.load_state_dict(load_state, strict=False)
    else:
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            print(f"[Warn] load_state_dict strict=True failed: {exc}. Retrying with strict=False.")
            model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"\n🏗️ 模型就绪: classes={meta['num_classes']}, in_ch={meta['in_channels']}, fstn={arch.get('use_feature_stn', True)}")
    return model

# ---------------- 关键：严格复现训练预处理 ----------------
def _features_and_pos_for_contract(mesh10k_mm: pv.PolyData, meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # 位置（mm、norm）都基于 10k decimated 网格
    pos_mm10k = mesh10k_mm.cell_centers().points.astype(np.float32)
    diag = _compute_diag(pos_mm10k)
    center = mesh10k_mm.center

    pos_norm10k = pos_mm10k.copy()
    if meta["centered"]:
        pos_norm10k -= center
    if meta["div_by_diag"]:
        pos_norm10k /= diag

    # 特征与训练一致：先中心化到0，再抽取15维特征，再做z-score
    feat_mesh = mesh10k_mm.copy(deep=True)
    if meta["centered"]:
        feat_mesh.points -= feat_mesh.center
    feat_mesh, *_ = normalize_mesh_units(feat_mesh)  # 目前为直通实现
    feat_mesh = feat_mesh.triangulate()
    feats10k = extract_features(feat_mesh).astype(np.float32, copy=False)  # (N,15)

    if meta["zscore_apply"] and meta["mean"] is not None and meta["std"] is not None:
        feats10k = (feats10k - meta["mean"].reshape(1, -1)) / np.clip(meta["std"], 1e-6, None).reshape(1, -1)

    return feats10k, pos_norm10k, pos_mm10k, diag

def _select_6k_ids(pos_mm10k: np.ndarray, meta: dict) -> np.ndarray:
    # 优先复用训练 6k 采样索引
    ids_path = meta.get("train_ids_path")
    if ids_path and Path(ids_path).exists():
        ids = np.load(ids_path).astype(np.int64).reshape(-1)
        if ids.max() < pos_mm10k.shape[0]:
            print(f"[Check#3] 复用训练采样索引: {len(ids)}")
            return ids
    # 兜底：FPS（比纯随机稳定）
    print("[Check#3] 训练采样缺失，使用 FPS 采样 6000")
    return _fps(pos_mm10k, int(meta["sample_cells"]))

def _read_decimated(stem: str, meta: dict, raw_path: Path) -> pv.PolyData:
    cache_path = Path(meta.get("decim_cache_vtp") or "")
    if cache_path and cache_path.exists():
        # 校验 stem 匹配
        if cache_path.name.split(".c")[0] != stem:
            raise ValueError(f"契约 decim_cache_vtp 与输入不一致: {cache_path.name} vs {stem}")
        print(f"[Check#2] 使用 decimated 缓存: {cache_path.name}")
        return pv.read(str(cache_path))
    # 回退：直接读原始（若本机也有同名缓存，会被 PyVista 透明读取）
    print(f"[Check#2] 使用原始: {raw_path.name}")
    return pv.read(str(raw_path))

def _infer_one(model: iMeshSegNet, meta: dict, inp_path: Path, out_dir: Path, device: torch.device):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp_path.stem

    # 读取 full 原始网格（用于最终着色保存）
    orig = pv.read(str(inp_path))
    orig_centers = orig.cell_centers().points.astype(np.float32)

    # 读取 10k decimated（契约优先，其次回退）
    mesh10k = _read_decimated(stem, meta, inp_path)
    if mesh10k.n_cells != int(meta["target_cells"]):
        print(f"[Warn] decimated cells ≠ 契约 target ({mesh10k.n_cells} vs {meta['target_cells']})，按实际N继续")

    # 提取 10k 特征与位置（严格对齐训练）
    feats10k, pos_norm10k, pos_mm10k, diag10k = _features_and_pos_for_contract(mesh10k, meta)

    # 选 6k 索引、组装张量
    ids6k = _select_6k_ids(pos_mm10k, meta)
    x = torch.from_numpy(feats10k[ids6k].T).unsqueeze(0).to(device)    # (B,15,N6)
    p = torch.from_numpy(pos_norm10k[ids6k].T).unsqueeze(0).to(device) # (B,3,N6)

    # 前向
    with torch.no_grad():
        logits = model(x, p)  # (B,C,N6)
    prob6k = torch.softmax(logits, dim=1)[0].cpu().numpy().T  # (N6,C)

    # 后处理：6k -> 10k -> full
    cfg = PPConfig(seed_conf_th=0.70, gc_beta=40.0, svm_max_train=5000)
    lab10k, lab_full = postprocess_6k_10k_full(
        prob6k=prob6k,
        pos_mm6=pos_mm10k[ids6k],
        pos_mm10=pos_mm10k,
        pos_mm_full=orig_centers,
        cfg=cfg
    )

    # 保存着色
    out_10k = out_dir / f"{stem}_10k_colored.vtp"
    out_full = out_dir / f"{stem}_full_colored.vtp"
    _apply_colors(mesh10k.copy(deep=True), lab10k).save(out_10k, binary=True)
    _apply_colors(orig.copy(deep=True),    lab_full).save(out_full, binary=True)
    print(f"✅ 已保存: {out_10k.name}, {out_full.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, meta = _load_pipeline(args.ckpt, args)
    device = torch.device(args.device)
    model = _build_model(args.ckpt, device, meta)

    files: List[Path] = []
    if args.input.is_dir():
        files = sorted([p for p in args.input.glob("*.vtp")] + [p for p in args.input.glob("*.stl")])
    else:
        files = [args.input]

    for fp in files:
        _infer_one(model, meta, fp, args.out, device)

if __name__ == "__main__":
    main()
