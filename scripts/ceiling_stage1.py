#!/usr/bin/env python3
"""
Ceiling analysis for Stage-1 segmentation.

Given a ground-truth (or manually corrected) mesh with per-cell labels,
this script skips the iMeshSegNet forward pass and injects perfect logits
directly into the post-processing pipeline (6k→10k→full).

The goal is to measure the performance upper-bound imposed by the
post-processing stack itself (exact vs. full pipeline with ICM/SVM/etc.).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

# Local imports
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from iMeshSegNet.m0_dataset import (  # type: ignore[import]
    SINGLE_ARCH_NUM_CLASSES,
    _build_single_arch_label_maps,
    _decim_cache_path,
    remap_labels_single_arch,
)
from iMeshSegNet.m2_infer import (  # type: ignore[import]
    _features_and_pos_for_contract,
    _load_pipeline,
    _read_decimated,
    _select_6k_ids,
)
from iMeshSegNet.m3_postprocess import (  # type: ignore[import]
    PPConfig,
    build_cell_adjacency,
    postprocess_6k_10k_full,
)
from iMeshSegNet import m3_postprocess as m3pp  # type: ignore[import]


@dataclass
class CeilingResult:
    variant: str
    logs: Dict
    lab10k: np.ndarray
    lab_full: np.ndarray
    dsc_per_class: Dict[int, float]
    mean_dsc: float


def _extract_orig_ids(mesh10k: pv.PolyData) -> np.ndarray:
    for key in ("vtkOriginalCellIds", "orig_cell_ids"):
        if key in mesh10k.cell_data:
            return np.asarray(mesh10k.cell_data[key]).astype(np.int64, copy=False)
    raise RuntimeError("decimated mesh is missing vtkOriginalCellIds/orig_cell_ids")


def _load_assignments(base_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    assign_indices = assign_weights = assign_ids = None
    knn_path = base_path.with_suffix(".assign_knn.npz")
    if knn_path.exists():
        try:
            with np.load(str(knn_path)) as knn:
                idx_arr = knn["indices"]
                w_arr = knn["weights"]
            assign_indices = idx_arr.astype(np.int64, copy=idx_arr.dtype != np.int64)
            assign_weights = w_arr.astype(np.float32, copy=w_arr.dtype != np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] failed to read {knn_path.name}: {exc}")
    assign_path = base_path.with_suffix(".assign.npy")
    if assign_path.exists():
        try:
            arr = np.load(str(assign_path))
            assign_ids = arr.astype(np.int64, copy=arr.dtype != np.int64)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] failed to read {assign_path.name}: {exc}")
    return assign_ids, assign_indices, assign_weights


def _labels_to_logits(labels: np.ndarray, num_classes: int, margin: float = 50.0) -> np.ndarray:
    logits = np.full((labels.shape[0], num_classes), -margin, dtype=np.float32)
    logits[np.arange(labels.shape[0]), labels.astype(np.int64)] = margin
    return logits


def _compute_dsc(gt: np.ndarray, pred: np.ndarray, classes: Iterable[int]) -> Dict[int, float]:
    dsc: Dict[int, float] = {}
    for cls in classes:
        gt_mask = gt == cls
        pred_mask = pred == cls
        denom = int(gt_mask.sum() + pred_mask.sum())
        if denom == 0:
            dsc[cls] = float("nan")
            continue
        inter = int(np.logical_and(gt_mask, pred_mask).sum())
        dsc[cls] = 2.0 * inter / denom
    return dsc


def _summarise_dsc(dsc_per_class: Dict[int, float], tooth_ids: Sequence[int]) -> float:
    values = [dsc_per_class.get(cls, float("nan")) for cls in tooth_ids]
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _apply_colors(mesh: pv.PolyData, labels: np.ndarray) -> pv.PolyData:
    mesh = mesh.copy(deep=True)
    mesh.cell_data["PredLabel"] = labels.astype(np.int32, copy=False)
    return mesh


def run_variant(
    variant: str,
    use_exact: bool,
    meta: Dict,
    cfg_base: PPConfig,
    pos6_mm: np.ndarray,
    logits6: np.ndarray,
    pos10_mm: np.ndarray,
    pos_full_mm: np.ndarray,
    normals6: Optional[np.ndarray],
    normals10: Optional[np.ndarray],
    full_adj,
    adj10,
    orig_ids: np.ndarray,
    assign_ids: Optional[np.ndarray],
    assign_indices: Optional[np.ndarray],
    assign_weights: Optional[np.ndarray],
    cfg_override: Optional[Dict[str, object]] = None,
    pairwise_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    cfg = PPConfig(**vars(cfg_base))
    if use_exact:
        cfg.knn_10k = 1
        cfg.knn_full = 1
        cfg.seed_conf_th = 0.0
        cfg.bg_seed_th = 0.0
        cfg.gc_beta = 0.0
        cfg.gc_iterations = 0
        cfg.full_gc_enabled = False
        cfg.fill_radius = 0.0
        cfg.min_component_size = 0
        cfg.min_component_size_full = 0
        cfg.tiny_component_size = 0
        cfg.gingiva_dilate_iters = 0
    else:
        kk = meta.get("knn_k", {})
        cfg.knn_10k = max(int(kk.get("to10k", cfg.knn_10k)), 1)
        cfg.knn_full = max(int(kk.get("tofull", cfg.knn_full)), 1)
        cfg.gingiva_label = int(meta.get("gingiva_label", cfg.gingiva_label))
        cfg.gc_beta = float(meta.get("gc_beta", cfg.gc_beta))
        cfg.gc_k = int(meta.get("gc_k", cfg.gc_k))
        cfg.gc_iterations = int(meta.get("gc_iterations", cfg.gc_iterations))
        cfg.full_gc_lambda = float(meta.get("full_gc_lambda", cfg.full_gc_lambda))
        cfg.full_gc_iterations = int(meta.get("full_gc_iterations", cfg.full_gc_iterations))
        cfg.full_gc_enabled = bool(meta.get("full_gc_enabled", cfg.full_gc_enabled))
        cfg.fill_radius = float(meta.get("fill_radius", cfg.fill_radius))
        cfg.min_component_size = int(meta.get("min_component_size", cfg.min_component_size))
        cfg.min_component_size_full = int(meta.get("min_component_size_full", cfg.min_component_size_full))
        cfg.clean_component_neighbors = int(meta.get("clean_component_neighbors", cfg.clean_component_neighbors))
        cfg.seed_conf_th = float(meta.get("seed_conf_th", cfg.seed_conf_th))
        cfg.bg_seed_th = float(meta.get("bg_seed_th", cfg.bg_seed_th))
        cfg.low_conf_threshold = float(meta.get("low_conf_threshold", cfg.low_conf_threshold))
        cfg.low_conf_neighbors = int(meta.get("low_conf_neighbors", cfg.low_conf_neighbors))
        cfg.low_conf_delta_th = float(meta.get("low_conf_delta_th", cfg.low_conf_delta_th))
        cfg.svm_max_train = int(meta.get("svm_max_train", cfg.svm_max_train))
        cfg.gingiva_dilate_iters = int(meta.get("gingiva_dilate_iters", cfg.gingiva_dilate_iters))
        cfg.gingiva_dilate_thresh = float(meta.get("gingiva_dilate_thresh", cfg.gingiva_dilate_thresh))
        cfg.gingiva_dilate_k = int(meta.get("gingiva_dilate_k", cfg.gingiva_dilate_k))
        cfg.gingiva_protect_seeds = bool(meta.get("gingiva_protect_seeds", cfg.gingiva_protect_seeds))
        cfg.gingiva_protect_conf = float(meta.get("gingiva_protect_conf", cfg.gingiva_protect_conf))
        cfg.tiny_component_size = int(meta.get("tiny_component_size", cfg.tiny_component_size))
        if cfg_override:
            for key, value in cfg_override.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    patched_pairwise = False
    original_pairwise = None
    if pairwise_mode and hasattr(m3pp, "_pairwise_weights_dihedral"):
        original_pairwise = m3pp._pairwise_weights_dihedral

        def _pairwise_patch(pos_mm: np.ndarray, normals: Optional[np.ndarray], edges: np.ndarray) -> np.ndarray:
            if edges.size == 0:
                return np.zeros(0, dtype=np.float32)
            ci = pos_mm[edges[:, 0]]
            cj = pos_mm[edges[:, 1]]
            phi = np.linalg.norm(ci - cj, axis=1)
            phi = np.clip(phi, 1e-6, None)
            if normals is None or normals.shape[0] != pos_mm.shape[0]:
                return (1.0 / phi).astype(np.float32)

            ni = normals[edges[:, 0]]
            nj = normals[edges[:, 1]]
            ni = ni / np.clip(np.linalg.norm(ni, axis=1, keepdims=True), 1e-6, None)
            nj = nj / np.clip(np.linalg.norm(nj, axis=1, keepdims=True), 1e-6, None)
            cos = np.clip(np.sum(ni * nj, axis=1), -1.0, 1.0)
            theta = np.arccos(np.clip(cos, -1.0, 1.0))
            w_base = -np.log(np.clip(theta / np.pi, 1e-4, 1.0)) * phi

            cross_vec = np.cross(ni, nj)
            dir_vec = cj - ci
            is_convex = np.sum(cross_vec * dir_vec, axis=1) > 0.0
            if pairwise_mode == "flip_convex":
                is_convex = ~is_convex

            beta_ij = 1.0 + np.abs(cos)
            if pairwise_mode == "coef1":
                coef = np.ones_like(beta_ij)
            else:
                coef = np.where(is_convex, 30.0 * beta_ij, 1.0)

            w = w_base * coef
            w = np.clip(w, 0.0, np.percentile(w, 99.5))
            return w.astype(np.float32, copy=False)

        m3pp._pairwise_weights_dihedral = _pairwise_patch  # type: ignore[assignment]
        patched_pairwise = True

    try:
        lab10k, lab_full, logs = postprocess_6k_10k_full(
            pos6=pos6_mm,
            logits6=logits6,
            pos10=pos10_mm,
            pos_full=pos_full_mm,
            normals6=normals6,
            normals10=normals10,
            orig_cell_ids=orig_ids,
            assign_ids=assign_ids,
            assign_indices=assign_indices,
            assign_weights=assign_weights,
            cfg=cfg,
            full_adjacency=full_adj,
            adjacency10=adj10,
        )
    finally:
        if patched_pairwise and original_pairwise is not None:
            m3pp._pairwise_weights_dihedral = original_pairwise  # type: ignore[assignment]

    if use_exact:
        # exact replay keeps 10k labels as truth; propagate deterministically
        if assign_indices is not None and assign_weights is not None and assign_indices.shape[0] == pos_full_mm.shape[0]:
            neigh_labels = lab10k[assign_indices]
            weights = assign_weights
            vote_bins = np.zeros((assign_indices.shape[0], np.max(lab10k) + 1), dtype=np.float32)
            for k in range(assign_indices.shape[1]):
                lbl = neigh_labels[:, k]
                np.add.at(vote_bins, (np.arange(vote_bins.shape[0]), lbl), weights[:, k])
            lab_full = vote_bins.argmax(axis=1).astype(np.int32, copy=False)
        elif assign_ids is not None and assign_ids.shape[0] == pos_full_mm.shape[0]:
            lab_full = lab10k[assign_ids]

    print(f"[Ceiling][{variant}] conf10_mean={logs.get('conf10_mean', -1):.3f} | seed_ratio={logs.get('seed_ratio', 0.0):.3f}")
    return lab10k, lab_full.astype(np.int32, copy=False), logs


def run_ceiling(case_path: Path, ckpt_path: Path, out_root: Optional[Path]) -> List[CeilingResult]:
    if not case_path.exists():
        raise FileNotFoundError(f"case mesh not found: {case_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"segmentation checkpoint not found: {ckpt_path}")

    _, meta = _load_pipeline(ckpt_path)

    mesh_full = pv.read(str(case_path))
    try:
        mesh_full_tri = mesh_full.triangulate()
    except Exception:
        mesh_full_tri = mesh_full
    full_adj = build_cell_adjacency(mesh_full_tri)
    pos_full_mm = mesh_full.cell_centers().points.astype(np.float32)

    raw_labels = np.asarray(mesh_full.cell_data["Label"], dtype=np.int64)
    single_maps = _build_single_arch_label_maps(0, 15, False)
    gt_full = remap_labels_single_arch(raw_labels, case_path, single_maps)

    mesh10k = _read_decimated(case_path.stem, meta, case_path)
    try:
        mesh10k_tri = mesh10k.triangulate()
    except Exception:
        mesh10k_tri = mesh10k
    adj10 = build_cell_adjacency(mesh10k_tri)

    feats10k, pos_norm10k, pos_mm10k, _, normals10k = _features_and_pos_for_contract(mesh10k, meta)
    ids6k = _select_6k_ids(pos_mm10k, meta)
    pos6_mm = pos_mm10k[ids6k]
    normals6 = normals10k[ids6k] if normals10k is not None else None

    orig_ids = _extract_orig_ids(mesh10k)
    gt_10k = gt_full[orig_ids]
    gt_6k = gt_10k[ids6k]

    num_classes = int(meta.get("num_classes", SINGLE_ARCH_NUM_CLASSES))
    logits6 = _labels_to_logits(gt_6k, num_classes)

    hint = meta.get("decim_cache_vtp")
    if hint:
        base_path = Path(hint)
    else:
        base_path = _decim_cache_path(case_path, int(meta.get("target_cells", pos10_mm.shape[0])))
    assign_ids, assign_indices, assign_weights = _load_assignments(base_path)
    if assign_ids is None and assign_indices is None:
        raise FileNotFoundError(
            f"Decimation assignment cache missing for {case_path.stem}: expected {base_path.with_suffix('.assign.npy')}"
        )

    cfg_base = PPConfig()

    variants = [
        ("exact", True, None, None),
        ("full", False, None, None),
        ("s1_gc_off", False, {"gc_beta": 0.0, "gc_iterations": 0}, None),
        ("s2_fullgc_off", False, {"full_gc_enabled": False, "full_gc_lambda": 0.0, "full_gc_iterations": 0}, None),
        ("s3_dihedral_flat", False, None, "coef1"),
        ("s4_clean_off", False, {"min_component_size": 0, "min_component_size_full": 0, "enforce_single_component": False}, None),
        ("s5_gingiva_off", False, {"gingiva_dilate_iters": 0, "gingiva_protect_conf": 0.98}, None),
    ]
    results: List[CeilingResult] = []
    for variant, use_exact, overrides, pairwise_mode in variants:
        lab10k, lab_full, logs = run_variant(
            variant=variant,
            use_exact=use_exact,
            meta=meta,
            cfg_base=cfg_base,
            pos6_mm=pos6_mm,
            logits6=logits6,
            pos10_mm=pos_mm10k,
            pos_full_mm=pos_full_mm,
            normals6=normals6,
            normals10=normals10k,
            full_adj=full_adj,
            adj10=adj10,
            orig_ids=orig_ids,
            assign_ids=assign_ids,
            assign_indices=assign_indices,
            assign_weights=assign_weights,
            cfg_override=overrides,
            pairwise_mode=pairwise_mode,
        )

        dsc = _compute_dsc(gt_full, lab_full, range(1, num_classes))
        mean_dsc = _summarise_dsc(dsc, range(1, num_classes - 1))
        results.append(
            CeilingResult(
                variant=variant,
                logs=logs,
                lab10k=lab10k,
                lab_full=lab_full,
                dsc_per_class=dsc,
                mean_dsc=mean_dsc,
            )
        )

        if out_root is not None:
            out_root.mkdir(parents=True, exist_ok=True)
            suffix = f"{case_path.stem}_{variant}"
            _apply_colors(mesh10k, lab10k).save(out_root / f"{suffix}_10k.vtp", binary=True)
            _apply_colors(mesh_full, lab_full).save(out_root / f"{suffix}_full.vtp", binary=True)

    return results, gt_full


def print_report(case_path: Path, results: Sequence[CeilingResult]) -> None:
    print(f"\n=== Ceiling analysis for {case_path.name} ===")
    for res in results:
        print(f"\n[{res.variant.upper()}]")
        print(f"  Mean DSC (teeth 1-14): {res.mean_dsc:.4f}")
        gingiva_dsc = res.dsc_per_class.get(15, float('nan'))
        if not np.isnan(gingiva_dsc):
            print(f"  Gingiva DSC: {gingiva_dsc:.4f}")
        top_worst = sorted(
            ((cls, d) for cls, d in res.dsc_per_class.items() if not np.isnan(d) and cls <= 14),
            key=lambda kv: kv[1],
        )[:3]
        if top_worst:
            print("  Worst teeth classes:", ", ".join(f"{cls}:{d:.4f}" for cls, d in top_worst))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 ceiling analysis via GT logits injection.")
    ap.add_argument("--case", type=Path, required=True, help="Path to ground-truth mesh (.vtp).")
    ap.add_argument(
        "--seg-ckpt",
        type=Path,
        default=ROOT / "outputs" / "segmentation" / "final_pt" / "best.pt",
        help="Stage-1 checkpoint (only used for pipeline meta).",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional directory to save recolored meshes.",
    )
    args = ap.parse_args()

    results, _ = run_ceiling(args.case, args.seg_ckpt, args.output_root)
    print_report(args.case, results)


if __name__ == "__main__":
    main()
