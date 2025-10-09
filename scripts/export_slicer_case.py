#!/usr/bin/env python
"""
Export per-tooth NPZ samples into 3D Slicer friendly VTP files.

Usage:
  python scripts/export_slicer_case.py --case 1 --arch L \
      --samples-dir datasets/landmarks_dataset/cooked/p0/samples \
      --landmark-def datasets/landmarks_dataset/cooked/landmark_def.json \
      --out-dir datasets/landmarks_dataset/cooked/p0/reports

Outputs (per tooth, e.g. t31):
  <out_dir>/<case>_<arch>/t31_roi.vtp          ROI points with heatmap scalars
  <out_dir>/<case>_<arch>/t31_landmarks.vtp    Landmark points (if available)
  <out_dir>/<case>_<arch>/t31_meta.json        Metadata snapshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import numpy as np
import pyvista as pv


def _normalise_case(case: str) -> str:
    case = case.strip()
    if case.isdigit():
        return f"{int(case):03d}"
    return case


def _load_landmark_def(path: Path | None) -> tuple[Dict[str, List[str]], Dict[str, any]]:
    if not path or not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    templates = data.get("templates", {})
    per_tooth = data.get("per_tooth", {})
    return {"templates": templates, "per_tooth": per_tooth}, data


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)


def _resolve_landmark_names(spec: Dict[str, any], tooth_label: str, count: int) -> List[str]:
    default = [f"lm_{i:02d}" for i in range(count)]
    if not spec:
        return default
    per_tooth = spec.get("per_tooth", {})
    templates = spec.get("templates", {})

    entry = None
    for key in (tooth_label, tooth_label.lower(), tooth_label.upper()):
        if key in per_tooth:
            entry = per_tooth[key]
            break
    if isinstance(entry, dict):
        order = entry.get("order")
        if isinstance(order, list) and order:
            names = [str(n) for n in order]
            return names[:count] if len(names) >= count else names + default[len(names):count]
        tpl = entry.get("template")
        if isinstance(tpl, str):
            entry = tpl
    if isinstance(entry, str):
        tpl_names = templates.get(entry)
        if isinstance(tpl_names, list) and tpl_names:
            names = [str(n) for n in tpl_names]
            return names[:count] if len(names) >= count else names + default[len(names):count]
    return default


def _make_roi_poly(points: np.ndarray, heatmap: np.ndarray, names: Sequence[str]) -> pv.PolyData:
    poly = pv.PolyData(points)
    heatmap_max = heatmap.max(axis=0)
    poly.point_data["heatmap_max"] = heatmap_max.astype(np.float32)
    poly.set_active_scalars("heatmap_max")
    for idx, channel in enumerate(heatmap):
        name = names[idx] if idx < len(names) else f"heatmap_{idx:02d}"
        poly.point_data[_safe_name(name)] = channel.astype(np.float32)
    return poly


def _make_landmark_poly(landmarks: np.ndarray, names: Sequence[str]) -> pv.PolyData | None:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.size == 0:
        return None
    valid = np.all(np.isfinite(landmarks), axis=1)
    if not valid.any():
        return None
    pts = landmarks[valid]
    poly = pv.PolyData()
    poly.points = pts
    verts = np.column_stack([np.ones((pts.shape[0],), dtype=np.int64), np.arange(pts.shape[0], dtype=np.int64)])
    poly.verts = verts
    label_arr = []
    for idx, flag in enumerate(valid):
        if not flag:
            continue
        name = names[idx] if idx < len(names) else f"lm_{idx:02d}"
        label_arr.append(name)
    poly["label"] = np.array(label_arr, dtype=np.str_)
    poly["index"] = np.arange(len(label_arr), dtype=np.int32)
    return poly


def _write_landmark_fcsv(
    path: Path,
    landmarks: np.ndarray,
    names: Sequence[str],
    *,
    offset: Optional[np.ndarray] = None,
    prefix: Optional[str] = None,
) -> None:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.size == 0:
        return
    valid = np.all(np.isfinite(landmarks), axis=1)
    if not valid.any():
        return
    header = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    lines = []
    for idx, flag in enumerate(valid):
        if not flag:
            continue
        name = names[idx] if idx < len(names) else f"lm_{idx:02d}"
        if prefix:
            name = f"{prefix}_{name}"
        x, y, z = landmarks[idx]
        if offset is not None:
            x, y, z = (np.asarray([x, y, z], dtype=np.float32) + offset).tolist()
        line = [
            f"vtkMRMLMarkupsFiducialNode_{idx}",
            f"{x:.6f}",
            f"{y:.6f}",
            f"{z:.6f}",
            "0", "0", "0", "1",  # orientation quaternion
            "1",  # vis
            "1",  # sel
            "0",  # lock
            name,
            "",
            "",
        ]
        lines.append(",".join(line))
    path.write_text("\n".join(header + lines), encoding="utf-8")


def _write_markups_fcsv_generic(path: Path, points: np.ndarray, labels: Sequence[str]) -> None:
    if points is None or len(points) == 0:
        return
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return
    header = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    lines = []
    for idx, (xyz, label) in enumerate(zip(points, labels)):
        line = [
            f"vtkMRMLMarkupsFiducialNode_{idx}",
            f"{xyz[0]:.6f}",
            f"{xyz[1]:.6f}",
            f"{xyz[2]:.6f}",
            "0", "0", "0", "1",
            "1",
            "1",
            "0",
            str(label),
            "",
            "",
        ]
        lines.append(",".join(line))
    path.write_text("\n".join(header + lines), encoding="utf-8")


def export_tooth(
    npz_path: Path,
    out_dir: Path,
    name_spec: Dict[str, any],
    arch_accum: Dict[str, any],
) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        meta_raw = data["meta"]
        if isinstance(meta_raw, np.ndarray):
            meta = meta_raw.item()
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {"raw_meta": str(meta_raw)}

        points_local = data["pos"].astype(np.float32)
        heatmap = data["y"].astype(np.float32)
        landmarks_local = data.get("landmarks")

    tooth_label = meta.get("tooth_label")
    if not tooth_label:
        fdi = meta.get("fdi")
        tooth_label = f"t{int(fdi)}" if fdi is not None else npz_path.stem.split("_")[-1]

    center = np.asarray(meta.get("center_mm", [0.0, 0.0, 0.0]), dtype=np.float32)

    points_global = points_local + center
    landmarks_global = None
    if landmarks_local is not None:
        landmarks_global = landmarks_local + center

    names = _resolve_landmark_names(name_spec, tooth_label, heatmap.shape[0])

    roi_poly = _make_roi_poly(points_global, heatmap, names)
    roi_path = out_dir / f"{tooth_label}_roi.vtp"
    roi_poly.save(roi_path)

    if landmarks_global is not None:
        lm_poly = _make_landmark_poly(landmarks_global, names)
        if lm_poly is not None:
            lm_path = out_dir / f"{tooth_label}_landmarks.vtp"
            lm_poly.save(lm_path)
        _write_landmark_fcsv(
            out_dir / f"{tooth_label}_landmarks.fcsv",
            landmarks_local,
            names,
            offset=center,
        )

    meta_out = dict(meta)
    meta_out.setdefault("source_npz", npz_path.as_posix())
    (out_dir / f"{tooth_label}_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # accumulate arch-level data
    N = points_global.shape[0]
    arch_accum.setdefault("points", []).append(points_global)
    arch_accum.setdefault("heatmap_max", []).append(heatmap.max(axis=0))
    arch_accum.setdefault("tooth_id", []).append(
        np.full(N, int(meta.get("fdi", 0)), dtype=np.int32)
    )
    arch_accum.setdefault("tooth_label", []).extend([tooth_label] * N)

    if landmarks_global is not None:
        valid = np.all(np.isfinite(landmarks_global), axis=1)
        for idx, flag in enumerate(valid):
            if not flag:
                continue
            label = f"{tooth_label}_{names[idx] if idx < len(names) else f'lm_{idx:02d}'}"
            arch_accum.setdefault("landmark_points", []).append(landmarks_global[idx])
            arch_accum.setdefault("landmark_labels", []).append(label)


def export_arch(case_out: Path, case_id: str, arch: str, arch_accum: Dict[str, any]) -> None:
    points_list = arch_accum.get("points", [])
    if not points_list:
        return
    points = np.concatenate(points_list, axis=0)
    poly = pv.PolyData(points)

    heatmap_max = np.concatenate(arch_accum.get("heatmap_max", []), axis=0)
    poly.point_data["heatmap_max"] = heatmap_max.astype(np.float32)
    poly.set_active_scalars("heatmap_max")

    tooth_ids = np.concatenate(arch_accum.get("tooth_id", []), axis=0)
    poly.point_data["tooth_id"] = tooth_ids.astype(np.int32)

    tooth_labels = np.array(arch_accum.get("tooth_label", []), dtype=np.str_)
    if tooth_labels.size == points.shape[0]:
        poly.point_data["tooth_label"] = tooth_labels

    arch_roi_path = case_out / f"{case_id}_{arch}_roi.vtp"
    poly.save(arch_roi_path)

    lm_labels = arch_accum.get("landmark_labels", [])
    lm_points = arch_accum.get("landmark_points", [])
    if lm_labels and lm_points:
        lm_pts = np.vstack(lm_points)
        lm_poly = pv.PolyData()
        lm_poly.points = lm_pts
        verts = np.column_stack([np.ones((lm_pts.shape[0],), dtype=np.int64), np.arange(lm_pts.shape[0], dtype=np.int64)])
        lm_poly.verts = verts
        lm_poly["label"] = np.array(lm_labels, dtype=np.str_)
        lm_poly["index"] = np.arange(len(lm_labels), dtype=np.int32)
        arch_lm_path = case_out / f"{case_id}_{arch}_landmarks.vtp"
        lm_poly.save(arch_lm_path)
        _write_markups_fcsv_generic(case_out / f"{case_id}_{arch}_landmarks.fcsv", lm_pts, lm_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NPZ samples to VTP for 3D Slicer inspection.")
    parser.add_argument("--case", required=True, help="Case ID (e.g. 1 or 001).")
    parser.add_argument("--arch", required=True, help="Arch (e.g. L or U).")
    parser.add_argument("--samples-dir", default="datasets/landmarks_dataset/cooked/p0/samples", help="Directory with *.npz.")
    parser.add_argument("--landmark-def", default=None, help="Path to landmark_def.json for naming.")
    parser.add_argument("--out-dir", default="datasets/landmarks_dataset/cooked/p0/reports", help="Output root directory.")
    args = parser.parse_args()

    case_id = _normalise_case(args.case)
    arch = args.arch.upper()
    samples_dir = Path(args.samples_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    spec, _ = _load_landmark_def(Path(args.landmark_def) if args.landmark_def else None)

    pattern = f"{case_id}_{arch}_t*.npz"
    npz_files = sorted(samples_dir.glob(pattern))
    if not npz_files:
        raise FileNotFoundError(f"No samples found for case {case_id} arch {arch} under {samples_dir}")

    case_out = out_root / f"{case_id}_{arch}"
    case_out.mkdir(parents=True, exist_ok=True)
    arch_accum: Dict[str, any] = {}

    for npz_path in npz_files:
        export_tooth(npz_path, case_out, spec, arch_accum)
    export_arch(case_out, case_id, arch, arch_accum)
    print(f"Exported {len(npz_files)} teeth to {case_out}")


if __name__ == "__main__":
    main()
