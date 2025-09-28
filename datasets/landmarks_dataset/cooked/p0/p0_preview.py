#!/usr/bin/env python3
"""Preview utility for a single p0 tooth sample heatmap channel."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv


def load_landmark_def(root: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    candidate = root / "../landmark_def.json"
    if not candidate.exists():
        candidate = root / "landmark_def.json"
    with candidate.resolve().open("r", encoding="utf-8") as fh:
        lm_def = json.load(fh)
    return lm_def.get("templates", {}), lm_def.get("per_tooth", {})


def resolve_sample_path(root: Path, sample_arg: str) -> Path:
    path = Path(sample_arg)
    if path.is_file():
        return path
    if not path.suffix:
        path = path.with_suffix(".npz")
    candidate = root / "samples" / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot find sample {sample_arg}")


def select_channel(mask: np.ndarray, names: List[str], target: str) -> Tuple[int, str]:
    valid_idx = np.where(mask > 0.5)[0]
    labelled = []
    for order, idx in enumerate(valid_idx):
        name = names[order] if order < len(names) else f"idx{idx}"
        labelled.append((idx, name))
    if not labelled:
        raise ValueError("Sample has no valid landmarks (mask empty)")

    if target is None:
        return labelled[0]

    target_lower = target.lower()
    # Try exact name match.
    for idx, name in labelled:
        if name.lower() == target_lower:
            return idx, name

    # Try interpreting as integer order or index.
    try:
        value = int(target)
    except ValueError as exc:
        raise ValueError(f"Target '{target}' not recognised") from exc

    if 0 <= value < len(labelled):
        return labelled[value]

    for idx, name in labelled:
        if idx == value:
            return idx, name

    raise ValueError(f"Target '{target}' does not match any channel")


def main():
    parser = argparse.ArgumentParser(description="Export heatmap preview for a single landmark channel")
    parser.add_argument("--sample", required=True, help="Sample file name or path")
    parser.add_argument("--target", default=None, help="Landmark name or index (defaults to first valid)")
    parser.add_argument("--output", default=None, help="Output VTP file path (default: reports/<sample>_<target>.vtp)")
    parser.add_argument("--sphere-radius", type=float, default=0.4, help="Radius (mm) for landmark spheres")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sample_path = resolve_sample_path(root, args.sample)

    data = np.load(sample_path, allow_pickle=True)
    pos = data["pos"]
    y = data["y"]
    mask = data["loss_mask"]
    landmarks = data["landmarks"]
    meta = data.get("meta")
    meta_dict = meta.item() if isinstance(meta, np.ndarray) and meta.dtype == object else {}

    templates, per_tooth = load_landmark_def(root)

    names: List[str] = []
    tooth_key = None
    if meta_dict:
        tooth_key = f"t{meta_dict.get('fdi')}"
        template_key = per_tooth.get(tooth_key)
        if template_key:
            names = templates.get(template_key, [])

    channel_idx, channel_name = select_channel(mask, names, args.target if args.target is None else str(args.target))

    heat = y[channel_idx]
    peak_idx = int(np.argmax(heat))
    peak_val = float(heat[peak_idx])
    pred_point = pos[peak_idx]
    gt_point = landmarks[channel_idx]
    gt_valid = np.all(np.isfinite(gt_point))
    distance = float(np.linalg.norm(pred_point - gt_point)) if gt_valid else None

    cloud = pv.PolyData(pos)
    cloud["heat"] = heat.astype(np.float32)
    denom = float(heat.max()) if heat.size else 1.0
    if denom <= 0.0:
        denom = 1.0
    cloud["heat_norm"] = (heat / denom).astype(np.float32)
    cloud["category"] = np.zeros(cloud.n_points, dtype=np.uint8)

    extras = []
    if gt_valid:
        sphere_gt = pv.Sphere(radius=args.sphere_radius, center=gt_point)
        sphere_gt["heat"] = np.full(sphere_gt.n_points, denom, dtype=np.float32)
        sphere_gt["heat_norm"] = np.ones(sphere_gt.n_points, dtype=np.float32)
        sphere_gt["category"] = np.full(sphere_gt.n_points, 1, dtype=np.uint8)
        extras.append(sphere_gt)

    sphere_pred = pv.Sphere(radius=args.sphere_radius * 0.9, center=pred_point)
    sphere_pred["heat"] = np.full(sphere_pred.n_points, peak_val, dtype=np.float32)
    sphere_pred["heat_norm"] = np.ones(sphere_pred.n_points, dtype=np.float32)
    sphere_pred["category"] = np.full(sphere_pred.n_points, 2, dtype=np.uint8)
    extras.append(sphere_pred)

    combined = cloud.copy()
    for extra in extras:
        combined = combined.merge(extra, merge_points=False)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        target_tag = channel_name.replace("/", "-") if channel_name else f"idx{channel_idx}"
        out_path = reports_dir / f"{sample_path.stem}_{target_tag}_preview.vtp"
    if out_path.suffix.lower() != ".vtp":
        out_path = out_path.with_suffix(".vtp")

    combined.save(str(out_path), binary=True)

    print(f"Sample: {sample_path.name}")
    print(f"Channel: {channel_idx} ({channel_name})")
    print(f"Peak heat value: {peak_val:.6f} at point #{peak_idx}")
    if gt_valid and distance is not None:
        print(f"Distance to GT: {distance:.4f} mm")
    else:
        print("Ground-truth landmark is invalid (NaN)")
    print(f"VTP written to: {out_path}")


if __name__ == "__main__":
    main()
