#!/usr/bin/env python3
"""Sanity checks for p0 tooth landmark samples."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

THRESHOLDS_MM: Tuple[float, ...] = (1.0, 2.0, 3.0)


def compute_distance_stats(distances: List[float]) -> Dict[str, float]:
    if not distances:
        return {"mean": None, "median": None, "max": None}
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def compute_pck(distances: List[float], thresholds: Iterable[float]) -> Dict[str, float]:
    if not distances:
        return {f"pck@{thr}mm": None for thr in thresholds}
    arr = np.asarray(distances, dtype=np.float64)
    out: Dict[str, float] = {}
    for thr in thresholds:
        out[f"pck@{thr}mm"] = float((arr <= thr).mean())
    return out


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def to_serialisable(value):
    if value is None:
        return None
    if isinstance(value, (float, int, bool, str)):
        return value
    if isinstance(value, np.generic):
        v = value.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    if isinstance(value, dict):
        return {k: to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serialisable(v) for v in value]
    return value


def analyse_sample(npz_path: Path, templates: Dict[str, List[str]], per_tooth: Dict[str, str]):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"]
    pos = data["pos"]
    y = data["y"]
    mask = data["loss_mask"]
    landmarks = data["landmarks"]
    sample_indices = data.get("sample_indices")
    meta = data.get("meta")
    meta_dict = meta.item() if isinstance(meta, np.ndarray) and meta.dtype == object else {}

    valid_idx = np.where(mask > 0.5)[0]
    names: List[str] = []
    if meta_dict:
        tooth_key = f"t{meta_dict.get('fdi')}"
        template_key = per_tooth.get(tooth_key)
        if template_key:
            names = templates.get(template_key, [])
    points = int(pos.shape[0])
    shapes_ok = x.shape[0] == pos.shape[0] == y.shape[1]

    per_landmark = []
    distances: List[float] = []
    peak_values: List[float] = []
    missing_preds = 0

    for order, idx in enumerate(valid_idx):
        lm_name = names[order] if order < len(names) else f"idx{idx}"
        heat = y[idx]
        peak_i = int(np.argmax(heat))
        peak_val = float(heat[peak_i])
        peak_values.append(peak_val)
        lm_coord = landmarks[idx]
        record = {
            "channel": int(idx),
            "name": lm_name,
            "peak": peak_val,
            "pred_point_index": peak_i,
        }
        if not np.all(np.isfinite(lm_coord)):
            record["status"] = "invalid_gt"
            record["distance"] = None
            missing_preds += 1
            per_landmark.append(record)
            continue
        if peak_val <= 0.0:
            record["status"] = "no_response"
            record["distance"] = None
            missing_preds += 1
            per_landmark.append(record)
            continue
        pred = pos[peak_i]
        dist = float(np.linalg.norm(pred - lm_coord))
        record["status"] = "ok"
        record["distance"] = dist
        distances.append(dist)
        per_landmark.append(record)

    dist_stats = compute_distance_stats(distances)
    pck_stats = compute_pck(distances, THRESHOLDS_MM)

    center_pos = float(np.linalg.norm(pos.mean(axis=0)))
    center_lm = None
    if valid_idx.size:
        center_lm = float(np.linalg.norm(np.nanmean(landmarks[valid_idx], axis=0)))

    per_sample = {
        "sample": npz_path.name,
        "case_id": meta_dict.get("case_id"),
        "arch": meta_dict.get("arch"),
        "fdi": meta_dict.get("fdi"),
        "tooth_id": meta_dict.get("tooth_id"),
        "points": points,
        "features_shape": list(map(int, x.shape)),
        "pos_shape": list(map(int, pos.shape)),
        "heat_shape": list(map(int, y.shape)),
        "landmarks_shape": list(map(int, landmarks.shape)),
        "valid_landmarks": int(len(valid_idx)),
        "shapes_ok": bool(shapes_ok),
        "sample_indices_unique": bool(sample_indices is not None and np.unique(sample_indices).size == sample_indices.size) if sample_indices is not None else None,
        "nan_in_x": bool(np.isnan(x).any()),
        "nan_in_pos": bool(np.isnan(pos).any()),
        "nan_in_y": bool(np.isnan(y).any()),
        "nan_in_landmarks": bool(np.isnan(landmarks[valid_idx]).any()) if len(valid_idx) else False,
        "mask_binary": bool(np.all((mask < 1e-6) | (np.abs(mask - 1.0) < 1e-6))),
        "heat_min": float(np.nanmin(y)),
        "heat_max": float(np.nanmax(y)),
        "heat_mean": float(np.nanmean(y)),
        "pos_center_norm": center_pos,
        "landmark_center_norm": center_lm,
        "distance_mean": dist_stats["mean"],
        "distance_median": dist_stats["median"],
        "distance_max": dist_stats["max"],
        "missing_preds": int(missing_preds),
        "pck@1mm": pck_stats["pck@1.0mm"],
        "pck@2mm": pck_stats["pck@2.0mm"],
        "pck@3mm": pck_stats["pck@3.0mm"],
        "landmark_checks": per_landmark,
    }

    return per_sample, distances, peak_values, missing_preds, len(valid_idx)


def main():
    parser = argparse.ArgumentParser(description="Run sanity checks on p0 samples.")
    parser.add_argument("--samples-dir", default=None, help="Directory containing *.npz samples (default: ./samples)")
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern for samples")
    parser.add_argument("--report-dir", default=None, help="Where to write reports (default: ./reports)")
    parser.add_argument("--report-stem", default="p0_sanity", help="Output file stem")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    samples_dir = Path(args.samples_dir) if args.samples_dir else root / "samples"
    report_dir = Path(args.report_dir) if args.report_dir else root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    landmark_def_path = root / "../landmark_def.json"
    if not landmark_def_path.exists():
        landmark_def_path = root / "landmark_def.json"
    with landmark_def_path.resolve().open("r", encoding="utf-8") as fh:
        lm_def = json.load(fh)
    templates = lm_def.get("templates", {})
    per_tooth = lm_def.get("per_tooth", {})

    sample_paths = sorted(samples_dir.glob(args.pattern))
    if not sample_paths:
        raise SystemExit(f"No samples found in {samples_dir} with pattern {args.pattern}")

    per_samples = []
    all_distances: List[float] = []
    all_peaks: List[float] = []
    total_missing = 0
    total_valid = 0

    for npz_path in sample_paths:
        sample_metrics, dists, peaks, missing, valid_count = analyse_sample(npz_path, templates, per_tooth)
        per_samples.append(sample_metrics)
        all_distances.extend(dists)
        all_peaks.extend(peaks)
        total_missing += missing
        total_valid += valid_count

    dist_summary = compute_distance_stats(all_distances)
    pck_summary = compute_pck(all_distances, THRESHOLDS_MM)
    peak_summary = summarise([p for p in all_peaks if np.isfinite(p)])

    summary = {
        "num_samples": len(per_samples),
        "total_valid_landmarks": int(total_valid),
        "distance_mean": dist_summary["mean"],
        "distance_median": dist_summary["median"],
        "distance_max": dist_summary["max"],
        "pck@1mm": pck_summary["pck@1.0mm"],
        "pck@2mm": pck_summary["pck@2.0mm"],
        "pck@3mm": pck_summary["pck@3.0mm"],
        "peak_min": peak_summary["min"],
        "peak_max": peak_summary["max"],
        "peak_mean": peak_summary["mean"],
        "missing_predictions": int(total_missing),
        "reports": {},
    }

    stem = args.report_stem
    json_path = report_dir / f"{stem}.json"
    csv_path = report_dir / f"{stem}.csv"
    summary["reports"] = {
        "json": json_path.name,
        "csv": csv_path.name,
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"summary": to_serialisable(summary), "samples": to_serialisable(per_samples)}, fh, indent=2, ensure_ascii=False)

    fieldnames = [
        "sample",
        "case_id",
        "arch",
        "fdi",
        "tooth_id",
        "points",
        "valid_landmarks",
        "distance_mean",
        "distance_median",
        "distance_max",
        "pck@1mm",
        "pck@2mm",
        "pck@3mm",
        "heat_min",
        "heat_max",
        "heat_mean",
        "pos_center_norm",
        "landmark_center_norm",
        "missing_preds",
        "nan_in_x",
        "nan_in_pos",
        "nan_in_y",
        "nan_in_landmarks",
        "shapes_ok",
        "sample_indices_unique",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_samples:
            writer.writerow({key: to_serialisable(row.get(key)) for key in fieldnames})

    print(f"Wrote summary JSON to {json_path}")
    print(f"Wrote per-sample CSV to {csv_path}")


if __name__ == "__main__":
    main()
