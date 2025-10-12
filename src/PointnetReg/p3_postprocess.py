"""Post-process PointNetReg inference outputs into case-level aggregates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

POINTNETREG_ROOT = Path("outputs/pointnetreg")
DEFAULT_INFER_ROOT = POINTNETREG_ROOT / "infer"
DEFAULT_POST_ROOT = POINTNETREG_ROOT / "postprocess"


VALID_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


@dataclass
class LandmarkResult:
    name: str
    coord: Optional[np.ndarray]
    score: Optional[float]
    margin: Optional[float]
    flags: Dict[str, bool]


@dataclass
class ToothResult:
    tooth_id: str
    landmarks: Dict[str, LandmarkResult]
    flags: Dict[str, Any]


@dataclass
class CaseAggregate:
    case_id: str
    meta: Dict[str, Any]
    teeth: Dict[str, ToothResult]
    stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_landmark_def(path: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    data = load_json(path)
    if not data:
        raise FileNotFoundError(f"Cannot read landmark definition: {path}")
    templates = data.get("templates", {})
    per_tooth = data.get("per_tooth", {})
    mapping: Dict[str, List[str]] = {}
    for tooth, tpl in per_tooth.items():
        names = templates.get(tpl, [])
        mapping[tooth.lower()] = list(names)
        mapping[tooth.upper()] = list(names)
    ordered = [tid for tid in VALID_TOOTH_IDS if tid in mapping]
    if not ordered:
        ordered = sorted({t.lower() for t in mapping})
    return mapping, ordered


def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if key in base:
            if isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value
        else:
            base[key] = value
    return base


def detect_case_id(path: Path, data: Dict[str, Any]) -> str:
    for key in ("case_id", "case", "id", "name"):
        if key in data and data[key]:
            return str(data[key])
    meta = data.get("meta")
    if isinstance(meta, dict):
        for key in ("case_id", "case", "id", "name", "basename"):
            if key in meta and meta[key]:
                return str(meta[key])
    return path.stem


def ensure_iterable_dict(obj: Any, names: List[str]) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {str(k): obj[k] for k in obj}
    if isinstance(obj, list) or isinstance(obj, tuple):
        res: Dict[str, Any] = {}
        for idx, name in enumerate(names):
            if idx < len(obj):
                res[name] = obj[idx]
        return res
    return {}


def to_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size != 3:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32)


def to_float_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return float(val)


def get_offset(tooth_meta: Dict[str, Any], case_meta: Dict[str, Any]) -> Optional[np.ndarray]:
    candidates = [tooth_meta, case_meta]
    keys = ("origin_mm", "offset_mm", "center_mm", "origin", "offset", "center", "shift")
    for container in candidates:
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key in container:
                arr = to_float_array(container[key])
                if arr is not None:
                    return arr
    return None


def get_bounds(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(meta, dict):
        return None
    for key in ("bounds_mm", "bbox_mm", "bounds", "bbox"):
        if key in meta:
            raw = np.asarray(meta[key], dtype=np.float64).reshape(-1)
            if raw.size == 6 and np.all(np.isfinite(raw)):
                bounds = raw.astype(np.float32)
                return bounds
    return None


def normalise_scalar_map(container: Dict[str, Any], keys: Iterable[str], names: List[str]) -> Dict[str, Optional[float]]:
    for key in keys:
        if key in container:
            raw_map = ensure_iterable_dict(container[key], names)
            return {name: to_float_scalar(raw_map.get(name)) for name in names}
    metrics = container.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics:
                raw_map = ensure_iterable_dict(metrics[key], names)
                return {name: to_float_scalar(raw_map.get(name)) for name in names}
    return {name: None for name in names}


def detect_secondary_scores(container: Dict[str, Any], names: List[str]) -> Dict[str, Optional[float]]:
    return normalise_scalar_map(container, ("second_scores", "top2", "top2_values", "score_top2"), names)


def detect_primary_scores(container: Dict[str, Any], names: List[str]) -> Dict[str, Optional[float]]:
    return normalise_scalar_map(container, ("scores", "score", "peak_scores", "max_values", "top1", "prob", "prob_top1", "heatmap_peaks"), names)


def detect_margin_scores(container: Dict[str, Any], names: List[str]) -> Dict[str, Optional[float]]:
    return normalise_scalar_map(container, ("margins", "margin", "score_margin", "top1_minus_top2", "delta"), names)


def build_landmark_results(
    tooth_id: str,
    tooth_data: Dict[str, Any],
    names: List[str],
    tooth_meta: Dict[str, Any],
    case_meta: Dict[str, Any],
    bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, LandmarkResult], Dict[str, Any]]:
    landmarks: Dict[str, LandmarkResult] = {}
    global_map = ensure_iterable_dict(tooth_data.get("landmarks_global") or tooth_data.get("global"), names)
    local_map = ensure_iterable_dict(tooth_data.get("landmarks_local") or tooth_data.get("local"), names)
    primary_scores = detect_primary_scores(tooth_data, names)
    margins_raw = detect_margin_scores(tooth_data, names)
    secondary_scores = detect_secondary_scores(tooth_data, names)

    offset = get_offset(tooth_data.get("meta", {}), case_meta)
    if offset is None:
        offset = get_offset(tooth_data, case_meta)

    missing = []
    nans = []
    oob = []
    converted_from_local = []

    total_present = 0

    for name in names:
        coord = None
        flags = {
            "from_local": False,
            "missing": False,
            "nan": False,
            "out_of_bounds": False,
        }

        raw_global = global_map.get(name)
        if raw_global is not None:
            coord = to_float_array(raw_global)
        if coord is None and name in local_map:
            local_raw = to_float_array(local_map.get(name))
            if local_raw is not None:
                if offset is not None:
                    coord = local_raw + offset
                    flags["from_local"] = True
                else:
                    coord = local_raw
                    flags["from_local"] = True
        if coord is None:
            flags["missing"] = True
            missing.append(name)
        else:
            if not np.all(np.isfinite(coord)):
                flags["nan"] = True
                nans.append(name)
            else:
                total_present += 1
                if bounds is not None:
                    xmin, xmax, ymin, ymax, zmin, zmax = bounds.tolist()
                    if not (xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax and zmin <= coord[2] <= zmax):
                        flags["out_of_bounds"] = True
                        oob.append(name)
                if flags["from_local"]:
                    converted_from_local.append(name)

        score = primary_scores.get(name)
        margin_val = margins_raw.get(name)
        if margin_val is None and score is not None:
            second = secondary_scores.get(name)
            if second is not None:
                margin_val = score - second

        landmarks[name] = LandmarkResult(name=name, coord=coord, score=score, margin=margin_val, flags=flags)

    tooth_flags: Dict[str, Any] = {
        "missing_landmarks": missing,
        "nan_landmarks": nans,
        "out_of_bounds": oob,
        "converted_from_local": converted_from_local,
        "total_expected": len(names),
        "total_present": total_present,
    }

    return landmarks, tooth_flags


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------


def collect_inference_records(infer_root: Path) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    if not infer_root.exists():
        raise FileNotFoundError(f"Inference directory not found: {infer_root}")

    for json_path in infer_root.rglob("*.json"):
        data = load_json(json_path)
        if not isinstance(data, dict):
            continue
        case_id = detect_case_id(json_path, data)
        case_entry = cases.setdefault(case_id, {"meta": {}, "teeth": {}, "sources": []})
        case_entry["sources"].append(str(json_path))
        if "meta" in data and isinstance(data["meta"], dict):
            merge_dict(case_entry["meta"], data["meta"])

        if "predictions" in data and isinstance(data["predictions"], dict):
            for tooth_id, tooth_data in data["predictions"].items():
                tooth_entry = case_entry["teeth"].setdefault(tooth_id, {})
                if isinstance(tooth_data, dict):
                    merge_dict(tooth_entry, tooth_data)
            continue

        tooth_id = data.get("tooth") or data.get("tooth_id")
        if tooth_id:
            tooth_entry = case_entry["teeth"].setdefault(str(tooth_id), {})
            merge_dict(tooth_entry, data)

    return cases


def aggregate_case(case_id: str, case_data: Dict[str, Any], defs: Dict[str, List[str]], order: List[str]) -> CaseAggregate:
    case_meta = case_data.get("meta", {})
    bounds = get_bounds(case_meta)
    teeth_results: Dict[str, ToothResult] = {}

    total_expected = 0
    total_present = 0
    missing_teeth: List[str] = []

    for tooth_id in order:
        names = defs.get(tooth_id) or defs.get(tooth_id.upper()) or defs.get(tooth_id.lower())
        if not names:
            continue
        tooth_key_variants = [tooth_id, tooth_id.upper(), tooth_id.lower()]
        tooth_record: Optional[Dict[str, Any]] = None
        for variant in tooth_key_variants:
            if variant in case_data.get("teeth", {}):
                tooth_record = case_data["teeth"][variant]
                break
        if tooth_record is None:
            missing_teeth.append(tooth_id)
            landmarks = {
                name: LandmarkResult(
                    name=name,
                    coord=None,
                    score=None,
                    margin=None,
                    flags={"missing": True, "from_local": False, "nan": False, "out_of_bounds": False},
                )
                for name in names
            }
            tooth_flags = {
                "missing_tooth": True,
                "missing_landmarks": names,
                "nan_landmarks": [],
                "out_of_bounds": [],
                "converted_from_local": [],
                "total_expected": len(names),
                "total_present": 0,
            }
        else:
            tooth_meta = {}
            if isinstance(tooth_record.get("meta"), dict):
                tooth_meta = tooth_record.get("meta", {})
            landmarks_map, tooth_flags = build_landmark_results(
                tooth_id,
                tooth_record,
                names,
                tooth_meta,
                case_meta,
                bounds,
            )
            tooth_flags["missing_tooth"] = False
            if tooth_meta:
                if "arch" in tooth_meta and tooth_meta["arch"]:
                    tooth_flags.setdefault("arch", tooth_meta["arch"])
                if "fdi" in tooth_meta:
                    tooth_flags.setdefault("fdi", tooth_meta["fdi"])
                tooth_flags.setdefault("meta", tooth_meta)
            landmarks = landmarks_map

        total_expected += len(names)
        total_present += tooth_flags.get("total_present", 0)

        teeth_results[tooth_id] = ToothResult(tooth_id=tooth_id, landmarks=landmarks, flags=tooth_flags)

    stats = {
        "total_expected": total_expected,
        "total_present": total_present,
        "coverage": float(total_present) / float(total_expected) if total_expected else 0.0,
        "missing_teeth": missing_teeth,
    }

    return CaseAggregate(case_id=case_id, meta=case_meta, teeth=teeth_results, stats=stats)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _group_landmark_control_points(case: CaseAggregate, include_all: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if include_all:
        groups["ALL"] = []
    for tooth_id, tooth in case.teeth.items():
        arch_flag = ""
        if isinstance(tooth.flags, dict):
            arch_value = tooth.flags.get("arch")
            if arch_value:
                arch_flag = str(arch_value).upper()
        fdi = tooth_id[1:] if tooth_id.lower().startswith("t") and len(tooth_id) > 1 else tooth_id
        for name, landmark in tooth.landmarks.items():
            if landmark.coord is None or landmark.flags.get("nan", False):
                continue
            point = {
                "label": f"{fdi}{name}",
                "position": landmark.coord.tolist(),
            }
            if include_all:
                groups.setdefault("ALL", []).append(point.copy())
            if arch_flag in {"U", "L"}:
                groups.setdefault(arch_flag, []).append(point.copy())
    return groups


def export_case_json(case: CaseAggregate, out_dir: Path) -> List[Path]:
    ORIENTATION = [-1.0, 0.0, 0.0,
                   0.0, -1.0, 0.0,
                   0.0, 0.0, 1.0]
    DISPLAY_BLOCK = {
        "visibility": True,
        "opacity": 1.0,
        "color": [0.4, 1.0, 1.0],
        "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
        "activeColor": [0.4, 1.0, 0.0],
        "propertiesLabelVisibility": False,
        "pointLabelsVisibility": True,
        "textScale": 3.0,
        "glyphType": "Sphere3D",
        "glyphScale": 1.0,
        "glyphSize": 5.0,
        "useGlyphScale": True,
        "sliceProjection": False,
        "sliceProjectionUseFiducialColor": True,
        "sliceProjectionOutlinedBehindSlicePlane": False,
        "sliceProjectionColor": [1.0, 1.0, 1.0],
        "sliceProjectionOpacity": 0.6,
        "lineThickness": 0.2,
        "lineColorFadingStart": 1.0,
        "lineColorFadingEnd": 10.0,
        "lineColorFadingSaturation": 1.0,
        "lineColorFadingHueOffset": 0.0,
        "handlesInteractive": False,
        "translationHandleVisibility": True,
        "rotationHandleVisibility": True,
        "scaleHandleVisibility": False,
        "interactionHandleScale": 3.0,
        "snapMode": "toVisibleSurface",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_landmark_control_points(case, include_all=True)
    out_paths: List[Path] = []
    for key, points in grouped.items():
        if not points:
            continue
        control_points = [
            {
                "id": str(idx + 1),
                "label": entry["label"],
                "description": "",
                "associatedNodeID": "",
                "position": entry["position"],
                "orientation": ORIENTATION,
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
            for idx, entry in enumerate(points)
        ]
        payload = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "controlPoints": control_points,
                    "display": DISPLAY_BLOCK,
                }
            ],
        }
        filename = f"{case.case_id}.json" if key == "ALL" else f"{case.case_id}_{key}.json"
        out_path = out_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        out_paths.append(out_path)
    return out_paths


def export_case_csv(case: CaseAggregate, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case.case_id}.csv"
    header = [
        "case_id", "tooth_id", "landmark", "x", "y", "z",
        "score", "margin", "missing_tooth", "missing", "nan", "out_of_bounds", "from_local",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for tooth_id, tooth in case.teeth.items():
            for name, landmark in tooth.landmarks.items():
                coord = landmark.coord.tolist() if landmark.coord is not None else [None, None, None]
                writer.writerow([
                    case.case_id,
                    tooth_id,
                    name,
                    coord[0],
                    coord[1],
                    coord[2],
                    landmark.score,
                    landmark.margin,
                    tooth.flags.get("missing_tooth", False),
                    landmark.flags.get("missing", False),
                    landmark.flags.get("nan", False),
                    landmark.flags.get("out_of_bounds", False),
                    landmark.flags.get("from_local", False),
                ])
    return out_path


def export_case_ply(case: CaseAggregate, out_dir: Path) -> Optional[Path]:
    points = []
    labels = []
    for tooth_id, tooth in case.teeth.items():
        for name, landmark in tooth.landmarks.items():
            if landmark.coord is not None and not landmark.flags.get("nan", False):
                points.append(landmark.coord)
                labels.append(f"{tooth_id}_{name}")
    if not points:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case.case_id}.ply"
    pts = np.vstack(points).astype(np.float32)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for point in pts:
            fh.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    return out_path


def export_case_mrk(case: CaseAggregate, out_dir: Path, split_by_arch: bool = True) -> List[Path]:
    ORIENTATION = [-1.0, 0.0, 0.0,
                   0.0, -1.0, 0.0,
                   0.0, 0.0, 1.0]
    DISPLAY_BLOCK = {
        "visibility": True,
        "opacity": 1.0,
        "color": [0.4, 1.0, 1.0],
        "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
        "activeColor": [0.4, 1.0, 0.0],
        "propertiesLabelVisibility": False,
        "pointLabelsVisibility": True,
        "textScale": 3.0,
        "glyphType": "Sphere3D",
        "glyphScale": 1.0,
        "glyphSize": 5.0,
        "useGlyphScale": True,
        "sliceProjection": False,
        "sliceProjectionUseFiducialColor": True,
        "sliceProjectionOutlinedBehindSlicePlane": False,
        "sliceProjectionColor": [1.0, 1.0, 1.0],
        "sliceProjectionOpacity": 0.6,
        "lineThickness": 0.2,
        "lineColorFadingStart": 1.0,
        "lineColorFadingEnd": 10.0,
        "lineColorFadingSaturation": 1.0,
        "lineColorFadingHueOffset": 0.0,
        "handlesInteractive": False,
        "translationHandleVisibility": True,
        "rotationHandleVisibility": True,
        "scaleHandleVisibility": False,
        "interactionHandleScale": 3.0,
        "snapMode": "toVisibleSurface",
    }

    grouped = _group_landmark_control_points(case, include_all=True)
    if split_by_arch:
        groups = {k: v for k, v in grouped.items() if k == "ALL" or k in {"U", "L"}}
    else:
        groups = {"ALL": grouped.get("ALL", [])}

    out_paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(groups.values()):
        return out_paths

    for key, pts in groups.items():
        if not pts:
            continue
        control_points = [
            {
                "id": str(idx + 1),
                "label": entry["label"],
                "description": "",
                "associatedNodeID": "",
                "position": entry["position"],
                "orientation": ORIENTATION,
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
            for idx, entry in enumerate(pts)
        ]
        markup = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "controlPoints": control_points,
                    "display": DISPLAY_BLOCK,
                }
            ],
        }
        if key == "ALL":
            filename = f"{case.case_id}.mrk.json"
        else:
            filename = f"{case.case_id}_{key}.json"
        out_path = out_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(markup, fh, ensure_ascii=False, indent=2)
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PointNetReg inference outputs into case-level files.")
    parser.add_argument("--infer-root", type=Path, default=DEFAULT_INFER_ROOT)
    parser.add_argument("--landmark-def", type=Path, default=Path("datasets/landmarks_dataset/cooked/landmark_def.json"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_POST_ROOT)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--export-ply", action="store_true")
    parser.add_argument("--export-mrk", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--cases", type=str, default=None, help="Optional comma-separated list of case IDs to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defs, order = load_landmark_def(args.landmark_def)
    records = collect_inference_records(args.infer_root)

    selected_cases: Iterable[str]
    if args.cases:
        selected_set = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_cases = [cid for cid in records if cid in selected_set]
    else:
        selected_cases = sorted(records.keys())

    out_dir = args.out_dir
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"
    out_ply_dir = out_dir / "ply"
    out_mrk_dir = out_dir / "markups"

    summaries: List[str] = []

    if not selected_cases:
        print("No cases found to process.")
        return

    for case_id in selected_cases:
        case_data = records[case_id]
        case = aggregate_case(case_id, case_data, defs, order)
        export_case_json(case, out_json_dir)
        if args.export_csv:
            export_case_csv(case, out_csv_dir)
        if args.export_ply:
            export_case_ply(case, out_ply_dir)
        if args.export_mrk:
            export_case_mrk(case, out_mrk_dir)

        stat = case.stats
        summary = (
            f"Case {case.case_id}: {stat['total_present']}/{stat['total_expected']} landmarks "
            f"({stat['coverage']*100:.1f}% coverage). Missing teeth: {stat['missing_teeth']}"
        )
        summaries.append(summary)
        print(summary)

    if args.log_path:
        log_path = args.log_path
    else:
        log_path = out_dir / "postprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        for line in summaries:
            fh.write(line + "\n")
        fh.write(f"Processed {len(summaries)} cases.\n")


if __name__ == "__main__":
    main()
