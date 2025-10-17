#!/usr/bin/env python3
"""
Batch pipeline:
1. Run MeshSegNet (step6 post-processing with pygco) on selected cases.
2. Materialise predicted VTPs under a staging raw directory.
3. Invoke PointNet-Reg p5_generate on those cases.
4. Compare landmark predictions with ground-truth markups to compute metrics.
5. Emit a markdown report with aggregated results.

This script mirrors the logic requested in the Codex task and is intended
for ad-hoc evaluation. It depends on vedo, torch, numpy and pygco (optional).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

try:
    from pygco import cut_from_graph  # type: ignore
except Exception as exc:  # pragma: no cover - fallback only triggered when pygco missing
    import warnings

    warnings.warn(f"pygco unavailable ({exc}); falling back to unary argmin.")

    def cut_from_graph(edges, unaries, pairwise):
        return np.argmin(unaries, axis=1).astype(np.int32)

import vedo  # type: ignore

# Add repo paths
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Ensure MeshSegNet & PointNetReg are importable
import sys

if str(SRC_ROOT / "MeshSegNet" / "models") not in sys.path:
    sys.path.append(str(SRC_ROOT / "MeshSegNet" / "models"))
if str(SRC_ROOT / "PointnetReg") not in sys.path:
    sys.path.append(str(SRC_ROOT / "PointnetReg"))

from meshsegnet import MeshSegNet  # type: ignore
from p5_generate import main as pointnet_generate_main  # type: ignore

LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (160, 160, 160),
    1: (255, 69, 0),
    2: (255, 165, 0),
    3: (255, 215, 0),
    4: (154, 205, 50),
    5: (34, 139, 34),
    6: (46, 139, 87),
    7: (72, 209, 204),
    8: (70, 130, 180),
    9: (65, 105, 225),
    10: (138, 43, 226),
    11: (199, 21, 133),
    12: (255, 105, 180),
    13: (205, 92, 92),
    14: (255, 140, 0),
    15: (255, 228, 196),
}
DEFAULT_COLOR = np.array([90, 90, 90], dtype=np.uint8)

MODEL_FILES = {
    "U": "MeshSegNet_Max_15_classes_72samples_lr1e-2_best.pth",
    "L": "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.pth",
}


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab in np.unique(labels):
        rgb[labels == lab] = np.asarray(LABEL_COLORS.get(int(lab), DEFAULT_COLOR), dtype=np.uint8)
    return rgb


def attach_label_data(mesh_obj, labels: np.ndarray):
    """Attach predicted labels & colors to a vedo mesh (cells + points)."""
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    colors = labels_to_rgb(labels)
    mesh_obj.celldata["Label"] = labels
    mesh_obj.celldata["PredictedID"] = labels
    mesh_obj.celldata["RGB"] = colors

    n_points = mesh_obj.npoints
    point_labels = np.zeros(n_points, dtype=np.int32)
    point_colors = np.zeros((n_points, 3), dtype=np.uint8)

    faces = np.asarray(mesh_obj.cells, dtype=np.int32)
    adjacency = [[] for _ in range(n_points)]
    for cid, face in enumerate(faces):
        for pid in face:
            adjacency[pid].append(cid)

    for pid, cid_list in enumerate(adjacency):
        if not cid_list:
            continue
        labs = labels[cid_list]
        uniq, cnt = np.unique(labs, return_counts=True)
        lab = int(uniq[np.argmax(cnt)])
        point_labels[pid] = lab
        point_colors[pid] = np.asarray(LABEL_COLORS.get(lab, DEFAULT_COLOR), dtype=np.uint8)

    mesh_obj.pointdata["Label"] = point_labels
    mesh_obj.pointdata["PredictedID"] = point_labels
    mesh_obj.pointdata["RGB"] = point_colors
    return mesh_obj


def infer_arch_from_name(path: Path) -> str:
    stem = path.stem.lower()
    tokens = stem.replace("-", "_").split("_")
    for token in reversed(tokens):
        if token in {"u", "upper", "max", "maxilla"}:
            return "U"
        if token in {"l", "lower", "man", "mandible"}:
            return "L"
    if stem.endswith("u"):
        return "U"
    if stem.endswith("l"):
        return "L"
    raise ValueError(f"无法从文件 {path} 推断颌别（需含 U/L 标记）")


def load_mesh(path: Path):
    mesh = vedo.load(str(path))
    if mesh is None:
        raise RuntimeError(f"无法加载网格：{path}")
    return mesh


@dataclass
class SegmentationResult:
    case_id: str
    arch: str
    source: Path
    refined_vtp: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def predict_single_mesh(
    mesh_path: Path,
    arch: str,
    model_dir: Path,
    device: torch.device,
    model_cache: Dict[str, nn.Module],
    output_dir: Path,
) -> Path:
    """Run MeshSegNet inference + pygco refinement + KNN upsampling."""
    arch = arch.upper()
    if arch not in MODEL_FILES:
        raise ValueError(f"未知颌别 {arch}，无法选择模型")

    if arch not in model_cache:
        ckpt_path = model_dir / MODEL_FILES[arch]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少模型权重 {ckpt_path}")
        model = MeshSegNet(num_classes=15, num_channels=15).to(device, dtype=torch.float)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        model.eval()
        model_cache[arch] = model
    else:
        model = model_cache[arch]

    ensure_dir(output_dir)
    mesh = load_mesh(mesh_path)

    # Downsample (limit to 20k faces)
    target_num = 20000
    ratio = min(1.0, target_num / float(mesh.ncells) if mesh.ncells else 1.0)
    mesh_d = mesh.clone()
    if ratio < 1.0:
        mesh_d.decimate(fraction=ratio)
    predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

    points = mesh_d.points.copy()
    mean_cell_centers = mesh_d.center_of_mass()
    points[:, 0:3] -= mean_cell_centers[0:3]

    ids = np.array(mesh_d.cells, dtype=np.int32)
    cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype="float32")

    mesh_d.compute_normals()
    normals = mesh_d.celldata["Normals"]

    barycenters = mesh_d.cell_centers().points.copy()
    barycenters -= mean_cell_centers[0:3]

    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        if stds[i] == 0:
            stds[i] = 1.0
        if nstds[i] == 0:
            nstds[i] = 1.0
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / max(1e-6, (maxs[i] - mins[i]))
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))

    A_S = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
    A_L = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
    from scipy.spatial import distance_matrix  # local import to avoid heavy startup

    D = distance_matrix(X[:, 9:12], X[:, 9:12])
    A_S[D < 0.1] = 1.0
    rowsum = np.sum(A_S, axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    A_S = A_S / rowsum

    A_L[D < 0.2] = 1.0
    rowsum = np.sum(A_L, axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    A_L = A_L / rowsum

    X_t = torch.from_numpy(X.T.reshape(1, X.shape[1], X.shape[0])).to(device, dtype=torch.float)
    A_S_t = torch.from_numpy(A_S.reshape(1, *A_S.shape)).to(device, dtype=torch.float)
    A_L_t = torch.from_numpy(A_L.reshape(1, *A_L.shape)).to(device, dtype=torch.float)

    with torch.no_grad():
        tensor_prob_output = model(X_t, A_S_t, A_L_t).to(device, dtype=torch.float)
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(15):
        predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1) == i_label] = i_label

    mesh2 = mesh_d.clone()
    mesh2 = attach_label_data(mesh2, predicted_labels_d)
    down_out = output_dir / f"{mesh_path.stem}_d_predicted.vtp"
    vedo.write(mesh2, str(down_out))

    round_factor = 70
    patch_prob_output = np.clip(patch_prob_output, 1.0e-6, 1.0)
    unaries = -round_factor * np.log10(patch_prob_output)
    unaries = unaries.astype(np.int32).reshape(-1, 15)

    pairwise = (1 - np.eye(15, dtype=np.int32))
    normals = mesh_d.celldata["Normals"].copy()
    barycenters = mesh_d.cell_centers().points.copy()
    cell_ids = np.asarray(mesh_d.cells, dtype=np.int32)

    lambda_c = 18
    edges = np.empty([1, 3], order="C")
    for i_node in range(cells.shape[0]):
        nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
        nei_id = np.where(nei == 2)
        for i_nei in nei_id[0][:]:
            if i_node < i_nei:
                cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                cos_theta /= max(1e-6, np.linalg.norm(normals[i_node, 0:3]) * np.linalg.norm(normals[i_nei, 0:3]))
                cos_theta = np.clip(cos_theta, -0.9999, 0.9999)
                theta = math.acos(cos_theta)
                phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                if theta > math.pi / 2.0:
                    weight = -np.log10(theta / math.pi) * phi
                else:
                    beta = 1 + np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                    weight = -beta * np.log10(theta / math.pi) * phi
                edges = np.concatenate((edges, np.array([i_node, i_nei, weight]).reshape(1, 3)), axis=0)
    edges = np.delete(edges, 0, 0)
    edges[:, 2] *= lambda_c * round_factor
    edges = edges.astype(np.int32, copy=False)

    refine_labels = cut_from_graph(edges, unaries, pairwise).reshape([-1, 1])
    mesh3 = mesh_d.clone()
    mesh3 = attach_label_data(mesh3, refine_labels)
    refined_ds_path = output_dir / f"{mesh_path.stem}_d_predicted_refined.vtp"
    vedo.write(mesh3, str(refined_ds_path))

    if mesh.ncells > 50000:
        ratio = 50000 / float(mesh.ncells)
        mesh.decimate(fraction=ratio)

    barycenters = mesh3.cell_centers().points.copy()
    fine_barycenters = mesh.cell_centers().points.copy()

    neigh = KNeighborsClassifier(n_neighbors=1, weights="distance")
    neigh.fit(barycenters, np.ravel(refine_labels))
    fine_labels = neigh.predict(fine_barycenters).reshape(-1, 1)

    mesh_final = attach_label_data(mesh.clone(), fine_labels)
    refined_path = output_dir / f"{mesh_path.stem}_predicted_refined.vtp"
    vedo.write(mesh_final, str(refined_path))
    return refined_path


def collect_arch_meshes(raw_root: Path, case_id: str) -> Dict[str, Path]:
    case_dir = raw_root / case_id.lstrip("0")
    if not case_dir.exists():
        case_dir = raw_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"未找到病例目录 {case_id} ({case_dir})")
    meshes: Dict[str, Path] = {}
    # Strictly rely on STL meshes to avoid leaking refined ground-truth VTPs.
    for mesh_path in case_dir.glob("*.stl"):
        arch = infer_arch_from_name(mesh_path)
        meshes[arch] = mesh_path
    if not meshes:
        raise FileNotFoundError(f"{case_id} 未发现 .stl 网格（需要上/下颌 STL 文件）")
    return meshes


def run_meshsegnet(
    raw_root: Path,
    cases: Sequence[str],
    model_dir: Path,
    seg_output_root: Path,
    staging_raw_root: Path,
    device: torch.device,
) -> List[SegmentationResult]:
    model_cache: Dict[str, nn.Module] = {}
    results: List[SegmentationResult] = []
    for case in cases:
        meshes = collect_arch_meshes(raw_root, case)
        case_stage_dir = staging_raw_root / case
        if case_stage_dir.exists():
            shutil.rmtree(case_stage_dir)
        case_stage_dir.mkdir(parents=True, exist_ok=True)

        for arch, mesh_path in meshes.items():
            refined_path = predict_single_mesh(
                mesh_path=mesh_path,
                arch=arch,
                model_dir=model_dir,
                device=device,
                model_cache=model_cache,
                output_dir=seg_output_root / case,
            )
            target = case_stage_dir / f"{case}_{arch.upper()}.vtp"
            shutil.copy2(refined_path, target)
            results.append(
                SegmentationResult(
                    case_id=case,
                    arch=arch.upper(),
                    source=mesh_path,
                    refined_vtp=target,
                )
            )
    return results


def load_landmarks_from_json(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    markups = data.get("markups", [])
    if not markups:
        return {}
    ctrl_pts = markups[0].get("controlPoints", [])
    out: Dict[str, np.ndarray] = {}
    for cp in ctrl_pts:
        label = cp.get("label")
        pos = cp.get("position")
        if label and pos and len(pos) == 3:
            out[label] = np.asarray(pos, dtype=np.float32)
    return out


def aggregate_ground_truth(raw_root: Path, case: str) -> Dict[str, np.ndarray]:
    case_dir = raw_root / case.lstrip("0")
    if not case_dir.exists():
        case_dir = raw_root / case
    upper = next(case_dir.glob("*_U.json"))
    lower = next(case_dir.glob("*_L.json"))
    gt = {}
    gt.update(load_landmarks_from_json(upper))
    gt.update(load_landmarks_from_json(lower))
    return gt


def aggregate_predictions(json_path: Path) -> Dict[str, np.ndarray]:
    return load_landmarks_from_json(json_path)


@dataclass
class CaseMetrics:
    case_id: str
    matched: int
    total_gt: int
    mae_mm: float
    p95_mm: float
    hit05: float
    hit10: float


def compute_case_metrics(
    case: str,
    pred_json: Path,
    raw_root: Path,
) -> CaseMetrics:
    preds = aggregate_predictions(pred_json)
    gt = aggregate_ground_truth(raw_root, case)
    errors: List[float] = []
    hit05 = 0
    hit10 = 0
    for label, gt_pos in gt.items():
        pred_pos = preds.get(label)
        if pred_pos is None:
            continue
        err = float(np.linalg.norm(pred_pos - gt_pos))
        errors.append(err)
        if err <= 0.5:
            hit05 += 1
        if err <= 1.0:
            hit10 += 1
    total_gt = len(gt)
    matched = len(errors)
    mae_mm = float(np.mean(errors)) if errors else float("nan")
    p95_mm = float(np.percentile(errors, 95)) if errors else float("nan")
    hit05_ratio = hit05 / total_gt if total_gt else 0.0
    hit10_ratio = hit10 / total_gt if total_gt else 0.0
    return CaseMetrics(
        case_id=case,
        matched=matched,
        total_gt=total_gt,
        mae_mm=mae_mm,
        p95_mm=p95_mm,
        hit05=hit05_ratio,
        hit10=hit10_ratio,
    )


def render_markdown_report(
    cases: List[CaseMetrics],
    md_path: Path,
    pointnet_out_dir: Path,
    seg_results: List[SegmentationResult],
) -> None:
    mae_values = [c.mae_mm for c in cases if not math.isnan(c.mae_mm)]
    p95_values = [c.p95_mm for c in cases if not math.isnan(c.p95_mm)]

    lines = [
        "# 10-case Evaluation",
        "",
        f"- PointNet-Reg output directory: `{pointnet_out_dir}`",
        f"- Segmentation staging raw: `{seg_results[0].refined_vtp.parent.parent}`" if seg_results else "- Segmentation data: (none)",
        "",
        "| Case | Matched / GT | MAE (mm) | P95 (mm) | Hit@0.5 | Hit@1.0 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cm in cases:
        lines.append(
            f"| {cm.case_id} | {cm.matched} / {cm.total_gt} | "
            f"{cm.mae_mm:.3f} | {cm.p95_mm:.3f} | {cm.hit05:.3%} | {cm.hit10:.3%} |"
        )

    if mae_values:
        lines.extend(
            [
                "",
                "## Aggregated",
                "",
                f"- Mean MAE: {np.mean(mae_values):.3f} mm",
                f"- Median MAE: {np.median(mae_values):.3f} mm",
                f"- Std MAE: {np.std(mae_values):.3f} mm",
                f"- Mean P95: {np.mean(p95_values):.3f} mm",
                f"- Hit@0.5 avg: {np.mean([c.hit05 for c in cases]):.3%}",
                f"- Hit@1.0 avg: {np.mean([c.hit10 for c in cases]):.3%}",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MeshSegNet + PointNet-Reg evaluation on selected cases.")
    parser.add_argument("--cases", nargs="+", required=True, help="Case IDs (e.g., 001 004 007).")
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "datasets" / "landmarks_dataset" / "raw")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "src" / "MeshSegNet" / "models")
    parser.add_argument("--seg-out", type=Path, default=REPO_ROOT / "outputs" / "seg_point_eval" / "vtp")
    parser.add_argument("--staging-raw", type=Path, default=REPO_ROOT / "outputs" / "seg_point_eval" / "raw")
    parser.add_argument("--pointnet-out", type=Path, default=REPO_ROOT / "outputs" / "seg_point_eval" / "pointnet")
    parser.add_argument("--md-path", type=Path, default=REPO_ROOT / "outputs" / "seg_point_eval" / "report.md")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-seg", action="store_true", help="Skip MeshSegNet inference (reuse staged VTPs).")
    parser.add_argument("--skip-pointnet", action="store_true", help="Skip PointNet-Reg generation.")
    parser.add_argument("--skip-report", action="store_true", help="Skip markdown report generation.")
    return parser.parse_args()


def run_pointnet_reg(pointnet_out: Path, staging_raw: Path, cases: Sequence[str]) -> None:
    args = [
        "--input",
        str(staging_raw),
        "--out-dir",
        str(pointnet_out),
        "--cases",
        *cases,
        "--workers",
        "0",
        "--batch-size",
        "4",
    ]
    sys.argv = ["p5_generate.py"] + args
    pointnet_generate_main()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    staging_raw = args.staging_raw
    ensure_dir(args.seg_out)
    ensure_dir(staging_raw)
    ensure_dir(args.pointnet_out)

    seg_results: List[SegmentationResult] = []
    if not args.skip_seg:
        seg_results = run_meshsegnet(
            raw_root=args.raw_root,
            cases=args.cases,
            model_dir=args.model_dir,
            seg_output_root=args.seg_out,
            staging_raw_root=staging_raw,
            device=device,
        )
    else:
        # Populate seg_results from staged files for completeness
        for case in args.cases:
            case_dir = staging_raw / case
            if not case_dir.exists():
                raise FileNotFoundError(f"缺少 staged VTP：{case_dir}")
            for arch_file in case_dir.glob("*.vtp"):
                arch = infer_arch_from_name(arch_file)
                seg_results.append(
                    SegmentationResult(
                        case_id=case,
                        arch=arch,
                        source=arch_file,
                        refined_vtp=arch_file,
                    )
                )

    if not args.skip_pointnet:
        run_pointnet_reg(args.pointnet_out, staging_raw, args.cases)

    if args.skip_report:
        return

    case_metrics: List[CaseMetrics] = []
    for case in args.cases:
        json_path = args.pointnet_out / "json" / f"{case}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"缺少 PointNet-Reg 输出：{json_path}")
        cm = compute_case_metrics(case, json_path, args.raw_root)
        case_metrics.append(cm)

    render_markdown_report(case_metrics, args.md_path, args.pointnet_out, seg_results)
    print(f"Report saved to {args.md_path}")


if __name__ == "__main__":
    main()
