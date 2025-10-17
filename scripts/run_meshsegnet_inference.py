#!/usr/bin/env python3
"""Run MeshSegNet inference on a single STL/VTP mesh."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree, distance_matrix

# Allow running without installing as package
if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "src" / "MeshSegNet" / "models"
    if str(models_dir) not in sys.path:
        sys.path.append(str(models_dir))

from meshsegnet import MeshSegNet  # noqa: E402


def _ensure_tri_mesh(mesh: pv.PolyData) -> pv.PolyData:
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return mesh


def _decimate(mesh: pv.PolyData, target_cells: int) -> pv.PolyData:
    if mesh.n_cells <= target_cells:
        return mesh.copy(deep=True)
    reduction = 1.0 - (target_cells / float(mesh.n_cells))
    reduction = float(np.clip(reduction, 0.0, 0.99))
    decimated = mesh.decimate_pro(
        reduction,
        preserve_topology=True,
        feature_angle=45.0,
    )
    return decimated


def _build_inputs(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = mesh.copy(deep=True)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    pts = mesh.points.astype(np.float32, copy=True)
    faces = mesh.faces.reshape((-1, 4))[:, 1:]
    bary_raw = mesh.cell_centers().points.astype(np.float32, copy=True)
    normals = mesh.cell_data.get("Normals")
    if normals is None or len(normals) != mesh.n_cells:
        normals = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False).cell_data["Normals"]
    normals = np.ascontiguousarray(normals).astype(np.float32)

    center = mesh.center_of_mass()
    pts -= center
    bary_centered = bary_raw - center

    cells = pts[faces].reshape(mesh.n_cells, 9).astype(np.float32, copy=False)

    maxs = pts.max(axis=0)
    mins = pts.min(axis=0)
    means = pts.mean(axis=0)
    stds = pts.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    bary_norm = bary_centered.copy()
    normals_norm = normals.copy()

    for i in range(3):
        std = stds[i] if stds[i] > 1e-8 else 1.0
        cells[:, i] = (cells[:, i] - means[i]) / std
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / std
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / std

        span = maxs[i] - mins[i]
        span = span if span > 1e-8 else 1.0
        bary_norm[:, i] = (bary_norm[:, i] - mins[i]) / span

        nstd = nstds[i] if nstds[i] > 1e-8 else 1.0
        normals_norm[:, i] = (normals_norm[:, i] - nmeans[i]) / nstd

    feats = np.column_stack((cells, bary_norm, normals_norm)).astype(np.float32, copy=False)
    return feats, bary_norm, bary_raw


def _build_adj(bary_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = distance_matrix(bary_norm, bary_norm)
    a_s = (D < 0.1).astype(np.float32)
    a_l = (D < 0.2).astype(np.float32)

    # Row-normalize; safeguard empty neighborhoods.
    def _row_norm(mat: np.ndarray) -> np.ndarray:
        denom = mat.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 1e-8, 1.0, denom)
        return mat / denom

    return _row_norm(a_s), _row_norm(a_l)


def infer_single(
    mesh_path: Path,
    ckpt_path: Path,
    out_path: Path,
    device: torch.device,
    target_cells: int = 10000,
) -> None:
    mesh_path = mesh_path.resolve()
    ckpt_path = ckpt_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_mesh = _ensure_tri_mesh(pv.read(mesh_path))
    full_mesh = full_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    dec_mesh = _decimate(full_mesh, target_cells)
    dec_mesh = dec_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    feats, bary_norm, bary_raw = _build_inputs(dec_mesh)
    adj_s, adj_l = _build_adj(bary_norm)

    X = feats.transpose(1, 0)[None, ...]  # (1, 15, N)
    A_S = adj_s[None, ...]
    A_L = adj_l[None, ...]

    X_t = torch.from_numpy(X).to(device)
    A_S_t = torch.from_numpy(A_S).to(device)
    A_L_t = torch.from_numpy(A_L).to(device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)

    model = MeshSegNet(num_classes=state["output_conv.bias"].shape[0], num_channels=X.shape[1])
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        probs = model(X_t, A_S_t, A_L_t).cpu().numpy()[0]  # (N, num_classes)

    labels_dec = probs.argmax(axis=1).astype(np.int32)

    if dec_mesh.n_cells != full_mesh.n_cells:
        tree = cKDTree(bary_raw)
        full_centers = full_mesh.cell_centers().points.astype(np.float32, copy=True)
        _, idx = tree.query(full_centers, k=1)
        labels_full = labels_dec[idx]
    else:
        labels_full = labels_dec

    out_mesh = full_mesh.copy(deep=True)
    out_mesh.cell_data["PredictedID"] = labels_full.astype(np.int32, copy=False)
    out_mesh.save(out_path, binary=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run MeshSegNet inference on a mesh.")
    ap.add_argument("--input", type=Path, required=True, help="Input STL/VTP mesh.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint (.pth) file.")
    ap.add_argument("--output", type=Path, required=True, help="Output VTP path.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-cells", type=int, default=10000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    infer_single(args.input, args.ckpt, args.output, device, target_cells=args.target_cells)


if __name__ == "__main__":
    main()
