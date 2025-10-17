#!/usr/bin/env python3
"""MeshSegNet inference with graph-cut refinement and KNN upsampling."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pyvista as pv
import torch
from pygco import cut_from_graph
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier

# Ensure we can import the original MeshSegNet implementation
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "MeshSegNet" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

from meshsegnet import MeshSegNet  # noqa: E402


def _ensure_tri_mesh(mesh: pv.PolyData) -> pv.PolyData:
    return mesh if mesh.is_all_triangles else mesh.triangulate()


def _decimate(mesh: pv.PolyData, target_cells: int) -> pv.PolyData:
    if mesh.n_cells <= target_cells:
        return mesh.copy(deep=True)
    reduction = 1.0 - (target_cells / float(mesh.n_cells))
    reduction = float(np.clip(reduction, 0.0, 0.99))
    return mesh.decimate_pro(reduction, preserve_topology=True, feature_angle=45.0)


def _build_inputs(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return feats, bary_norm, bary_raw, normals


def _build_adj(bary_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = distance_matrix(bary_norm, bary_norm)
    a_s = (D < 0.1).astype(np.float32)
    a_l = (D < 0.2).astype(np.float32)

    def _row_norm(mat: np.ndarray) -> np.ndarray:
        denom = mat.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 1e-8, 1.0, denom)
        return mat / denom

    return _row_norm(a_s), _row_norm(a_l)


def _build_edge_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for cid, tri in enumerate(faces):
        edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        for u, v in edges:
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(cid)
    return edge_map


def _graph_cut_refine(
    probs: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    barycenters: np.ndarray,
    *,
    lambda_c: float = 30.0,
    round_factor: float = 100.0,
) -> np.ndarray:
    """Apply pygco graph cut refinement on decimated mesh predictions."""
    probs = np.clip(probs, 1.0e-6, 1.0)
    num_cells, num_classes = probs.shape
    unaries = (-round_factor * np.log10(probs)).astype(np.int32, copy=False)
    pairwise = (1 - np.eye(num_classes, dtype=np.int32))

    edge_map = _build_edge_map(faces)
    edges: List[Tuple[int, int, int]] = []

    normal_norms = np.linalg.norm(normals, axis=1)
    normal_norms = np.where(normal_norms > 1e-6, normal_norms, 1.0)

    for cells in edge_map.values():
        if len(cells) < 2:
            continue
        for i_idx in range(len(cells)):
            for j_idx in range(i_idx + 1, len(cells)):
                ci = cells[i_idx]
                cj = cells[j_idx]
                ni = normals[ci]
                nj = normals[cj]
                dot = np.dot(ni, nj) / (normal_norms[ci] * normal_norms[cj])
                dot = float(np.clip(dot, -0.9999, 0.9999))
                theta = np.arccos(dot)
                phi = float(np.linalg.norm(barycenters[ci] - barycenters[cj]))
                theta = max(theta, 1e-4)
                if theta > np.pi / 2.0:
                    weight = -np.log10(theta / np.pi) * phi
                else:
                    beta = 1.0 + abs(np.dot(ni, nj))
                    weight = -beta * np.log10(theta / np.pi) * phi
                weight = float(max(weight, 1e-6))
                edges.append((ci, cj, int(lambda_c * round_factor * weight)))

    if not edges:
        return probs.argmax(axis=1).astype(np.int32, copy=False)

    edges_arr = np.asarray(edges, dtype=np.int32)
    refined = cut_from_graph(edges_arr, unaries, pairwise)
    return refined.astype(np.int32, copy=False)


def _upsample_labels(
    dec_centers: np.ndarray,
    dec_labels: np.ndarray,
    full_centers: np.ndarray,
    *,
    n_neighbors: int = 15,
) -> np.ndarray:
    k = min(int(n_neighbors), dec_centers.shape[0])
    if k <= 0:
        return dec_labels
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    clf.fit(dec_centers, dec_labels)
    return clf.predict(full_centers).astype(np.int32, copy=False)


def infer_single(
    mesh_path: Path,
    ckpt_path: Path,
    out_path: Path,
    *,
    device: torch.device,
    target_cells: int = 10000,
    lambda_c: float = 30.0,
    round_factor: float = 100.0,
    knn_neighbors: int = 15,
) -> None:
    mesh_path = mesh_path.resolve()
    ckpt_path = ckpt_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_mesh = _ensure_tri_mesh(pv.read(mesh_path))
    full_mesh = full_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    dec_mesh = _decimate(full_mesh, target_cells)
    dec_mesh = dec_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    feats, bary_norm, bary_raw, normals = _build_inputs(dec_mesh)
    adj_s, adj_l = _build_adj(bary_norm)

    X = feats.transpose(1, 0)[None, ...]  # (1, F, N)
    A_S = adj_s[None, ...]
    A_L = adj_l[None, ...]

    X_t = torch.from_numpy(X).to(device)
    A_S_t = torch.from_numpy(A_S).to(device)
    A_L_t = torch.from_numpy(A_L).to(device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)

    num_classes = state["output_conv.bias"].shape[0]
    model = MeshSegNet(num_classes=num_classes, num_channels=X.shape[1])
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        probs = model(X_t, A_S_t, A_L_t).cpu().numpy()[0]  # (N, num_classes)

    faces = dec_mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int32, copy=False)
    dec_centers = dec_mesh.cell_centers().points.astype(np.float32, copy=False)

    refined_dec = _graph_cut_refine(
        probs,
        faces,
        normals,
        dec_centers,
        lambda_c=lambda_c,
        round_factor=round_factor,
    )

    full_centers = full_mesh.cell_centers().points.astype(np.float32, copy=False)
    refined_full = _upsample_labels(
        dec_centers,
        refined_dec,
        full_centers,
        n_neighbors=knn_neighbors,
    )

    out_mesh = full_mesh.copy(deep=True)
    out_mesh.cell_data["PredictedID"] = refined_full.astype(np.int32, copy=False)
    out_mesh.save(out_path, binary=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MeshSegNet inference with refinement.")
    ap.add_argument("--input", type=Path, required=True, help="Input STL/VTP mesh.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint (.pth).")
    ap.add_argument("--output", type=Path, required=True, help="Output VTP path.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-cells", type=int, default=10000)
    ap.add_argument("--lambda-c", type=float, default=30.0)
    ap.add_argument("--round-factor", type=float, default=100.0)
    ap.add_argument("--knn-neighbors", type=int, default=15)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    infer_single(
        args.input,
        args.ckpt,
        args.output,
        device=device,
        target_cells=args.target_cells,
        lambda_c=args.lambda_c,
        round_factor=args.round_factor,
        knn_neighbors=args.knn_neighbors,
    )


if __name__ == "__main__":
    main()
