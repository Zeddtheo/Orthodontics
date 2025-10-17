from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier
from vedo import Mesh, load as vedo_load, write as vedo_write

from MeshSegNet.models.meshsegnet import MeshSegNet

try:  # pragma: no cover - optional dependency
    from pygco import cut_from_graph as graph_cut
except Exception as exc:  # noqa: BLE001
    warnings.warn(f"pygco unavailable ({exc}); falling back to argmin smoothing.")

    def graph_cut(edges: np.ndarray, unaries: np.ndarray, pairwise: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.argmin(unaries, axis=1).astype(np.int32)


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


MODEL_CANDIDATES = {
    "U": [
        "MeshSegNet_Max_15_classes_72samples_lr1e-2_best.pth",
        "MeshSegNet_Max_best.pth",
        "maxilla.pth",
    ],
    "L": [
        "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.pth",
        "MeshSegNet_Man_best.pth",
        "mandible.pth",
    ],
}


def infer_arch_from_name(filename: Path) -> str:
    base = filename.stem.lower()
    tokens = base.replace("-", "_").split("_")
    for token in reversed(tokens):
        if token in {"u", "upper", "max", "maxilla"}:
            return "U"
        if token in {"l", "lower", "man", "mandible"}:
            return "L"
    if base.endswith("u"):
        return "U"
    if base.endswith("l"):
        return "L"
    raise ValueError(f"无法从文件名 {filename.name} 推断颌别（需包含 U/L 或 upper/lower 标记）")


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1).astype(np.int32, copy=False)
    rgb = np.empty((labels.size, 3), dtype=np.uint8)
    for lab in np.unique(labels):
        rgb[labels == lab] = np.asarray(LABEL_COLORS.get(int(lab), tuple(DEFAULT_COLOR)), dtype=np.uint8)
    return rgb


def _attach_label_data(mesh: Mesh, labels: np.ndarray) -> Mesh:
    labels = labels.reshape(-1).astype(np.int32, copy=False)
    colors = _labels_to_rgb(labels)
    mesh.celldata["Label"] = labels
    mesh.celldata["PredLabel"] = labels
    mesh.celldata["PredictedID"] = labels
    mesh.celldata["RGB"] = colors

    n_points = mesh.npoints
    point_labels = np.zeros(n_points, dtype=np.int32)
    point_colors = np.zeros((n_points, 3), dtype=np.uint8)

    faces = np.asarray(mesh.cells, dtype=np.int32)
    adjacency = [[] for _ in range(n_points)]
    for cid, face in enumerate(faces):
        for pid in face:
            adjacency[pid].append(cid)

    for pid, cell_ids in enumerate(adjacency):
        if not cell_ids:
            continue
        labs = labels[cell_ids]
        uniq, cnt = np.unique(labs, return_counts=True)
        lab = int(uniq[np.argmax(cnt)])
        point_labels[pid] = lab
        point_colors[pid] = LABEL_COLORS.get(lab, tuple(DEFAULT_COLOR))

    mesh.pointdata["Label"] = point_labels
    mesh.pointdata["PredLabel"] = point_labels
    mesh.pointdata["PredictedID"] = point_labels
    mesh.pointdata["RGB"] = point_colors.astype(np.uint8, copy=False)
    return mesh


@dataclass
class SegmentationResult:
    arch: str
    seg_vtp: Path
    polydata_path: Path


class MeshSegNetRunner:
    def __init__(self, model_dir: Path, device: torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, MeshSegNet] = {}

    def _load_model(self, arch: str) -> MeshSegNet:
        if arch in self._cache:
            return self._cache[arch]
        ckpt_path = None
        for name in MODEL_CANDIDATES.get(arch, []):
            candidate = (self.model_dir / name).resolve()
            if candidate.exists():
                ckpt_path = candidate
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"缺少 MeshSegNet 权重文件，请在 {self.model_dir} 下提供 {MODEL_CANDIDATES.get(arch, [])}"
            )

        model = MeshSegNet(num_classes=15, num_channels=15).to(self.device)
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        self._cache[arch] = model
        return model

    def _prepare_downsampled(self, mesh: Mesh, target_cells: int = 10000) -> Tuple[Mesh, np.ndarray]:
        mesh_d = mesh.clone()
        if mesh_d.ncells > target_cells:
            fraction = max(0.0, min(1.0, target_cells / mesh_d.ncells))
            mesh_d.decimate(fraction=fraction)
        predicted_labels = np.zeros((mesh_d.ncells, 1), dtype=np.int32)
        return mesh_d, predicted_labels

    def _forward_meshsegnet(self, model: MeshSegNet, mesh_d: Mesh) -> np.ndarray:
        device = self.device
        points = mesh_d.points.copy()
        origin = mesh_d.center_of_mass()
        points[:, :3] -= origin[:3]

        cell_ids = np.asarray(mesh_d.cells, dtype=np.int32)
        cells = points[cell_ids].reshape(mesh_d.ncells, 9).astype(np.float32, copy=False)

        mesh_d.compute_normals()
        normals = mesh_d.celldata["Normals"].astype(np.float32, copy=False)
        barycenters = mesh_d.cell_centers().points.copy().astype(np.float32, copy=False)
        barycenters -= origin[:3]

        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)
        stds[stds == 0] = 1.0
        nstds[nstds == 0] = 1.0
        span = maxs - mins
        span[span == 0] = 1.0

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / span[i]
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        features = np.column_stack((cells, barycenters, normals)).astype(np.float32, copy=False)

        coords = features[:, 9:12]
        dist = distance_matrix(coords, coords)
        a_s = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)
        a_l = np.zeros_like(a_s)
        a_s[dist < 0.1] = 1.0
        a_l[dist < 0.2] = 1.0

        row_sum_s = a_s.sum(axis=1, keepdims=True)
        row_sum_s[row_sum_s == 0] = 1.0
        a_s /= row_sum_s

        row_sum_l = a_l.sum(axis=1, keepdims=True)
        row_sum_l[row_sum_l == 0] = 1.0
        a_l /= row_sum_l

        X = torch.from_numpy(features.T.astype(np.float32, copy=False)).unsqueeze(0).to(device)
        a_s_t = torch.from_numpy(a_s).unsqueeze(0).to(device)
        a_l_t = torch.from_numpy(a_l).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X, a_s_t, a_l_t).to(device)
        return logits.cpu().numpy()

    def _graphcut_refine(
        self,
        mesh_d: Mesh,
        cells: np.ndarray,
        logits_np: np.ndarray,
        round_factor: float = 70.0,
    ) -> np.ndarray:
        num_classes = logits_np.shape[-1]
        probs = np.clip(logits_np[0], 1.0e-6, 1.0)
        unaries = (-round_factor * np.log10(probs)).astype(np.int32)
        pairwise = (1 - np.eye(num_classes, dtype=np.int32))

        normals = mesh_d.celldata["Normals"].copy()
        barycenters = mesh_d.cell_centers().points.copy()
        faces = np.asarray(mesh_d.cells, dtype=np.int32)

        edges_list: list[Tuple[int, int, int]] = []
        lam_c = 18
        for i in range(cells.shape[0]):
            hits = np.sum(np.isin(faces, faces[i]), axis=1)
            neighbors = np.where(hits == 2)[0]
            for j in neighbors:
                if i >= j:
                    continue
                n_i = normals[i, :3]
                n_j = normals[j, :3]
                denom = np.linalg.norm(n_i) * np.linalg.norm(n_j)
                if denom == 0:
                    cos_theta = 1.0
                else:
                    cos_theta = np.dot(n_i, n_j) / denom
                cos_theta = max(min(cos_theta, 0.9999), -0.9999)
                theta = math.acos(cos_theta)
                phi = np.linalg.norm(barycenters[i] - barycenters[j])
                if theta > math.pi / 2:
                    weight = -np.log10(theta / math.pi) * phi
                else:
                    beta = 1 + abs(np.dot(n_i, n_j))
                    weight = -beta * np.log10(theta / math.pi) * phi
                edges_list.append((i, j, int(weight * lam_c * round_factor)))

        if not edges_list:
            return np.argmax(probs, axis=1).astype(np.int32).reshape(-1, 1)

        edges = np.asarray(edges_list, dtype=np.int32)
        refined = graph_cut(edges, unaries, pairwise)
        return refined.reshape(-1, 1)

    def _upsample_labels(
        self,
        coarse_mesh: Mesh,
        fine_mesh: Mesh,
        coarse_labels: np.ndarray,
    ) -> np.ndarray:
        coarse_centers = coarse_mesh.cell_centers().points.copy()
        fine_centers = fine_mesh.cell_centers().points.copy()
        classifier = KNeighborsClassifier(n_neighbors=1, weights="distance")
        classifier.fit(coarse_centers, coarse_labels.ravel())
        fine_labels = classifier.predict(fine_centers).reshape(-1, 1)
        return fine_labels

    def segment(self, mesh_path: Path, output_dir: Path, arch_override: Optional[str] = None) -> SegmentationResult:
        mesh_path = Path(mesh_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        arch = arch_override.upper() if isinstance(arch_override, str) else infer_arch_from_name(mesh_path)
        model = self._load_model(arch)

        mesh = vedo_load(str(mesh_path))
        mesh_d, predicted_labels_d = self._prepare_downsampled(mesh)
        logits_np = self._forward_meshsegnet(model, mesh_d)

        predicted_labels_d[:] = np.argmax(logits_np[0], axis=1).reshape(-1, 1)
        mesh_d = _attach_label_data(mesh_d, predicted_labels_d)

        refined_labels = self._graphcut_refine(mesh_d, np.asarray(mesh_d.cells, dtype=np.int32), logits_np)
        mesh_refined = mesh_d.clone()
        mesh_refined = _attach_label_data(mesh_refined, refined_labels)

        if mesh.ncells > 50000:
            fraction = max(0.0, min(1.0, 50000 / mesh.ncells))
            mesh.decimate(fraction=fraction)

        upsampled_labels = self._upsample_labels(mesh_refined, mesh, refined_labels)
        mesh_final = _attach_label_data(mesh, upsampled_labels)

        seg_vtp = output_dir / f"{mesh_path.stem}_seg.vtp"
        vedo_write(mesh_final, str(seg_vtp))

        return SegmentationResult(
            arch=arch,
            seg_vtp=seg_vtp,
            polydata_path=seg_vtp,
        )
