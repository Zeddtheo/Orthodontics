from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier


def _ensure_repo_root() -> Tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "vendor"
    meshsegnet_pkg = vendor_root / "MeshSegNet"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    if str(meshsegnet_pkg) not in sys.path:
        sys.path.insert(0, str(meshsegnet_pkg))
    datasets_root = vendor_root / "datasets" / "landmarks_dataset"
    return repo_root, vendor_root, datasets_root


def _infer_arch_from_name(filename: str) -> str:
    base = Path(filename).stem.lower()
    tokens = base.replace('-', '_').split('_')
    for token in reversed(tokens):
        if token in {'u', 'upper', 'max', 'maxilla'}:
            return 'U'
        if token in {'l', 'lower', 'man', 'mandible'}:
            return 'L'
    if base.endswith('u'):
        return 'U'
    if base.endswith('l'):
        return 'L'
    raise ValueError(f"无法从文件名 {filename} 推断颌别（需包含 U/L 或 upper/lower 标记）")

def _load_meshsegnet_components():
    from MeshSegNet.models.meshsegnet import MeshSegNet  # type: ignore  # noqa: WPS433
    from MeshSegNet.models import utils  # type: ignore  # noqa: WPS433

    model_candidates = {
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
    return MeshSegNet, utils, model_candidates


MODEL_CACHE: Dict[str, torch.nn.Module] = {}
DEVICE: Optional[torch.device] = None
TOOTHMAP_CACHE: Optional[Dict[str, Dict[str, int]]] = None


PALETTE: Dict[int, Tuple[int, int, int]] = {
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

ARCH_INDEX_FIX: Dict[str, Dict[int, int]] = {
    "U": {
        1: 7,
        2: 6,
        3: 5,
        4: 4,
        5: 3,
        6: 2,
        7: 1,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
    },
    "L": {
        1: 14,
        2: 13,
        3: 12,
        4: 11,
        5: 10,
        6: 9,
        7: 8,
        8: 1,
        9: 2,
        10: 3,
        11: 4,
        12: 5,
        13: 6,
        14: 7,
    },
}

COLORIZE_PALETTE = np.array(
    [
        [200, 200, 200],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
        [128, 255, 0],
        [255, 0, 128],
        [0, 255, 128],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ],
    dtype=np.uint8,
)


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab in np.unique(labels):
        rgb[labels == lab] = np.asarray(PALETTE.get(int(lab), (90, 90, 90)), dtype=np.uint8)
    return rgb


def _attach_label_data(mesh_obj, labels: np.ndarray):
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    colors = _labels_to_rgb(labels)
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
        point_colors[pid] = np.asarray(PALETTE.get(lab, (90, 90, 90)), dtype=np.uint8)

    mesh_obj.pointdata["Label"] = point_labels
    mesh_obj.pointdata["PredictedID"] = point_labels
    mesh_obj.pointdata["RGB"] = point_colors
    return mesh_obj


def _colorize_by_palette(labels: np.ndarray) -> np.ndarray:
    flat = labels.astype(np.int32, copy=False).reshape(-1)
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    uniq = sorted({int(v) for v in np.unique(flat)})
    lut = {
        lab: COLORIZE_PALETTE[idx % len(COLORIZE_PALETTE)]
        for idx, lab in enumerate(uniq)
    }
    rgb = np.vstack([lut[int(val)] for val in flat]).astype(np.uint8)
    return rgb


def _map_labels_to_fdi(labels: np.ndarray, arch: str, arch_map: Dict[str, int]) -> np.ndarray:
    shape = labels.shape
    flat = labels.astype(np.int32, copy=False).reshape(-1)
    if flat.size == 0:
        return flat.reshape(shape)
    inv_map = {int(v): int(k) for k, v in arch_map.items()}
    fix_map = ARCH_INDEX_FIX.get(arch.upper(), {})
    mapped = np.empty_like(flat, dtype=np.int32)
    for idx, raw in enumerate(flat):
        raw_int = int(raw)
        if raw_int <= 0:
            mapped[idx] = raw_int
            continue
        remapped = fix_map.get(raw_int, raw_int)
        mapped[idx] = inv_map.get(remapped, remapped)
    return mapped.reshape(shape)


def _array_from_points(obj) -> np.ndarray:
    if hasattr(obj, "vertices"):
        vertices = obj.vertices
        if isinstance(vertices, np.ndarray):
            return vertices
    attr = getattr(obj, "points", None)
    if isinstance(attr, np.ndarray):
        return attr
    if callable(attr):
        arr = attr()
        return np.asarray(arr)
    return np.asarray(obj)


def _cells_array(mesh_obj) -> np.ndarray:
    cells_attr = getattr(mesh_obj, 'cells', None)
    if callable(cells_attr):
        return np.asarray(cells_attr())
    return np.asarray(cells_attr)

def _ensure_device(utils):
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available():
        gpu_id = utils.get_avail_gpu()
        torch.cuda.set_device(gpu_id)
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE


def _load_toothmap(datasets_root: Path) -> Dict[str, Dict[str, int]]:
    global TOOTHMAP_CACHE
    if TOOTHMAP_CACHE is not None:
        return TOOTHMAP_CACHE
    toothmap_path = datasets_root / "cooked" / "toothmap.json"
    if toothmap_path.exists():
        TOOTHMAP_CACHE = json.loads(toothmap_path.read_text(encoding="utf-8"))
    else:
        TOOTHMAP_CACHE = {}
    return TOOTHMAP_CACHE


def run_meshsegnet(
    stl_path: Path,
    out_vtp: Path,
    model_dir: Path,
    arch_override: Optional[str] = None,
    tmp_dir: Optional[Path] = None,
) -> str:
    repo_root, vendor_root, datasets_root = _ensure_repo_root()

    import vedo  # type: ignore

    MeshSegNet, utils, model_candidates = _load_meshsegnet_components()
    try:
        from pygco import cut_from_graph  # type: ignore
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(f"pygco unavailable ({exc}); fallback to unary argmin.")

        def cut_from_graph(edges, unaries, pairwise):  # type: ignore
            return np.argmin(unaries, axis=1).astype(np.int32)

    toothmap = _load_toothmap(datasets_root)

    arch = arch_override.upper() if arch_override else _infer_arch_from_name(stl_path.name)
    ckpt_path = None
    for candidate in model_candidates.get(arch, []):
        path = model_dir / candidate
        if path.exists():
            ckpt_path = path
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"缺少 MeshSegNet 权重，请在 {model_dir} 下提供 {model_candidates.get(arch, [])}")

    device = _ensure_device(utils)

    if arch not in MODEL_CACHE:
        num_classes = 15
        num_channels = 15
        model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float32)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device, dtype=torch.float32)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        model.eval()
        MODEL_CACHE[arch] = model
    else:
        model = MODEL_CACHE[arch]

    mesh = vedo.load(str(stl_path))
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="meshsegnet_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Predicting Sample filename: {stl_path.name}")
        print("\tDownsampling...")
        mesh_d = mesh.clone()
        target_cells = 20000
        if mesh_d.ncells > target_cells:
            ratio = target_cells / mesh_d.ncells
            mesh_d.decimate(fraction=ratio)
        predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

        print("\tPredicting...")
        points = _array_from_points(mesh_d).reshape(-1, 3).copy()
        mean_cell_centers = mesh_d.center_of_mass()
        points[:, 0:3] -= mean_cell_centers[0:3]

        cells_array = _cells_array(mesh_d)
        ids = np.asarray(cells_array)
        cells = points[ids].reshape(mesh_d.ncells, 9).astype("float32")

        mesh_d.compute_normals()
        normals = mesh_d.celldata["Normals"]
        bary_pts = mesh_d.cell_centers()
        bary_points = _array_from_points(bary_pts)
        barycenters = bary_points.reshape(-1, 3).copy()
        barycenters -= mean_cell_centers[0:3]

        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        A_S = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
        A_L = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
        D = distance_matrix(X[:, 9:12], X[:, 9:12])
        A_S[D < 0.1] = 1.0
        A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))
        A_L[D < 0.2] = 1.0
        A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])
        X = torch.from_numpy(X).to(device, dtype=torch.float)
        A_S = torch.from_numpy(A_S.reshape([1, A_S.shape[0], A_S.shape[1]])).to(device, dtype=torch.float)
        A_L = torch.from_numpy(A_L.reshape([1, A_L.shape[0], A_L.shape[1]])).to(device, dtype=torch.float)

        with torch.no_grad():
            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
        patch_prob_output = tensor_prob_output.cpu().numpy()

        for i_label in range(15):
            predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1) == i_label] = i_label
        if os.environ.get("MESHSEGNET_DEBUG"):
            uniq, cnt = np.unique(predicted_labels_d, return_counts=True)
            print(f"\tDebug coarse labels: {dict(zip(uniq.tolist(), cnt.tolist()))}")
        mesh2 = mesh_d.clone()
        mesh2 = _attach_label_data(mesh2, predicted_labels_d)

        print("\tRefining by pygco...")
        patch_prob_output = np.clip(patch_prob_output, 1.0e-6, 1.0)
        round_factor = 70
        unaries = (-round_factor * np.log10(patch_prob_output)).astype(np.int32).reshape(-1, 15)
        pairwise = (1 - np.eye(15, dtype=np.int32))
        background_bias = int(round_factor * 0.4)
        unaries[:, 0] += background_bias
        coarse_probs = patch_prob_output.reshape(-1, 15)

        normals = mesh_d.celldata["Normals"].copy()
        coarse_centers = _array_from_points(mesh_d.cell_centers()).reshape(-1, 3)
        cell_ids = _cells_array(mesh_d).astype(np.int32)

        edges = []
        lambda_c = 18
        for i_node in range(cells.shape[0]):
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei == 2)[0]
            for i_nei in nei_id:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, :3], normals[i_nei, :3]) / (
                        np.linalg.norm(normals[i_node, :3]) * np.linalg.norm(normals[i_nei, :3])
                    )
                    cos_theta = np.clip(cos_theta, -0.9999, 0.9999)
                    theta = math.acos(cos_theta)
                    phi = np.linalg.norm(coarse_centers[i_node, :] - coarse_centers[i_nei, :])
                    if theta > math.pi / 2.0:
                        weight = -np.log10(theta / math.pi) * phi
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, :3], normals[i_nei, :3]))
                    weight = -beta * np.log10(theta / math.pi) * phi
                edges.append((i_node, i_nei, int(weight * lambda_c * round_factor)))

        if edges:
            edges_arr = np.asarray(edges, dtype=np.int32)
            refine_labels = cut_from_graph(edges_arr, unaries, pairwise).reshape([-1, 1])
        else:
            refine_labels = np.argmax(coarse_probs, axis=1).reshape([-1, 1])
        if os.environ.get("MESHSEGNET_DEBUG"):
            uniq, cnt = np.unique(refine_labels, return_counts=True)
            print(f"\tDebug refined labels: {dict(zip(uniq.tolist(), cnt.tolist()))}")

        mesh3 = mesh_d.clone()
        mesh3 = _attach_label_data(mesh3, refine_labels)

        print("\tUpsampling...")
        if mesh.ncells > 50000:
            ratio = 50000 / mesh.ncells
            mesh.decimate(fraction=ratio)

        knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        coarse_barycenters = _array_from_points(mesh3.cell_centers()).reshape(-1, 3)
        fine_barycenters = _array_from_points(mesh.cell_centers()).reshape(-1, 3)
        knn.fit(coarse_barycenters, refine_labels.ravel())
        fine_labels = knn.predict(fine_barycenters).reshape(-1, 1)
        if os.environ.get("MESHSEGNET_DEBUG"):
            uniq, cnt = np.unique(fine_labels, return_counts=True)
            print(f"\tDebug fine labels: {dict(zip(uniq.tolist(), cnt.tolist()))}")

        mesh_cells_raw = _cells_array(mesh)
        fine_faces = np.asarray(mesh_cells_raw, dtype=np.int32)
        lab = fine_labels.reshape(-1)
        gum_label = 0

        adjacency = [[] for _ in range(mesh.ncells)]
        edge_owner: Dict[Tuple[int, int], int] = {}
        for ci, face in enumerate(fine_faces):
            for e in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                e = tuple(sorted(e))
                if e in edge_owner:
                    cj = edge_owner[e]
                    adjacency[ci].append(cj)
                    adjacency[cj].append(ci)
                else:
                    edge_owner[e] = ci

        visited = np.zeros(mesh.ncells, dtype=bool)
        max_hole = 300
        for start in range(mesh.ncells):
            if visited[start] or lab[start] != gum_label:
                continue
            comp = [start]
            visited[start] = True
            queue = [start]
            while queue:
                u = queue.pop()
                for v in adjacency[u]:
                    if not visited[v] and lab[v] == gum_label:
                        visited[v] = True
                        queue.append(v)
                        comp.append(v)
            if len(comp) > max_hole:
                continue
            neighbors = []
            for u in comp:
                for v in adjacency[u]:
                    if lab[v] != gum_label:
                        neighbors.append(lab[v])
            if neighbors:
                maj = np.bincount(np.asarray(neighbors, dtype=np.int32)).argmax()
                lab[comp] = maj
        fine_labels = lab.reshape(-1, 1)

        mesh = _attach_label_data(mesh, fine_labels)
        if toothmap:
            arch_key = arch.upper()
            arch_map = toothmap.get(arch_key, {})
            if arch_map:
                if "Label" in mesh.celldata.keys():
                    cell_labels = np.asarray(mesh.celldata["Label"], dtype=np.int32)
                    mapped = _map_labels_to_fdi(cell_labels, arch_key, arch_map)
                    mesh.celldata["Label"] = mapped
                    mesh.celldata["PredictedID"] = mapped.copy()
                    mesh.celldata["RGB"] = _colorize_by_palette(mapped)
                if "Label" in mesh.pointdata.keys():
                    point_labels = np.asarray(mesh.pointdata["Label"], dtype=np.int32)
                    mapped_p = _map_labels_to_fdi(point_labels, arch_key, arch_map)
                    mesh.pointdata["Label"] = mapped_p
                    mesh.pointdata["PredictedID"] = mapped_p.copy()
                    mesh.pointdata["RGB"] = _colorize_by_palette(mapped_p)

        vedo.write(mesh, str(out_vtp))
    finally:
        if tmp_dir is None and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
    return arch


def main() -> None:
    parser = argparse.ArgumentParser(description="MeshSegNet 推理包装器（原 step6 脚本移植版）")
    parser.add_argument("--upper_stl", help="上颌 STL")
    parser.add_argument("--upper_out", help="上颌 VTP 输出路径")
    parser.add_argument("--lower_stl", help="下颌 STL")
    parser.add_argument("--lower_out", help="下颌 VTP 输出路径")
    parser.add_argument("--model_dir", default="models/meshsegnet", help="MeshSegNet 权重目录")
    parser.add_argument("--upper_arch", choices=["U", "u"], help="上颌颌别覆盖")
    parser.add_argument("--lower_arch", choices=["L", "l"], help="下颌颌别覆盖")
    # backward compatibility
    parser.add_argument("--stl", help="兼容旧接口的 STL")
    parser.add_argument("--out_vtp", help="兼容旧接口的输出路径")
    parser.add_argument("--arch", choices=["U", "L", "u", "l"], help="兼容旧接口颌别")
    parser.add_argument("--tmp_dir", help="可选：临时目录覆盖")
    args = parser.parse_args()

    tasks = []
    if args.upper_stl and args.upper_out:
        tasks.append(("U", Path(args.upper_stl), Path(args.upper_out), args.upper_arch))
    if args.lower_stl and args.lower_out:
        tasks.append(("L", Path(args.lower_stl), Path(args.lower_out), args.lower_arch))
    if not tasks and args.stl and args.out_vtp:
        tasks.append((None, Path(args.stl), Path(args.out_vtp), args.arch))

    if not tasks:
        raise ValueError("至少需要提供一对 --upper_stl/--upper_out 或 --lower_stl/--lower_out")

    model_dir = Path(args.model_dir)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else None

    for default_arch, stl_path, out_vtp, override_arch in tasks:
        if not stl_path.exists():
            raise FileNotFoundError(f"STL not found: {stl_path}")
        arch_hint = override_arch or default_arch
        start = time.time()
        arch = run_meshsegnet(stl_path, out_vtp, model_dir, arch_hint, tmp_dir)
        duration = time.time() - start
        print(f"✅ MeshSegNet 完成 {stl_path.name} (arch={arch}) -> {out_vtp} 用时 {duration:.2f}s")


if __name__ == "__main__":
    main()
