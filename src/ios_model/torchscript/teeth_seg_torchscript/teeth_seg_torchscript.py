# %%
import os
import numpy as np
try:
    import open3d
except ImportError:
    open3d = None
import scipy.sparse as sp
import torch
import trimesh
import vedo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from graphcut import graph_cut
    _graphcut_import_error = None
except Exception as exc:  # 捕获 pygco 缺失等情况
    graph_cut = None
    _graphcut_import_error = exc
from mesh_dataset import graph_preprocess
from sklearn.neighbors import KNeighborsClassifier
from torch_cluster import knn_graph
from torch_geometric.utils import to_scipy_sparse_matrix


# %%
# --- 新增：通用网格加载器，支持 STL/PLY/OBJ/DRC ---
def load_mesh(path_mesh: str):
    ext = os.path.splitext(path_mesh)[1].lower()
    if ext == ".drc":
        import DracoPy  # 延迟导入，避免纯 STL 流程也要装 DracoPy

        with open(path_mesh, "rb") as f:
            mesh = DracoPy.decode(f.read())
        points = np.asarray(mesh.points, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return points, faces

    if ext in [".stl", ".ply", ".obj", ".off"]:
        mesh = trimesh.load(path_mesh, process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = tuple(mesh.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in scene: {path_mesh}")
            mesh = trimesh.util.concatenate(geometries)
        if hasattr(mesh, "faces") and len(mesh.faces) and mesh.faces.shape[1] != 3:
            mesh = mesh.triangulate()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return points, faces

    raise ValueError(f"Unsupported mesh format: {ext}")


def drc_decode(path_drc: str):
    points, faces = load_mesh(path_drc)
    return points, faces


def drc_encode(points: np.ndarray, faces: np.ndarray, path_drc: str, bit: int = 25):
    import DracoPy

    with open(path_drc, "wb") as drc_file:
        binary = DracoPy.encode(points, faces, quantization_bits=bit)
        drc_file.write(binary)


def downsample_mesh_open3d(points: np.ndarray, faces: np.ndarray, num_points: int):
    # faster than downsample_mesh_vedo, but the result is worse as the vedo was used during model training
    if open3d is None:
        raise ImportError("open3d is not available in this environment")
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(points)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh_downsample = open3d.geometry.simplify_quadric_decimation(
        mesh, target_number_of_triangles=num_points
    )
    points_downsample = np.asarray(mesh_downsample.vertices)
    faces_downsample = np.asarray(mesh_downsample.triangles)
    return points_downsample, faces_downsample


def downsample_mesh_vedo(points: np.ndarray, faces: np.ndarray, num_cells: int):
    # lead to better prediction details than downsample_mesh_open3d
    mesh = vedo.Mesh([points, faces])
    mesh_downsample = mesh.decimate(num_cells / mesh.ncells)
    points_attr = getattr(mesh_downsample, "points", None)
    if points_attr is None:
        raise AttributeError("vedo.Mesh missing points attribute after decimation")
    if callable(points_attr):
        points_attr = points_attr()
    points_downsample = np.asarray(points_attr)
    faces_attr = getattr(mesh_downsample, "faces", None)
    if faces_attr is None:
        faces_attr = getattr(mesh_downsample, "cells", None)
    if faces_attr is None:
        raise AttributeError("vedo.Mesh missing faces/cells attribute after decimation")
    if callable(faces_attr):
        faces_attr = faces_attr()
    faces_downsample = np.asarray(faces_attr, dtype=np.int64)
    return points_downsample, faces_downsample


def upsample(
    positions: np.ndarray,
    labels: np.ndarray,
    positions_upsample: np.ndarray,
    n_neighbors: int = 4,
):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    x = positions
    y = np.ravel(labels)
    neigh.fit(x, y)
    labels_upsample = neigh.predict(positions_upsample).astype("int")
    return labels_upsample


def remove_small_components(faces: np.ndarray, labels: np.ndarray):
    """
    Remove small connected components from the mesh.
    """
    edge_index = torch.LongTensor(trimesh.graph.face_adjacency(faces=faces).T)
    edge_row_label = labels[edge_index[0]]
    edge_col_label = labels[edge_index[1]]
    # Filter edges within the same cluster
    cluster_edge_mask = edge_col_label == edge_row_label
    cluster_edge_index = edge_index[:, cluster_edge_mask.flatten()]
    adj = to_scipy_sparse_matrix(cluster_edge_index, num_nodes=len(faces))
    num_components, component = sp.csgraph.connected_components(adj, connection="weak")

    for idx in np.unique(labels):
        # Skip the background label
        if idx == 0:
            continue
        mask = labels == idx
        uniques, counts = np.unique(component[mask], return_counts=True)
        # Get the most common unique component
        mode_unique = uniques[np.argmax(counts)]
        # Get the mask for the non-dominant components within the current label
        mask = (component != mode_unique) & mask
        # Set the labels of non-dominant components to 0 (background)
        labels[mask] = 0
    return labels


@torch.no_grad()
def predict(
    device: torch.device,
    model: torch.jit._script.RecursiveScriptModule,
    points_origin: np.ndarray,
    faces_origin: np.ndarray,
    downsample_num: int,
    graph_k: int,
):
    """
    Args:
        device: cude device
        model: torchscript model
        point_origin: origin mesh points
        faces_origin: origin mesh faces
        downsample_num: cell number of downsampled mesh
        graph_k: k of knn graph for mesh graph construction

    Returns:
        labels_downsample: predicted labels of downsampled mesh
        labels_origin: predicted labels of origin mesh
        points: downsampled mesh points
        faces: downsampled mesh faces
    """
    N = len(faces_origin)
    print("Start downsampling")
    if downsample_num is not None and N > downsample_num:
        points, faces = downsample_mesh_vedo(
            points_origin, faces_origin, downsample_num
        )
    else:
        points, faces = points_origin, faces_origin
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    positions = cells.mean(1)
    print("Start graph construction")
    graph, normals = graph_preprocess(
        points, faces, graph_k=graph_k, normalize_coordinate=True
    )
    graph.num_graphs = 1
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    if os.environ.get("TEETH_SEG_DEBUG"):
        x_np = graph.x.cpu().numpy()
        print(
            "Feature stats:",
            {"min": float(x_np.min()), "max": float(x_np.max()), "mean": float(x_np.mean()), "std": float(x_np.std())},
        )
        print("Edge index shape:", tuple(graph.edge_index.shape))
    graph = graph.to(device)
    print("Start model prediction")
    predictions, offsets = model(
        graph.num_graphs, graph.x, graph.pos, graph.edge_index, graph.batch
    )
    predictions = predictions.cpu().float().numpy()
    if os.environ.get("TEETH_SEG_DEBUG"):
        print(
            "Pred stats:",
            {"min": float(predictions.min()), "max": float(predictions.max()), "mean": float(predictions.mean())},
        )
        print("Sample logits:", predictions[0][:10].tolist())
        pred_labels_dbg = np.argmax(predictions, axis=1)
        uniq_dbg, cnt_dbg = np.unique(pred_labels_dbg, return_counts=True)
        print(
            "Pred label distribution:",
            {int(k): int(v) for k, v in zip(uniq_dbg, cnt_dbg)},
        )
    if graph_cut is not None:
        print("Start graphcut post-processing")
        labels = graph_cut(
            predictions,
            faces,
            positions,
            normals,
            round_factor=100,
            num_classes=17,
            lambda_c=30,
        )
    else:
        if _graphcut_import_error is not None:
            print(
                "Graphcut disabled, falling back to argmax labels:",
                repr(_graphcut_import_error),
            )
        labels = np.argmax(predictions, axis=1).astype(np.int32)
    labels = remove_small_components(faces, labels)
    if downsample_num is not None and N > downsample_num:
        # TODO: consider to apply the second graphcut for only edge regions for acceleration
        cells_origin = points_origin[
            faces_origin
        ]  # 3x3 dimensional coordinates of faces
        positions_origin = cells_origin.mean(1)
        print("Start upsampling")
        labels_origin = upsample(positions, labels, positions_origin, n_neighbors=1)
    else:
        labels_origin = labels
    return labels, labels_origin, points, faces


# %%
if __name__ == "__main__":
    ### Define Parameters Below ###
    jaw_type = "man"
    path_mesh = os.path.join(SCRIPT_DIR, "LowerJawScan.stl")  # 更新为待处理 STL/DRC/PLY/OBJ 路径
    device = torch.device("cuda:0")  # 无 GPU 可改为 torch.device("cpu")
    # Allow overriding via environment variables for batch runs
    jaw_type = os.environ.get("TEETH_SEG_JAW", jaw_type)
    path_mesh = os.environ.get("TEETH_SEG_INPUT", path_mesh)
    device_override = os.environ.get("TEETH_SEG_DEVICE")
    if device_override:
        device = torch.device(device_override)
    ### Don't change params below ###
    if jaw_type == "max":
        script_path = os.path.join(SCRIPT_DIR, "max_teeth_seg_model_script.pt")
    elif jaw_type == "man":
        script_path = os.path.join(SCRIPT_DIR, "man_teeth_seg_model_script.pt")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = torch.jit.load(script_path)
    model.to(device)
    model.eval()
    points_origin, faces_origin = load_mesh(path_mesh)
    labels, labels_origin, points, faces = predict(
        device,
        model,
        points_origin,
        faces_origin,
        downsample_num=50000,
        graph_k=8,
    )
    # Export segmentation to VTP with per-face labels and RGB colors
    mesh_out = vedo.Mesh([points_origin, faces_origin])
    labels_origin_arr = np.asarray(labels_origin, dtype=np.int32)
    mesh_out.celldata["teeth_label"] = labels_origin_arr
    palette = np.stack(
        [
            (labels_origin_arr * 13) % 256,
            (labels_origin_arr * 29) % 256,
            (labels_origin_arr * 53) % 256,
        ],
        axis=1,
    ).astype(np.uint8)
    mesh_out.celldata["RGB"] = palette
    out_vtp = os.path.splitext(path_mesh)[0] + "_labeled.vtp"
    vedo.write(mesh_out, out_vtp)
    print("Saved:", out_vtp)
