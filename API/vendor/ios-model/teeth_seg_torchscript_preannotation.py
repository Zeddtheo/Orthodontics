# %%
from pathlib import Path

import os

import numpy as np
import torch
import vedo
from mesh_dataset import graph_preprocess
from sklearn.neighbors import KNeighborsClassifier
from torch_cluster import knn_graph

from graphcut import graph_cut


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SEARCH_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent.parent / "models" / "ios-model",
]


def _resolve_model_path(filename: str) -> Path:
    for directory in MODEL_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Model weights {filename} not found in {MODEL_SEARCH_DIRS}")


# %%
def downsample_mesh_vedo(points: np.ndarray, faces: np.ndarray, num_points: int):
    # lead to better prediction details than downsample_mesh_open3d
    mesh = vedo.Mesh([points, faces])
    mesh_downsample = mesh.decimate(num_points / mesh.ncells)
    points_downsample = np.asarray(mesh_downsample.points())
    faces_downsample = np.asarray(mesh_downsample.faces())
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
    N = len(points_origin)
    if downsample_num is not None and N > downsample_num:
        points, faces = downsample_mesh_vedo(
            points_origin, faces_origin, downsample_num
        )
    else:
        points, faces = points_origin, faces_origin
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    positions = cells.mean(1)
    graph, normals = graph_preprocess(points, faces, graph_k=graph_k)
    graph.num_graphs = 1
    graph.to(device)
    predictions, offsets = model(
        graph.num_graphs, graph.x, graph.pos, graph.edge_index, graph.batch
    )
    predictions = predictions.cpu().float().numpy()
    labels = graph_cut(
        predictions,
        faces,
        positions,
        normals,
        round_factor=100,
        num_classes=17,
        lambda_c=30,
    )
    if downsample_num is not None and N > downsample_num:
        # TODO: consider to apply the second graphcut for only edge regions for acceleration
        cells_origin = points_origin[
            faces_origin
        ]  # 3x3 dimensional coordinates of faces
        positions_origin = cells_origin.mean(1)
        labels_origin = upsample(positions, labels, positions_origin, n_neighbors=1)
    else:
        labels_origin = labels
    return labels, labels_origin, points, faces


# %%
if __name__ == "__main__":
    import glob
    from tqdm import tqdm

    ### Define Parameters Below ###
    device = torch.device("cuda:5")
    torch.cuda.set_device(device)
    max_script_path = _resolve_model_path("max_teeth_seg_model_script.pt")
    man_script_path = _resolve_model_path("man_teeth_seg_model_script.pt")
    max_model = torch.jit.load(str(max_script_path))
    man_model = torch.jit.load(str(man_script_path))
    max_model.to(device)
    max_model.eval()
    man_model.to(device)
    man_model.eval()
    vtp_list = glob.glob("../../data/intraoral_scanners/ios_0608/implant/*.vtp")
    save_root = "../../data/intraoral_scanners/ios_0608/preannotation/"
    for path in tqdm(vtp_list):
        filename = os.path.basename(path)
        if "max" in path:
            model = max_model
        elif "man" in path:
            model = man_model
        mesh = vedo.load(path)
        points_origin = np.asarray(mesh.points())
        faces_origin = np.asarray(mesh.faces())
        labels, labels_origin, points, faces = predict(
            device,
            model,
            points_origin,
            faces_origin,
            downsample_num=50000,
            graph_k=8,
        )
        mesh.celldata["Label"] = labels_origin
        mesh.write(os.path.join(save_root, filename))

# %%
