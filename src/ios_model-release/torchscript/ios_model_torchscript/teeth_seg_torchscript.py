# %%
import DracoPy
import numpy as np
import torch
import vedo
from graphcut import graph_cut
from mesh_dataset import graph_preprocess
from sklearn.neighbors import KNeighborsClassifier
from torch_cluster import knn_graph


# %%
def drc_decode(path_drc: str):
    with open(path_drc, "rb") as drc_file:
        mesh = DracoPy.decode(drc_file.read())
        points = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces)
    return points, faces


def drc_encode(points: np.ndarray, faces: np.ndarray, path_drc: str, bit: int = 25):
    with open(path_drc, "wb") as drc_file:
        binary = DracoPy.encode(points, faces, quantization_bits=bit)
        drc_file.write(binary)


def downsample_mesh_vedo(points: np.ndarray, faces: np.ndarray, num_cells: int):
    # lead to better prediction details than downsample_mesh_open3d
    mesh = vedo.Mesh([points, faces])
    mesh_downsample = mesh.decimate(num_cells / mesh.ncells)
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
    graph.to(device)
    print("Start model prediction")
    predictions, offsets = model(
        graph.num_graphs, graph.x, graph.edge_index, graph.batch
    )
    predictions = predictions.cpu().float().numpy()
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
    path_drc = "./LowerJawScan.drc"
    device = torch.device("cuda:3")
    ### Don't change params below ###
    if jaw_type == "max":
        script_path = "./max_teeth_seg_model_script.pt"
    elif jaw_type == "man":
        script_path = "./man_teeth_seg_model_script.pt"
    torch.cuda.set_device(device)
    model = torch.jit.load(script_path)
    model.to(device)
    model.eval()
    points_origin, faces_origin = drc_decode(path_drc)
    labels, labels_origin, points, faces = predict(
        device,
        model,
        points_origin,
        faces_origin,
        downsample_num=50000,
        graph_k=8,
    )
