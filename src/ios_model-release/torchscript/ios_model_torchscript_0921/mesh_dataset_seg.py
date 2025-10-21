# %%
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import k_hop_subgraph

EPS = 1e-16


def preprocess_cell_features_seg(points: np.ndarray, faces: np.ndarray):
    # Make a copy of the points array to avoid in-place operation on the original array
    points_copy = points.copy()
    points_copy -= points_copy.mean(0)  # move centroid to the origin
    cells = points_copy[faces]  # 3x3 dimensional coordinates of faces
    positions = cells.mean(1)  # 3d coordinates of cell centers

    # customized normal calculation; the vtk/vedo build-in function will change number of points
    v1 = cells[:, 0] - cells[:, 1]
    v2 = cells[:, 1] - cells[:, 2]
    normals = np.cross(v1, v2)
    normal_length = np.linalg.norm(normals, axis=1)
    normals[:, 0] /= normal_length[:] + EPS
    normals[:, 1] /= normal_length[:] + EPS
    normals[:, 2] /= normal_length[:] + EPS
    normals = (normals - normals.mean(axis=0)) / normals.std(axis=0)
    return cells, positions, normals


def graph_preprocess_seg(points: np.ndarray, faces: np.ndarray, graph_k: int = 10):
    cells, positions, normals = preprocess_cell_features_seg(points, faces)
    X = np.column_stack((cells.reshape(-1, 9), positions, normals))
    edge_index = torch_geometric.nn.knn_graph(
        torch.FloatTensor(positions), k=graph_k, loop=True
    )
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = torch_geometric.data.Data(
        x=torch.FloatTensor(X),
        edge_index=edge_index,
        pos=torch.FloatTensor(positions),
    )
    return graph, normals
