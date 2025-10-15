# %%
import numpy as np
import torch
import torch_geometric
from torch_cluster import knn_graph

EPS = 1e-16


def preprocess_cell_features(
    points: torch.Tensor,
    cells: torch.Tensor,
    positions: torch.Tensor,
    normalize_coordinate: bool,
):
    # customized normal calculation; the vtk/vedo build-in function will change number of points
    v1 = cells[:, 0] - cells[:, 1]
    v2 = cells[:, 1] - cells[:, 2]
    normals = torch.cross(v1, v2)
    normal_length = torch.norm(normals, dim=1)
    normals /= normal_length[:, None] + EPS
    if normalize_coordinate:
        cells_norm = (cells - points.min(dim=0).values) / (
            points.max(dim=0).values - points.min(dim=0).values
        )
        positions_norm = (positions - points.min(dim=0).values) / (
            points.max(dim=0).values - points.min(dim=0).values
        )
        x = torch.cat((positions_norm, normals, cells_norm.view(-1, 9)), dim=1)
    else:
        x = torch.cat((positions, normals, cells.view(-1, 9)), dim=1)
    return x, normals


def graph_preprocess(
    points: np.ndarray, faces: np.ndarray, graph_k: int, normalize_coordinate: bool
):
    points = torch.FloatTensor(points)  # 3d coordinates of points
    faces = torch.LongTensor(faces)  # 3-point index of each face
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    positions = cells.mean(1)  # 3d coordinates of cell centers
    x, normals = preprocess_cell_features(
        points, cells, positions, normalize_coordinate
    )
    edge_index = torch_geometric.nn.knn_graph(positions, k=graph_k, loop=True)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
    )
    graph.pos = positions
    return graph, normals.cpu().numpy()


def graph_preprocess_w_label(
    points: np.ndarray,
    faces: np.ndarray,
    cell_labels: np.ndarray,
    graph_k: int,
    normalize_coordinate: bool,
):
    graph, normals = graph_preprocess(points, faces, graph_k, normalize_coordinate)
    cell_labels = torch.LongTensor(cell_labels).reshape(-1, 1)
    cell_label_norm = cell_labels / 16
    # add cell labels as input features
    graph.x = torch.cat((graph.x, cell_label_norm), dim=1)
    graph.cell_label = cell_labels
    return graph


# %%
