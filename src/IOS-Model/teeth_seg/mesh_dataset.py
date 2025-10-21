# %%
import numpy as np
import torch
import torch_geometric
import trimesh
from torch_geometric.utils import k_hop_subgraph

EPS = 1e-16


def label_cell_region(faces: np.ndarray, cell_label: torch.Tensor):
    edge_index = torch.LongTensor(trimesh.graph.face_adjacency(faces=faces).T)
    edge_row_label = cell_label[edge_index[0]]
    edge_col_label = cell_label[edge_index[1]]
    cluster_teeth_edge_mask = (edge_col_label == edge_row_label) & (edge_col_label != 0)
    cluster_teeth_edges = edge_index[:, cluster_teeth_edge_mask.flatten()]
    boundary_gingiva_edge_mask = (edge_col_label != edge_row_label) & (
        edge_col_label * edge_row_label == 0
    )
    boundary_gingiva_edges = edge_index[
        :,
        boundary_gingiva_edge_mask.flatten(),
    ]
    boundary_teeth_edge_mask = (edge_col_label != edge_row_label) & (
        edge_col_label * edge_row_label != 0
    )
    boundary_teeth_edges = edge_index[
        :,
        boundary_teeth_edge_mask.flatten(),
    ]
    cluster_teeth_cell_idx = torch.unique(cluster_teeth_edges)
    boundary_gingiva_cell_idx = torch.unique(boundary_gingiva_edges)
    boundary_teeth_cell_idx = torch.unique(boundary_teeth_edges)

    enhanced_boundary_gingiva_cell_idx, _, _, _ = k_hop_subgraph(
        node_idx=boundary_gingiva_cell_idx,
        num_hops=2,
        edge_index=edge_index,
    )
    enhanced_boundary_teeth_cell_idx, _, _, _ = k_hop_subgraph(
        node_idx=boundary_teeth_cell_idx,
        num_hops=2,
        edge_index=edge_index,
    )
    cell_region_label = torch.zeros_like(cell_label)
    cell_region_label[cluster_teeth_cell_idx] = 1
    cell_region_label[torch.unique(enhanced_boundary_gingiva_cell_idx)] = 2
    cell_region_label[torch.unique(enhanced_boundary_teeth_cell_idx)] = 3
    return cell_region_label


def preprocess_cell_features(points: np.ndarray, faces: np.ndarray):
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


def graph_preprocess(points: np.ndarray, faces: np.ndarray, graph_k: int = 10):
    cells, positions, normals = preprocess_cell_features(points, faces)
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
