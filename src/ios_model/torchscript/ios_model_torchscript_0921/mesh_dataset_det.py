# %%
import numpy as np
import torch
import torch_geometric
import vedo
from torch_geometric.utils import k_hop_subgraph


def preprocess_cell_features_det(mesh: vedo.mesh.Mesh, coordinate_scale: int):
    mesh.compute_normals()
    normals = torch.FloatTensor(mesh.celldata["Normals"])  # normals of cells
    points = torch.FloatTensor(mesh.points())  # 3d coordinates of points
    # move to center and rescale, need reversed operation in postprocess
    points = (points - points.mean(0)) / coordinate_scale
    faces = torch.LongTensor(mesh.faces())  # 3-point index of each face
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    centers = cells.mean(1)  # 3d coordinates of cell centers
    x = torch.cat((centers, normals, cells.view(-1, 9)), dim=1)
    return x, points


def graph_preprocess_det(
    mesh: vedo.mesh.Mesh,
    graph_k: int,
    coordinate_scale: int,
):
    x, points = preprocess_cell_features_det(mesh, coordinate_scale)
    x_pos = x[:, :3]  # preprocessed cell center positions
    edge_index = torch_geometric.nn.knn_graph(x_pos, k=graph_k, loop=True)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
    )
    return graph
