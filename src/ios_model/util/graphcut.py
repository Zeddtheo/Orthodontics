# %%
import numpy as np
import scipy.sparse as sp
import torch
import trimesh
import vedo
from pygco import cut_from_graph
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from tqdm import tqdm

# %%
EPS = 1e-16


def get_boundary_cells(faces, labels, num_hops):
    edge_index = torch.LongTensor(trimesh.graph.face_adjacency(faces=faces).T)
    edge_row_label = labels[edge_index[0]]
    edge_col_label = labels[edge_index[1]]
    boundary_edge_mask = edge_col_label != edge_row_label
    boundary_edges = edge_index[:, boundary_edge_mask.flatten()]
    boundary_cell_idx = torch.unique(boundary_edges)
    # Get the k-hop subgraph of the boundary cells
    boundary_cell_idx, _, _, _ = k_hop_subgraph(
        node_idx=boundary_cell_idx,
        num_hops=num_hops,
        edge_index=edge_index,
    )
    return boundary_cell_idx


# %%
def graph_cut(
    probs: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    round_factor: int = 100,
    num_classes: int = 17,
    lambda_c: int = 30,
    num_hops: int = 6,
):
    labels = np.argmax(probs, axis=1)
    boundary_cell_idx = get_boundary_cells(faces, labels, num_hops=num_hops)

    # only apply graph cut on boundary cells
    probs = probs[boundary_cell_idx]
    faces = faces[boundary_cell_idx]
    positions = positions[boundary_cell_idx]
    normals = normals[boundary_cell_idx]

    probs[probs < 1.0e-6] = 1.0e-6
    unaries = -round_factor * np.log10(probs)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # parawise
    pairwise = 1 - np.eye(num_classes, dtype=np.int32)

    # edges
    source, target = trimesh.graph.face_adjacency(faces=faces).T
    p1 = np.einsum("ij,ij->i", normals[source], normals[target])
    p2 = np.linalg.norm(normals[source], axis=1)
    p3 = np.linalg.norm(normals[target], axis=1)
    cos_theta = p1 / (p2 * p3 + EPS)
    cos_theta = np.clip(cos_theta, -1, 0.9999)
    theta = np.arccos(cos_theta)
    phi = np.linalg.norm(positions[source] - positions[target], axis=1)
    beta = 1 + np.linalg.norm(p1.reshape(-1, 1), axis=1)
    concave_feature = -np.log10(theta / np.pi) * phi
    edge_feature = -beta * np.log10(theta / np.pi) * phi
    edge_feature[theta > np.pi / 2.0] = concave_feature[theta > np.pi / 2.0]
    edge_feature *= lambda_c * round_factor
    edges = np.ascontiguousarray(
        np.stack([source, target, edge_feature]).astype(np.int32).T
    )
    labels[boundary_cell_idx] = cut_from_graph(edges, unaries, pairwise)
    return labels


def remove_small_components(
    faces: np.ndarray, labels: np.ndarray, min_cell_num: int = 500
):
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
        if max(counts) > min_cell_num:
            # Get the most common unique component
            mode_unique = uniques[np.argmax(counts)]
            # Get the mask for the non-dominant components within the current label
            mask = (component != mode_unique) & mask
        # Set the labels of non-dominant components to 0 (background), or remove it completely if the ROI is too small
        labels[mask] = 0
    return labels
