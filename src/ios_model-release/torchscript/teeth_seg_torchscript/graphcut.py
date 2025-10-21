# %%
import numpy as np
import trimesh
from pygco import cut_from_graph

# %%
EPS = 1e-16


def graph_cut(
    predictions:np.ndarray,
    faces:np.ndarray,
    positions:np.ndarray,
    normals:np.ndarray,
    round_factor:int=100,
    num_classes:int=17,
    lambda_c:int=30,
):
    predictions[predictions < 1.0e-6] = 1.0e-6

    # unaries
    unaries = -round_factor * np.log10(predictions)
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
    labels = cut_from_graph(edges, unaries, pairwise)
    return labels
