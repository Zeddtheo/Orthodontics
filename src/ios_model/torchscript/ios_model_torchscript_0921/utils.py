# %%
import DracoPy
import numpy as np
import scipy.sparse as sp
import torch
import trimesh
import vedo
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.utils import to_scipy_sparse_matrix


# %%
def drc_decode(path_drc: str):
    with open(path_drc, "rb") as drc_file:
        mesh = DracoPy.decode(drc_file.read())
        points = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces)
    return points, faces


def downsample_mesh_vedo(points: np.ndarray, faces: np.ndarray, num_cells: int):
    mesh = vedo.Mesh([points, faces])
    mesh_downsample = mesh.decimate(num_cells / mesh.ncells)
    points_downsample = np.asarray(mesh_downsample.points())
    faces_downsample = np.asarray(mesh_downsample.faces())
    return mesh_downsample, points_downsample, faces_downsample


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
    probs_upsample = neigh.predict_proba(positions_upsample)
    labels_upsample = neigh.predict(positions_upsample)
    # in case of missing labels
    unique_labels = np.unique(labels)
    probs_upsample_full = np.zeros((len(positions_upsample), 17))
    probs_upsample_full[:, unique_labels] = probs_upsample
    return probs_upsample_full, labels_upsample


def extract_roi(points: np.ndarray, faces: np.ndarray, selected_face_mask: np.ndarray):
    face_selected = faces[selected_face_mask]
    points_idx_selected = np.unique(face_selected)
    point_selected = points[points_idx_selected]
    # reindex points
    points_idx_mapping = {i: idx for idx, i in enumerate(points_idx_selected)}
    face_selected = np.vectorize(points_idx_mapping.get)(face_selected)
    return point_selected, face_selected


def pose_pca(points: np.ndarray, faces: np.ndarray, labels: np.ndarray):
    teeth_points_idx = np.unique(faces[labels > 0])
    teeth_points = points[teeth_points_idx]
    pca = PCA(n_components=3)
    pca.fit(teeth_points)
    points_pca = pca.transform(points)
    teeth_points_pca = points_pca[teeth_points_idx]
    z_min = np.min(points_pca[:, 2])
    z_max = np.max(points_pca[:, 2])
    teeth_z_min = np.min(teeth_points_pca[:, 2])
    teeth_z_max = np.max(teeth_points_pca[:, 2])
    return pca, z_min, z_max, teeth_z_min, teeth_z_max
