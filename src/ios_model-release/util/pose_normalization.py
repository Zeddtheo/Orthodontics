# %%
import glob
import os

import numpy as np
import vedo
from sklearn.decomposition import PCA


def extract_roi(points: np.ndarray, faces: np.ndarray, selected_face_mask: np.ndarray):
    face_selected = faces[selected_face_mask]
    points_idx_selected = np.unique(face_selected)
    point_selected = points[points_idx_selected]
    # reindex points
    points_idx_mapping = {i: idx for idx, i in enumerate(points_idx_selected)}
    face_selected = np.vectorize(points_idx_mapping.get)(face_selected)
    return point_selected, face_selected


# %%
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


def load_mesh(mesh_path: str, label_key: str = "Label"):
    mesh = vedo.load(mesh_path)
    points = mesh.points()
    faces = np.array(mesh.faces())
    labels = mesh.celldata[label_key]
    return points, faces, labels


def cut_mesh(
    points: np.ndarray, faces: np.ndarray, pca: PCA, z_min: float, z_max: float
):
    points_pca = pca.transform(points)
    centers_pca = points_pca[faces].mean(axis=1)
    selected_face_mask = (centers_pca[:, 2] > z_min) & (centers_pca[:, 2] < z_max)
    point_selected, face_selected = extract_roi(points, faces, selected_face_mask)
    return point_selected, face_selected


# %%
vtp_mesh_path = (
    "../data/intraoral_scanners/demo_test/strau_testing/Chris Birth_pred.vtp"
)
points, faces, labels = load_mesh(vtp_mesh_path, label_key="Label_pred")
pca, z_min, z_max, teeth_z_min, teeth_z_max = pose_pca(points, faces, labels)

if (z_max - z_min) > 2 * (teeth_z_max - teeth_z_min):
    z_delta = 0.5 * (teeth_z_max - teeth_z_min)
    stl_mesh_path = "../data/intraoral_scanners/demo_test/strau_testing/Chris Birth.stl"
    points_stl, faces_stl, _ = load_mesh(stl_mesh_path)
    point_stl_selected, face_stl_selected = cut_mesh(
        points_stl, faces_stl, pca, teeth_z_min - z_delta, teeth_z_max + z_delta
    )
    mesh_stl_cut = vedo.Mesh([point_stl_selected, face_stl_selected])
    mesh_stl_cut.write(stl_mesh_path.replace(".stl", "_cut.stl"))
# %%
