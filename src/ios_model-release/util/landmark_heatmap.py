# %%
import glob
import os

import numpy as np
import pandas as pd
import vedo
from landmark_fcsv_preprocess import fcsv_preprocess
from tqdm import tqdm


# %%
def extract_roi(mesh, face_idx_selected):
    points = mesh.points()
    faces = mesh.cells()  # 3-point index of each face
    face_selected = np.array(faces)[face_idx_selected.tolist()]
    select_points_idx = np.unique(face_selected)
    select_points = points[select_points_idx]
    # reindex points
    points_idx_mapping = {i: idx for idx, i in enumerate(select_points_idx)}
    face_selected = np.vectorize(points_idx_mapping.get)(face_selected)
    roi_mesh = vedo.Mesh([select_points, face_selected])
    for key in mesh.celldata.keys():
        roi_mesh.celldata[key] = mesh.celldata[key][face_idx_selected]
    return roi_mesh


# TODO: we can use graph to propagate heatmap value to k-hop neighborhood
def heatmap(landmark_location, points, faces, std, threshold):
    """
    Computes a heatmap for a given landmark location and a mesh.

    Args:
        landmark_location (numpy.ndarray): The location of the landmark in 3D space.
        points (numpy.ndarray): The 3D coordinates of the mesh vertices.
        faces (numpy.ndarray): The mesh faces.
        std (float): The standard deviation of the Gaussian kernel used to compute the heatmap.
        threshold (float): The maximum distance at which the heatmap value is non-zero.

    Returns:
        closest_point_idx (int): The index of the closest vertex to the landmark.
        heatmap_value (numpy.ndarray): The heatmap values for each vertex of the mesh.
    """
    distances_square = np.sum((landmark_location - points) ** 2, axis=1)
    closest_point_idx = np.argmin(distances_square)
    face_mean_distances_square = np.array(
        [distances_square[face].mean() for face in faces]
    )
    heatmap_value = np.exp(-face_mean_distances_square / (2 * (std**2)))
    lowest_value = np.exp(-threshold / (2 * (std**2)))
    heatmap_value[heatmap_value < lowest_value] = 0

    return closest_point_idx, heatmap_value


def get_heatmap(
    landmark_category, landmark_csv, teeth_idx, points, faces, std, threshold
):
    category_heatmap = np.zeros(len(faces))
    category_location = np.ones([3]) * 1e8
    category_point_idx = -1
    category_csv = landmark_csv[
        landmark_csv["label"] == f"{landmark_category}{teeth_idx}"
    ]
    if len(category_csv) == 1:
        landmark_location = category_csv[["x", "y", "z"]].values.astype("float")[0]
        closest_point_idx, heatmap_value = heatmap(
            landmark_location, points, faces, std, threshold
        )
        category_heatmap = heatmap_value
        category_location = landmark_location
        category_point_idx = closest_point_idx
    return category_heatmap, category_location, category_point_idx


def process_mesh(mesh, landmark_csv, unique_landmark, std, threshold):
    points = mesh.points()
    faces = mesh.cells()
    landmark_heatmap_dict = {
        landmark_category: np.zeros([mesh.ncells])
        for landmark_category in unique_landmark
    }
    # no landmark information for gingiva
    landmark_location_dict = {
        landmark_category: [np.ones([3]) * 1e8] for landmark_category in unique_landmark
    }
    landmark_point_idx_dict = {
        landmark_category: [-1] for landmark_category in unique_landmark
    }
    for teeth_idx in range(1, 17):
        selected_face_idx = np.where(mesh.celldata["Label"] == teeth_idx)[0]
        if len(selected_face_idx) == 0:
            for landmark_category in unique_landmark:
                landmark_location_dict[landmark_category].append(np.ones([3]) * 1e8)
                landmark_point_idx_dict[landmark_category].append(-1)
            continue
        face_selected = np.array(faces)[selected_face_idx]
        select_points_idx = np.unique(face_selected)
        point_selected = points[select_points_idx]
        # reindex points
        points_idx_mapping = {i: idx for idx, i in enumerate(select_points_idx)}
        face_selected = np.vectorize(points_idx_mapping.get)(face_selected)
        for landmark_category in unique_landmark:
            category_heatmap, category_location, category_point_idx = get_heatmap(
                landmark_category,
                landmark_csv,
                teeth_idx,
                point_selected,
                face_selected,
                std,
                threshold,
            )
            # map point idx back to original order
            category_point_idx = (
                select_points_idx[category_point_idx]
                if category_point_idx != -1
                else category_point_idx
            )
            landmark_heatmap_dict[landmark_category][
                selected_face_idx
            ] = category_heatmap
            landmark_location_dict[landmark_category].append(category_location)
            landmark_point_idx_dict[landmark_category].append(category_point_idx)
    for k, v in landmark_heatmap_dict.items():
        mesh.celldata[f"{k}_heatmap"] = v
    for k, v in landmark_location_dict.items():
        mesh.metadata[f"{k}_location"] = np.concatenate(v)
    for k, v in landmark_point_idx_dict.items():
        mesh.metadata[f"{k}_point_idx"] = np.array(v)
    return mesh


# %%
if __name__ == "__main__":
    unique_landmark_max = [
        "BTP",
        "BTT",
        "CCT",
        "CTF",
        "DBT",
        "DCP",
        "DTT",
        "IEP",
        "LGP",
        "MBT",
        "MCP",
        "MTT",
        "PGP",
    ]

    unique_landmark_man = [
        "BTP",
        "BTT",
        "CCT",
        "CTF",
        "DBT",
        "DCP",
        "DPP",
        "DTT",
        "IEP",
        "LGP",
        "MBH",
        "MBT",
        "MCP",
        "MTT",
        "PGP",
    ]
    std = 1
    threshold = 4
    jaw_type = "man"
    fcsv_root = f"../../ios_data/{jaw_type}/fcsvs/"
    if jaw_type == "max":
        unique_landmark = unique_landmark_max
    elif jaw_type == "man":
        unique_landmark = unique_landmark_man
    load_teeth_label = False
    if load_teeth_label:
        teeth_label_csv = pd.read_csv(
            f"../../ios_data/teeth_labels/ios_{jaw_type}_teeth_labels.csv"
        )
    mesh_root = f"../../ios_data/{jaw_type}/vtps/"
    save_root = f"../data/intraoral_scanners/{jaw_type}/landmark_heatmap_std{std}_threshold{threshold}/"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    fcsv_path_list = glob.glob(os.path.join(fcsv_root, "*.fcsv"))
    for fcsv_path in tqdm(fcsv_path_list):
        filename = fcsv_path.split("/")[-1].split(".")[0]
        mesh_path = os.path.join(mesh_root, f"{filename}.vtp")
        mesh = vedo.load(mesh_path)
        landmark_csv = fcsv_preprocess(fcsv_path)
        mesh = process_mesh(mesh, landmark_csv, unique_landmark, std, threshold)
        if load_teeth_label:
            if f"{filename}.vtp" not in teeth_label_csv.file_name.values:
                continue
            _, abrasion, twisted, tilted, ectopic = teeth_label_csv[
                teeth_label_csv.file_name == f"{filename}.vtp"
            ].values[0]
            for k, v in zip(
                ["abrasion", "twisted", "tilted", "ectopic"],
                [abrasion, twisted, tilted, ectopic],
            ):
                unique_teeth_idx = np.unique(mesh.celldata["Label"]).astype("int")
                if type(v) == str:
                    target_teeth = v.replace("，", ",").split(",")
                    target_teeth = [int(i) for i in target_teeth if i != ""]
                elif type(v) == float or type(v) == int:
                    if np.isnan(v):
                        target_teeth = []
                    else:
                        target_teeth = [int(v)]
                mesh.metadata[k] = np.isin(unique_teeth_idx, target_teeth).astype("int")
        vedo.write(mesh, os.path.join(save_root, f"./{filename}.vtp"))

# %%
