# %%
import glob
import os

import numpy as np
import vedo
from stl import mesh


def combined_stl(mesh_path_list):
    meshes = [mesh.Mesh.from_file(path) for path in mesh_path_list]
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
    return combined


def merge_stl_to_vtp(mesh_path_list):
    merged_mesh_points = []
    merged_mesh_cells = []
    merged_mesh_cell_labels = []
    for mesh_path in mesh_path_list:
        mesh = vedo.load(mesh_path)
        idx = mesh_path.split("/")[-1].split("_")[1]
        if len(merged_mesh_points) == 0:
            merged_mesh_points = mesh.points()
        else:
            merged_mesh_points = np.concatenate([merged_mesh_points, mesh.points()])
        new_mesh_cells = mesh.cells()
        if len(merged_mesh_cells) == 0:
            merged_mesh_cells = new_mesh_cells
        else:
            merged_mesh_cells = np.concatenate(
                [merged_mesh_cells, new_mesh_cells + np.max(merged_mesh_cells) + 1]
            )
        new_mesh_cell_labels = np.array([teeth_idx_mapping[idx]] * len(new_mesh_cells))
        if len(merged_mesh_cell_labels) == 0:
            merged_mesh_cell_labels = new_mesh_cell_labels
        else:
            merged_mesh_cell_labels = np.concatenate(
                [merged_mesh_cell_labels, new_mesh_cell_labels]
            )
    merged_mesh = vedo.Mesh([merged_mesh_points, merged_mesh_cells])
    merged_mesh.celldata["Label"] = merged_mesh_cell_labels
    return merged_mesh


# %%
if __name__ == "__main__":
    teeth_idx_mapping = {
        "11": 7,
        "12": 6,
        "13": 5,
        "14": 4,
        "15": 3,
        "16": 2,
        "17": 1,
        "18": 15,
        "21": 8,
        "22": 9,
        "23": 10,
        "24": 11,
        "25": 12,
        "26": 13,
        "27": 14,
        "28": 16,
        "31": 7,
        "32": 6,
        "33": 5,
        "34": 4,
        "35": 3,
        "36": 2,
        "37": 1,
        "38": 15,
        "41": 8,
        "42": 9,
        "43": 10,
        "44": 11,
        "45": 12,
        "46": 13,
        "47": 14,
        "48": 16,
    }
    stl_root = '../data/implant_design/标准牙列模型/18_ZRS-00/xia/'
    root_idx = stl_root.split('/')[-3].split('_')[0]
    mesh_path_list = glob.glob(os.path.join(stl_root, "*.stl"))
    save_path = os.path.join(stl_root, f"standard_{root_idx}_man.vtp")
    merged_vtp = merge_stl_to_vtp(mesh_path_list)
    vedo.write(merged_vtp, save_path)

# %%
