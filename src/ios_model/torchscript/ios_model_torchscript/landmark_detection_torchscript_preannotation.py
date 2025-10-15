# %%
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import vedo
from mesh_dataset import graph_preprocess_w_label
from torch_cluster import knn_graph
from tqdm import tqdm


# %%
def mesh_decode(path_mesh: str):
    mesh = vedo.load(path_mesh)
    points = np.array(mesh.points())
    faces = np.array(mesh.faces())
    labels = np.array(mesh.celldata["Label"].astype("int32"))
    return points, faces, labels


def extract_teeth_roi(points: np.ndarray, faces: np.ndarray, labels: np.ndarray):
    face_idx_selected = np.where(labels > 0)[0]  # remove gingiva
    face_selected = faces[face_idx_selected]
    label_selected = labels[face_idx_selected]
    points_idx_selected = np.unique(face_selected)
    point_selected = points[points_idx_selected]
    # reindex points
    points_idx_mapping = {i: idx for idx, i in enumerate(points_idx_selected)}
    face_selected = np.vectorize(points_idx_mapping.get)(face_selected)
    return point_selected, face_selected, label_selected


# %%
if __name__ == "__main__":
    ### Define Parameters Below ###
    jaw_type = "max"
    save_root = "./"
    path_mesh_list = glob(
        "../../data/intraoral_scanners/max/landmark_heatmap_std1_threshold4/*.vtp"
    )
    device = torch.device("cuda:0")
    ### Don't change params below ###
    if jaw_type == "max":
        script_path = "./max_landmark_detection_model_script.pt"
    elif jaw_type == "man":
        script_path = "./man_landmark_detection_model_script.pt"
    graph_k = 12
    torch.cuda.set_device(device)
    model = torch.jit.load(script_path)
    model.to(device)
    model.eval()
    landmark_teeth_dict = OrderedDict(
        [
            ("CCT", [5, 10]),
            ("CTF", list(range(1, 17))),
            ("DCP", list(range(1, 17))),
            ("IEP", [6, 7, 8, 9]),
            ("LGP", list(range(1, 17))),
            ("MCP", list(range(1, 17))),
            ("PGP", list(range(1, 17))),
        ]
    )
    # %%
    for path_mesh in tqdm(path_mesh_list[:2]):
        mesh_name = path_mesh.split("/")[-1].split(".")[0]
        points, faces, labels = mesh_decode(path_mesh)
        point_selected, face_selected, label_selected = extract_teeth_roi(
            points, faces, labels
        )
        cell_selected = point_selected[face_selected]
        position_selected = cell_selected.mean(1)
        graph = graph_preprocess_w_label(
            point_selected,
            face_selected,
            label_selected,
            graph_k=graph_k,
            normalize_coordinate=True,
        )
        graph.num_graphs = 1
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        graph.to(device)
        teeth_batch = graph.batch * 17 + graph.cell_label.flatten()
        with torch.no_grad():
            probs = model(graph.x, graph.edge_index, graph.batch, teeth_batch)
        location_outputs = []
        mesh_outputs = vedo.Mesh([point_selected, face_selected])
        mesh_outputs.celldata["Label"] = label_selected
        unique_teeth = torch.unique(teeth_batch).cpu().numpy()
        for idx, (category, teeth_list) in enumerate(landmark_teeth_dict.items()):
            prob = probs[0, idx]
            mesh_outputs.celldata[f"{category}_prob"] = prob.cpu().numpy()
            for teeth_idx in set(teeth_list) & set(unique_teeth):
                batch_idx = torch.where(teeth_batch == teeth_idx)[0]
                batch_prob = prob[batch_idx]
                pred_location_idx_batch = torch.argmax(batch_prob)
                pred_location_idx = batch_idx[pred_location_idx_batch]
                x, y, z = position_selected[pred_location_idx.item()]
                location_outputs.append([teeth_idx, category, x, y, z])
        mesh_outputs.write(f"./{mesh_name}.vtp")
        df = pd.DataFrame(
            location_outputs, columns=["teeth_idx", "category", "x", "y", "z"]
        )
        df.to_csv(f"./{mesh_name}.csv", index=False)

# %%
