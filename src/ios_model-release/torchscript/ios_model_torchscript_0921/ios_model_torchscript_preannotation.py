# %%
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import vedo
from mesh_dataset_det import graph_preprocess_det
from mesh_dataset_seg import graph_preprocess_seg
from torch_cluster import knn_graph
from ios_model_torchscript import teeth_segmentation, cut_mesh
from utils import (
    downsample_mesh_vedo,
    drc_decode,
    extract_roi,
    pose_pca,
    remove_small_components,
    upsample,
)


# %%
@torch.no_grad()
def landmark_detection(
    device: torch.device,
    model_det: torch.jit._script.RecursiveScriptModule,
    mesh: vedo.mesh.Mesh,
    prob: torch.Tensor,
    pred: torch.Tensor,
    graph_k: int,
    coordinate_scale: int,
):
    graph = graph_preprocess_det(
        mesh,
        graph_k,
        coordinate_scale,
    )
    graph.num_graphs = 1
    graph.batch = torch.zeros(graph.x.shape[0]).long().to(device)
    graph.to(device)
    x_pos = graph.x[:, :3]
    z, _, _ = model_det.extract_feature(
        graph.x, x_pos, graph.edge_index, graph.batch, run_predictor=False
    )
    heatmap = model_det.run_detector(
        z, x_pos, graph.edge_index, graph.batch, prob, pred
    )
    batch_teeth = pred.cpu()
    unique_teeth = torch.unique(batch_teeth).numpy()
    landmark_outputs = {}
    points_mean = mesh.points().mean(axis=0)
    for idx, (category, teeth_list) in enumerate(landmark_teeth_dict.items()):
        category_heatmap = heatmap[:, idx]
        mesh.celldata[f"{category}_heatmap_pred"] = category_heatmap.cpu().numpy()
        for teeth_idx in set(teeth_list) & set(unique_teeth):
            batch_idx = torch.where(batch_teeth == teeth_idx)[0]
            batch_prob = category_heatmap[batch_idx]
            pred_location_idx_batch = torch.argmax(batch_prob)
            pred_location_idx = batch_idx[pred_location_idx_batch]
            pred_location = x_pos[pred_location_idx].cpu().numpy()
            pred_location = pred_location * coordinate_scale + points_mean
            if teeth_idx not in landmark_outputs:
                landmark_outputs[teeth_idx] = {category: pred_location.tolist()}
            else:
                landmark_outputs[teeth_idx][category] = pred_location.tolist()
    return mesh, landmark_outputs


@torch.no_grad()
def predict(
    device: torch.device,
    model_seg: torch.jit._script.RecursiveScriptModule,
    model_det: torch.jit._script.RecursiveScriptModule,
    points_origin: np.ndarray,
    faces_origin: np.ndarray,
    downsample_num: int,
    graph_k: int,
    coordinate_scale: int,
):
    """
    Args:
        device: cude device
        model_seg: torchscript model for teeth segmentation
        model_det: torchscript model for landmark detection
        point_origin: origin mesh points
        faces_origin: origin mesh faces
        downsample_num: cell number of downsampled mesh
        graph_k: k of knn graph for mesh graph construction

    Returns:
        labels_downsample: predicted labels of downsampled mesh
        labels_origin: predicted labels of origin mesh
        points: downsampled mesh points
        faces: downsampled mesh faces
    """
    print("Cut mesh")
    selected_face_mask, point_selected, face_selected = cut_mesh(
        device,
        model_seg,
        points_origin,
        faces_origin,
        downsample_num,
        graph_k,
    )
    print("Teeth segmentation")
    mesh, probs_ds, preds_ds, preds_selected = teeth_segmentation(
        device,
        model_seg,
        point_selected,
        face_selected,
        downsample_num,
        graph_k,
        upsample_flag=True,
    )
    mesh.celldata["Label_pred"] = preds_ds
    print("Landmark detection")
    mesh, landmark_outputs = landmark_detection(
        device,
        model_det,
        mesh,
        probs_ds,
        torch.LongTensor(preds_ds).to(device),
        graph_k,
        coordinate_scale,
    )
    preds_origin = np.zeros(len(faces_origin), dtype=int)
    preds_origin[selected_face_mask] = preds_selected
    mesh_origin = vedo.Mesh([points_origin, faces_origin])
    mesh_origin.celldata["Label_pred"] = preds_origin
    return mesh, mesh_origin, landmark_outputs


# %%
if __name__ == "__main__":
    ### Define Parameters Below ###
    jaw_type = "man"
    mesh_path = "../../data/intraoral_scanners/demo_test/strau_testing/John Doe.stl"
    save_root = "./"
    device = torch.device("cuda:9")
    torch.cuda.set_device(device)
    ### Don't change params below ###
    if jaw_type == "max":
        seg_script_path = "./max_teeth_seg_model_script.pt"
        det_script_path = "./max_multitask_model_script.pt"
    elif jaw_type == "man":
        seg_script_path = "./man_teeth_seg_model_script.pt"
        det_script_path = "./man_multitask_model_script.pt"
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
    model_seg = torch.jit.load(seg_script_path)
    model_det = torch.jit.load(det_script_path)
    model_seg.to(device)
    model_det.to(device)
    model_seg.eval()
    model_det.eval()
    mesh = vedo.load(mesh_path)
    points_origin = np.array(mesh.points())
    faces_origin = np.array(mesh.faces())
    mesh, mesh_origin, landmark_outputs = predict(
        device,
        model_seg,
        model_det,
        points_origin,
        faces_origin,
        downsample_num=50000,
        graph_k=8,
        coordinate_scale=10,
    )

    # %%
    mesh_name = mesh_path.split("/")[-1].split(".")[0]
    mesh.write(os.path.join(save_root, f"{mesh_name}_pred.vtp"))
    mesh_origin.write(os.path.join(save_root, f"{mesh_name}_origin_pred.vtp"))
    locations = []
    for teeth_idx, values in landmark_outputs.items():
        for category, [x, y, z] in values.items():
            locations.append([teeth_idx, category, x, y, z])
    df = pd.DataFrame(locations, columns=["teeth_idx", "category", "x", "y", "z"])
    df.to_csv(os.path.join(save_root, f"{mesh_name}.csv"), index=False)

# %%
