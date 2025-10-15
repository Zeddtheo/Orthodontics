# %%
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import vedo
from graphcut import graph_cut
from mesh_dataset_det import graph_preprocess_det
from mesh_dataset_seg import graph_preprocess_seg, preprocess_cell_features_seg
from torch_cluster import knn_graph
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
def teeth_segmentation(
    device: torch.device,
    model_seg: torch.jit._script.RecursiveScriptModule,
    points_origin: np.ndarray,
    faces_origin: np.ndarray,
    downsample_num: int,
    graph_k: int,
    upsample_flag: bool,
):
    downsample_flag = downsample_num is not None and len(faces_origin) > downsample_num
    if downsample_flag:
        mesh, points, faces = downsample_mesh_vedo(
            points_origin, faces_origin, downsample_num
        )
    else:
        points, faces = points_origin, faces_origin
        mesh = vedo.Mesh([points, faces])
    cells = points[faces]
    positions = cells.mean(1)
    graph, normals = graph_preprocess_seg(points, faces, graph_k=graph_k)
    graph.num_graphs = 1
    graph.batch = torch.zeros(graph.x.shape[0]).long().to(device)
    graph.to(device)
    probs, offsets = model_seg(
        graph.num_graphs, graph.x, graph.pos, graph.edge_index, graph.batch
    )
    preds = graph_cut(
        probs.cpu().numpy(),
        faces,
        positions,
        normals,
        round_factor=100,
        num_classes=17,
        lambda_c=30,
        num_hops=-1,
    )
    preds = remove_small_components(faces, preds, min_cell_num=500)
    if downsample_flag and upsample_flag:
        cells_origin = points_origin[faces_origin]
        positions_origin = cells_origin.mean(1)
        _, _, normals_origin = preprocess_cell_features_seg(points_origin, faces_origin)
        probs_origin, preds_origin = upsample(
            positions, preds, positions_origin, n_neighbors=1
        )
        preds_origin = graph_cut(
            probs_origin,
            faces_origin,
            positions_origin,
            normals_origin,
            round_factor=100,
            num_classes=17,
            lambda_c=30,
            num_hops=-1,
        )
    else:
        preds_origin = preds
    return mesh, probs, preds, preds_origin


def cut_mesh(
    device: torch.device,
    model_seg: torch.jit._script.RecursiveScriptModule,
    points_origin: np.ndarray,
    faces_origin: np.ndarray,
    downsample_num: int,
    graph_k: int,
):
    mesh, probs, preds, _ = teeth_segmentation(
        device,
        model_seg,
        points_origin,
        faces_origin,
        downsample_num,
        graph_k,
        upsample_flag=False,
    )
    points = mesh.points()
    faces = np.array(mesh.faces())
    pca, z_min, z_max, teeth_z_min, teeth_z_max = pose_pca(points, faces, preds)
    # if the z axis range is too large, cut the mesh based on teeth length
    if (z_max - z_min) > 2 * (teeth_z_max - teeth_z_min):
        z_delta = 0.5 * (teeth_z_max - teeth_z_min)
        points_origin_pca = pca.transform(points_origin)
        centers_origin_pca = points_origin_pca[faces_origin].mean(axis=1)
        selected_face_mask = (centers_origin_pca[:, 2] > teeth_z_min - z_delta) & (
            centers_origin_pca[:, 2] < teeth_z_max + z_delta
        )
        point_selected, face_selected = extract_roi(
            points_origin, faces_origin, selected_face_mask
        )
    else:
        point_selected, face_selected = points_origin, faces_origin
        selected_face_mask = np.ones(len(faces_origin), dtype=bool)
    return selected_face_mask, point_selected, face_selected


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
    return landmark_outputs


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
    print("Landmark detection")
    landmark_outputs = landmark_detection(
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
    return preds_origin, landmark_outputs


if __name__ == "__main__":
    ### Define Parameters Below ###
    jaw_type = "man"
    path_drc = "./LowerJawScan.drc"
    device = torch.device("cuda:3")
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
    points_origin, faces_origin = drc_decode(path_drc)
    preds_origin, landmark_outputs = predict(
        device,
        model_seg,
        model_det,
        points_origin,
        faces_origin,
        downsample_num=50000,
        graph_k=8,
        coordinate_scale=10,
    )
