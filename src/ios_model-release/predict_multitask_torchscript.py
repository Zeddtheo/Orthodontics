# %%
import glob
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import vedo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader.CellGraph_dataset import graph_preprocess
from model.CellDGCN import MeshDGCN
from util import logger
from util.graphcut import graph_cut, remove_small_components
from util.metrics import *


@torch.no_grad()
def predict(mesh_path, model):
    mesh = vedo.load(mesh_path)
    graph, mesh = graph_preprocess(
        mesh,
        transform=False,
        graph_k=graph_k,
        test=True,
        sample_num=None,
        return_mesh=True,
        coordinate_scale=coordinate_scale,
        multitask=True,
    )
    graph.num_graphs = 1
    graph.batch = torch.zeros(graph.x.shape[0]).long().to(device)
    # send mini-batch to device
    graph.to(device)
    # forward + backward + optimize
    x_pos = graph.x[:, :3]
    z, prob, offset = model.extract_feature(
        graph.x, x_pos, graph.edge_index, graph.batch, run_predictor=True
    )
    pred = torch.argmax(prob, dim=1)
    mesh.celldata["Label_pred"] = pred.cpu().numpy()
    if "Normals" not in mesh.celldata.keys():
        mesh.compute_normals()
    normals = mesh.celldata["Normals"]
    faces = np.array(mesh.faces())
    pred_graphcut = graph_cut(
        prob.cpu().numpy(),
        faces,
        x_pos.cpu().numpy(),
        normals,
        round_factor=100,
        num_classes=17,
        lambda_c=30,
    )
    pred_graphcut_refined = torch.LongTensor(
        remove_small_components(faces, pred_graphcut, min_cell_num=500)
    )
    mesh.celldata["Label_pred_graphcut"] = pred_graphcut
    mesh.celldata["Label_pred_graphcut_refined"] = pred_graphcut_refined
    heatmap_pred = model.run_detector(
        z, x_pos, graph.edge_index, graph.batch, prob, pred_graphcut_refined.to(device)
    )
    batch_teeth = pred_graphcut_refined
    unique_teeth = torch.unique(batch_teeth).numpy()
    location_outputs = []
    for idx, (category, teeth_list) in enumerate(landmark_teeth_dict.items()):
        category_heatmap_pred = heatmap_pred[:, idx]
        mesh.celldata[f"{category}_heatmap_pred"] = category_heatmap_pred.cpu().numpy()
        for teeth_idx in set(teeth_list) & set(unique_teeth):
            batch_idx = torch.where(batch_teeth == teeth_idx)[0]
            batch_prob = category_heatmap_pred[batch_idx]
            pred_location_idx_batch = torch.argmax(batch_prob)
            pred_location_idx = batch_idx[pred_location_idx_batch]
            x, y, z = x_pos[pred_location_idx].cpu().numpy()
            location_outputs.append([teeth_idx, category, x, y, z])
    df = pd.DataFrame(
        location_outputs, columns=["teeth_idx", "category", "x", "y", "z"]
    )
    return mesh, df


# %%
if __name__ == "__main__":
    device = "cuda:9"
    torch.cuda.set_device(device)  # assign which gpu will be used (only linux works)
    checkpoint_folder = "/raid/zichenwang/random/ios_model/checkpoint/multitask_man_5w_20230731_00-32-25/"
    weight_path = os.path.join(checkpoint_folder, "weight/best_model.tar")
    script_path = os.path.join(
        checkpoint_folder, "weight/man_multitask_model_script.pt"
    )
    in_channels = 15
    extractor_hidden_channels = 64
    predictor_hidden_channels = 256
    detector_hidden_channels = 256
    out_channels = 17
    graph_k = 8
    coordinate_scale = 10
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

    # set model
    model = MeshDGCN(
        in_channels,
        extractor_hidden_channels,
        predictor_hidden_channels,
        detector_hidden_channels,
        out_channels,
        list(landmark_teeth_dict.keys()),
    )
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    model.eval()
    model_script = torch.jit.script(model)
    torch.jit.save(model_script, script_path)
    del model
    # %%
    model_script = torch.jit.load(script_path)
    model_script.to(device)
    model_script.eval()

    data_path_list = np.load(
        f"./data/intraoral_scanners/adult_man_vtp_5w_list_multitask.npy"
    )
    train_list, test_list = train_test_split(
        data_path_list, train_size=0.9, shuffle=True, random_state=0
    )
    train_list, val_list = train_test_split(
        train_list, train_size=0.9, shuffle=True, random_state=0
    )
    # %%
    mesh_path = "./data/intraoral_scanners/demo_test/3_下颌8颗缺失/3_l.vtp"
    mesh, df = predict(mesh_path, model_script)
    pos = df[["x", "y", "z"]].values
    points_mean = mesh.points().mean(axis=0)
    pos = pos * coordinate_scale + points_mean
    df[["x", "y", "z"]] = pos
    mesh.write("test.vtp")
    df.to_csv("test.csv", index=False)
# %%
