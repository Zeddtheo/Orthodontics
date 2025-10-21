# %%
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import vedo
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader.CellGraph_dataset import graph_preprocess
from model.CellDGCN import MeshDGCN
from util import logger
from util.metrics import *

# %%
if __name__ == "__main__":
    device = "cuda:5"
    torch.cuda.set_device(device)  # assign which gpu will be used (only linux works)

    data_path_list = np.load(
        f"./data/intraoral_scanners/adult_man_vtp_5w_list_multitask.npy"
    )
    train_list, test_list = train_test_split(
        data_path_list, train_size=0.9, shuffle=True, random_state=0
    )
    train_list, val_list = train_test_split(
        train_list, train_size=0.9, shuffle=True, random_state=0
    )

    in_channels = 15
    extractor_hidden_channels = 64
    predictor_hidden_channels = 256
    detector_hidden_channels = 256
    out_channels = 17
    graph_k = 8
    coordinate_scale = 10
    class_weights = torch.ones(out_channels).to(device, dtype=torch.float)
    landmark_list = [
        "CCT",
        "CTF",
        "DCP",
        "IEP",
        "LGP",
        "MCP",
        "PGP",
    ]
    checkpoint_path = "/raid/zichenwang/random/ios_model/checkpoint/multitask_man_5w_20230731_00-32-25/weight/best_model.tar"
    checkpoint = torch.load(checkpoint_path)

    # set model
    model = MeshDGCN(
        in_channels,
        extractor_hidden_channels,
        predictor_hidden_channels,
        detector_hidden_channels,
        out_channels,
        landmark_list,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # %%
    dsc_list = []
    landmark_distance_list_dict = {category: [] for category in landmark_list}
    with torch.no_grad():
        for path in tqdm(test_list[17:19]):
            mesh = vedo.load(path)
            graph, mesh = graph_preprocess(
                mesh,
                transform=False,
                graph_k=graph_k,
                test=False,
                sample_num=None,
                return_mesh=True,
                coordinate_scale=coordinate_scale,
                multitask=True,
            )
            graph.num_graphs = 1
            graph.batch = torch.zeros(graph.x.shape[0]).long().to(device)
            # send mini-batch to device
            graph.to(device)
            one_hot_labels = nn.functional.one_hot(
                graph.cell_label.flatten(), num_classes=out_channels
            )
            # forward + backward + optimize
            x_pos = graph.x[:, :3]
            prob, offset_pred, heatmap_pred = model(
                graph.x, x_pos, graph.edge_index, graph.batch
            )
            pred = torch.argmax(prob, dim=1)
            offset_gt = get_offset_gt(offset_pred, graph, out_channels)
            dsc = weighting_DSC(prob, one_hot_labels, class_weights)
            dsc_list.append(dsc.item())
            batch_teeth = graph.batch * 17 + pred
            teeth_idx_pred = torch.unique(batch_teeth)
            teeth_idx_pred = teeth_idx_pred[teeth_idx_pred > 0]  # exclude gingiva
            for idx, category in enumerate(landmark_list):
                category_heatmap_pred = heatmap_pred[:, idx]
                category_heatmap_gt = graph[f"{category}_heatmap"]
                landmark_pos_pred = (
                    torch.ones([graph.num_graphs * 17, 3]).to(device) * 1e8
                )
                for i in teeth_idx_pred:
                    batch_idx = torch.where(batch_teeth == i)[0]
                    batch_prob = category_heatmap_pred[batch_idx]
                    pred_location_idx_batch = torch.argmax(batch_prob)
                    pred_location_idx = batch_idx[pred_location_idx_batch]
                    landmark_pos_pred[i] = x_pos[pred_location_idx]
                valid_mask_pred = torch.zeros(
                    graph.num_graphs * 17, dtype=torch.bool
                ).to(device)
                valid_mask_pred[teeth_idx_pred] = True
                landmark_pos_gt = graph[f"{category}_pos"]
                valid_mask_gt = landmark_pos_gt.mean(dim=1) < 1e3
                valid_mask = valid_mask_pred & valid_mask_gt
                if valid_mask.sum() == 0:  # no valid landmark
                    pred_distance = np.nan
                else:
                    landmark_pos_gt = landmark_pos_gt[valid_mask]
                    landmark_pos_pred = landmark_pos_pred[valid_mask]
                    pred_distance = (
                        (
                            landmark_pos_pred * coordinate_scale
                            - landmark_pos_gt * coordinate_scale
                        )
                        .pow(2)
                        .sum(1)
                        .sqrt()
                        .mean(0)
                    ).item()
                landmark_distance_list_dict[category].append(pred_distance)
    # %%
    print(np.mean(dsc_list))
    for k, v in landmark_distance_list_dict.items():
        print(np.mean(np.array(v)[~np.isnan(v)]))
