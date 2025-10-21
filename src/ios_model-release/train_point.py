# %%
import glob
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from dataloader.CellPoint_dataset import CellPoint_Dataset
from model.MeshSegPoint import MeshSegPoint
from util import logger
from util.metrics import *

# %%
if __name__ == "__main__":
    device = "cuda:3"
    torch.cuda.set_device(device)  # assign which gpu will be used (only linux works)
    jaw_type = "man"
    data_path_list = glob.glob(
        f"./data/intraoral_scanners/man/landmark_heatmap_std1_threshold4/*.vtp"
    )
    cell_num = "5w"
    model_name = f"{jaw_type}_{cell_num}"
    resume_checkpoint_path = "./checkpoint/landmark_detection_man_5w_20230705_01-19-41/weight/latest_checkpoint.tar"
    num_channels = 16  # number of features
    num_epochs = 500
    num_workers = 8
    graph_k = 12
    train_batch_size = 12
    val_batch_size = 12
    lr = 0.0005
    landmark_categories = [
        "CCT",
        "CTF",
        "DCP",
        "IEP",
        "LGP",
        "MCP",
        "PGP",
    ]
    # set model
    model = MeshSegPoint(
        class_names=landmark_categories,
        num_channels=num_channels,
        with_dropout=True,
        dropout_p=0.1,
    )
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs / 10, eta_min=lr / 100)
    # resume training if checkpoint exists
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resuming training from epoch {epoch}")
    experiment_name = f"landmark_detection_{model_name}"
    debug = False
    if not debug:
        checkpoint_root = "./checkpoint/"
        copy_file_list = [
            "./util/metrics.py",
            "./dataloader/CellGraph_dataset.py",
            "./dataloader/CellPoint_dataset.py",
            "./model/MeshSegPoint.py",
            "./train_point.py",
        ]
        writer, checkpoint_folder = logger.create_logger(
            experiment_name, checkpoint_root, copy_file_list
        )
        print(f"Checkpoint Folder = {checkpoint_folder}")
    train_list, test_list = train_test_split(
        data_path_list, train_size=0.9, shuffle=True, random_state=0
    )
    train_list, val_list = train_test_split(
        train_list, train_size=0.9, shuffle=True, random_state=0
    )
    # set dataset
    train_dataset = CellPoint_Dataset(
        data_list=np.array(train_list),
        return_mesh=False,
        graph_k=graph_k,
        transform=True,
        normalize_coordinate=True,
    )
    val_dataset = CellPoint_Dataset(
        data_list=np.array(val_list),
        return_mesh=False,
        graph_k=graph_k,
        transform=False,
        normalize_coordinate=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    train_log_var_name_list = ["loss"] + [
        f"{category}_mse_loss" for category in landmark_categories
    ]
    val_log_var_name_list = (
        ["loss", "total_pred_distance"]
        + [f"{category}_mse_loss" for category in landmark_categories]
        + [f"{category}_pred_distance" for category in landmark_categories]
    )

    best_val_loss = 1e8

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print("Training model...")
    for epoch in range(num_epochs):
        # training
        model.train()
        train_epoch_logger = logger.EpochLogger(name_list=train_log_var_name_list)
        for i_batch, graph in enumerate(train_loader):
            global_batch_idx = epoch * len(train_loader) + i_batch
            # send mini-batch to device
            graph.to(device)
            # zero the parameter gradients
            opt.zero_grad()
            # forward + backward + optimize
            teeth_batch = graph.batch * 17 + graph.cell_label.flatten()
            outputs = model(graph.x, graph.edge_index, graph.batch, teeth_batch)
            loss = torch.FloatTensor([0]).to(device)
            for idx, category in enumerate(landmark_categories):
                prob = outputs[0, idx]
                heatmap = graph[f"{category}_heatmap"]
                # symmetric landmarks (e.g. DCP and MCP)
                if category in ["DCP", "MCP"]:
                    roi_mask = (graph["DCP_heatmap"] + graph[f"MCP_heatmap"]) > 0
                elif category in ["PGP", "LGP"]:
                    roi_mask = (graph["PGP_heatmap"] + graph[f"LGP_heatmap"]) > 0
                else:
                    roi_mask = heatmap > 0
                mse_loss = weighted_mse_loss(
                    prob,
                    heatmap,
                    weights="balanced",
                    roi_mask=roi_mask,
                )
                train_epoch_logger.set_batch_log(
                    names=[f"{category}_mse_loss"],
                    values=[mse_loss.item()],
                    numbers=[1],
                )
                loss += mse_loss
            loss.backward()
            opt.step()
            scheduler.step(epoch + i_batch / len(train_loader))
            current_lr = scheduler.get_last_lr()[0]

            train_epoch_logger.set_batch_log(
                names=["loss"],
                values=[loss.item()],
                numbers=[1],
            )
            train_batch_log = train_epoch_logger.get_batch_log()
            train_batch_log.update({"lr": current_lr})
            if not debug:
                writer.add_scalars(
                    "train_batch", train_batch_log, global_step=global_batch_idx
                )
            else:
                print(train_batch_log)

        # record losses and metrics
        train_epoch_log = train_epoch_logger.get_epoch_log()
        if not debug:
            writer.add_scalars("train_epoch", train_epoch_log, global_step=epoch)

        # validation
        model.eval()
        with torch.no_grad():
            val_epoch_logger = logger.EpochLogger(name_list=val_log_var_name_list)
            for i_batch, graph in enumerate(val_loader):
                global_batch_idx = epoch * len(val_loader) + i_batch
                # send mini-batch to device
                graph.to(device)
                # forward + backward + optimize
                teeth_batch = graph.batch * 17 + graph.cell_label.flatten()
                outputs = model(graph.x, graph.edge_index, graph.batch, teeth_batch)
                total_mean_pred_distance = torch.FloatTensor([0]).to(device)
                loss = torch.FloatTensor([0]).to(device)
                for idx, category in enumerate(landmark_categories):
                    prob = outputs[0, idx]
                    heatmap = graph[f"{category}_heatmap"]
                    # symmetric landmarks (e.g. DCP and MCP)
                    if category in ["DCP", "MCP"]:
                        roi_mask = (graph["DCP_heatmap"] + graph[f"MCP_heatmap"]) > 0
                    elif category in ["PGP", "LGP"]:
                        roi_mask = (graph["PGP_heatmap"] + graph[f"LGP_heatmap"]) > 0
                    else:
                        roi_mask = heatmap > 0
                    mse_loss = weighted_mse_loss(
                        prob,
                        heatmap,
                        weights="balanced",
                        roi_mask=roi_mask,
                    )
                    loss += mse_loss
                    location = graph[f"{category}_location"].view(-1, 3)
                    # ignore missing landmark locations
                    location_mask = location.mean(dim=1) < 1e3
                    pred_location_idx_list = []
                    for i in torch.unique(teeth_batch):
                        batch_idx = torch.where(teeth_batch == i)[0]
                        batch_prob = prob[batch_idx]
                        pred_location_idx_batch = torch.argmax(batch_prob)
                        pred_location_idx = batch_idx[pred_location_idx_batch]
                        pred_location_idx_list.append(pred_location_idx.item())
                    pred_location = graph.pos[pred_location_idx_list]
                    mean_pred_distance = torch.sqrt(
                        torch.sum(
                            (pred_location[location_mask] - location[location_mask])
                            ** 2,
                            dim=1,
                        )
                    ).mean()
                    val_epoch_logger.set_batch_log(
                        names=[f"{category}_mse_loss", f"{category}_pred_distance"],
                        values=[mse_loss.item(), mean_pred_distance.item()],
                        numbers=[1] * 2,
                    )
                    total_mean_pred_distance += mean_pred_distance
                val_epoch_logger.set_batch_log(
                    names=["loss", "total_pred_distance"],
                    values=[
                        loss.item(),
                        total_mean_pred_distance.item(),
                    ],
                    numbers=[1] * 2,
                )
                val_batch_log = val_epoch_logger.get_batch_log()
                if not debug:
                    writer.add_scalars(
                        "val_batch", val_batch_log, global_step=global_batch_idx
                    )

            # record losses and metrics
            val_epoch_log = val_epoch_logger.get_epoch_log()
            if not debug:
                writer.add_scalars("val_epoch", val_epoch_log, global_step=epoch)
        if not debug:
            # save the checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                os.path.join(f"{checkpoint_folder}/weight", "latest_checkpoint.tar"),
            )
            # save the best model
            if best_val_loss > val_epoch_log["loss"]:
                best_val_loss = val_epoch_log["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                    },
                    os.path.join(f"{checkpoint_folder}/weight", "best_model.tar"),
                )

# %%
