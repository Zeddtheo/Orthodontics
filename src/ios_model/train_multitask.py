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

from dataloader.CellGraph_dataset import CellGraph_Dataset
from model.CellDGCN import MeshDGCN
from util import logger
from util.metrics import *


def run_batch(graph, device, landmark_list, epoch_logger):
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
    dice_loss = Generalized_Dice_Loss(prob, one_hot_labels, class_weights)
    offset_distance_loss = distance_loss(offset_pred, offset_gt)
    loss_predictor = dice_loss + offset_distance_loss * offset_loss_weight
    loss_detector = torch.FloatTensor([0]).to(device)
    batch_teeth = graph.batch * 17 + pred
    teeth_idx_pred = torch.unique(batch_teeth)
    teeth_idx_pred = teeth_idx_pred[teeth_idx_pred > 0]  # exclude gingiva
    for idx, category in enumerate(landmark_list):
        category_heatmap_pred = heatmap_pred[:, idx]
        category_heatmap_gt = graph[f"{category}_heatmap"]
        # symmetric landmarks (e.g. DCP and MCP)
        if category in ["DCP", "MCP"]:
            roi_mask = (graph["DCP_heatmap"] + graph[f"MCP_heatmap"]) > 0
        elif category in ["PGP", "LGP"]:
            roi_mask = (graph["PGP_heatmap"] + graph[f"LGP_heatmap"]) > 0
        else:
            roi_mask = category_heatmap_gt > 0
        mse_loss = weighted_mse_loss(
            category_heatmap_pred,
            category_heatmap_gt,
            weights="balanced",
            roi_mask=roi_mask,
        )
        loss_detector += mse_loss

        landmark_pos_pred = torch.ones([graph.num_graphs * 17, 3]).to(device) * 1e8
        for i in teeth_idx_pred:
            batch_idx = torch.where(batch_teeth == i)[0]
            batch_prob = category_heatmap_pred[batch_idx]
            pred_location_idx_batch = torch.argmax(batch_prob)
            pred_location_idx = batch_idx[pred_location_idx_batch]
            landmark_pos_pred[i] = x_pos[pred_location_idx]

        valid_mask_pred = torch.zeros(graph.num_graphs * 17, dtype=torch.bool).to(
            device
        )
        valid_mask_pred[teeth_idx_pred] = True
        landmark_pos_gt = graph[f"{category}_pos"]
        valid_mask_gt = landmark_pos_gt.mean(dim=1) < 1e3
        valid_mask = valid_mask_pred & valid_mask_gt
        if valid_mask.sum() > 0:  # valid landmark exists
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
            )
            epoch_logger.set_batch_log(
                names=[f"{category}_mse_loss", f"{category}_pred_distance"],
                values=[mse_loss.item(), pred_distance.item()],
                numbers=[1] * 2,
            )
    loss = loss_predictor + loss_detector
    epoch_logger.set_batch_log(
        names=[
            "loss",
            "dsc",
            "dice_loss",
            "offset_distance_loss",
            "heatmap_mse_loss",
        ],
        values=[
            loss.item(),
            dsc.item(),
            dice_loss.item(),
            offset_distance_loss.item(),
            loss_detector.item(),
        ],
        numbers=[1] * 5,
    )
    return loss, epoch_logger


# %%
if __name__ == "__main__":
    # Train multi-task model for teeth segmentation and landmark detection
    # Create logging folder
    jaw_type = "max"
    cell_num = "5w"
    model_name = f"{jaw_type}_{cell_num}"
    experiment_name = f"multitask_{model_name}"
    debug = False
    if not debug:
        checkpoint_root = "./checkpoint/"
        copy_file_list = [
            "./util/metrics.py",
            "./dataloader/CellGraph_dataset.py",
            "./model/CellDGCN.py",
            "./train_multitask.py",
        ]
        writer, checkpoint_folder = logger.create_logger(
            experiment_name, checkpoint_root, copy_file_list
        )
        print(f"Checkpoint Folder = {checkpoint_folder}")
    # %%
    device = "cuda:2"
    torch.cuda.set_device(device)  # assign which gpu will be used (only linux works)

    data_path_list = np.load(
        f"./data/intraoral_scanners/adult_{jaw_type}_vtp_5w_list_multitask.npy"
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
    train_sample_num = None
    val_sample_num = None
    num_epochs = 500
    num_workers = 8
    graph_k = 8
    train_batch_size = 7
    val_batch_size = 7
    coordinate_scale = 10
    lr = 0.001
    offset_loss_weight = 0.5
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
    pretrained_path = None
    freeze_pretrain = False

    # set dataset
    train_dataset = CellGraph_Dataset(
        train_list,
        sample_num=train_sample_num,
        test=False,
        return_mesh=False,
        graph_k=graph_k,
        transform=True,
        coordinate_scale=coordinate_scale,
        multitask=True,
    )
    val_dataset = CellGraph_Dataset(
        val_list,
        sample_num=val_sample_num,
        test=False,
        return_mesh=False,
        graph_k=graph_k,
        transform=False,
        coordinate_scale=coordinate_scale,
        multitask=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # set model
    model = MeshDGCN(
        in_channels,
        extractor_hidden_channels,
        predictor_hidden_channels,
        detector_hidden_channels,
        out_channels,
        landmark_list,
    )
    # can load and freeze pretrained teeth segmentation model to stabilize training
    if pretrained_path is not None:
        pretrained_checkpoint = torch.load(pretrained_path)
        model.load_state_dict(pretrained_checkpoint["model_state_dict"], strict=False)
        print(f"Load pretrained model weights from {pretrained_path}")
        if freeze_pretrain:
            for param in model.extractor.parameters():
                param.requires_grad = False
            for param in model.predictor.parameters():
                param.requires_grad = False
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = CosineAnnealingWarmRestarts(
        opt, T_0=num_epochs // 5, T_mult=1, eta_min=lr / 100
    )

    log_var_name_list = (
        [
            "loss",
            "dsc",
            "dice_loss",
            "offset_distance_loss",
            "heatmap_mse_loss",
        ]
        + [f"{landmark}_mse_loss" for landmark in landmark_list]
        + [f"{landmark}_pred_distance" for landmark in landmark_list]
    )

    best_val_loss = 1e8

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print("Training model...")
    for epoch in range(num_epochs):
        # training
        model.train()
        train_epoch_logger = logger.EpochLogger(name_list=log_var_name_list)
        for i_batch, graph in enumerate(train_loader):
            global_batch_idx = epoch * len(train_loader) + i_batch
            loss, train_epoch_logger = run_batch(
                graph, device, landmark_list, train_epoch_logger
            )
            # zero the parameter gradients
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(epoch + i_batch / len(train_loader))
            current_lr = scheduler.get_last_lr()[0]

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
            val_epoch_logger = logger.EpochLogger(name_list=log_var_name_list)
            for i_batch, graph in enumerate(val_loader):
                global_batch_idx = epoch * len(val_loader) + i_batch
                loss, val_epoch_logger = run_batch(
                    graph, device, landmark_list, val_epoch_logger
                )
                val_batch_log = val_epoch_logger.get_batch_log()
                if not debug:
                    writer.add_scalars(
                        "val_batch", val_batch_log, global_step=global_batch_idx
                    )
                else:
                    print(val_batch_log)

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
