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
from model.CellDGCN import CellDGCN
from util import logger
from util.metrics import *

# %%
if __name__ == "__main__":
    # Create logging folder
    jaw_type = "max"
    cell_num = "5w"
    model_name = f"{jaw_type}_{cell_num}"
    experiment_name = f"teeth_segmentation_{model_name}"
    debug = False
    if not debug:
        checkpoint_root = "./checkpoint/"
        copy_file_list = [
            "./util/metrics.py",
            "./dataloader/CellGraph_dataset.py",
            "./model/CellDGCN.py",
            "./train_seg.py",
        ]
        writer, checkpoint_folder = logger.create_logger(
            experiment_name, checkpoint_root, copy_file_list
        )
        print(f"Checkpoint Folder = {checkpoint_folder}")
    # %%
    device = "cuda:1"
    torch.cuda.set_device(device)  # assign which gpu will be used (only linux works)

    data_path_list = np.load(
        f"./data/intraoral_scanners/adult_{jaw_type}_vtp_5w_list.npy"
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
    out_channels = 17
    train_sample_num = None
    val_sample_num = None
    num_epochs = 500
    num_workers = 8
    graph_k = 8
    train_batch_size = 9
    val_batch_size = 9
    lr = 0.001
    offset_loss_weight = 0.5
    class_weights = torch.ones(out_channels).to(device, dtype=torch.float)
    # class_weights[[1, 2, 13, 14]] = 2
    # class_weights[[15, 16]] = 4  # wisdom teeth

    # set dataset
    train_dataset = CellGraph_Dataset(
        train_list,
        sample_num=train_sample_num,
        test=False,
        return_mesh=False,
        graph_k=graph_k,
        transform=True,
        coordinate_scale=10,
    )
    val_dataset = CellGraph_Dataset(
        val_list,
        sample_num=val_sample_num,
        test=False,
        return_mesh=False,
        graph_k=graph_k,
        transform=False,
        coordinate_scale=10,
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
    model = CellDGCN(
        in_channels,
        extractor_hidden_channels,
        predictor_hidden_channels,
        out_channels,
    )
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = CosineAnnealingWarmRestarts(
        opt, T_0=num_epochs // 5, T_mult=1, eta_min=lr / 100
    )
    log_var_name_list = [
        "loss",
        "dsc",
        "dice_loss",
        "offset_distance_loss",
    ]

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
            # send mini-batch to device
            graph.to(device)
            one_hot_labels = nn.functional.one_hot(
                graph.cell_label.flatten(), num_classes=out_channels
            )

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            x_pos = graph.x[:, :3]
            prob, offset_pred = model(graph.x, x_pos, graph.edge_index, graph.batch)
            offset_gt = get_offset_gt(offset_pred, graph, out_channels)
            dice_loss = Generalized_Dice_Loss(prob, one_hot_labels, class_weights)
            offset_distance_loss = distance_loss(offset_pred, offset_gt)
            loss = dice_loss + offset_distance_loss * offset_loss_weight
            dsc = weighting_DSC(prob, one_hot_labels, class_weights)
            loss.backward()
            opt.step()
            scheduler.step(epoch + i_batch / len(train_loader))
            current_lr = scheduler.get_last_lr()[0]

            train_epoch_logger.set_batch_log(
                names=log_var_name_list,
                values=[
                    loss.item(),
                    dsc.item(),
                    dice_loss.item(),
                    offset_distance_loss.item(),
                ],
                numbers=[1] * len(log_var_name_list),
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
            val_epoch_logger = logger.EpochLogger(name_list=log_var_name_list)
            for i_batch, graph in enumerate(val_loader):
                global_batch_idx = epoch * len(val_loader) + i_batch
                # send mini-batch to device
                graph.to(device)
                one_hot_labels = nn.functional.one_hot(
                    graph.cell_label.flatten(), num_classes=out_channels
                )
                # forward + backward + optimize
                x_pos = graph.x[:, :3]
                prob, offset_pred = model(graph.x, x_pos, graph.edge_index, graph.batch)
                offset_gt = get_offset_gt(offset_pred, graph, out_channels)
                dice_loss = Generalized_Dice_Loss(prob, one_hot_labels, class_weights)
                offset_distance_loss = distance_loss(offset_pred, offset_gt)
                loss = dice_loss + offset_distance_loss * offset_loss_weight
                dsc = weighting_DSC(prob, one_hot_labels, class_weights)
                val_epoch_logger.set_batch_log(
                    names=log_var_name_list,
                    values=[
                        loss.item(),
                        dsc.item(),
                        dice_loss.item(),
                        offset_distance_loss.item(),
                    ],
                    numbers=[1] * len(log_var_name_list),
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
