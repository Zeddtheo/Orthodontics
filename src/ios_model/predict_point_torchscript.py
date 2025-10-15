# %%
import glob
import os

import torch
import vedo
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataloader.CellPoint_dataset import graph_preprocess
from model.MeshSegPoint import MeshSegPoint
from util.metrics import *

# %%
if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    landmark_categories = [
        "CCT",
        "CTF",
        "DCP",
        "IEP",
        "LGP",
        "MCP",
        "PGP",
    ]
    checkpoint_folder = "./checkpoint/landmark_detection_max_5w_20230706_10-41-46"
    weight_path = os.path.join(checkpoint_folder, "weight/best_model.tar")
    script_path = os.path.join(
        checkpoint_folder, "weight/max_landmark_detection_model_script.pt"
    )
    output_folder = os.path.join(checkpoint_folder, "script_output")
    if not os.path.exists(output_folder):
        os.system(f"mkdir {output_folder}")

    graph_k = 12
    data_path_list = []
    data_path_list = glob.glob(
        f"./data/intraoral_scanners/max/landmark_heatmap_std1_threshold4/*.vtp"
    )
    train_list, test_list = train_test_split(
        data_path_list, train_size=0.9, shuffle=True, random_state=0
    )
    train_list, val_list = train_test_split(
        train_list, train_size=0.9, shuffle=True, random_state=0
    )

    MeshSegPoint_checkpoint = torch.load(weight_path, map_location="cpu")

    num_channels = 16  # number of features
    num_workers = 8

    # set model
    model = MeshSegPoint(
        class_names=landmark_categories,
        num_channels=num_channels,
        with_dropout=True,
        dropout_p=0.1,
    )
    model.to(device)
    model.load_state_dict(MeshSegPoint_checkpoint["model_state_dict"])
    model.eval()
    model_script = torch.jit.script(model)
    torch.jit.save(model_script, script_path)
    # %%
    del model
    model_script = torch.jit.load(script_path)
    model_script.to(device)
    model_script.eval()
    # validation
    landmark_distance_dict = {name: [] for name in landmark_categories}
    with torch.no_grad():
        for idx in tqdm(range(len(test_list))):
            mesh_path = test_list[idx]
            mesh_name = mesh_path.split("/")[-1].split(".")[0]
            mesh = vedo.load(mesh_path)
            graph, mesh = graph_preprocess(
                mesh,
                transform=False,
                graph_k=graph_k,
                return_mesh=True,
                normalize_coordinate=True,
            )
            graph.num_graphs = 1
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            # send mini-batch to device
            graph.to(device)
            # forward + backward + optimize
            teeth_batch = graph.batch * 17 + graph.cell_label.flatten()
            outputs = model_script(graph.x, graph.edge_index, graph.batch, teeth_batch)
            for idx, category in enumerate(landmark_categories):
                prob = outputs[0, idx]
                mesh.celldata[f"{category}_prob"] = prob.cpu().numpy()
                location = graph[f"{category}_location"].view(-1, 3)
                # ignore missing landmark locations
                location_mask = location.mean(dim=1) < 1e3
                if location_mask.sum().item() > 0:
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
                    landmark_distance_dict[category].append(mean_pred_distance.item())
            # vedo.write(mesh, os.path.join(output_folder, "{}.vtp".format(mesh_name)))
    # %%
    distance_list = []
    for k, v in landmark_distance_dict.items():
        print(f"{k}: {np.mean(v):.3f}, {np.std(v):.3f}")
        distance_list.append(round(np.mean(v), 3))
    print(distance_list)
    all_landmark_distance = np.concatenate(list(landmark_distance_dict.values()))
    print(
        f"All: {np.mean(all_landmark_distance):.3f}, {np.std(all_landmark_distance):.3f}"
    )

# %%
