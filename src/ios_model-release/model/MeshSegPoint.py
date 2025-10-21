# %%
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class GlobalAttention_jittable(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention_jittable, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

    def forward(self, x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # batch might not be in order
        size = batch.max().item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        return out


class MeshSegPoint(nn.Module):
    """
    Landmark detection model
    """

    def __init__(
        self,
        class_names: List[str],
        num_channels: int = 15,
        with_dropout: bool = True,
        dropout_p: float = 0.5,
    ):
        super(MeshSegPoint, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        # GLM-1 (graph-contrained learning modulus)
        self.gcn_1 = EdgeConv(nn=torch.nn.Linear(64 * 2, 32), aggr="max").jittable()
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv2 = torch.nn.Conv1d(32 + 32, 64, 1)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 256, 1)
        # GLM-2 (graph-contrained learning modulus)
        self.gcn_2 = EdgeConv(nn=torch.nn.Linear(256 * 2, 128), aggr="max").jittable()
        self.gcn_3 = EdgeConv(nn=torch.nn.Linear(128 * 2, 128), aggr="max").jittable()
        self.glm2_conv1_1 = torch.nn.Conv1d(256, 128, 1)
        self.glm2_conv2 = torch.nn.Conv1d(128 * 3, 256, 1)
        # Jaw-level global pooling
        self.jaw_pool = GlobalAttention_jittable(gate_nn=torch.nn.Linear(256, 1))
        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(256 * 2, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        # Teeth_level glocal pooling
        self.teeth_pool = GlobalAttention_jittable(gate_nn=torch.nn.Linear(128, 1))
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        jaw_batch: torch.Tensor,
        teeth_batch: torch.Tensor,
    ):
        x = x.T.unsqueeze(0)
        # MLP-1
        x = F.relu(self.mlp1_conv1(x))
        x = F.relu(self.mlp1_conv2(x))
        # GLM-1
        sap = F.relu(self.gcn_1(x.squeeze(0).T, edge_index)).T.unsqueeze(0)
        x = F.relu(self.glm1_conv1_1(x))
        x = torch.cat([x, sap], dim=1)
        x = F.relu(self.glm1_conv2(x))
        # MLP-2
        x = F.relu(self.mlp2_conv1(x))
        x = F.relu(self.mlp2_conv2(x))
        x_mlp2 = F.relu(self.mlp2_conv3(x))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        sap_1 = F.relu(self.gcn_2(x_mlp2.squeeze(0).T, edge_index)).T.unsqueeze(0)
        sap_2 = F.relu(self.gcn_3(sap_1.squeeze(0).T, edge_index)).T.unsqueeze(0)
        x = F.relu(self.glm2_conv1_1(x_mlp2))
        x = torch.cat([x, sap_1, sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_conv2(x))
        # Jaw-level global pooling
        jaw_features = self.jaw_pool(x_glm2.squeeze(0).T, batch=jaw_batch)
        x_jaw_features = jaw_features[jaw_batch]
        x_glm2 = x_glm2 + x_jaw_features.T.unsqueeze(0)
        # Dense fusion
        x = torch.cat([x_mlp2, x_glm2], dim=1)
        # MLP-3
        x = F.relu(self.mlp3_conv1(x))
        x = F.relu(self.mlp3_conv2(x))
        x = F.relu(self.mlp3_conv3(x))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_conv4(x))
        # Teeth-level global pooling
        # TODO: add layernorm within teeth batch, normalize prob within each single teeth meshes
        teeth_features = self.teeth_pool(x.squeeze(0).T, batch=teeth_batch)
        x_teeth_features = teeth_features[teeth_batch]
        x = x + x_teeth_features.T.unsqueeze(0)
        # output
        x = self.output_conv(x)
        # TODO: maybe use softmax instead of sigmoid to ensure non-overlap between landmarks?
        # TODO: when generate heatmap, use smaller std to ensure non-overlap heatmaps
        x = self.sigmoid(x)
        return x


# %%
if __name__ == "__main__":
    import glob

    import numpy as np
    import pandas as pd
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    from CellPoint_dataset import CellPoint_Dataset

    device = "cuda:0"
    train_list = np.array(
        glob.glob("./data/intraoral_scanners/ios_landmarks/teeth_bbox_heatmaps/*.vtp")
    )

    dataset = CellPoint_Dataset(
        data_list=train_list,
        sample_num=-1,
        test=False,
        return_mesh=False,
        graph_k=8,
        transform=True,
    )
    loader = DataLoader(
        dataset, batch_size=3, shuffle=False, num_workers=8, drop_last=False
    )
    graph = next(iter(loader))
    graph.to(device)
    model = MeshSegPoint()
    model.to(device)
    output = model(graph)

# %%
