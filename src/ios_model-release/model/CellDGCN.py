# %%
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv, InstanceNorm, global_max_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from model.jittable_functions import to_undirected


@torch.jit.script
def get_iden(k: int, batchsize: int, device: torch.device):
    iden = torch.eye(k, device=device).flatten().view(1, k * k).repeat(batchsize, 1)
    return iden


class LinearSequential(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSequential, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Linear(in_channels, out_channels)
        self.norm = InstanceNorm(out_channels)
        self.act = nn.LeakyReLU()

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = self.conv(x)
        x = self.norm(x, batch)
        x = self.act(x)
        return x


class STNkd(nn.Module):
    def __init__(self, k):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = LinearSequential(k, 64)
        self.conv2 = LinearSequential(64, 128)
        self.conv3 = LinearSequential(128, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.act = nn.LeakyReLU()

    def forward(self, x, batch):
        x = self.conv1(x, batch)
        x = self.conv2(x, batch)
        x = self.conv3(x, batch)
        x = global_max_pool(x, batch)  # pooling across points per graph batch
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        iden = get_iden(self.k, x.shape[0], x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class GraphSequential(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSequential, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = EdgeConv(
            nn=torch.nn.Linear(in_channels * 2, out_channels)
        ).jittable()
        self.norm = InstanceNorm(out_channels)
        self.act = nn.LeakyReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.norm(x, batch)
        x = self.act(x)
        return x


class GlobalAttention_jittable(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention_jittable, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

    def forward(self, x: Tensor, batch: Tensor, size: Optional[int] = None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # batch might not be in order
        size = batch.max().item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        return out


class PointDGCN(nn.Module):
    """
    General cell graph feature extractor, concatenating features from cell-level and jaw-level
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ):
        super(PointDGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv0 = LinearSequential(in_channels, hidden_channels)
        self.fstn = STNkd(k=hidden_channels)
        self.conv1 = GraphSequential(hidden_channels, hidden_channels)
        self.conv2 = GraphSequential(hidden_channels, hidden_channels)
        self.conv3 = GraphSequential(hidden_channels, hidden_channels)
        self.conv4 = GraphSequential(hidden_channels, hidden_channels)
        self.jaw_pool = GlobalAttention_jittable(
            gate_nn=torch.nn.Linear(hidden_channels * 4, 1)
        )

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.fstn.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        x = self.conv0(x, batch)
        trans_feat_batch = self.fstn(x, batch)  # [batch, k, k]
        trans_feat = trans_feat_batch[batch]  # [x.shape[0], k, k]
        x = torch.matmul(x.unsqueeze(1), trans_feat)  # [x.shape[0], 1, k]
        x = x.squeeze(1)  # [x.shape[0], k]
        x = self.conv1(x, edge_index, batch)
        xs = [x]
        x = self.conv2(x, edge_index, batch)
        xs.append(x)
        x = self.conv3(x, edge_index, batch)
        xs.append(x)
        x = self.conv4(x, edge_index, batch)
        xs.append(x)
        x = torch.cat(xs, dim=1)
        jaw_feature_pool = self.jaw_pool(x, batch)
        jaw_features_x = jaw_feature_pool[batch]
        x = torch.cat([x, jaw_features_x], dim=1)
        return x


class MultiHeadPredictor(nn.Module):
    """
    Multi-head predictor for teeth segmentation, cell-level classification and offset regression
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ):
        super(MultiHeadPredictor, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.offset_conv1 = LinearSequential(in_channels, hidden_channels)
        self.offset_conv2 = LinearSequential(hidden_channels, hidden_channels // 2)
        self.offset_conv3 = nn.Linear(hidden_channels // 2, 3)

        self.semantic_conv1 = LinearSequential(in_channels, hidden_channels)
        self.semantic_conv2 = GraphSequential(hidden_channels, hidden_channels)
        self.semantic_conv3 = LinearSequential(hidden_channels, hidden_channels // 2)
        self.semantic_conv4 = nn.Linear(hidden_channels // 2, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, batch: Tensor):
        offset = self.offset_conv1(x, batch)
        offset = self.offset_conv2(offset, batch)
        offset = self.offset_conv3(offset)
        pos_offset = pos + offset
        offset_edge_index = knn_graph(pos_offset, k=4, batch=batch, loop=True)
        offset_edge_index = to_undirected(offset_edge_index)
        x = self.semantic_conv1(x, batch)
        x = self.semantic_conv2(x, offset_edge_index, batch)
        x = self.semantic_conv3(x, batch)
        x = self.semantic_conv4(x)
        x = self.softmax(x)
        return x, offset


class CellDGCN(nn.Module):
    """
    Teeth segmentation model based on PointDGCN (which can serve as general cell graph feature extractor)
    """

    def __init__(
        self,
        in_channels: int,
        extractor_hidden_channels: int,
        predictor_hidden_channels: int,
        out_channels: int,
    ):
        super(CellDGCN, self).__init__()
        self.in_channels = in_channels
        self.extractor_hidden_channels = extractor_hidden_channels
        self.predictor_hidden_channels = predictor_hidden_channels
        self.out_channels = out_channels
        self.extractor = PointDGCN(in_channels, extractor_hidden_channels)
        self.predictor = MultiHeadPredictor(
            extractor_hidden_channels * 4 * 2,
            predictor_hidden_channels,
            out_channels,
        )

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, batch: Tensor):
        x = self.extractor(x, edge_index, batch)
        x, offset = self.predictor(x, pos, edge_index, batch)
        return x, offset


class MeshDGCN(nn.Module):
    """
    Multi-task model for teeth segmentation and landmark detection
    """

    def __init__(
        self,
        in_channels: int,
        extractor_hidden_channels: int,
        predictor_hidden_channels: int,
        detector_hidden_channels: int,
        out_channels: int,
        landmark_list: List[str],
    ):
        super(MeshDGCN, self).__init__()
        self.in_channels = in_channels
        self.extractor_hidden_channels = extractor_hidden_channels
        self.predictor_hidden_channels = predictor_hidden_channels
        self.detector_hidden_channels = detector_hidden_channels
        self.out_channels = out_channels
        self.landmark_list = landmark_list
        self.extractor = PointDGCN(
            in_channels, extractor_hidden_channels
        )  # cell-level feature extractor
        self.predictor = MultiHeadPredictor(
            extractor_hidden_channels * 4 * 2,
            predictor_hidden_channels,
            out_channels,
        )  # teeth index classifier for teeth segmentation
        self.heatmap_conv1 = LinearSequential(
            extractor_hidden_channels * 4 * 2, detector_hidden_channels
        )
        self.heatmap_conv2 = LinearSequential(
            detector_hidden_channels, detector_hidden_channels // 2
        )

        self.teeth_pool = GlobalAttention_jittable(
            gate_nn=torch.nn.Linear(detector_hidden_channels // 2, 1)
        )
        self.heatmap_conv3 = LinearSequential(
            detector_hidden_channels + self.out_channels, detector_hidden_channels // 2
        )
        self.heatmap_conv4 = nn.Linear(
            detector_hidden_channels // 2, len(landmark_list)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, batch_jaw: Tensor):
        x = self.extractor(x, edge_index, batch_jaw)
        prob, offset = self.predictor(x, pos, edge_index, batch_jaw)
        pred = torch.argmax(prob, dim=1)
        batch_teeth = batch_jaw * self.out_channels + pred
        x = self.heatmap_conv1(x, batch_jaw)
        x = self.heatmap_conv2(x, batch_jaw)
        teeth_feature_pool = self.teeth_pool(x, batch_teeth)
        teeth_features_x = teeth_feature_pool[batch_teeth]
        x = torch.cat([x, teeth_features_x, prob], dim=1)
        x = self.heatmap_conv3(x, batch_jaw)
        x = self.heatmap_conv4(x)
        heatmap = self.sigmoid(x)

        return prob, offset, heatmap

    @torch.jit.export
    def extract_feature(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch_jaw: Tensor,
        run_predictor: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        z = self.extractor(x, edge_index, batch_jaw)
        if not run_predictor:
            return z, None, None
        else:
            prob, offset = self.predictor(z, pos, edge_index, batch_jaw)
            return z, prob, offset

    @torch.jit.export
    def run_detector(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch_jaw: Tensor,
        prob: Optional[Tensor],
        pred: Optional[Tensor],
    ):
        # use refined predicted prob or labels as input
        if prob is None:
            prob, offset = self.predictor(z, pos, edge_index, batch_jaw)
        if pred is None:
            pred = torch.argmax(prob, dim=1)
        batch_teeth = batch_jaw * self.out_channels + pred
        z = self.heatmap_conv1(z, batch_jaw)
        z = self.heatmap_conv2(z, batch_jaw)
        teeth_feature_pool = self.teeth_pool(z, batch_teeth)
        teeth_features_z = teeth_feature_pool[batch_teeth]
        z = torch.cat([z, teeth_features_z, prob], dim=1)
        z = self.heatmap_conv3(z, batch_jaw)
        z = self.heatmap_conv4(z)
        heatmap = self.sigmoid(z)
        return heatmap


class MeshDGCN_torchscript(nn.Module):
    def __init__(
        self,
        model_torchscript: torch.jit._script.RecursiveScriptModule,
    ):
        super(MeshDGCN_torchscript, self).__init__()
        self.model_torchscript = model_torchscript

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch_jaw: Tensor,
        prob: Optional[Tensor],
        pred: Optional[Tensor],
    ):
        # use refined predicted prob or labels as input
        x = self.model_torchscript.extractor(x, edge_index, batch_jaw)
        if prob is None:
            prob, offset = self.model_torchscript.predictor(
                x, pos, edge_index, batch_jaw
            )
        if pred is None:
            pred = torch.argmax(prob, dim=1)
        batch_teeth = batch_jaw * 17 + pred
        x = self.model_torchscript.heatmap_conv1(x, batch_jaw)
        x = self.model_torchscript.heatmap_conv2(x, batch_jaw)
        teeth_feature_pool = self.model_torchscript.teeth_pool(x, batch_teeth)
        teeth_features_x = teeth_feature_pool[batch_teeth]
        x = torch.cat([x, teeth_features_x, prob], dim=1)
        x = self.model_torchscript.heatmap_conv3(x, batch_jaw)
        x = self.model_torchscript.heatmap_conv4(x)
        heatmap = self.model_torchscript.sigmoid(x)
        return heatmap


# %%
if __name__ == "__main__":
    import sys

    import numpy as np

    sys.path.append("../")
    from glob import glob

    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    from dataloader.CellGraph_dataset import CellGraph_Dataset

    train_list = np.load(
        "../data/intraoral_scanners/adult_max_vtp_5w_list_multitask.npy"
    )

    dataset = CellGraph_Dataset(
        data_list=train_list,
        sample_num=None,
        test=False,
        return_mesh=False,
        graph_k=8,
        transform=True,
        coordinate_scale=10,
        multitask=True,
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False
    )
    graph = next(iter(loader))

    # %%
    device = "cuda:8"
    landmark_list = ["DCP", "MCP", "LGP", "PGP"]
    model = MeshDGCN(
        in_channels=15,
        extractor_hidden_channels=64,
        predictor_hidden_channels=256,
        detector_hidden_channels=256,
        out_channels=17,
        landmark_list=landmark_list,
    )
    model.to(device)
    model = torch.jit.script(model)
    graph.to(device)
    x_pos = graph.x[:, :3]
    prob, offset, heatmap = model(graph.x, x_pos, graph.edge_index, graph.batch)
    # use two-steps to decouple predictor and detector
    z, prob, offset = model.extract_feature(
        graph.x, x_pos, graph.edge_index, graph.batch, return_prob=True
    )
    heatmap = model.forward_refined(z, x_pos, graph.edge_index, graph.batch, prob, None)
# %%
