import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Small utils
# ------------------------------------------------------------
def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-NN graph indices for each sample in a batch.
    Args:
        pos: (B, 3, N) positions in *the unified arch coordinate frame*.
        k:   number of neighbors.
    Returns:
        idx: (B, N, k) neighbor indices for each center i.
    """
    B, C, N = pos.shape
    assert C == 3, "pos must be (B,3,N)"
    # pairwise distance: (B, N, N)
    # NOTE: for large N you may replace with faiss/approx or batched chunking
    dists = torch.cdist(pos.transpose(1, 2), pos.transpose(1, 2), p=2)  # (B,N,N)
    # exclude self by taking topk on negative distance or mask diag
    # take smallest k+1 then drop self
    knn_d, knn_i = torch.topk(dists, k=k + 1, dim=-1, largest=False, sorted=False)
    # drop the self index (distance 0)
    idx = knn_i[..., 1:]
    return idx  # (B,N,k)


def index_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather features by neighbor indices.
    Args:
        x:   (B, C, N)
        idx: (B, N, k)
    Returns:
        out: (B, C, N, k)
    """
    B, C, N = x.shape
    k = idx.shape[-1]
    # expand batch dim for gather
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B,C,N,k)
    # gather along last dim from N
    x_expand = x.unsqueeze(-1).expand(-1, -1, -1, k)     # (B,C,N,k)
    out = torch.gather(x_expand, 2, idx_expand)
    return out


# ------------------------------------------------------------
# Spatial/Feature Transformers (STN3d / STNkd)
# ------------------------------------------------------------
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 9)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):  # (B,3,N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(B, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):  # (B,k,N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# ------------------------------------------------------------
# EdgeConv-based GLM (iMeshSegNet version)
#   h_θ([x_i, x_j - x_i]) with k-NN graph (kshort/klong)
# ------------------------------------------------------------
class EdgeConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """EdgeConv forward.
        Args:
            x:   (B, C, N) features
            idx: (B, N, k) neighbor indices
        Returns:
            out: (B, out_ch, N)
        """
        B, C, N = x.shape
        k = idx.size(-1)
        x_j = index_points(x, idx)           # (B,C,N,k)
        x_i = x.unsqueeze(-1).expand(-1, -1, -1, k)  # (B,C,N,k)
        e_ij = torch.cat([x_i, x_j - x_i], dim=1)    # (B,2C,N,k)
        out = self.mlp(e_ij)                 # (B,out_ch,N,k)
        out = torch.max(out, dim=-1)[0]      # (B,out_ch,N) max over k
        return out


class GLMEdgeConv(nn.Module):
    """Graph-constrained Learning Module (EdgeConv flavor).
    It computes two EdgeConv streams with different neighborhoods (short/long),
    then concatenates them.
    """
    def __init__(self, in_ch: int, out_ch: int, k_short: int = 6, k_long: int = 12):
        super().__init__()
        self.k_short = k_short
        self.k_long = k_long
        self.ec_short = EdgeConv(in_ch, out_ch // 2)
        self.ec_long = EdgeConv(in_ch, out_ch // 2)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B,3,N) for kNN graph construction
        idx_s = knn_graph(pos, self.k_short)  # (B,N,ks)
        idx_l = knn_graph(pos, self.k_long)   # (B,N,kl)
        xs = self.ec_short(x, idx_s)          # (B,out/2,N)
        xl = self.ec_long(x, idx_l)           # (B,out/2,N)
        return torch.cat([xs, xl], dim=1)     # (B,out,N)


# ------------------------------------------------------------
# (Optional) Legacy SAP-based GLM for ablation/backward-compat
# ------------------------------------------------------------
class GLMSAP(nn.Module):
    """Legacy GLM using symmetric average pooling over given adjacency.
    Expected to be slower/more memory hungry; kept for A/B tests.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch // 2, 1)
        self.bn1 = nn.BatchNorm1d(out_ch // 2)
        self.conv2 = nn.Conv1d(in_ch, out_ch // 2, 1)
        self.bn2 = nn.BatchNorm1d(out_ch // 2)

    def forward(self, x: torch.Tensor, a_s: torch.Tensor, a_l: torch.Tensor) -> torch.Tensor:
        # x: (B,C,N); a_*: (B,N,N) row-stochastic adjacency (short/long)
        xs = torch.bmm(x, a_s.transpose(1, 2))  # (B,C,N)
        xl = torch.bmm(x, a_l.transpose(1, 2))  # (B,C,N)
        xs = F.relu(self.bn1(self.conv1(xs)))
        xl = F.relu(self.bn2(self.conv2(xl)))
        return torch.cat([xs, xl], dim=1)  # (B,out,N)


# ------------------------------------------------------------
# iMeshSegNet (EdgeConv-enabled)
#   Input  : (B, 15, N) features per cell
#   Pos    : (B, 3,  N) cell centroids (for kNN)
#   Output : (B, nclass, N) logits per cell
# ------------------------------------------------------------
class iMeshSegNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 66,        # dataset-specific label count (0-65)
        glm_impl: str = "edgeconv",   # 'edgeconv' | 'sap'
        k_short: int = 6,
        k_long: int = 12,
        with_dropout: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.glm_impl = glm_impl
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 (feature lifting to 64)
        self.mlp1_conv1 = nn.Conv1d(15, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_conv2 = nn.Conv1d(64, 64, 1)
        self.mlp1_bn2 = nn.BatchNorm1d(64)

        # Feature transform (STNkd on 64-d)
        self.fstn = STNkd(k=64)

        # GLM-1
        if glm_impl == "edgeconv":
            self.glm1 = GLMEdgeConv(64, 128, k_short=k_short, k_long=k_long)  # -> 128
        else:
            self.glm1 = GLMSAP(64, 128)

        # MLP-2
        self.mlp2_conv1 = nn.Conv1d(128, 128, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(128)
        self.mlp2_conv2 = nn.Conv1d(128, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)

        # GLM-2
        if glm_impl == "edgeconv":
            self.glm2 = GLMEdgeConv(128, 128, k_short=k_short, k_long=k_long)  # -> 128
        else:
            self.glm2 = GLMSAP(128, 128)

        # Dense fusion: concat pooled features from stages + per-point feat
        fusion_in = 64 + 128 + 128 + 128  # g1+g_glm1+g_mlp2+g_glm2
        self.fuse_fc = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # MLP-3 (point-wise classifier)
        self.classifier = nn.Sequential(
            nn.Conv1d(128 + 128, 256, 1),  # concat(globals, last local=glm2)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p) if with_dropout else nn.Identity(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p) if with_dropout else nn.Identity(),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        a_s: Optional[torch.Tensor] = None,
        a_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward
        Args:
            x:   (B, 15, N) input features (z-score normalized)
            pos: (B, 3,  N) cell centroids in unified arch frame (for kNN)
            a_s/a_l: optional (B,N,N) adjacencies for legacy SAP mode
        Returns:
            logits: (B, num_classes, N)
        """
        B, C, N = x.shape
        assert C == 15, "Input feature must be 15-D"
        # ----- MLP-1 -----
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))   # (B,64,N)
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))   # (B,64,N)

        # ----- Feature Transform -----
        x_t = x.transpose(2, 1)                         # (B,N,64)
        trans = self.fstn(x)                            # (B,64,64)
        x_t = torch.bmm(x_t, trans)                     # (B,N,64)
        x = x_t.transpose(2, 1)                         # (B,64,N)
        g1 = torch.max(x, dim=-1)[0]                    # (B,64)

        # ----- GLM-1 -----
        if self.glm_impl == "edgeconv":
            y1 = self.glm1(x, pos)                      # (B,128,N)
        else:
            assert a_s is not None and a_l is not None, "SAP mode requires a_s/a_l"
            y1 = self.glm1(x, a_s, a_l)
        g2 = torch.max(y1, dim=-1)[0]                   # (B,128)

        # ----- MLP-2 -----
        y2 = F.relu(self.mlp2_bn1(self.mlp2_conv1(y1))) # (B,128,N)
        y2 = F.relu(self.mlp2_bn2(self.mlp2_conv2(y2))) # (B,128,N)
        g3 = torch.max(y2, dim=-1)[0]                   # (B,128)

        # ----- GLM-2 -----
        if self.glm_impl == "edgeconv":
            y3 = self.glm2(y2, pos)                     # (B,128,N)
        else:
            y3 = self.glm2(y2, a_s, a_l)
        g4 = torch.max(y3, dim=-1)[0]                   # (B,128)

        # ----- Dense Fusion (global) -----
        g = torch.cat([g1, g2, g3, g4], dim=1)          # (B,448)
        g = self.fuse_fc(g)                              # (B,128)
        g = g.unsqueeze(-1).repeat(1, 1, N)             # (B,128,N)

        # ----- Classifier (point-wise) -----
        feat = torch.cat([y3, g], dim=1)                # (B,128+128,N)
        logits = self.classifier(feat)                  # (B,num_classes,N)
        return logits


if __name__ == "__main__":
    # quick shape sanity test
    B, N = 2, 4096
    x = torch.randn(B, 15, N)
    pos = torch.randn(B, 3, N)
    net = iMeshSegNet(num_classes=66, glm_impl="edgeconv", k_short=6, k_long=12)
    with torch.cuda.amp.autocast(enabled=False):
        out = net(x, pos)
    print("logits:", out.shape)  # (B,15,N)
