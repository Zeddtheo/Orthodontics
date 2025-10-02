import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Small utils
# ------------------------------------------------------------
def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-NN graph with explicit self-loop for each center.
    Args:
        pos: (B, 3, N) positions in *the unified arch coordinate frame*.
        k:   number of neighbors (including self).
    Returns:
        idx: (B, N, k) neighbor indices for each center i.
    """
    B, C, N = pos.shape
    assert C == 3, "pos must be (B,3,N)"
    if k <= 0:
        raise ValueError("k must be positive for knn_graph")
    dists = torch.cdist(pos.transpose(1, 2), pos.transpose(1, 2), p=2)  # (B,N,N)
    knn_i = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False).indices  # (B,N,k)
    self_idx = torch.arange(N, device=pos.device).view(1, N, 1)
    knn_i[..., 0:1] = self_idx  # ensure self-loop present
    return knn_i


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


class GLM1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 6):
        super().__init__()
        self.k = k
        self.ec = EdgeConv(in_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        idx_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if idx_k is None:
            idx_k = knn_graph(pos, self.k)
        return self.ec(x, idx_k)


class GLMEdgeConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 512, k_short: int = 6, k_long: int = 12):
        super().__init__()
        if out_ch % 2 != 0:
            raise ValueError("out_ch must be even for GLMEdgeConv")
        self.k_short = k_short
        self.k_long = k_long
        mid = out_ch // 2
        self.ec_short = EdgeConv(in_ch, mid)
        self.ec_long = EdgeConv(in_ch, mid)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        idx_s: Optional[torch.Tensor] = None,
        idx_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if idx_s is None:
            idx_s = knn_graph(pos, self.k_short)
        if idx_l is None:
            idx_l = knn_graph(pos, self.k_long)
        xs = self.ec_short(x, idx_s)
        xl = self.ec_long(x, idx_l)
        feat = torch.cat([xs, xl], dim=1)
        return self.fusion_conv(feat)

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
        num_classes: int = 15,        # 背景 + 14 颗上颌牙
        glm_impl: str = "edgeconv",   # "edgeconv" | "sap"
        k_short: int = 6,             # 默认短邻域 k=6
        k_long: int = 12,            # 默认长邻域 k=12
        with_dropout: bool = False,
        dropout_p: float = 0.1,
        use_feature_stn: bool = True,  # 启用 64×64 特征变换
    ):
        super().__init__()
        self.num_classes = num_classes
        self.glm_impl = glm_impl
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p
        self.use_feature_stn = use_feature_stn
        self.k_short = k_short
        self.k_long = k_long

        # Feature transformation module (FTM) -> 64 channels
        self.ftm = nn.Sequential(
            nn.Conv1d(15, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.fstn = STNkd(k=64) if self.use_feature_stn else None

        if glm_impl == "edgeconv":
            self.glm1 = GLM1(64, 128, k=k_short)
            
            # MLP-2: 论文描述的中间特征提取模块（GLM-1 到 GLM-2 之间）
            self.mlp2 = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )
            
            # GLM-2 输入维度调整为 512（接收 MLP-2 的输出）
            self.glm2 = GLMEdgeConv(512, out_ch=512, k_short=k_short, k_long=k_long)
        else:
            self.glm1 = GLMSAP(64, 128)
            self.mlp2 = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )
            self.glm2 = GLMSAP(512, 512)

        self.gmp1 = nn.AdaptiveMaxPool1d(1)
        self.gmp2 = nn.AdaptiveMaxPool1d(1)
        self.gmp3 = nn.AdaptiveMaxPool1d(1)
        self.gmp4 = nn.AdaptiveMaxPool1d(1)
        self.gap5 = nn.AdaptiveAvgPool1d(1)

        # 密集融合：FTM(64) + GLM-1(128) + MLP-2(512) + GLM-2(512)
        fusion_in = 64 + 128 + 512 + 512
        self.fuse_fc = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        def _make_dropout() -> nn.Module:
            if self.with_dropout and self.dropout_p > 0:
                return nn.Dropout(self.dropout_p)
            return nn.Identity()

        # 分类器输入：FTM(64) + GLM-1(128) + MLP-2(512) + GLM-2(512) + Globals(128)
        classifier_in_ch = 64 + 128 + 512 + 512 + 128
        self.classifier = nn.Sequential(
            nn.Conv1d(classifier_in_ch, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            _make_dropout(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            _make_dropout(),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        a_s: Optional[torch.Tensor] = None,
        a_l: Optional[torch.Tensor] = None,
        idx_k6: Optional[torch.Tensor] = None,
        idx_k12: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, N = x.shape
        assert C == 15, "Input feature must be 15-D"

        x = self.ftm(x)
        if self.fstn is not None:
            trans = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)

        if self.glm_impl == "edgeconv":
            if idx_k6 is None:
                idx_k6 = knn_graph(pos, self.k_short)
            if idx_k12 is None:
                idx_k12 = knn_graph(pos, self.k_long)
            y1 = self.glm1(x, pos, idx_k=idx_k6)
            
            # MLP-2: 中间特征变换
            y2 = self.mlp2(y1)
            
            # GLM-2: 输入从 y1 改为 y2
            y3 = self.glm2(y2, pos, idx_s=idx_k6, idx_l=idx_k12)
        else:
            if a_s is None or a_l is None:
                raise ValueError("SAP mode requires adjacency matrices a_s and a_l")
            y1 = self.glm1(x, a_s, a_l)
            
            # MLP-2: 中间特征变换
            y2 = self.mlp2(y1)
            
            # GLM-2: 输入从 y1 改为 y2
            y3 = self.glm2(y2, a_s, a_l)

        assert x.shape[1] == 64, f"Expected FTM output 64 channels, got {x.shape[1]}"
        assert y1.shape[1] == 128, f"Expected GLM-1 output 128 channels, got {y1.shape[1]}"
        assert y2.shape[1] == 512, f"Expected MLP-2 output 512 channels, got {y2.shape[1]}"
        assert y3.shape[1] == 512, f"Expected GLM-2 output 512 channels, got {y3.shape[1]}"

        # 密集融合：从各阶段提取全局特征
        g1 = self.gmp1(x).squeeze(-1)       # FTM: 64
        g2 = self.gmp2(y1).squeeze(-1)      # GLM-1: 128
        g3 = self.gmp3(y2).squeeze(-1)      # MLP-2: 512
        g4 = self.gmp4(y3).squeeze(-1)      # GLM-2 (max): 512
        g5 = self.gap5(y3).squeeze(-1)      # GLM-2 (avg): 512 (可选，论文未明确)
        # 论文使用 g1+g2+g3+g4，总计 64+128+512+512=1216
        globals_feat = torch.cat([g1, g2, g3, g4], dim=1)
        globals_feat = self.fuse_fc(globals_feat)  # 1216 -> 128
        globals_feat = globals_feat.unsqueeze(-1).repeat(1, 1, N)

        # 密集连接：FTM + GLM-1 + MLP-2 + GLM-2 + Globals
        # 64 + 128 + 512 + 512 + 128 = 1344
        feat = torch.cat([x, y1, y2, y3, globals_feat], dim=1)
        assert feat.shape[1] == 1344, f"Expected fused feature dim 1344 (64+128+512+512+128), got {feat.shape[1]}"

        logits = self.classifier(feat)
        return logits

if __name__ == "__main__":
    # quick shape sanity test
    B, N = 2, 4096
    x = torch.randn(B, 15, N)
    pos = torch.randn(B, 3, N)
    net = iMeshSegNet(num_classes=15, glm_impl="edgeconv", k_short=6, k_long=12)
    with torch.cuda.amp.autocast(enabled=False):
        out = net(x, pos)
    print("logits:", out.shape)  # (B,15,N)


