# pointnetreg.py
# A lightweight PointNet-Reg backbone for ROI landmark heatmap regression.
# Focus: clean model only (no dataset/loss/decoder here).

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- utils -------------------------

def make_norm(ch: int, kind: str = "gn") -> nn.Module:
    """'bn' for BatchNorm1d, 'gn' for GroupNorm (more stable on small batches)."""
    if kind == "bn":
        return nn.BatchNorm1d(ch)
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ChannelDropout1d(nn.Dropout2d):
    """Channel-wise dropout for 1D feature maps: (B,C,N)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.unsqueeze(-1)).squeeze(-1)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------- TNet (xyz only) -------------------------

class TNet(nn.Module):
    """3x3 input alignment (applies only on xyz)."""
    def __init__(self, k: int = 3, norm: str = "gn"):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1);   self.bn1 = make_norm(64, norm)
        self.conv2 = nn.Conv1d(64, 128, 1); self.bn2 = make_norm(128, norm)
        self.conv3 = nn.Conv1d(128, 256, 1); self.bn3 = make_norm(256, norm)
        self.fc1 = nn.Linear(256, 128);     self.bn4 = make_norm(128, norm)
        self.fc2 = nn.Linear(128, 64);      self.bn5 = make_norm(64, norm)
        self.fc3 = nn.Linear(64, k * k)
        self.last_A: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))          # (B,256,N)
        g = torch.max(x, dim=2).values               # (B,256)
        g = F.relu(self.bn4(self.fc1(g)))
        g = F.relu(self.bn5(self.fc2(g)))
        A = self.fc3(g)                              # (B,9)
        I = torch.eye(self.k, device=A.device).view(1, -1).expand(B, -1)
        A = (A + I).view(B, self.k, self.k)         # residual → near identity
        self.last_A = A
        return A


# ------------------------- PointNet-Reg -------------------------

class PointNetReg(nn.Module):
    """
    PointNet-Reg backbone for heatmap regression on ROI meshes/point-sets.

    Args
    ----
    in_channels : int
        Input feature dims per point; first 3 dims must be xyz. (C >= 3)
    num_landmarks : int
        Number of landmark channels (L) for this tooth.
    use_tnet : bool
        If True, align xyz with a small TNet.
    norm : {'gn','bn'}
        Normalization kind. 'gn' is preferred for small batch sizes.
    dropout_p : float
        Channel-wise dropout prob in the head.
    return_logits : bool
        If True, return raw logits (preferred for BCEWithLogits); else sigmoid.

    Shapes
    ------
    Input  : (B, C, N)
    Output : (B, L, N)
    """
    def __init__(
        self,
        in_channels: int,
        num_landmarks: int,
        use_tnet: bool = True,
        norm: str = "gn",
        dropout_p: float = 0.0,
        return_logits: bool = True,
    ):
        super().__init__()
        assert in_channels >= 3, "in_channels must include xyz as first 3 dims."
        self.in_channels = in_channels
        self.num_landmarks = num_landmarks
        self.use_tnet = use_tnet
        self.return_logits = return_logits

        self.tnet = TNet(k=3, norm=norm) if use_tnet else None

        # local shared MLP: C -> 64 -> 128 -> 256
        self.conv1 = nn.Conv1d(in_channels, 64, 1);   self.bn1 = make_norm(64, norm)
        self.conv2 = nn.Conv1d(64, 128, 1);           self.bn2 = make_norm(128, norm)
        self.conv3 = nn.Conv1d(128, 256, 1);          self.bn3 = make_norm(256, norm)

        # global descriptor: 256 -> 512 -> GMP
        self.convg = nn.Conv1d(256, 512, 1);          self.bng = make_norm(512, norm)

        # fusion head: (256 local + 512 global = 768) -> 256 -> L
        self.head1 = nn.Conv1d(768, 256, 1);          self.bnh1 = make_norm(256, norm)
        self.dropout = ChannelDropout1d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.head2 = nn.Conv1d(256, num_landmarks, 1)

        self.act = nn.ReLU(inplace=True)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,N) with xyz in x[:, :3, :]
        B, C, N = x.shape
        xyz, extras = x[:, :3, :], x[:, 3:, :]

        if self.use_tnet:
            A = self.tnet(xyz)             # (B,3,3)
            xyz = torch.bmm(A, xyz)        # align xyz
            if extras.numel() > 0:
                rotated_chunks = []
                for start in range(0, extras.shape[1], 3):
                    chunk = extras[:, start:start + 3, :]
                    if chunk.shape[1] == 3:
                        rotated_chunks.append(torch.bmm(A, chunk))
                    else:
                        rotated_chunks.append(chunk)
                extras = torch.cat(rotated_chunks, dim=1)

        x = torch.cat([xyz, extras], dim=1)   # (B,C,N)

        # local
        x = self.act(self.bn1(self.conv1(x)))     # (B,64,N)
        x = self.act(self.bn2(self.conv2(x)))     # (B,128,N)
        x = self.act(self.bn3(self.conv3(x)))     # (B,256,N)
        feat_local = x

        # global
        g = self.act(self.bng(self.convg(x)))     # (B,512,N)
        g = torch.max(g, dim=2, keepdim=True).values
        g = g.repeat(1, 1, N)                     # (B,512,N)

        # fusion + head
        h = torch.cat([feat_local, g], dim=1)     # (B,768,N)
        h = self.act(self.bnh1(self.head1(h)))    # (B,256,N)
        h = self.dropout(h)
        logits = self.head2(h)                    # (B,L,N)

        return logits if self.return_logits else torch.sigmoid(logits)

    @property
    def last_tnet_matrix(self) -> Optional[torch.Tensor]:
        return None if not self.use_tnet else self.tnet.last_A
