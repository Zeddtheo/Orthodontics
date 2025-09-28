# module4_postprocess.py
# Minimal postprocess for iMeshSegNet pipeline (Module-4)
# 功能：将 Module-3 的低分辨率粗分割（*_coarse.vtp, cell_data["PredLabel"]）
#      回投到原始高分辨率网格，并做轻量平滑+小岛清理，输出高分辨率结果。
#
# 用法（单个文件）：
#   python module4_postprocess.py \
#       --orig datasets/new_arch/001_L.vtp \
#       --coarse runs/infer_coarse/001_L_coarse.vtp \
#       --out runs/post_full
#
# 用法（文件夹批量；按文件名 stem 匹配 *_coarse.vtp）：
#   python module4_postprocess.py \
#       --orig datasets/new_arch/ \
#       --coarse runs/infer_coarse/ \
#       --out runs/post_full --ext .vtp .stl
#
# 依赖：torch, numpy, pyvista   （与项目现有依赖一致）

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import pyvista as pv


# ---------------- 小工具 ----------------
def _gather_files(path: Path, exts: List[str]) -> List[Path]:
    if path.is_file():
        return [path]
    out = []
    for e in exts:
        out += sorted(path.rglob(f"*{e}"))
    return out

def _match_coarse_for(orig_file: Path, coarse_root: Path) -> Optional[Path]:
    # 约定：coarse 文件名为 "<stem>_coarse.vtp"
    c = coarse_root / f"{orig_file.stem}_coarse.vtp"
    return c if c.exists() else None

def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)

def _compute_cell_centers(mesh: pv.PolyData) -> np.ndarray:
    return mesh.cell_centers().points.astype(np.float32)  # (N,3)

def _compute_cell_normals(mesh: pv.PolyData) -> np.ndarray:
    m2 = mesh.copy()
    m2.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    return m2.cell_normals.astype(np.float32)  # (N,3)

def _knn_map_labels(src_pos: torch.Tensor,  # (Ns,3) coarse
                    src_lab: torch.Tensor,  # (Ns,)
                    dst_pos: torch.Tensor,  # (Nd,3) full
                    k: int = 3,
                    chunk: int = 8192) -> torch.Tensor:
    """
    将 coarse 标签回投到 full：对每个 dst，找 src 的 kNN，做距离加权多数投票。
    返回 (Nd,) int 标签。
    """
    Ns = src_pos.size(0)
    Nd = dst_pos.size(0)
    # 预防极端小样本
    k = min(k, Ns)
    votes = torch.zeros((Nd, src_lab.max().item() + 1), device=src_pos.device)
    # 分块以控内存
    for s in range(0, Nd, chunk):
        e = min(Nd, s + chunk)
        block = dst_pos[s:e]                       # (B,3)
        # 距离 (B,Ns)
        d = torch.cdist(block.unsqueeze(0), src_pos.unsqueeze(0), p=2).squeeze(0)
        knn_d, knn_i = torch.topk(d, k=k, dim=1, largest=False)
        knn_lab = src_lab.index_select(0, knn_i.view(-1)).view(-1, k)  # (B,k)
        # 距离加权：w = 1 / (eps + d)
        w = 1.0 / (1e-6 + knn_d)
        # 聚合到 votes
        for j in range(k):
            lj = knn_lab[:, j]
            wj = w[:, j]
            votes[s:e].index_add_(1, lj, wj.unsqueeze(1))
    pred = torch.argmax(votes, dim=1)
    return pred  # (Nd,)

def _build_knn_graph(pos: torch.Tensor, k: int = 8, chunk: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 kNN 图：indices (N,k) 与对应的权重 (N,k)。
    权重基于距离的 RBF：w = exp(-d^2 / (2*sigma^2))，sigma 取平均距离/√2。
    """
    N = pos.size(0)
    idx_all = []
    dist_all = []
    # 取一个小样本估 sigma
    sidx = torch.randperm(N, device=pos.device)[:min(4096, N)]
    dsamp = torch.cdist(pos.index_select(0, sidx).unsqueeze(0), pos.index_select(0, sidx).unsqueeze(0), p=2).squeeze(0)
    sigma = torch.mean(torch.topk(dsamp, k=min(k+1, dsamp.size(1)), largest=False).values[:, 1:]) + 1e-6
    sigma2 = (sigma * sigma * 0.5)
    # 分块 knn
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d = torch.cdist(pos[s:e].unsqueeze(0), pos.unsqueeze(0), p=2).squeeze(0)  # (B,N)
        knn_d, knn_i = torch.topk(d, k=k+1, dim=1, largest=False)  # +1 包含自身
        knn_d = knn_d[:, 1:]
        knn_i = knn_i[:, 1:]
        idx_all.append(knn_i)
        dist_all.append(knn_d)
    idx = torch.cat(idx_all, dim=0)               # (N,k)
    w = torch.exp(-(torch.cat(dist_all, dim=0) ** 2) / sigma2).clamp_min(1e-6)
    return idx, w

def _smooth_labels(pos: torch.Tensor, normals: torch.Tensor, labels: torch.Tensor,
                   iters: int = 2, k: int = 8, normal_gamma: float = 2.0) -> torch.Tensor:
    """
    轻量平滑：kNN 邻域加权多数投票；权重 = RBF(距离) * (cos夹角)^gamma。
    """
    idx, w_d = _build_knn_graph(pos, k=k)      # (N,k)
    N = pos.size(0)
    C = int(labels.max().item()) + 1
    lab = labels.clone()
    # 预先计算法向余弦权
    nbr_n = normals.index_select(0, idx.view(-1)).view(N, k, 3)
    cos = torch.clamp((nbr_n * normals.unsqueeze(1)).sum(-1), 0.0, 1.0)  # (N,k) 负相关抑制
    w = (w_d * (cos ** normal_gamma)).clamp_min(1e-6)
    for _ in range(max(0, iters)):
        votes = torch.zeros((N, C), device=pos.device)
        nbr_lab = lab.index_select(0, idx.view(-1)).view(N, k)
        for j in range(k):
            lj = nbr_lab[:, j]
            wj = w[:, j]
            votes.index_add_(1, lj, wj.unsqueeze(1))
        lab = torch.argmax(votes, dim=1)
    return lab

def _connected_components(idx: torch.Tensor) -> List[List[int]]:
    """
    kNN 图上的连通域（无向）。返回每个连通域的索引列表。
    """
    N, k = idx.shape
    # 构建无向邻接
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in idx[i].tolist():
            adj[i].append(j)
            adj[j].append(i)
    seen = [False] * N
    comps = []
    for i in range(N):
        if seen[i]:
            continue
        q = [i]; seen[i] = True; cur = [i]
        while q:
            u = q.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps

def _cleanup_small_islands(pos: torch.Tensor, labels: torch.Tensor,
                           k: int = 8, min_faces: int = 25) -> torch.Tensor:
    """
    小岛清理：对每个类别内的连通域，若面片数 < min_faces，则将其标签改为
    邻域多数（跨类邻域）。
    """
    new_lab = labels.clone()
    # 全图 kNN，后面也会用到
    idx, _ = _build_knn_graph(pos, k=k)
    # 每个类做一次
    classes = torch.unique(labels).tolist()
    for c in classes:
        if c == 0:  # 背景/牙龈可视需要时再清
            continue
        mask = (new_lab == c)
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            continue
        # 在子集上做连通域：用原图邻接筛子图
        sub_map = {int(i): t for t, i in enumerate(ids.tolist())}
        # 建子图邻接
        sub_adj = [[] for _ in range(len(ids))]
        for t, i in enumerate(ids.tolist()):
            for j in idx[i].tolist():
                if int(j) in sub_map:
                    sub_adj[t].append(sub_map[int(j)])
        # 连通域
        seen = [False] * len(ids)
        for s in range(len(ids)):
            if seen[s]:
                continue
            stack = [s]; seen[s] = True; comp = [s]
            while stack:
                u = stack.pop()
                for v in sub_adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
                        comp.append(v)
            # 判断大小
            if len(comp) < min_faces:
                # 将这团小岛的点改成跨类邻居多数标签
                island_ids = torch.tensor([ids[t] for t in comp], device=labels.device)
                # 找跨类邻居标签
                nbr = idx.index_select(0, island_ids).reshape(-1)
                nbr_lab = new_lab.index_select(0, nbr)
                # 过滤掉同类标签，取多数
                keep = (nbr_lab != c)
                if keep.any():
                    vals, counts = torch.unique(nbr_lab[keep], return_counts=True)
                    tgt = vals[counts.argmax()]
                    new_lab.index_copy_(0, island_ids, tgt.repeat(island_ids.numel()))
    return new_lab


# ---------------- 主流程 ----------------
def process_one(orig_path: Path, coarse_path: Path, out_dir: Path,
                device: torch.device, k_map: int, k_smooth: int,
                smooth_iters: int, normal_gamma: float, min_faces: int) -> Tuple[str, Dict]:
    # 1) 读取原始与粗分割
    full_mesh = pv.read(str(orig_path))
    full_mesh = full_mesh.triangulate()
    coarse_mesh = pv.read(str(coarse_path))
    if "PredLabel" not in coarse_mesh.cell_data:
        raise ValueError(f"{coarse_path.name} 缺少 cell_data['PredLabel']。")
    # 2) 取质心/法向
    pos_full = _compute_cell_centers(full_mesh)      # (Nd,3)
    nrm_full = _compute_cell_normals(full_mesh)      # (Nd,3)
    pos_coarse = _compute_cell_centers(coarse_mesh)  # (Ns,3)
    lab_coarse = coarse_mesh.cell_data["PredLabel"].astype(np.int64)  # (Ns,)

    # 3) 粗→全 回投（kNN 多数投票）
    with torch.no_grad():
        dev = device
        src_pos = _to_torch(pos_coarse, dev).float()
        src_lab = _to_torch(lab_coarse, dev).long()
        dst_pos = _to_torch(pos_full, dev).float()
        pred_full = _knn_map_labels(src_pos, src_lab, dst_pos, k=k_map)  # (Nd,)

        # 4) 轻量平滑：kNN 加权多数+法向相似
        normals = _to_torch(nrm_full, dev).float()
        pred_smooth = _smooth_labels(dst_pos, normals, pred_full, iters=smooth_iters,
                                     k=k_smooth, normal_gamma=normal_gamma)

        # 5) 小岛清理（最小面片阈值）
        pred_clean = _cleanup_small_islands(dst_pos, pred_smooth, k=max(k_smooth, 8), min_faces=min_faces)

    # 6) 写回高分辨率网格
    full_mesh.cell_data["PredLabel_full"] = pred_full.cpu().numpy().astype(np.int32)
    full_mesh.cell_data["PredLabel_smooth"] = pred_clean.cpu().numpy().astype(np.int32)

    stem = orig_path.stem
    out_vtp = out_dir / f"{stem}_full.vtp"
    full_mesh.save(str(out_vtp), binary=True)

    # 7) 摘要
    unique, counts = np.unique(full_mesh.cell_data["PredLabel_smooth"], return_counts=True)
    cls_hist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    summary = dict(
        file=str(orig_path.name),
        coarse=str(coarse_path.name),
        cells=int(full_mesh.n_cells),
        classes=cls_hist,
        notes="labels from coarse->full via kNN mapping; smoothed & small-island cleaned."
    )
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return stem, summary


def main():
    ap = argparse.ArgumentParser("Module-4 Postprocess (coarse->full, smooth & cleanup)")
    ap.add_argument("--orig", type=str, required=True, help="原始高分辨率牙弓路径（文件或目录）")
    ap.add_argument("--coarse", type=str, required=True, help="Module-3 输出目录或对应 coarse 文件")
    ap.add_argument("--out", type=str, default="outputs/segmentation/module4_post", help="输出目录")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ext", nargs="*", default=[".vtp", ".stl"])
    # 超参（都很轻量，默认即可）
    ap.add_argument("--k-map", type=int, default=3, help="coarse->full 回投的 kNN k 值")
    ap.add_argument("--k-smooth", type=int, default=8, help="平滑阶段的 kNN k 值")
    ap.add_argument("--smooth-iters", type=int, default=2, help="平滑迭代次数")
    ap.add_argument("--normal-gamma", type=float, default=2.0, help="法向相似权的幂指数")
    ap.add_argument("--min-faces", type=int, default=25, help="小岛清理的最小面片数阈值")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")

    orig_p = Path(args.orig)
    coarse_p = Path(args.coarse)

    # 构建任务列表
    if orig_p.is_file():
        coarse_file = coarse_p if coarse_p.is_file() else _match_coarse_for(orig_p, coarse_p)
        if coarse_file is None:
            raise FileNotFoundError(f"未找到匹配的 coarse 文件：{orig_p.stem}_coarse.vtp")
        pairs = [(orig_p, coarse_file)]
    else:
        orig_files = _gather_files(orig_p, [e.lower() for e in args.ext])
        pairs = []
        for of in orig_files:
            cf = _match_coarse_for(of, coarse_p) if coarse_p.is_dir() else coarse_p
            if cf is not None and cf.exists():
                pairs.append((of, cf))
        if not pairs:
            raise FileNotFoundError("未匹配到任何 (orig, coarse) 文件对。")

    print(f"[Post] pairs={len(pairs)} device={device}  k-map={args.k_map}  k-smooth={args.k_smooth} iters={args.smooth_iters}")

    for of, cf in pairs:
        try:
            stem, _ = process_one(of, cf, out_dir, device,
                                  args.k_map, args.k_smooth, args.smooth_iters,
                                  args.normal_gamma, args.min_faces)
            print(f"  ✓ {stem} -> saved *_full.vtp + *_summary.json")
        except Exception as e:
            print(f"  ✗ {of.name} failed: {e}")

    print(f"[Done] outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
