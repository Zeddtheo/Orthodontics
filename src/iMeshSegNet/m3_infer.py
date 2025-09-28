# module3_infer.py
# Minimal, single-file inference for iMeshSegNet (Module-3, coarse/low-res output)
# Usage examples:
#   python module3_infer.py --ckpt outputs/segmentation_model/best.pt \
#       --input datasets/new_arch/001_L.vtp --stats outputs/segmentation/stats.npz \
#       --out runs/infer_coarse
#
#   # 批量目录
#   python module3_infer.py --ckpt outputs/segmentation_model/best.pt \
#       --input datasets/new_arch/ --stats outputs/segmentation/stats.npz \
#       --out runs/infer_coarse --ext .vtp .stl

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import pyvista as pv

from imeshsegnet import iMeshSegNet
from m0_dataset import (
    SEG_NUM_CLASSES,
    extract_features,
    load_arch_frames,
    load_stats,
    normalize_mesh_units,
)


# ---------------- Utils ----------------
def _lookup_arch_frame(stem: str, frames: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if not frames:
        return None
    if stem in frames:
        return frames[stem]
    base = stem.split("_")[0]
    return frames.get(base)

@torch.no_grad()
def _load_model(ckpt: Path, num_classes: int, device: torch.device) -> iMeshSegNet:
    model = iMeshSegNet(
        num_classes=num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=False,
    )
    state = torch.load(str(ckpt), map_location="cpu")
    key = "model" if isinstance(state, dict) and "model" in state else \
          "state_dict" if isinstance(state, dict) and "state_dict" in state else None
    if key is None:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state[key])
    model.to(device).eval()
    return model

def _subset_cells(mesh: pv.PolyData, cell_ids: np.ndarray) -> pv.PolyData:
    # 从 decimated 网格中抽取若干 cell 组成一个低分辨率的粗糙网格
    sub = mesh.extract_cells(cell_ids.astype(np.int32))
    sub = sub.clean()  # 去掉悬挂拓扑等
    return sub

def _infer_single_mesh(
    mesh_path: Path,
    model: iMeshSegNet,
    mean: np.ndarray,
    std: np.ndarray,
    arch_frames: Dict[str, torch.Tensor],
    device: torch.device,
    target_cells: int,
    sample_cells: int,
    num_classes: int,
) -> Tuple[pv.PolyData, np.ndarray]:
    """
    Returns:
        sample_mesh:  采样后的粗糙网格（用于落盘展示）
        pred_labels:  (Ns,) 逐 cell 预测
    """
    mesh = pv.read(str(mesh_path))
    mesh.points -= mesh.center
    mesh, scale_factor, diag_before, diag_after = normalize_mesh_units(mesh)
    if scale_factor != 1.0:
        print(
            f"  -> scaled {mesh_path.name} from diag={diag_before:.4f} to {diag_after:.2f} (mm)",
            flush=True,
        )
    mesh = mesh.triangulate()

    # 2) 网格抽取（把 cell 数量压到 target_cells 附近）
    if mesh.n_cells > target_cells:
        reduction = 1.0 - (target_cells / float(mesh.n_cells))
        mesh = mesh.decimate_pro(reduction, feature_angle=45, preserve_topology=True)

    # 3) 特征提取（与训练一致：9点坐标 + 3法向 + 3相对位置 = 15D）
    feats = extract_features(mesh).astype(np.float32)        # (Nd, 15)
    pos_raw = mesh.cell_centers().points.astype(np.float32)  # (Nd, 3)
    scale_pos = diag_after if diag_after > 1e-6 else 1.0
    pos_raw = pos_raw / scale_pos
    Nd = feats.shape[0]

    # 4) 随机采样到 sample_cells（得到粗糙低分辨率网格）
    if Nd > sample_cells:
        ids = np.random.permutation(Nd)[:sample_cells]
        feats = feats[ids]
        pos_raw = pos_raw[ids]
        sample_mesh = _subset_cells(mesh, ids)
    else:
        sample_mesh = mesh

    # 5) z-score（与训练 stats 一致），以及 arch frame（若可用）
    feats_t = torch.from_numpy(feats)  # (Ns,15)
    feats_t = (feats_t - torch.from_numpy(mean)) / torch.from_numpy(std)
    feats_t = feats_t.transpose(0, 1).contiguous().unsqueeze(0)  # (1,15,Ns)

    pos_t = torch.from_numpy(pos_raw)  # (Ns,3)
    frame = _lookup_arch_frame(mesh_path.stem, arch_frames)
    if frame is not None:
        # frame: (3,3), pos: (Ns,3)
        pos_t = (frame @ pos_t.T).T
    pos_t = pos_t.transpose(0, 1).contiguous().unsqueeze(0)      # (1,3,Ns)

    feats_t = feats_t.to(device, non_blocking=True).float()
    pos_t = pos_t.to(device, non_blocking=True).float()

    # 6) 推理（一次前向）
    logits = model(feats_t, pos_t)         # (1,C,Ns)
    pred = torch.argmax(logits, dim=1)     # (1,Ns)
    pred_np = pred.squeeze(0).cpu().numpy().astype(np.int32)

    # 7) 写入到采样网格的 cell_data 以便直接可视化
    sample_mesh.cell_data["PredLabel"] = pred_np
    return sample_mesh, pred_np

def _gather_inputs(input_path: Path, exts: List[str]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in exts:
        files.extend(sorted(input_path.rglob(f"*{ext}")))
    return files


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("Module-3 Inference (coarse/low-res)")
    ap.add_argument("--ckpt", type=str, required=True, help="path to best.pt")
    ap.add_argument("--input", type=str, required=True, help="file or directory of new scans (.vtp/.stl)")
    ap.add_argument("--stats", type=str, required=True, help="stats.npz for z-score (same as training)")
    ap.add_argument("--out", type=str, default="outputs/segmentation/module3_infer", help="output directory")
    ap.add_argument("--arch-frames", type=str, default=None, help="optional JSON of arch frames (3x3 or 4x4)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-classes", type=int, default=SEG_NUM_CLASSES)
    ap.add_argument("--target-cells", type=int, default=25000)
    ap.add_argument("--sample-cells", type=int, default=8192)
    ap.add_argument("--ext", nargs="*", default=[".vtp", ".stl"], help="valid extensions when input is a folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")
    model = _load_model(Path(args.ckpt), num_classes=args.num_classes, device=device)

    mean, std = load_stats(Path(args.stats))
    arch_frames = load_arch_frames(Path(args.arch_frames)) if args.arch_frames else {}

    inputs = _gather_inputs(Path(args.input), [e.lower() for e in args.ext])
    if not inputs:
        raise FileNotFoundError(f"No input files found under: {args.input}")

    print(f"[Infer] files={len(inputs)}  device={device}  target_cells={args.target_cells}  sample_cells={args.sample_cells}")

    for f in inputs:
        try:
            sample_mesh, pred = _infer_single_mesh(
                mesh_path=f,
                model=model,
                mean=mean,
                std=std,
                arch_frames=arch_frames,
                device=device,
                target_cells=args.target_cells,
                sample_cells=args.sample_cells,
                num_classes=args.num_classes,
            )
            # 导出：低分辨率VTP（含 PredLabel）与 NPY
            stem = f.stem
            out_npy = out_dir / f"{stem}_pred.npy"
            # UnstructuredGrid needs .vtu whereas PolyData can use .vtp
            mesh_ext = ".vtp" if isinstance(sample_mesh, pv.PolyData) else ".vtu"
            out_mesh = out_dir / f"{stem}_coarse{mesh_ext}"

            sample_mesh.save(str(out_mesh), binary=True)
            np.save(str(out_npy), pred)
            print(f"  ✓ {stem} -> {out_mesh.name} (cells={sample_mesh.n_cells})")
        except Exception as e:
            print(f"  ✗ {f.name} failed: {e}")

    print(f"[Done] outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
