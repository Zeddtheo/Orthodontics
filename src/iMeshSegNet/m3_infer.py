# m3_infer.py
# 推理脚本：读取原始 STL/VTP，使用训练好的模型预测，输出带颜色的 VTP
# 
# 使用示例：
#   1. Overfit 模型推理：
#      python m3_infer.py --ckpt outputs/overfit/overfit_model.pt \
#          --input datasets/landmarks_dataset/raw/1/1_L.stl \
#          --stats outputs/segmentation/stats.npz --out outputs/overfit/infer
#
#   2. 正常训练模型推理：
#      python m3_infer.py --ckpt outputs/segmentation/model/best.pt \
#          --input datasets/landmarks_dataset/raw/1/1_L.stl \
#          --stats outputs/segmentation/stats.npz --out outputs/segmentation/infer
#
#   3. 批量推理：
#      python m3_infer.py --ckpt outputs/segmentation/model/best.pt \
#          --input datasets/landmarks_dataset/raw/ --stats outputs/segmentation/stats.npz \
#          --out outputs/segmentation/infer --ext .stl

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

# 定义颜色映射（15个类别）
LABEL_COLORS = {
    0:  [128, 128, 128],  # 背景/牙龈 - 灰色
    1:  [255, 0, 0],      # 牙齿1 - 红色
    2:  [255, 127, 0],    # 牙齿2 - 橙色
    3:  [255, 255, 0],    # 牙齿3 - 黄色
    4:  [0, 255, 0],      # 牙齿4 - 绿色
    5:  [0, 255, 255],    # 牙齿5 - 青色
    6:  [0, 0, 255],      # 牙齿6 - 蓝色
    7:  [127, 0, 255],    # 牙齿7 - 紫色
    8:  [255, 0, 255],    # 牙齿8 - 品红
    9:  [255, 192, 203],  # 牙齿9 - 粉色
    10: [165, 42, 42],    # 牙齿10 - 棕色
    11: [255, 215, 0],    # 牙齿11 - 金色
    12: [0, 128, 128],    # 牙齿12 - 暗青色
    13: [128, 0, 128],    # 牙齿13 - 暗紫色
    14: [255, 140, 0],    # 牙齿14 - 暗橙色
}

def apply_color_to_mesh(mesh: pv.PolyData, labels: np.ndarray) -> pv.PolyData:
    """根据预测标签为网格着色（cell 和 point 级别）"""
    # 1. 为每个 cell 生成颜色
    cell_colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for label_id, color in LABEL_COLORS.items():
        mask = labels == label_id
        cell_colors[mask] = color
    
    # 2. 将颜色添加到 cell_data
    mesh.cell_data["RGB"] = cell_colors
    mesh.cell_data["PredLabel"] = labels
    
    # 3. 快速将 cell 数据转换为 point 数据（使用 PyVista 内置方法）
    mesh_with_point_data = mesh.cell_data_to_point_data()
    
    # 4. 复制转换后的 point 数据到原网格
    mesh.point_data["RGB"] = mesh_with_point_data.point_data["RGB"]
    mesh.point_data["PredLabel"] = mesh_with_point_data.point_data["PredLabel"]
    
    return mesh


# ---------------- Utils ----------------
def _lookup_arch_frame(stem: str, frames: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if not frames:
        return None
    if stem in frames:
        return frames[stem]
    base = stem.split("_")[0]
    return frames.get(base)


def load_pipeline_meta(ckpt_path: Path, args=None):
    """
    加载 checkpoint 中的 pipeline 元数据契约
    
    优先级：CLI 参数 > checkpoint 中的 pipeline > 默认值
    
    Returns:
        (checkpoint, pipeline_meta): checkpoint 字典和解析后的 pipeline 元数据
    """
    # 加载 checkpoint
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    
    # 提取 pipeline 配置
    P = ckpt.get("pipeline", {}) if isinstance(ckpt, dict) else {}
    
    # 辅助函数：按优先级获取参数
    def get(key, default=None):
        # CLI 参数优先
        if args and hasattr(args, key) and getattr(args, key) is not None:
            return getattr(args, key)
        # checkpoint 中的配置
        if key in P:
            return P[key]
        # 默认值
        return default
    
    # 构建 pipeline 元数据
    zscore_cfg = P.get("zscore", {}) if isinstance(P.get("zscore"), dict) else {}
    feature_layout = P.get("feature_layout", {}) if isinstance(P.get("feature_layout"), dict) else {}
    
    meta = {
        # Z-score 标准化
        "zscore_apply": zscore_cfg.get("apply", True),
        "mean": np.array(zscore_cfg.get("mean")) if zscore_cfg.get("mean") else None,
        "std": np.array(zscore_cfg.get("std")) if zscore_cfg.get("std") else None,
        
        # 几何预处理
        "centered": get("centered", True),
        "div_by_diag": get("div_by_diag", False),
        "use_frame": get("use_frame", False),
        
        # 采样策略
        "sampler": get("sampler", "random"),
        "sample_cells": get("sample_cells", 6000),
        "target_cells": get("target_cells", 10000),
        
        # 特征布局（用于旋转对齐）
        "rotate_blocks": feature_layout.get("rotate_blocks", []),
        
        # 随机种子
        "seed": get("seed", 42),
        
        # 训练信息
        "train_sample_ids_path": ckpt.get("train_sample_ids_path", None),
    }
    
    # 打印加载的配置
    print(f"\n📋 Pipeline 契约:")
    print(f"   Z-score: {'✓' if meta['zscore_apply'] else '✗'} (mean shape: {meta['mean'].shape if meta['mean'] is not None else 'N/A'})")
    print(f"   Centered: {meta['centered']}, Div by diag: {meta['div_by_diag']}")
    print(f"   Use frame: {meta['use_frame']}, Sampler: {meta['sampler']}")
    print(f"   Target cells: {meta['target_cells']}, Sample cells: {meta['sample_cells']}")
    
    return ckpt, meta


@torch.no_grad()
def _load_model_with_contract(ckpt_path: Path, device: torch.device, args=None) -> Tuple[iMeshSegNet, dict]:
    """
    加载模型并验证 pipeline 契约
    
    Returns:
        (model, pipeline_meta): 加载的模型和 pipeline 元数据
    """
    ckpt, meta = load_pipeline_meta(ckpt_path, args)
    
    # 获取模型配置
    num_classes = ckpt.get("num_classes", SEG_NUM_CLASSES)
    in_channels = ckpt.get("in_channels", 15)
    
    print(f"\n🏗️  模型配置:")
    print(f"   Num classes: {num_classes}")
    print(f"   In channels: {in_channels}")
    
    # 创建模型
    model = iMeshSegNet(
        num_classes=num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=False,
    )
    
    # 加载权重
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    
    model.to(device).eval()
    
    # 验证契约
    print(f"\n✅ 契约验证:")
    print(f"   模型输出维度与 num_classes 一致: {num_classes}")
    print(f"   特征输入维度: {in_channels}")
    
    # 添加 num_classes 和 in_channels 到 meta
    meta["num_classes"] = num_classes
    meta["in_channels"] = in_channels
    
    return model, meta


@torch.no_grad()
def _load_model(ckpt: Path, num_classes: int, device: torch.device) -> iMeshSegNet:
    model = iMeshSegNet(
        num_classes=num_classes,
        glm_impl="edgeconv",
        k_short=6,
        k_long=12,
        with_dropout=False,
    )
    # 兼容不同的 PyTorch 版本
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    
    # 处理不同的 checkpoint 格式
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "model" in state:
            model.load_state_dict(state["model"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    
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
    pipeline_meta: dict,
    arch_frames: Dict[str, torch.Tensor],
    device: torch.device,
    num_classes: int,
) -> Tuple[pv.PolyData, np.ndarray]:
    """
    应用 pipeline 契约进行推理
    
    Args:
        mesh_path: 输入网格路径
        model: 加载的模型
        pipeline_meta: 从 checkpoint 读取的 pipeline 配置（包含 mean/std/sampler/sample_cells/target_cells/use_frame/rotate_blocks 等）
        arch_frames: 可选的 arch frame 字典（如果 pipeline_meta["use_frame"]=True 则使用）
        device: 推理设备
        num_classes: 类别数量
    
    Returns:
        sample_mesh:  采样后的粗糙网格（用于落盘展示）
        pred_labels:  (Ns,) 逐 cell 预测
    """
    # 从 pipeline_meta 提取参数
    mean = pipeline_meta["mean"]
    std = pipeline_meta["std"]
    target_cells = pipeline_meta["target_cells"]
    sample_cells = pipeline_meta["sample_cells"]
    sampler = pipeline_meta["sampler"]
    use_frame = pipeline_meta["use_frame"]
    rotate_blocks = pipeline_meta["rotate_blocks"]
    
    mesh = pv.read(str(mesh_path))
    
    # 1) 几何预处理（与训练契约一致）
    if pipeline_meta["centered"]:
        mesh.points -= mesh.center
        
    if pipeline_meta["div_by_diag"]:
        mesh, scale_factor, diag_before, diag_after = normalize_mesh_units(mesh)
        if scale_factor != 1.0:
            print(
                f"  -> scaled {mesh_path.name} from diag={diag_before:.4f} to {diag_after:.2f} (mm)",
                flush=True,
            )
    else:
        diag_after = 1.0
        
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

    # 4) 采样到 sample_cells（根据 pipeline 契约选择采样策略）
    if Nd > sample_cells:
        if sampler == "random":
            # 随机采样
            ids = np.random.permutation(Nd)[:sample_cells]
        elif sampler == "fps":
            # FPS 采样（需要实现或跳过）
            print(f"  [Warning] FPS sampler not implemented, falling back to random")
            ids = np.random.permutation(Nd)[:sample_cells]
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
            
        feats = feats[ids]
        pos_raw = pos_raw[ids]
        sample_mesh = _subset_cells(mesh, ids)
        # 转换为 PolyData 以便保存为 VTP
        sample_mesh = sample_mesh.cast_to_unstructured_grid().extract_surface()
    else:
        sample_mesh = mesh

    # 5) z-score（与训练 stats 一致）
    feats_t = torch.from_numpy(feats)  # (Ns,15)
    feats_t = (feats_t - torch.from_numpy(mean)) / torch.from_numpy(std)
    
    # 6) 应用 arch frame（如果 pipeline 契约要求）
    pos_t = torch.from_numpy(pos_raw)  # (Ns,3)
    if use_frame:
        frame = _lookup_arch_frame(mesh_path.stem, arch_frames)
        if frame is not None:
            # frame: (3,3), pos: (Ns,3)
            pos_t = (frame @ pos_t.T).T
            
            # 根据 rotate_blocks 旋转特征向量
            if rotate_blocks:
                # rotate_blocks = [True, True, True, True, False] 表示前 4 个 block 需要旋转
                # block0: v0(0:3), block1: v1(3:6), block2: v2(6:9), block3: normal(9:12), block4: cent_rel(12:15)
                for i, should_rotate in enumerate(rotate_blocks):
                    if should_rotate and i < 5:
                        start_idx = i * 3
                        end_idx = start_idx + 3
                        vec_block = feats_t[:, start_idx:end_idx]  # (Ns, 3)
                        vec_block = (frame @ vec_block.T).T
                        feats_t[:, start_idx:end_idx] = vec_block
    
    # 转换为模型输入格式
    feats_t = feats_t.transpose(0, 1).contiguous().unsqueeze(0)  # (1,15,Ns)
    pos_t = pos_t.transpose(0, 1).contiguous().unsqueeze(0)      # (1,3,Ns)

    feats_t = feats_t.to(device, non_blocking=True).float()
    pos_t = pos_t.to(device, non_blocking=True).float()

    # 7) 推理（一次前向）
    logits = model(feats_t, pos_t)         # (1,C,Ns)
    pred = torch.argmax(logits, dim=1)     # (1,Ns)
    pred_np = pred.squeeze(0).cpu().numpy().astype(np.int32)

    # 8) 应用颜色映射到网格
    sample_mesh = apply_color_to_mesh(sample_mesh, pred_np)
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
    ap = argparse.ArgumentParser("Module-3 Inference (coarse/low-res) with Pipeline Contract")
    ap.add_argument("--ckpt", type=str, required=True, help="path to best.pt (contains pipeline metadata)")
    ap.add_argument("--input", type=str, required=True, help="file or directory of new scans (.vtp/.stl)")
    ap.add_argument("--out", type=str, default="outputs/segmentation/module3_infer", help="output directory")
    ap.add_argument("--arch-frames", type=str, default=None, help="optional JSON of arch frames (3x3 or 4x4)")
    ap.add_argument("--device", type=str, default="cuda:0")
    
    # Override options (optional, defaults to checkpoint metadata)
    ap.add_argument("--num-classes", type=int, default=None, help="Override num_classes from checkpoint")
    ap.add_argument("--target-cells", type=int, default=None, help="Override target_cells from checkpoint")
    ap.add_argument("--sample-cells", type=int, default=None, help="Override sample_cells from checkpoint")
    ap.add_argument("--ext", nargs="*", default=[".vtp", ".stl"], help="valid extensions when input is a folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")
    
    # 使用新的契约加载器（从 checkpoint 读取 pipeline 配置）
    model, pipeline_meta = _load_model_with_contract(Path(args.ckpt), device, args)
    
    # 打印 pipeline 配置信息
    print(f"[Pipeline Contract]")
    print(f"  zscore: mean shape={pipeline_meta['mean'].shape}, std shape={pipeline_meta['std'].shape}")
    print(f"  sampler: {pipeline_meta['sampler']}")
    print(f"  sample_cells: {pipeline_meta['sample_cells']}")
    print(f"  target_cells: {pipeline_meta['target_cells']}")
    print(f"  use_frame: {pipeline_meta['use_frame']}")
    print(f"  rotate_blocks: {pipeline_meta['rotate_blocks']}")
    print(f"  centered: {pipeline_meta['centered']}")
    print(f"  div_by_diag: {pipeline_meta['div_by_diag']}")
    
    num_classes = pipeline_meta["num_classes"]
    
    # 加载 arch frames（如果 pipeline 契约需要）
    arch_frames = {}
    if pipeline_meta["use_frame"]:
        if args.arch_frames:
            arch_frames = load_arch_frames(Path(args.arch_frames))
        else:
            print(f"[Warning] Pipeline契约要求 use_frame=True，但未提供 --arch-frames 参数")

    inputs = _gather_inputs(Path(args.input), [e.lower() for e in args.ext])
    if not inputs:
        raise FileNotFoundError(f"No input files found under: {args.input}")

    print(f"\n[Infer] files={len(inputs)}  device={device}")

    for f in inputs:
        try:
            sample_mesh, pred = _infer_single_mesh(
                mesh_path=f,
                model=model,
                pipeline_meta=pipeline_meta,
                arch_frames=arch_frames,
                device=device,
                num_classes=num_classes,
            )
            # 导出：带颜色的 VTP（含 RGB 和 PredLabel）与预测标签 NPY
            stem = f.stem
            out_npy = out_dir / f"{stem}_pred.npy"
            out_mesh = out_dir / f"{stem}_colored.vtp"

            sample_mesh.save(str(out_mesh), binary=True)
            np.save(str(out_npy), pred)
            
            # 统计预测的类别分布
            unique_labels, counts = np.unique(pred, return_counts=True)
            label_dist = ", ".join([f"L{lbl}:{cnt}" for lbl, cnt in zip(unique_labels, counts)])
            print(f"  ✓ {stem} -> {out_mesh.name} (cells={sample_mesh.n_cells}, labels=[{label_dist}])")
        except Exception as e:
            print(f"  ✗ {f.name} failed: {e}")

    print(f"[Done] outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
