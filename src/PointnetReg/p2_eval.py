from pathlib import Path
import argparse
import json
from typing import Optional, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg
from tooth_groups import TOOTH_GROUPS, get_group_teeth, get_all_tooth_ids, is_valid_tooth_id

DEFAULT_TOOTH_IDS = get_all_tooth_ids()


def _load_pos_and_gt(npz_path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "pos" in data:
            pos = data["pos"].astype(np.float32)
        else:
            x = data["x"]
            pos = (x[:, :3] if x.shape[-1] >= 3 else x[:, 0:3]).astype(np.float32)
        if pos.shape[0] < pos.shape[1]:
            pos = pos.T
        lm = data.get("landmarks")
        mask = data.get("loss_mask", data.get("mask"))
        if lm is None:
            return pos, None
        if mask is not None:
            lm = lm[mask.astype(bool)]
        return pos, lm.astype(np.float32)


def eval_one_target(
    root: str, 
    ckpt_root: str, 
    target: str, 
    mode: str,
    features: str = "all", 
    batch_size: int = 8, 
    workers: int = 2, 
    out_dir: str = "runs_eval"
) -> Dict:
    """评估单个目标（牙位或组）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建数据集配置
    if mode == "per_tooth":
        if not is_valid_tooth_id(target):
            raise ValueError(f"Invalid tooth_id: {target}")
        cfg = DatasetConfig(
            root=root,
            file_patterns=(f"*_{target}.npz", f"*_{target.upper()}.npz"),
            features=features,
            select_landmarks="active",
            ensure_constant_L=True
        )
        ckpt_path = Path(ckpt_root) / "per_tooth" / target / "best.pt"
    elif mode == "per_group":
        if target not in TOOTH_GROUPS:
            raise ValueError(f"Invalid group name: {target}")
        cfg = DatasetConfig(
            root=root,
            group=target,
            features=features,
            select_landmarks="active",
            ensure_constant_L=True
        )
        ckpt_path = Path(ckpt_root) / "per_group" / target / "best.pt"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    dataset = P0PointNetRegDataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_p0)

    sample = dataset[0]
    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        use_tnet=True,
        return_logits=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    coord_errors: list[np.ndarray] = []
    heatmap_mse: list[np.ndarray] = []
    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x)
            heatmap_mse.append(torch.mean((pred - y) ** 2, dim=(1, 2)).cpu().numpy())
            idx = torch.argmax(pred, dim=-1).cpu().numpy()
            for b, meta in enumerate(batch["meta"]):
                npz_path = meta["path"] if isinstance(meta, dict) else meta
                pos, lm_gt = _load_pos_and_gt(npz_path)
                lm_pred = pos[idx[b]]
                if lm_gt is None or lm_gt.shape[0] != lm_pred.shape[0]:
                    rows.append([Path(npz_path).name] + [np.nan] * lm_pred.shape[0])
                    continue
                err = np.linalg.norm(lm_pred - lm_gt, axis=1)
                coord_errors.append(err)
                rows.append([Path(npz_path).name] + err.tolist())

    coord_stack = np.concatenate(coord_errors, axis=0) if coord_errors else np.array([])
    mse_stack = np.concatenate(heatmap_mse, axis=0) if heatmap_mse else np.array([])
    # 分析包含的牙位信息（特别是对于组模式）
    analyzed_teeth = set()
    if mode == "per_group":
        for batch in loader:
            for meta in batch["meta"]:
                tooth_id = meta.get("tooth_id") if isinstance(meta, dict) else None
                if tooth_id:
                    analyzed_teeth.add(tooth_id)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_p0)  # 重置loader
    
    summary = {
        "target": target,
        "mode": mode,
        "num_samples": len(dataset),
        "analyzed_teeth": sorted(list(analyzed_teeth)) if analyzed_teeth else [target],
        "C": int(sample["x"].shape[0]),
        "L": int(sample["y"].shape[0]),
        "coord_mm": {
            "mean": float(np.nanmean(coord_stack)) if coord_stack.size else None,
            "median": float(np.nanmedian(coord_stack)) if coord_stack.size else None,
            "pck@1mm": float(np.nanmean(coord_stack <= 1.0)) if coord_stack.size else None,
            "pck@2mm": float(np.nanmean(coord_stack <= 2.0)) if coord_stack.size else None,
            "pck@3mm": float(np.nanmean(coord_stack <= 3.0)) if coord_stack.size else None,
        },
        "heatmap_mse": {
            "mean": float(np.nanmean(mse_stack)) if mse_stack.size else None,
            "median": float(np.nanmedian(mse_stack)) if mse_stack.size else None,
        },
    }

    output_dir = Path(out_dir) / mode / target
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_case_errors.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        header = ["case"] + [f"lm_{i:02d}" for i in range(sample["y"].shape[0])]
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(x) for x in row) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if summary["coord_mm"]["mean"] is not None:
        teeth_info = f" ({len(analyzed_teeth)} teeth)" if mode == "per_group" else ""
        print(
            f"[{target}]{teeth_info} samples={len(dataset)} mean={summary['coord_mm']['mean']:.4f} "
            f"median={summary['coord_mm']['median']:.4f} P@2mm={summary['coord_mm']['pck@2mm']:.3f}"
        )
    else:
        print(f"[{target}] samples={len(dataset)} (no GT coordinates)")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PointNetReg Evaluation with Group Support")
    
    # 基础参数
    parser.add_argument("--root", type=str, default="datasets/landmarks_dataset/cooked/p0/samples_consistent",
                       help="Dataset root directory")
    parser.add_argument("--ckpt_root", type=str, default="runs_pointnetreg",
                       help="Checkpoint root directory")
    parser.add_argument("--out_dir", type=str, default="runs_eval",
                       help="Output directory for evaluation results")
    
    # 评估模式
    parser.add_argument("--mode", type=str, choices=["per_tooth", "per_group"], default="per_tooth",
                       help="Evaluation mode: per_tooth or per_group")
    parser.add_argument("--tooth", type=str, default="t31",
                       help="Comma-separated tooth IDs (for per_tooth mode)")
    parser.add_argument("--group", type=str, choices=list(TOOTH_GROUPS.keys()),
                       help="Group name (for per_group mode)")
    
    # 其他参数
    parser.add_argument("--features", type=str, choices=["all", "xyz"], default="all",
                       help="Feature type")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # 解析目标列表
    if args.mode == "per_tooth":
        args.targets = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
        # 验证牙位ID
        invalid_teeth = [t for t in args.targets if not is_valid_tooth_id(t)]
        if invalid_teeth:
            raise ValueError(f"Invalid tooth IDs: {invalid_teeth}")
    elif args.mode == "per_group":
        if not args.group:
            raise ValueError("--group is required when mode is per_group")
        args.targets = [args.group]
    
    return args


def main():
    args = parse_args()
    
    print(f"🔍 PointNetReg Evaluation")
    print(f"   Mode: {args.mode}")
    print(f"   Root: {args.root}")
    print(f"   Checkpoint root: {args.ckpt_root}")
    print(f"   Targets: {args.targets}")
    print()
    
    summaries = []
    for target in args.targets:
        print(f"📊 Evaluating {target} ({args.mode})")
        summary = eval_one_target(
            args.root, 
            args.ckpt_root, 
            target, 
            args.mode,
            args.features, 
            args.batch_size, 
            args.workers, 
            args.out_dir
        )
        summaries.append(summary)
        print()
    
    # 计算总体统计
    means = [s["coord_mm"]["mean"] for s in summaries if s["coord_mm"]["mean"] is not None]
    if len(means) > 1:
        print(f"📈 Overall Statistics:")
        print(f"   Average mean error: {sum(means) / len(means):.4f} mm")
        
        p2mm_scores = [s["coord_mm"]["pck@2mm"] for s in summaries if s["coord_mm"]["pck@2mm"] is not None]
        if p2mm_scores:
            print(f"   Average P@2mm: {sum(p2mm_scores) / len(p2mm_scores):.3f}")
    
    # 保存整体报告
    overall_report = {
        "mode": args.mode,
        "targets": args.targets,
        "summaries": summaries,
        "overall": {
            "mean_error": sum(means) / len(means) if means else None,
            "mean_p2mm": sum(p2mm_scores) / len(p2mm_scores) if 'p2mm_scores' in locals() and p2mm_scores else None,
        }
    }
    
    report_path = Path(args.out_dir) / f"{args.mode}_overall_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Overall report saved to: {report_path}")


if __name__ == "__main__":
    main()
