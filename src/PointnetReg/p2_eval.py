from pathlib import Path
import argparse
import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

DEFAULT_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


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


def eval_one_tooth(root, ckpt_root, tooth_id, features="all", batch_size=8, workers=2, out_dir="runs_eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DatasetConfig(
        root=root,
        file_patterns=(f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"),
        features=features,
        select_landmarks="active",
    )
    dataset = P0PointNetRegDataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_p0)

    sample = dataset[0]
    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        use_tnet=True,
        return_logits=False,
    ).to(device)
    ckpt_path = Path(ckpt_root) / tooth_id / "best.pt"
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
    summary = {
        "tooth": tooth_id,
        "num_samples": len(dataset),
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

    out_dir = Path(out_dir) / tooth_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_case_errors.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        header = ["case"] + [f"lm_{i:02d}" for i in range(sample["y"].shape[0])]
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(x) for x in row) + "\n")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if summary["coord_mm"]["mean"] is not None:
        print(
            f"[{tooth_id}] samples={len(dataset)} mean={summary['coord_mm']['mean']:.4f} "
            f"median={summary['coord_mm']['median']:.4f} P@2mm={summary['coord_mm']['pck@2mm']:.3f}"
        )
    else:
        print(f"[{tooth_id}] samples={len(dataset)} (no GT coordinates)")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/p0_npz")
    parser.add_argument("--ckpt_root", type=str, default="runs_pointnetreg")
    parser.add_argument("--tooth", type=str, default="t31")
    parser.add_argument("--features", type=str, choices=["all", "xyz"], default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="runs_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    teeth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
    summaries = [eval_one_tooth(args.root, args.ckpt_root, tooth, args.features, args.batch_size, args.workers, args.out_dir) for tooth in teeth]
    means = [s["coord_mm"]["mean"] for s in summaries if s["coord_mm"]["mean"] is not None]
    if len(means) > 1:
        print(f"[ALL] avg mean error: {sum(means) / len(means):.4f} mm")


if __name__ == "__main__":
    main()
