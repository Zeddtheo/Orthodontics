# module2_eval.py
# Minimal, single-file evaluator for iMeshSegNet (Module-2)
# Usage:
#   python module2_eval.py --ckpt runs/train/best.pt --split val --out runs/eval_val
# Options:
#   --device cuda:0 | cpu
#   --num-classes 33
#   --hd-max-points 5000         # HD每类最多采样点数（双向）
#   --dataset-fn m0_dataset:get_dataloaders  # 可按需更换

import argparse, csv, json, time, importlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- import your model (from Module-1) ----
from imeshsegnet import iMeshSegNet
from m0_dataset import SEG_NUM_CLASSES


# ---------------- Metric utils ----------------
def _safe_div(n, d):
    return float(n) / float(d) if d != 0 else 0.0

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    y_true, y_pred: (N,) int [0..C-1]
    return: (C, C) matrix; rows=true, cols=pred
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)
    idx = yt * num_classes + yp
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    cm += binc.reshape(num_classes, num_classes)
    return cm

def per_class_stats_from_cm(cm: np.ndarray) -> List[Dict]:
    """
    Returns list of dicts with TP/FP/FN/support and DSC/SEN/PPV per class.
    """
    C = cm.shape[0]
    res = []
    for c in range(C):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        support = cm[c, :].sum()
        dsc = _safe_div(2 * TP, 2 * TP + FP + FN)
        sen = _safe_div(TP, TP + FN)
        ppv = _safe_div(TP, TP + FP)
        res.append(dict(
            cls=c, TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
            support=int(support), DSC=dsc, SEN=sen, PPV=ppv
        ))
    return res

def macro_micro_from_stats(stats: List[Dict], ignore_bg: int = 0) -> Dict:
    # macro（排除背景类0）
    sel = [s for s in stats if s["cls"] != ignore_bg and s["support"] > 0]
    if len(sel) == 0:  # fallback
        sel = [s for s in stats if s["cls"] != ignore_bg]
    macro = {
        "macro_DSC": float(np.mean([s["DSC"] for s in sel])) if sel else 0.0,
        "macro_SEN": float(np.mean([s["SEN"] for s in sel])) if sel else 0.0,
        "macro_PPV": float(np.mean([s["PPV"] for s in sel])) if sel else 0.0,
    }
    # micro（合并非背景类）
    TP = sum(s["TP"] for s in stats if s["cls"] != ignore_bg)
    FP = sum(s["FP"] for s in stats if s["cls"] != ignore_bg)
    FN = sum(s["FN"] for s in stats if s["cls"] != ignore_bg)
    micro = {
        "micro_DSC": _safe_div(2 * TP, 2 * TP + FP + FN),
        "micro_SEN": _safe_div(TP, TP + FN),
        "micro_PPV": _safe_div(TP, TP + FP),
    }
    return {**macro, **micro}

@torch.no_grad()
def hausdorff_per_class(pos: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
                        num_classes: int, max_points: int = 5000) -> Dict[int, float]:
    """
    Symmetric Hausdorff distance per class, computed on cell centroids (in the same arch frame).
    Returns: dict {cls: hd_value_in_mm or np.nan if undefined}
    Notes: Subsample to ≤ max_points per set to keep memory stable.
    """
    # pos: (3,N), y_*: (N,)
    P = pos.transpose(0, 1).contiguous()  # (N,3)
    res = {}
    for c in range(1, num_classes):  # 忽略0类
        idx_t = (y_true == c).nonzero(as_tuple=False).flatten()
        idx_p = (y_pred == c).nonzero(as_tuple=False).flatten()
        if idx_t.numel() == 0 or idx_p.numel() == 0:
            res[c] = float("nan")
            continue
        # subsample
        if idx_t.numel() > max_points:
            idx_t = idx_t[torch.randperm(idx_t.numel())[:max_points]]
        if idx_p.numel() > max_points:
            idx_p = idx_p[torch.randperm(idx_p.numel())[:max_points]]
        A = P.index_select(0, idx_t)  # (Nt,3)
        B = P.index_select(0, idx_p)  # (Np,3)
        # directed distances
        d_ab = torch.cdist(A, B, p=2)  # (Nt,Np)
        d_ba = d_ab.transpose(0, 1)    # (Np,Nt)
        # Hausdorff
        h_ab = d_ab.min(dim=1)[0].max().item()
        h_ba = d_ba.min(dim=1)[0].max().item()
        res[c] = float(max(h_ab, h_ba))
    return res


# ---------------- Eval core ----------------
def load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> iMeshSegNet:
    model = iMeshSegNet(num_classes=num_classes, glm_impl="edgeconv",
                        k_short=6, k_long=12, with_dropout=False)
    state = torch.load(ckpt_path, map_location="cpu")
    key = "model" if "model" in state else "state_dict" if "state_dict" in state else None
    if key is None:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state[key])
    model.to(device).eval()
    return model

def import_dataset_fn(spec: str):
    """
    spec: 'm0_dataset:get_dataloaders' or 'my_ds:build_eval_loader'
    Will try to adapt to either returning (train, val, test) or a single loader.
    """
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn

def build_eval_loader(spec: str, split: str) -> DataLoader:
    mod_name, fn_name = spec.split(":")
    fn = import_dataset_fn(spec)

    if mod_name in {"m0_dataset", "iMeshSegNet.m0_dataset", "src.iMeshSegNet.m0_dataset"} and fn_name == "get_dataloaders":
        from m0_dataset import DataConfig

        config = DataConfig(augment=False, shuffle=False)
        train_loader, val_loader = fn(config)
        loaders = {"train": train_loader, "val": val_loader}
        return loaders.get(split, val_loader)

    try:
        out = fn()
        if isinstance(out, (list, tuple)) and len(out) == 3:
            m = {"train": out[0], "val": out[1], "test": out[2]}
            return m[split]
        if isinstance(out, DataLoader):
            return out
        raise RuntimeError("Dataset function returned unsupported object.")
    except TypeError:
        out = fn(split=split)
        if isinstance(out, DataLoader):
            return out
        raise

def eval_once(args):
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model = load_model(Path(args.ckpt), num_classes=args.num_classes, device=device)

    loader = build_eval_loader(args.dataset_fn, args.split)

    overall_cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    per_class_hd_collect: List[np.ndarray] = []  # 每例一个 hd 数组（按类索引）
    case_rows = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                (x, pos), y, extra = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                x, pos, y, extra = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                (x, pos), y = batch
                extra = {}
            else:
                raise RuntimeError("Unexpected batch format; expected ((x,pos), y, extra) or (x,pos,y,extra).")

            assert x.ndim == 3 and pos.ndim == 3 and y.ndim == 2, "Shapes must be x:(B,15,N), pos:(B,3,N), y:(B,N)"
            B = x.shape[0]

            x = x.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x, pos)                 # (B,C,N)
            pred = logits.argmax(1)                # (B,N)

            for b in range(B):
                case_id = str(extra.get("case_id", f"case_{time.time_ns()}")) if isinstance(extra, dict) else f"case_{time.time_ns()}"
                y_b = y[b]                         # (N,)
                p_b = pred[b]
                pos_b = pos[b]                     # (3,N)

                # confusion
                cm = compute_confusion(_to_numpy(y_b), _to_numpy(p_b), args.num_classes)
                overall_cm += cm

                # per-class HD
                hd_map = hausdorff_per_class(pos_b, y_b, p_b, args.num_classes, max_points=args.hd_max_points)
                hd_arr = np.full((args.num_classes,), np.nan, dtype=np.float32)
                for k, v in hd_map.items():
                    hd_arr[k] = v
                per_class_hd_collect.append(hd_arr)

                # per-case macro（排除背景）
                stats = per_class_stats_from_cm(cm)
                macro = macro_micro_from_stats(stats, ignore_bg=0)
                macro_dsc = macro["macro_DSC"]
                # per-case HD（非背景的均值）
                hd_vals = [v for c, v in hd_map.items() if c != 0 and np.isfinite(v)]
                macro_hd = float(np.mean(hd_vals)) if len(hd_vals) else float("nan")

                case_rows.append(dict(
                    case_id=case_id,
                    faces=int(y_b.numel()),
                    macro_DSC=macro_dsc,
                    macro_HD=macro_hd
                ))

    # 汇总
    stats_all = per_class_stats_from_cm(overall_cm)
    macro_micro = macro_micro_from_stats(stats_all, ignore_bg=0)

    # HD 汇总（按类）
    if per_class_hd_collect:
        hd_mat = np.stack(per_class_hd_collect, axis=0)  # (num_cases, C)
        per_class_hd = np.nanmean(hd_mat, axis=0)        # (C,)
    else:
        per_class_hd = np.full((args.num_classes,), np.nan, dtype=np.float32)

    # ---- 写出 CSV / JSON ----
    # per_class.csv
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cls", "support", "TP", "FP", "FN", "DSC", "SEN", "PPV", "HD"])
        for s in stats_all:
            c = s["cls"]
            w.writerow([c, s["support"], s["TP"], s["FP"], s["FN"],
                        f"{s['DSC']:.6f}", f"{s['SEN']:.6f}", f"{s['PPV']:.6f}",
                        f"{per_class_hd[c]:.6f}" if np.isfinite(per_class_hd[c]) else "nan"])

    # overall.csv
    with open(out_dir / "overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in macro_micro.items():
            w.writerow([k, f"{v:.6f}"])
        w.writerow(["acc_overall", f"{(overall_cm.trace() / max(1, overall_cm.sum())):.6f}"])

    # confusion.csv
    with open(out_dir / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["true\\pred"] + [str(i) for i in range(args.num_classes)]
        w.writerow(header)
        for i in range(args.num_classes):
            w.writerow([str(i)] + [int(v) for v in overall_cm[i]])

    # cases.csv
    with open(out_dir / "cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "faces", "macro_DSC", "macro_HD"])
        w.writeheader()
        for r in case_rows:
            w.writerow({**r, "macro_DSC": f"{r['macro_DSC']:.6f}",
                           "macro_HD":  "nan" if not np.isfinite(r["macro_HD"]) else f"{r['macro_HD']:.6f}"})

    # summary.json
    summary = dict(
        ckpt=str(args.ckpt),
        split=args.split,
        num_classes=args.num_classes,
        n_cases=len(case_rows),
        metrics=macro_micro,
        notes="macro excludes class 0 (background), Hausdorff computed on centroids with subsampling."
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Eval done] {args.split}  cases={len(case_rows)}  out={str(out_dir)}  time={time.time()-t0:.1f}s")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("Module-2 Eval (minimal single-file)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    p.add_argument("--out", type=str, default="outputs/segmentation/module2_eval")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-classes", type=int, default=SEG_NUM_CLASSES)
    p.add_argument("--hd-max-points", type=int, default=5000)
    p.add_argument("--dataset-fn", type=str, default="m0_dataset:get_dataloaders",
                   help="Function to build eval DataLoader. "
                        "Default tries m0_dataset.get_dataloaders() and picks the split.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    eval_once(args)
